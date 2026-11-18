//! How often each layer of the head sits on its floor or its ceiling, counted inside the engine.
//!
//! **A build, not an option.** This module exists only under the `clip_stats` feature. A build
//! without it has no counter and no branch for one: `value_from_sums` is the function it was. A
//! build with it counts every evaluation the search makes, so it is slower and is for measuring,
//! not for playing. It is a feature rather than a debug build on purpose: a debug build is an
//! order of magnitude slower, and a search under a time limit visits other positions when it is
//! slower, so what it counted would not be what the release engine sees. The feature works in
//! either profile, and the two count the same (the arithmetic is the same; see the test in
//! `tests/clip_stats.rs`).
//!
//! **What is counted.** The values the head works on, after the clip, which is what the arithmetic
//! of this engine produced: the f16 tables widened, the first layer on `i16` or `i8` weights
//! against a rounded input, and dequantised before it is clipped. That is the difference from
//! the trainer's `activations` command, which runs the f32 net. A unit is at the floor when it is
//! `<= 0` and at the ceiling when it is `>= 1`, the trainer's definitions, so the two commands'
//! lines read side by side. The layers are the clipped blocks (power sums, pair table, second
//! moment, square field, the two perspectives laid end to end, the side to move first) and the two
//! hidden layers.
//!
//! **What it does not count.** There is no integer saturation to count in this head: the `i16`
//! layer's total is bounded to fit an `i32` (`qdense.rs`), and the `i8` layer's input stops at 127
//! so that `vpmaddubsw` cannot saturate (`qdense_i8.rs`). The clip is the only place a value is
//! cut off, and the clip is what this counts. The count is over *evaluations*, not distinct
//! positions: the same position evaluated twice is counted twice, as it is searched.
//!
//! **One shard a thread, so that counting does not slow the threads down by fighting over a cache
//! line.** A counter shared by every search thread is a cache line every thread writes, and at
//! sixteen threads that cost three times the speed of the engine (measured: 10.1 M nodes a second
//! without the counters, 3.3 M with a shared counter), which is not a measurement of the engine
//! any more. Each thread counts into a [`Shard`] of its own, which it alone writes, with a load
//! and a store and no locked instruction; [`summary`] adds the shards up. A thread that ends
//! hands its counts to the registry's `retired` shard, so a long run that starts threads does not
//! grow a shard for each.
//!
//! **Read it when the search is idle.** The shards are read with relaxed loads while their threads
//! may still be adding, so a read during a search is a total that is not one moment's. The driver
//! reads after joining the search, and `reset` is only for then too: a writer that is mid-add when
//! another thread zeroes its counter keeps its add.

use std::cell::OnceCell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use super::{
    BLOCK_LEN, FEATURE_LEN, FIELD_AT, FIELD_LEN, FirstLayer, HAS_FIELD, HAS_MOMENT, HAS_PAIR,
    HIDDEN, HIDDEN2, MOMENT_AT, MOMENT_LEN, PAIR, PAIR_AT, POWER_LEN,
};

/// A layer of the head whose values are clipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layer {
    Power,
    Pair,
    Moment,
    Field,
    Hidden,
    Hidden2,
}

impl Layer {
    /// The name the trainer's `activations` command gives the same layer.
    pub fn name(self) -> &'static str {
        match self {
            Layer::Power => "power sums",
            Layer::Pair => "pair table",
            Layer::Moment => "second moment",
            Layer::Field => "square field",
            Layer::Hidden => "hidden 1",
            Layer::Hidden2 => "hidden 2",
        }
    }

    /// Where a block starts in one perspective's share of the features, and how long it is. The
    /// hidden layers are not in the features and have no place there.
    fn block(self) -> Option<(usize, usize)> {
        match self {
            Layer::Power => Some((0, POWER_LEN)),
            Layer::Pair => Some((PAIR_AT, PAIR)),
            Layer::Moment => Some((MOMENT_AT, MOMENT_LEN)),
            Layer::Field => Some((FIELD_AT, FIELD_LEN)),
            Layer::Hidden | Layer::Hidden2 => None,
        }
    }
}

/// The layers this shape has, in the order the evaluation goes through them, each with its width:
/// a block is both perspectives' share of it.
fn layers() -> Vec<(Layer, usize)> {
    let mut found = vec![(Layer::Power, 2 * POWER_LEN)];
    if HAS_PAIR {
        found.push((Layer::Pair, 2 * PAIR));
    }
    if HAS_MOMENT {
        found.push((Layer::Moment, 2 * MOMENT_LEN));
    }
    if HAS_FIELD {
        found.push((Layer::Field, 2 * FIELD_LEN));
    }
    found.push((Layer::Hidden, HIDDEN));
    found.push((Layer::Hidden2, HIDDEN2));
    found
}

/// What the first dense layer ran on, as a number a shard can keep: it changes what the first
/// hidden layer's values are, so a count is not read without it.
fn first_layer_code(first: &FirstLayer) -> u8 {
    match first {
        FirstLayer::F32 => 0,
        FirstLayer::I16(_) => 1,
        FirstLayer::I8(_) => 2,
    }
}

fn first_layer_name(code: u8) -> &'static str {
    match code {
        0 => "f32",
        1 => "i16",
        _ => "i8",
    }
}

/// The counts of one layer, one pair of counters per unit.
struct Counts {
    layer: Layer,
    floor: Vec<AtomicU64>,
    ceiling: Vec<AtomicU64>,
    /// Values that were not a number. `clamp` leaves a `NaN` as it is, so without a count of its
    /// own one would read as a unit between the rails.
    non_finite: AtomicU64,
}

/// `cell += by`, for a cell only the calling thread writes: a load and a store, which is what a
/// plain add compiles to, where `fetch_add` is a locked instruction.
#[inline(always)]
fn bump(cell: &AtomicU64, by: u64) {
    cell.store(cell.load(Relaxed) + by, Relaxed);
}

impl Counts {
    fn new(layer: Layer, width: usize) -> Counts {
        let zeros = || (0..width).map(|_| AtomicU64::new(0)).collect();
        Counts {
            layer,
            floor: zeros(),
            ceiling: zeros(),
            non_finite: AtomicU64::new(0),
        }
    }

    /// Count `values`, which are the units from `first` on. Without a branch for each: the three
    /// tests are values, added.
    #[inline]
    fn count(&self, values: &[f32], first: usize) {
        let floor = &self.floor[first..first + values.len()];
        let ceiling = &self.ceiling[first..first + values.len()];
        let mut non_finite = 0u64;
        for ((&v, floor), ceiling) in values.iter().zip(floor).zip(ceiling) {
            bump(floor, u64::from(v <= 0.0));
            bump(ceiling, u64::from(v >= 1.0));
            non_finite += u64::from(v.is_nan());
        }
        if non_finite != 0 {
            bump(&self.non_finite, non_finite);
        }
    }

    fn clear(&self) {
        self.non_finite.store(0, Relaxed);
        for cell in self.floor.iter().chain(&self.ceiling) {
            cell.store(0, Relaxed);
        }
    }
}

/// One thread's counts. Only that thread adds to it ([`Shard::record`]); any thread may read it.
pub struct Shard {
    evaluations: AtomicU64,
    first_layer: AtomicU8,
    layers: Vec<Counts>,
}

impl Default for Shard {
    fn default() -> Shard {
        Shard::new()
    }
}

impl Shard {
    pub fn new() -> Shard {
        Shard {
            evaluations: AtomicU64::new(0),
            first_layer: AtomicU8::new(0),
            layers: layers()
                .into_iter()
                .map(|(layer, width)| Counts::new(layer, width))
                .collect(),
        }
    }

    /// Count one evaluation: the clipped blocks, and the two clipped hidden layers.
    ///
    /// Reads the arrays and never writes them, so the evaluation is the one it would have been.
    /// Not to be called from two threads at once on one shard: it is a load and a store.
    pub fn record(
        &self,
        first: &FirstLayer,
        features: &[f32; FEATURE_LEN],
        hidden: &[f32; HIDDEN],
        second: &[f32; HIDDEN2],
    ) {
        self.first_layer.store(first_layer_code(first), Relaxed);
        for counts in &self.layers {
            match counts.layer.block() {
                Some((start, len)) => {
                    // The two perspectives' shares are one layer, laid end to end.
                    for half in 0..2 {
                        let at = half * BLOCK_LEN + start;
                        counts.count(&features[at..at + len], half * len);
                    }
                }
                None if counts.layer == Layer::Hidden => counts.count(hidden, 0),
                None => counts.count(second, 0),
            }
        }
        bump(&self.evaluations, 1);
    }

    /// Add `other`'s counts to this one's. For the registry, which is the only writer of a shard
    /// it is merging into.
    fn merge(&self, other: &Shard) {
        bump(&self.evaluations, other.evaluations.load(Relaxed));
        self.first_layer
            .store(other.first_layer.load(Relaxed), Relaxed);
        for (mine, theirs) in self.layers.iter().zip(&other.layers) {
            bump(&mine.non_finite, theirs.non_finite.load(Relaxed));
            for (a, b) in mine.floor.iter().zip(&theirs.floor) {
                bump(a, b.load(Relaxed));
            }
            for (a, b) in mine.ceiling.iter().zip(&theirs.ceiling) {
                bump(a, b.load(Relaxed));
            }
        }
    }

    fn clear(&self) {
        self.evaluations.store(0, Relaxed);
        for counts in &self.layers {
            counts.clear();
        }
    }
}

/// One layer's counts, read out.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerSummary {
    pub layer: Layer,
    pub width: usize,
    /// Units at the floor, summed over the units and the evaluations.
    pub floor: u64,
    pub ceiling: u64,
    pub non_finite: u64,
    /// Units that were at the floor in every evaluation counted, and at the ceiling in every one.
    pub always_floor: usize,
    pub always_ceiling: usize,
}

/// Every shard, read out and added.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Summary {
    pub evaluations: u64,
    /// `f32`, `i16` or `i8`: what the first dense layer ran on in the last evaluation counted.
    pub first_layer: &'static str,
    pub layers: Vec<LayerSummary>,
}

impl Summary {
    /// The sum of `shards`.
    pub fn of(shards: &[&Shard]) -> Summary {
        let evaluations: u64 = shards.iter().map(|s| s.evaluations.load(Relaxed)).sum();
        // Any shard that counted has the precision; they are all the one network's.
        let first_layer = shards
            .iter()
            .find(|s| s.evaluations.load(Relaxed) > 0)
            .map_or(0, |s| s.first_layer.load(Relaxed));
        let layers = layers()
            .into_iter()
            .enumerate()
            .map(|(at, (layer, width))| {
                let add = |cells: fn(&Counts) -> &Vec<AtomicU64>| -> Vec<u64> {
                    (0..width)
                        .map(|unit| {
                            shards
                                .iter()
                                .map(|s| cells(&s.layers[at])[unit].load(Relaxed))
                                .sum()
                        })
                        .collect()
                };
                let (floor, ceiling) = (add(|c| &c.floor), add(|c| &c.ceiling));
                LayerSummary {
                    layer,
                    width,
                    floor: floor.iter().sum(),
                    ceiling: ceiling.iter().sum(),
                    non_finite: shards
                        .iter()
                        .map(|s| s.layers[at].non_finite.load(Relaxed))
                        .sum(),
                    // A unit that was never counted is not "always" anything.
                    always_floor: floor
                        .iter()
                        .filter(|&&f| evaluations > 0 && f == evaluations)
                        .count(),
                    always_ceiling: ceiling
                        .iter()
                        .filter(|&&c| evaluations > 0 && c == evaluations)
                        .count(),
                }
            })
            .collect();
        Summary {
            evaluations,
            first_layer: first_layer_name(first_layer),
            layers,
        }
    }

    /// The lines the `clipstats` command prints: one for the net and one for each layer, in the
    /// shape of the trainer's `activations` lines.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "clipstats first_layer={} evaluations={}",
            self.first_layer, self.evaluations
        )];
        for layer in &self.layers {
            lines.push(layer.line(self.evaluations));
        }
        lines
    }
}

impl LayerSummary {
    /// Share of unit-evaluations at the floor, at the ceiling and between, each in `[0, 1]`, or
    /// `None` when nothing was counted. A value that was not a number is in none of the three.
    pub fn shares(&self, evaluations: u64) -> Option<[f64; 3]> {
        let seen = evaluations * self.width as u64;
        (seen > 0).then(|| {
            let n = seen as f64;
            let between = seen - self.floor - self.ceiling - self.non_finite;
            [
                self.floor as f64 / n,
                self.ceiling as f64 / n,
                between as f64 / n,
            ]
        })
    }

    fn line(&self, evaluations: u64) -> String {
        let name = self.layer.name();
        let Some([floor, ceiling, between]) = self.shares(evaluations) else {
            return format!(
                "clipstats layer=\"{name}\" width={} no evaluations counted",
                self.width
            );
        };
        format!(
            "clipstats layer=\"{name}\" width={} evaluations={evaluations} floor={floor:.4} \
             ceiling={ceiling:.4} between={between:.4} dead_floor={} dead_ceiling={} live={} \
             non_finite={}",
            self.width,
            self.always_floor,
            self.always_ceiling,
            self.width - self.always_floor - self.always_ceiling,
            self.non_finite,
        )
    }
}

/// The shards of the threads that are running, and the counts of the ones that ended.
pub struct Registry {
    live: Mutex<Vec<Arc<Shard>>>,
    retired: Shard,
}

impl Default for Registry {
    fn default() -> Registry {
        Registry::new()
    }
}

impl Registry {
    pub fn new() -> Registry {
        Registry {
            live: Mutex::new(Vec::new()),
            retired: Shard::new(),
        }
    }

    /// A shard for the calling thread. It is read by [`Registry::summary`] from now on, and
    /// folded into the retired counts by [`Registry::retire`].
    pub fn register(&self) -> Arc<Shard> {
        let shard = Arc::new(Shard::new());
        self.live
            .lock()
            .expect("the registry's lock")
            .push(shard.clone());
        shard
    }

    /// A thread is done: its counts go to the retired shard and its own is dropped.
    pub fn retire(&self, shard: &Arc<Shard>) {
        // Held across the merge, so that a summary does not see the counts in both places or in
        // neither.
        let mut live = self.live.lock().expect("the registry's lock");
        self.retired.merge(shard);
        live.retain(|s| !Arc::ptr_eq(s, shard));
    }

    pub fn summary(&self) -> Summary {
        let live = self.live.lock().expect("the registry's lock");
        let mut shards: Vec<&Shard> = live.iter().map(|s| &**s).collect();
        shards.push(&self.retired);
        Summary::of(&shards)
    }

    pub fn reset(&self) {
        let live = self.live.lock().expect("the registry's lock");
        for shard in live.iter() {
            shard.clear();
        }
        self.retired.clear();
    }
}

/// A thread's hold on its shard: when the thread ends, the counts are handed on.
struct Local {
    shard: Arc<Shard>,
    registry: &'static Registry,
}

impl Drop for Local {
    fn drop(&mut self) {
        self.registry.retire(&self.shard);
    }
}

fn registry() -> &'static Registry {
    static GLOBAL: OnceLock<Registry> = OnceLock::new();
    GLOBAL.get_or_init(Registry::new)
}

thread_local! {
    static LOCAL: OnceCell<Local> = const { OnceCell::new() };
}

/// Count one evaluation into the calling thread's shard of the process-wide registry. This is what
/// `value_from_sums` calls.
pub fn record(
    first: &FirstLayer,
    features: &[f32; FEATURE_LEN],
    hidden: &[f32; HIDDEN],
    second: &[f32; HIDDEN2],
) {
    LOCAL.with(|local| {
        let local = local.get_or_init(|| Local {
            shard: registry().register(),
            registry: registry(),
        });
        local.shard.record(first, features, hidden, second);
    });
}

/// What every thread has counted so far, the ended ones' included.
pub fn summary() -> Summary {
    registry().summary()
}

/// Back to nothing counted, in every thread's shard.
pub fn reset() {
    registry().reset();
}
