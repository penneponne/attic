//! The counters of `clip_stats.rs`: that they count what the head produced, add up across
//! threads, and change nothing about the evaluation they watch.
//!
//! They run against a [`Registry`] and [`Shard`] of their own and never the process-wide ones,
//! because tests share a process and every evaluation in it counts into the global registry.

use std::sync::Arc;

use super::super::clip_stats::{Layer, Registry, Shard, Summary};
use super::super::forward::{dense_clip, features_from_sums};
use super::super::{
    Accumulator, BLOCK_LEN, FEATURE_LEN, FIELD_AT, FIELD_LEN, FirstLayer, HAS_FIELD, HAS_MOMENT,
    HAS_PAIR, HIDDEN, HIDDEN2, MOMENT_AT, MOMENT_LEN, PAIR, PAIR_AT, POWER_LEN, W1Precision,
    evaluate_value,
};
use super::fixtures::{game, synthetic};

/// The three arrays the head hands the counter for one position, worked out here from the pieces
/// the forward pass is made of, and not read from the counter's own path.
fn head(pos: &attic_state::Position) -> ([f32; FEATURE_LEN], [f32; HIDDEN], [f32; HIDDEN2]) {
    let net = synthetic();
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    let mut features = [0.0f32; FEATURE_LEN];
    features_from_sums(net, &acc, pos.side_to_move(), &mut features);
    let mut hidden = [0.0f32; HIDDEN];
    dense_clip(&net.w1, &net.b1, &features, &mut hidden);
    let mut second = [0.0f32; HIDDEN2];
    dense_clip(&net.w2, &net.b2, &hidden, &mut second);
    (features, hidden, second)
}

/// `(floor, ceiling)` of every layer, counted the plainest way: one unit at a time, with the
/// counter's own two tests spelled out again.
fn counted_by_hand(positions: &[attic_state::Position]) -> Vec<(Layer, u64, u64)> {
    let mut found: Vec<(Layer, u64, u64)> = vec![(Layer::Power, 0, 0)];
    if HAS_PAIR {
        found.push((Layer::Pair, 0, 0));
    }
    if HAS_MOMENT {
        found.push((Layer::Moment, 0, 0));
    }
    if HAS_FIELD {
        found.push((Layer::Field, 0, 0));
    }
    found.push((Layer::Hidden, 0, 0));
    found.push((Layer::Hidden2, 0, 0));
    let block_of = |layer: Layer| match layer {
        Layer::Power => (0, POWER_LEN),
        Layer::Pair => (PAIR_AT, PAIR),
        Layer::Moment => (MOMENT_AT, MOMENT_LEN),
        _ => (FIELD_AT, FIELD_LEN),
    };
    for pos in positions {
        let (features, hidden, second) = head(pos);
        for (layer, floor, ceiling) in &mut found {
            let values: Vec<f32> = match layer {
                Layer::Hidden => hidden.to_vec(),
                Layer::Hidden2 => second.to_vec(),
                other => {
                    let (start, len) = block_of(*other);
                    (0..2)
                        .flat_map(|half| {
                            features[half * BLOCK_LEN + start..half * BLOCK_LEN + start + len]
                                .to_vec()
                        })
                        .collect()
                }
            };
            for v in values {
                if v <= 0.0 {
                    *floor += 1;
                } else if v >= 1.0 {
                    *ceiling += 1;
                }
            }
        }
    }
    found
}

fn positions(plies: usize) -> Vec<attic_state::Position> {
    game(plies).into_iter().map(|(pos, _)| pos).collect()
}

fn record(shard: &Shard, pos: &attic_state::Position) {
    let (features, hidden, second) = head(pos);
    shard.record(&synthetic().first, &features, &hidden, &second);
}

#[test]
fn a_shard_counts_what_a_unit_at_a_time_count_gives() {
    let positions = positions(30);
    let shard = Shard::new();
    for pos in &positions {
        record(&shard, pos);
    }
    let summary = Summary::of(&[&shard]);
    assert_eq!(summary.evaluations, positions.len() as u64);
    let want = counted_by_hand(&positions);
    assert_eq!(summary.layers.len(), want.len());
    for (got, (layer, floor, ceiling)) in summary.layers.iter().zip(&want) {
        assert_eq!(got.layer, *layer);
        assert_eq!(got.floor, *floor, "{} floor", layer.name());
        assert_eq!(got.ceiling, *ceiling, "{} ceiling", layer.name());
        // The test net is one the clip can be seen in, so a count of nothing would be a counter
        // that counts nothing.
        assert!(got.floor + got.ceiling > 0, "{}", layer.name());
    }
}

#[test]
fn the_three_shares_of_a_layer_add_to_one() {
    let shard = Shard::new();
    for pos in &positions(20) {
        record(&shard, pos);
    }
    let summary = Summary::of(&[&shard]);
    for layer in &summary.layers {
        let [floor, ceiling, between] = layer.shares(summary.evaluations).expect("counted");
        assert!(
            (floor + ceiling + between - 1.0).abs() < 1e-12,
            "{}",
            layer.layer.name()
        );
        assert!((0.0..=1.0).contains(&between), "{}", layer.layer.name());
    }
}

/// Counting a game in two threads and adding is counting it in one. This is the property the
/// per-thread shards have to keep, and a counter that dropped a thread's counts would fail it.
#[test]
fn threads_add_up_to_what_one_thread_counts() {
    let positions = positions(40);
    let whole = Shard::new();
    for pos in &positions {
        record(&whole, pos);
    }

    let registry = Arc::new(Registry::new());
    let (left, right) = positions.split_at(positions.len() / 2);
    std::thread::scope(|scope| {
        for part in [left, right] {
            let registry = registry.clone();
            scope.spawn(move || {
                let shard = registry.register();
                for pos in part {
                    record(&shard, pos);
                }
                // Left running: the registry reads a live shard.
                std::mem::forget(shard);
            });
        }
    });
    assert_eq!(registry.summary(), Summary::of(&[&whole]));
}

/// A thread that ends hands its counts on, and a summary taken after sees them exactly once.
#[test]
fn a_thread_that_ends_leaves_its_counts_behind() {
    let positions = positions(25);
    let whole = Shard::new();
    for pos in &positions {
        record(&whole, pos);
    }

    let registry = Arc::new(Registry::new());
    let mid = positions.len() / 2;
    for part in [&positions[..mid], &positions[mid..]] {
        let shard = registry.register();
        for pos in part {
            record(&shard, pos);
        }
        // Seen once while it is live...
        registry.retire(&shard);
    }
    // ...and once after it is gone.
    assert_eq!(registry.summary(), Summary::of(&[&whole]));
}

#[test]
fn reset_forgets_everything() {
    let registry = Registry::new();
    let shard = registry.register();
    for pos in &positions(10) {
        record(&shard, pos);
    }
    assert!(registry.summary().evaluations > 0);
    registry.reset();
    let summary = registry.summary();
    assert_eq!(summary.evaluations, 0);
    for layer in &summary.layers {
        assert_eq!((layer.floor, layer.ceiling, layer.non_finite), (0, 0, 0));
        assert_eq!((layer.always_floor, layer.always_ceiling), (0, 0));
        assert!(layer.shares(0).is_none());
    }
    // And counts again afterwards.
    record(&shard, &attic_state::Position::startpos());
    assert_eq!(registry.summary().evaluations, 1);
}

#[test]
fn a_unit_on_a_rail_every_time_is_reported_as_dead() {
    let shard = Shard::new();
    let pos = attic_state::Position::startpos();
    // The same position again and again: a unit that is on a rail in it is on that rail every
    // time, so the units counted dead are exactly the ones on a rail in this one position.
    for _ in 0..5 {
        record(&shard, &pos);
    }
    let summary = Summary::of(&[&shard]);
    let (features, hidden, second) = head(&pos);
    let want_hidden = hidden.iter().filter(|&&v| v <= 0.0).count();
    let hidden_layer = summary
        .layers
        .iter()
        .find(|l| l.layer == Layer::Hidden)
        .expect("hidden 1");
    assert_eq!(hidden_layer.always_floor, want_hidden);
    let want_ceiling = second.iter().filter(|&&v| v >= 1.0).count();
    let second_layer = summary
        .layers
        .iter()
        .find(|l| l.layer == Layer::Hidden2)
        .expect("hidden 2");
    assert_eq!(second_layer.always_ceiling, want_ceiling);
    let power = &summary.layers[0];
    let want_power = (0..2)
        .flat_map(|half| features[half * BLOCK_LEN..half * BLOCK_LEN + POWER_LEN].to_vec())
        .filter(|&v| v <= 0.0)
        .count();
    assert_eq!(power.always_floor, want_power);
}

/// A layer with a `NaN` in it is not a layer with a unit between the rails.
#[test]
fn a_value_that_is_not_a_number_is_counted_on_its_own() {
    let shard = Shard::new();
    let (features, mut hidden, second) = head(&attic_state::Position::startpos());
    hidden[0] = f32::NAN;
    shard.record(&synthetic().first, &features, &hidden, &second);
    let summary = Summary::of(&[&shard]);
    let layer = summary
        .layers
        .iter()
        .find(|l| l.layer == Layer::Hidden)
        .expect("hidden 1");
    assert_eq!(layer.non_finite, 1);
    let [floor, ceiling, between] = layer.shares(1).expect("counted");
    // One unit in HIDDEN is in none of the three, so they fall short of one by exactly its share.
    assert!((floor + ceiling + between - (1.0 - 1.0 / HIDDEN as f64)).abs() < 1e-12);
}

#[test]
fn the_first_layer_the_net_ran_is_named() {
    let net = synthetic();
    let pos = attic_state::Position::startpos();
    let (features, hidden, second) = head(&pos);
    let names = [
        (FirstLayer::F32, "f32"),
        (
            FirstLayer::I16(super::super::QuantW1::new(&net.w1, HIDDEN, FEATURE_LEN)),
            "i16",
        ),
        (
            FirstLayer::I8(super::super::QuantW1I8::new(&net.w1, HIDDEN, FEATURE_LEN)),
            "i8",
        ),
    ];
    for (first, name) in names {
        let shard = Shard::new();
        shard.record(&first, &features, &hidden, &second);
        assert_eq!(Summary::of(&[&shard]).first_layer, name);
    }
}

/// The counter watches and does not touch: the value `evaluate_value` gives, which has the hook in
/// it, is the value of the same forward pass written out again without it, bit for bit, at each
/// precision of the first layer. (The pass written out is `value_from_sums` minus the hook, which
/// is the point: if the hook ever wrote to what it reads, the two would part.)
#[test]
fn counting_does_not_change_the_value() {
    let mut net = synthetic().replicate();
    let positions = positions(30);
    for precision in [W1Precision::F32, W1Precision::I16, W1Precision::I8] {
        net.use_first_layer(precision);
        for pos in &positions {
            let mut acc = Accumulator::new();
            acc.refresh(&net, pos);
            let mut features = [0.0f32; FEATURE_LEN];
            features_from_sums(&net, &acc, pos.side_to_move(), &mut features);
            let mut hidden = [0.0f32; HIDDEN];
            match &net.first {
                FirstLayer::F32 => dense_clip(&net.w1, &net.b1, &features, &mut hidden),
                FirstLayer::I16(q) => q.layer_clip(&features, &net.b1, &mut hidden),
                FirstLayer::I8(q) => q.layer_clip(&features, &net.b1, &mut hidden),
            }
            let mut second = [0.0f32; HIDDEN2];
            dense_clip(&net.w2, &net.b2, &hidden, &mut second);
            let want = net.value_b + super::super::dense::dot(&net.value_w, &second);
            assert_eq!(
                evaluate_value(&net, pos).to_bits(),
                want.to_bits(),
                "{precision:?}"
            );
        }
    }
}
