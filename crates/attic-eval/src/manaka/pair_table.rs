//! The pair table as the derive reads it: halves, both frames to a row.
//!
//! The file holds one table, `[PAIR_ROWS][PAIR]` halves, and both frames read it:
//! the row of a pair is `pair_row` of the two tokens as that frame numbers them.
//! A move touches the same pieces in both frames, so a derive reads, for every
//! pair it touches, one row for the black frame and another for the white one,
//! far apart in the table. This keeps the two together: row `r` here is the
//! file's row `r` (the pair as the black frame numbers it) followed by the file's
//! row for the same two pieces as the white frame numbers them. One read of
//! `2 · PAIR` halves serves both frames.
//!
//! The values stay halves, as the file has them, and are widened when they are
//! added (`pair.rs`). That is exact, so what the sums see is what an `f32` table
//! widened at load would give, and the table is half the size: `2 · PAIR` halves a
//! row is the `PAIR` `f32`s a row of the single-frame table took.
//!
//! A net read with `hm` keeps a second table of the same size, [`PairTable::mixed_cells`], for a position
//! whose two frames are folded differently. Its row `{a, b}` is the black half of row `{m(a), m(b)}`
//! followed by the white half of row `{a, b}` (`m` the reflection, `mirror_row`), so that such a position
//! too reads one row a pair for both frames, where the one table would have it read two and use half of
//! each.
//!
//! The table can also be held as bytes, half the size again: the reads of a derive
//! are rows picked at random from a table of 88 to 352 MB, so the bytes a row takes
//! are what the memory is asked for. [`PairPrecision`] names the two ways to round a
//! half to a byte, and `ATTIC_SYNN_PAIR` picks one when a network is loaded. The file
//! is the same either way; the rounding is done once, at load.

use crate::Aligned64;

use super::PAIR;
use super::reader::half_to_f32;
#[cfg(feature = "sym_token")]
use super::{HM, HM_NOPAIRFOLD, PAIR_ROWS};

/// Entries in a row of the table: both frames' channels.
pub(super) const ROW: usize = 2 * PAIR;

/// Bytes a row takes when each half has a scale of its own (`i8row`): the `i8`
/// entries, then one `f32` scale for the black frame's channels and one for the
/// white frame's.
pub(super) const SCALED_ROW: usize = ROW + 8;

/// How many levels a byte has on each side of zero: `-128` is left out so that
/// the range is symmetric.
const LEVELS: f32 = 127.0;

/// `127³`: a cube-rounded entry is `q³ · max / 127³` for `q` in `−127..=127`.
const CUBE_LEVELS: f64 = 127.0 * 127.0 * 127.0;

/// How the pair table is held, as `ATTIC_SYNN_PAIR` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairPrecision {
    /// The halves of the file, widened when they are added. The default.
    F16,
    /// A byte per entry, `q³` times a scale per channel: the entry is the signed
    /// cube root of its size against the channel's largest, rounded to 127 levels
    /// a side. Small entries get fine steps and large ones coarse, which is what
    /// the pair table's entries (mostly small, a few large) want; one scale for a
    /// channel costs nothing per row.
    I8,
    /// A byte per entry and a scale for each half of a row, `f32`: the entry is
    /// `q · scale`, 127 levels of the half's largest. Eight bytes more a row.
    I8Row,
}

impl PairPrecision {
    /// `i8` and `i8row`, in any case, name those two; anything else, or nothing,
    /// is `F16`, the default.
    pub fn from_name(name: Option<&str>) -> PairPrecision {
        match name {
            Some(n) if n.eq_ignore_ascii_case("i8") => PairPrecision::I8,
            Some(n) if n.eq_ignore_ascii_case("i8row") => PairPrecision::I8Row,
            _ => PairPrecision::F16,
        }
    }
}

/// What the table holds.
#[derive(Debug)]
pub(super) enum Cells {
    /// `[PAIR_ROWS][ROW]` halves.
    Halves(Aligned64<u16>),
    /// `[PAIR_ROWS][ROW]` bytes (`i8`), and the factor each channel's `q³` is
    /// multiplied by, the same for both frames.
    Cubes { q: Aligned64<u8>, scale: [f32; ROW] },
    /// `[PAIR_ROWS][SCALED_ROW]` bytes: a row's `ROW` entries (`i8`), then its two
    /// scales as little-endian `f32`.
    Scaled { q: Aligned64<u8> },
}

/// `[PAIR_ROWS][2][PAIR]` entries: for each pair as the black frame numbers it,
/// that pair's row of the file, then the row of the same two pieces as the white
/// frame numbers them. Empty in a shape without a pair table.
#[derive(Debug)]
pub struct PairTable {
    cells: Cells,
    /// `hm` only, and empty otherwise: the rows a position reads when one frame is folded and the other is
    /// not, in the same form as `cells`. See the module head.
    mixed: Cells,
}

impl PairTable {
    /// The table of a shape without one.
    pub(super) fn empty() -> PairTable {
        PairTable {
            cells: Cells::Halves(Aligned64::zeroed(0)),
            mixed: Cells::Halves(Aligned64::zeroed(0)),
        }
    }

    /// The table from the file's, `PAIR_ROWS × PAIR` little-endian halves that the
    /// reader has already found finite.
    ///
    /// Each pair `(lo, hi)` of the black frame, in the file's row order, takes
    /// its own row and then the row of `(other_frame(lo), other_frame(hi))`.
    #[cfg(feature = "sym_token")]
    pub(super) fn from_le_bytes(file: &[u8]) -> PairTable {
        use super::TOKENS;
        use super::pair::pair_row;
        use crate::tokens::{FeatureIndex, other_frame};

        assert_eq!(file.len(), PAIR_ROWS * PAIR * 2, "the file's pair table");
        let read = |row: usize, out: &mut [u16]| {
            let bytes = &file[row * PAIR * 2..(row + 1) * PAIR * 2];
            for (slot, two) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *slot = u16::from_le_bytes(*two);
            }
        };
        let mut halves = Aligned64::zeroed(PAIR_ROWS * 2 * PAIR);
        let mut row = 0;
        for lo in 0..TOKENS as FeatureIndex {
            for hi in lo + 1..TOKENS as FeatureIndex {
                let (black, white) =
                    halves[row * 2 * PAIR..(row + 1) * 2 * PAIR].split_at_mut(PAIR);
                read(row, black);
                read(pair_row(other_frame(lo), other_frame(hi)), white);
                row += 1;
            }
        }
        // Frames folded differently read the pair table at different places only when the pairs fold.
        let mixed = if HM && !HM_NOPAIRFOLD {
            mixed_halves(&halves)
        } else {
            Aligned64::zeroed(0)
        };
        PairTable {
            cells: Cells::Halves(halves),
            mixed: Cells::Halves(mixed),
        }
    }

    /// Rows, one per pair: `PAIR_ROWS`, or none without a pair table.
    pub fn rows(&self) -> usize {
        if PAIR == 0 {
            return 0;
        }
        match &self.cells {
            Cells::Halves(halves) => halves.len() / ROW,
            Cells::Cubes { q, .. } => q.len() / ROW,
            Cells::Scaled { q } => q.len() / SCALED_ROW,
        }
    }

    /// What the table holds, for the kernels that add rows (`pair.rs`).
    #[cfg(feature = "sym_token")]
    pub(super) fn cells(&self) -> &Cells {
        &self.cells
    }

    /// The rows for a position whose frames are folded differently, `hm` only: row `{a, b}` is the black
    /// channels of [`cells`](Self::cells)' row `{m(a), m(b)}` and the white channels of its row `{a, b}`.
    /// Empty without `hm`.
    #[cfg(feature = "sym_token")]
    pub(super) fn mixed_cells(&self) -> &Cells {
        &self.mixed
    }

    /// Every half, `[rows][2][PAIR]`. Only a table that has not been rounded to
    /// bytes has them.
    ///
    /// # Panics
    /// Panics if the table was rounded by [`use_precision`](Self::use_precision).
    #[cfg(test)]
    pub(super) fn halves(&self) -> &[u16] {
        match &self.cells {
            Cells::Halves(halves) => halves,
            _ => panic!("the pair table was rounded to bytes and no longer has its halves"),
        }
    }

    /// Which [`PairPrecision`] the table is held in.
    pub fn precision(&self) -> PairPrecision {
        match &self.cells {
            Cells::Halves(_) => PairPrecision::F16,
            Cells::Cubes { .. } => PairPrecision::I8,
            Cells::Scaled { .. } => PairPrecision::I8Row,
        }
    }

    /// Hold the table at `precision`: the halves of the file rounded to bytes, and
    /// the halves dropped.
    ///
    /// Rounding is one way. A table already rounded is left as it is, whatever
    /// `precision` says, since its halves are gone; [`PairPrecision::F16`] leaves
    /// the table alone.
    ///
    /// The second table of `hm` is rounded the same way. Its halves of a row are the main table's halves of
    /// rows, the same ones, so each rounds to the bytes it does there.
    pub fn use_precision(&mut self, precision: PairPrecision) {
        if PAIR == 0 {
            return;
        }
        if let Some(cells) = rounded(&self.cells, precision) {
            self.cells = cells;
        }
        if let Some(cells) = rounded(&self.mixed, precision) {
            self.mixed = cells;
        }
    }

    /// The file's entry at `row` and `channel`, widened: the table both frames
    /// read, as the file has it.
    ///
    /// # Panics
    /// Panics if `channel` is not below `PAIR` or `row` not below [`rows`](Self::rows).
    pub fn value(&self, row: usize, channel: usize) -> f32 {
        let channels = PAIR;
        assert!(channel < channels, "channel {channel} of {channels}");
        match &self.cells {
            Cells::Halves(halves) => half_to_f32(halves[row * ROW + channel])
                .expect("the reader found every half finite"),
            Cells::Cubes { q, scale } => {
                let level = f32::from(q[row * ROW + channel] as i8);
                level * level * level * scale[channel]
            }
            Cells::Scaled { q } => {
                let at = row * SCALED_ROW;
                let scale = f32::from_le_bytes(
                    q[at + ROW..at + ROW + 4]
                        .try_into()
                        .expect("a scale is four bytes"),
                );
                f32::from(q[at + channel] as i8) * scale
            }
        }
    }

    /// The file's table, `[PAIR_ROWS][PAIR]`, widened: for the tests that write a
    /// network out and read it back.
    #[cfg(test)]
    pub(super) fn file_table(&self) -> Vec<f32> {
        (0..self.rows())
            .flat_map(|row| (0..PAIR).map(move |channel| (row, channel)))
            .map(|(row, channel)| self.value(row, channel))
            .collect()
    }

    /// A second copy, for another NUMA node.
    pub(super) fn replicate(&self) -> PairTable {
        PairTable {
            cells: copy(&self.cells),
            mixed: copy(&self.mixed),
        }
    }
}

/// A copy of `cells`.
fn copy(cells: &Cells) -> Cells {
    match cells {
        Cells::Halves(halves) => Cells::Halves(Aligned64::from_slice(halves)),
        Cells::Cubes { q, scale } => Cells::Cubes {
            q: Aligned64::from_slice(q),
            scale: *scale,
        },
        Cells::Scaled { q } => Cells::Scaled {
            q: Aligned64::from_slice(q),
        },
    }
}

/// `cells` rounded to `precision`, or `None` when that leaves them as they are: [`PairPrecision::F16`],
/// cells already rounded (their halves are gone), or none at all.
fn rounded(cells: &Cells, precision: PairPrecision) -> Option<Cells> {
    let Cells::Halves(halves) = cells else {
        return None;
    };
    if halves.is_empty() {
        return None;
    }
    match precision {
        PairPrecision::F16 => None,
        PairPrecision::I8 => Some(cube_cells(halves)),
        PairPrecision::I8Row => Some(scaled_cells(halves)),
    }
}

/// `hm`'s second table from the first, `halves` (`[PAIR_ROWS][ROW]`): row `{a, b}` is the black half of row
/// `{m(a), m(b)}` and the white half of row `{a, b}`.
///
/// A position whose black frame is folded and white frame is not reads, for the pair `{a, b}`, the black
/// half of `{m(a), m(b)}` and the white half of `{a, b}`: this row. One whose white frame is folded and black
/// frame is not reads the black half of `{a, b}` and the white half of `{m(a), m(b)}`, which is this table's
/// row `{m(a), m(b)}`, since `m` is an involution. The reflection is a permutation of the tokens, so `{m(a),
/// m(b)}` is a pair, of two distinct tokens, whenever `{a, b}` is.
#[cfg(feature = "sym_token")]
fn mixed_halves(halves: &[u16]) -> Aligned64<u16> {
    use super::TOKENS;
    use super::pair::pair_row;
    use crate::tokens::{FeatureIndex, mirror_row};

    let mut mixed = Aligned64::zeroed(halves.len());
    let mut row = 0;
    for lo in 0..TOKENS as FeatureIndex {
        for hi in lo + 1..TOKENS as FeatureIndex {
            let reflected = pair_row(mirror_row(lo), mirror_row(hi)) * ROW;
            let (black, white) = mixed[row * ROW..(row + 1) * ROW].split_at_mut(PAIR);
            black.copy_from_slice(&halves[reflected..reflected + PAIR]);
            white.copy_from_slice(&halves[row * ROW + PAIR..(row + 1) * ROW]);
            row += 1;
        }
    }
    mixed
}

/// The level of `x` among the cubes: the `q` in `0..=127` for which `q³` is nearest
/// `size`, where `size` is `|x|` in units of the channel's largest times `127³`.
fn cube_level(size: f64) -> u8 {
    let guess = size.cbrt().round().clamp(0.0, 127.0) as i64;
    let distance = |q: i64| (size - (q * q * q) as f64).abs();
    (guess - 1..=guess + 1)
        .filter(|q| (0..=127).contains(q))
        .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
        .expect("the guess is a level") as u8
}

/// The table rounded by [`PairPrecision::I8`]. One look-up per half: for each
/// channel, the byte every one of the 65,536 halves rounds to, so that the table is
/// read once and no cube root is taken per entry.
// `as_chunks` refuses a chunk of zero, which is what `PAIR` is in a shape without a
// pair table; those never get here (`use_precision`), but they are compiled.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn cube_cells(halves: &[u16]) -> Cells {
    let size = |h: u16| half_to_f32(h).map_or(0.0, f32::abs);
    // A channel's largest entry, over both frames' rows (which are the same rows of
    // the file, so it is the file's channel).
    let mut largest = [0.0f32; PAIR];
    for row in halves.chunks_exact(PAIR) {
        for (m, &h) in largest.iter_mut().zip(row) {
            *m = m.max(size(h));
        }
    }
    let lookup: Vec<u8> = largest
        .iter()
        .flat_map(|&max| {
            (0..=u16::MAX).map(move |bits| {
                if max == 0.0 {
                    return 0;
                }
                let level = cube_level(f64::from(size(bits)) / f64::from(max) * CUBE_LEVELS);
                // The sign of a half is its top bit; a zero keeps level 0 either way.
                if bits & 0x8000 != 0 {
                    (level as i8).wrapping_neg() as u8
                } else {
                    level
                }
            })
        })
        .collect();
    let mut q = Aligned64::<u8>::zeroed(halves.len());
    for (dst, src) in q.chunks_exact_mut(PAIR).zip(halves.chunks_exact(PAIR)) {
        for ((d, &h), table) in dst.iter_mut().zip(src).zip(lookup.chunks_exact(65_536)) {
            *d = table[usize::from(h)];
        }
    }
    let per_channel: [f32; PAIR] =
        std::array::from_fn(|c| (f64::from(largest[c]) / CUBE_LEVELS) as f32);
    let mut scale = [0.0f32; ROW];
    scale[..PAIR].copy_from_slice(&per_channel);
    scale[PAIR..].copy_from_slice(&per_channel);
    Cells::Cubes { q, scale }
}

/// The table rounded by [`PairPrecision::I8Row`]: each half of a row to 127 levels
/// of its own largest entry.
// As `cube_cells`: `PAIR` may be zero where this is compiled but not called.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn scaled_cells(halves: &[u16]) -> Cells {
    // `ROW` is zero in a shape without a pair table, which never gets here; the
    // division is checked so that the build of that shape's tests compiles.
    let rows = halves.len().checked_div(ROW).unwrap_or(0);
    let mut q = Aligned64::<u8>::zeroed(rows * SCALED_ROW);
    for (dst, src) in q.chunks_exact_mut(SCALED_ROW).zip(halves.chunks_exact(ROW)) {
        let (entries, scales) = dst.split_at_mut(ROW);
        for (half, (entries, scale)) in src.chunks_exact(PAIR).zip(
            entries
                .chunks_exact_mut(PAIR)
                .zip(scales.chunks_exact_mut(4)),
        ) {
            let values: [f32; PAIR] = std::array::from_fn(|i| half_to_f32(half[i]).unwrap_or(0.0));
            let largest = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if largest == 0.0 {
                continue;
            }
            let step = largest / LEVELS;
            for (e, v) in entries.iter_mut().zip(values) {
                *e = ((v / step).round_ties_even().clamp(-LEVELS, LEVELS) as i8) as u8;
            }
            scale.copy_from_slice(&step.to_le_bytes());
        }
    }
    Cells::Scaled { q }
}
