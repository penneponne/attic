//! What a table over pairs of tokens needs from a token's number: which row a
//! pair is, and the add-or-subtract of the rows of one token against all the
//! others present, in both frames at once.
//!
//! The numbering itself is `tokens.rs`'s; this reads it and does not extend it,
//! so that the one file that knows how a token is numbered stays the only one.

use crate::tokens::FeatureIndex;

use super::accumulator::PairSum;
use super::pair_i8::{cube_totals, scaled_totals};
use super::pair_table::{Cells, ROW};
use super::{PAIR, PAIR_ROWS, SymNetwork, TOKENS};

/// The row of the pair `{a, b}` in the pair table.
///
/// The upper triangle in lexicographic order, `(0,1) (0,2) … (0,2343) (1,2) …`:
/// the pair with the lower token `lo` and the higher `hi` is `lo` full rows of
/// the triangle in, `TOKENS − 1`, `TOKENS − 2`, … long, then `hi − lo − 1` into
/// its own. The division is exact, since `lo` and `2 · TOKENS − lo − 1` are not
/// both odd, and the order of the arguments does not matter.
///
/// # Panics
/// Panics, in a debug build, if the two tokens are the same one, which a
/// position never has: a token is a distinct row in a given frame.
#[inline]
pub(super) fn pair_row(a: FeatureIndex, b: FeatureIndex) -> usize {
    debug_assert!(a != b, "a token does not pair with itself");
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (lo, hi) = (lo as usize, hi as usize);
    lo * (2 * TOKENS - lo - 1) / 2 + (hi - lo - 1)
}

/// The most tokens a position holds, which is the most `others` a touch has to read
/// rows against (a token is never its own other, so one fewer would do; the rest is
/// slack for the padding to whole registers).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const MAX_OTHERS: usize = crate::tokens::MAX_ACTIVE_FEATURES;

/// What [`row_offsets`] hands back: room for [`MAX_OTHERS`] offsets, in whole
/// registers of eight.
#[cfg(any(test, all(target_arch = "x86_64", target_feature = "avx2")))]
pub(super) const OFFSETS: usize = crate::tokens::MAX_ACTIVE_FEATURES.next_multiple_of(8);

/// `pair_row(token, other) · stride` for each of `others`, in order: where in the
/// table, in units of `stride`, each row starts. The entries past `others.len()`
/// are not meaningful.
///
/// The kernels that add rows read the whole list before they add any, so the
/// arithmetic that finds a row (a compare, a multiplication and a handful of
/// additions: more instructions than the row's eight loads) is done eight rows to a
/// register, ahead of the loads and out of their way. It is `pair_row`'s formula
/// exactly, in `u32`, which holds it: `lo · (2 · TOKENS − 1 − lo)` is under 5.5
/// million and the offset under `PAIR_ROWS · stride`, asserted to fit.
///
/// # Panics
/// Panics if `others` is longer than [`OFFSETS`], or if the table is so large that
/// an offset does not fit in a `u32`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(super) fn row_offsets(
    token: FeatureIndex,
    others: &[FeatureIndex],
    stride: usize,
) -> [u32; OFFSETS] {
    assert!(
        others.len() <= OFFSETS,
        "a position has at most {MAX_OTHERS} tokens"
    );
    assert!(
        PAIR_ROWS * stride <= u32::MAX as usize,
        "an offset fits a u32"
    );
    // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
    unsafe { row_offsets_avx2(token, others, stride as u32) }
}

/// [`row_offsets`] one row at a time, through [`pair_row`]: the reference the AVX2
/// path is held to.
#[cfg(test)]
pub(super) fn row_offsets_scalar(
    token: FeatureIndex,
    others: &[FeatureIndex],
    stride: usize,
) -> [u32; OFFSETS] {
    let mut out = [0u32; OFFSETS];
    for (slot, &other) in out.iter_mut().zip(others) {
        *slot = (pair_row(token, other) * stride) as u32;
    }
    out
}

/// [`row_offsets_scalar`] eight rows to a register.
///
/// The last register is loaded under a mask, so that no entry past `others` is
/// read; its unused lanes see a zero token and their results are not meaningful.
///
/// # Safety
/// The running CPU must support AVX2. The `cfg` that compiles this in names it,
/// and the build is for the host CPU, so it does.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn row_offsets_avx2(
    token: FeatureIndex,
    others: &[FeatureIndex],
    stride: u32,
) -> [u32; OFFSETS] {
    use std::arch::x86_64::*;

    let mut out = [0u32; OFFSETS];
    let this = _mm256_set1_epi32(token as i32);
    let across = _mm256_set1_epi32((2 * TOKENS - 1) as i32);
    let one = _mm256_set1_epi32(1);
    let stride = _mm256_set1_epi32(stride as i32);
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for at in (0..others.len()).step_by(LANES) {
        let left = others.len() - at;
        // SAFETY: `at + min(left, 8) <= others.len()`: a whole register is loaded
        // only when eight entries are left, and a masked one touches only the lanes
        // below `left`. `at + 8 <= OFFSETS` bounds the store, which is unaligned.
        let (lo, hi) = unsafe {
            let other = if left >= LANES {
                _mm256_loadu_si256(others.as_ptr().add(at).cast())
            } else {
                let mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(left as i32), lane);
                _mm256_maskload_epi32(others.as_ptr().add(at).cast(), mask)
            };
            (_mm256_min_epu32(this, other), _mm256_max_epu32(this, other))
        };
        // `lo · (2 · TOKENS − 1 − lo) / 2 + (hi − lo − 1)`; the product is even, so
        // the shift is the division.
        let triangle = _mm256_srli_epi32::<1>(_mm256_mullo_epi32(lo, _mm256_sub_epi32(across, lo)));
        let row = _mm256_add_epi32(triangle, _mm256_sub_epi32(_mm256_sub_epi32(hi, lo), one));
        // SAFETY: see above.
        unsafe {
            _mm256_storeu_si256(
                out.as_mut_ptr().add(at).cast(),
                _mm256_mullo_epi32(row, stride),
            );
        }
    }
    out
}

/// The group of channels the AVX2 path adds at once: a register of `f32`.
const LANES: usize = 8;

const _: () = assert!(
    PAIR.is_multiple_of(LANES),
    "pair channels come in whole groups of eight"
);

/// Add (`sign = 1`) or subtract (`sign = −1`) the pair rows of `token` against
/// every one of `others`, into both frames' sums.
///
/// `token` and `others` are numbered as the black frame numbers them, and
/// `sums` is `[black, white]`. A row of the table holds both frames' channels for
/// the pair (see [`PairTable`](super::PairTable)), so the white frame's sum comes from the same
/// reads. That is the white frame's own computation and not an approximation of
/// it: the white frame's tokens are the black frame's through `other_frame`, in
/// the same order (`tokens.rs`), so the rows it would read and the order it
/// would add them in are the ones read here.
///
/// `others` must not contain `token`. The caller keeps that by taking a token
/// out of the set before subtracting its pairs and putting it in after adding
/// them, which is why the accumulator removes before it touches a row.
///
/// The rows are added up in `f32` first, at most thirty-nine of them, and that
/// total goes into the `f64` sum once, so the `f64` sum takes one rounding per
/// token touched. The rows go to two totals in turn, even-numbered rows to one
/// and odd to the other, a last odd row to the first, and the two are added at
/// the end; two totals let two rows be added at once. [`row_totals_scalar`] is
/// that order written plainly, and the AVX2 path is held to it to the last bit.
#[inline]
pub(super) fn touch_pairs(
    sums: &mut [Vec<PairSum>; 2],
    net: &SymNetwork,
    token: FeatureIndex,
    others: &[FeatureIndex],
    sign: PairSum,
) {
    if PAIR == 0 {
        return;
    }
    let totals = row_totals(net.pair.cells(), token, others);
    for (frame, sum) in sums.iter_mut().enumerate() {
        add_frame(sum, &totals, frame, sign);
    }
}

/// `totals`' channels for `frame` (`0` black, `1` white) into that frame's pair sum.
#[inline]
fn add_frame(sum: &mut [PairSum], totals: &[f32; ROW], frame: usize, sign: PairSum) {
    let sum: &mut [PairSum; PAIR] = sum.try_into().expect("a pair sum is PAIR long");
    for (s, total) in sum
        .iter_mut()
        .zip(&totals[frame * PAIR..(frame + 1) * PAIR])
    {
        *s += sign * PairSum::from(*total);
    }
}

/// [`touch_pairs`] for a position read with `hm`: each frame's tokens are first reflected, or not, by that
/// frame's own fold (`folded[0]` black's, `folded[1]` white's), and the pair sums are those of the
/// reflected tokens. Only the frames `update` names are written.
///
/// `token` and `others` are numbered as the black frame numbers them and unreflected, as in
/// [`touch_pairs`]. A row of the table holds a pair's black channels and then the white channels of the
/// same two pieces. Reflecting a frame's tokens (`mirror_row`) commutes with turning them to the other
/// frame (`other_frame`), so the black frame's reflected pair `{m(a), m(b)}` is read in the black half of
/// row `{m(a), m(b)}`, and the white frame's reflected pair, in the white half of the row `{m(a), m(b)}`
/// again. The white half of that row is `T(other m(a), other m(b))`, which is the pair the white frame
/// reads once it has reflected its own tokens (`m(other(a)) = other(m(a))`).
///
/// Every fold reads one row a pair for both frames. Frames folded alike read the table at the tokens, or
/// at the tokens reflected; frames folded differently read the second table
/// ([`PairTable::mixed_cells`](super::PairTable::mixed_cells)), whose row `{a, b}` is the black half of `{m(a), m(b)}` and the white half
/// of `{a, b}`: at the tokens when black is the one folded, and at the tokens reflected when white is.
/// Each half is the half the frame's own read would give, from the same rows in the same order, so the
/// sums are the bits they would be if each frame read its rows alone, and a position none of whose frames
/// fold is the same bits it always was.
#[inline]
pub(super) fn touch_pairs_hm(
    sums: &mut [Vec<PairSum>; 2],
    net: &SymNetwork,
    token: FeatureIndex,
    others: &[FeatureIndex],
    sign: PairSum,
    folded: [bool; 2],
    update: [bool; 2],
) {
    use crate::tokens::{MAX_ACTIVE_FEATURES, mirror_row};

    if PAIR == 0 || !(update[0] || update[1]) {
        return;
    }
    // Held as halves, the table is read a block a pair, at the tokens as they are: each frame's channels from
    // the block's first row if the frame is not folded and its second if it is.
    if let Some(quad) = net.pair.quad() {
        let totals = quad_totals(quad, token, others, folded);
        for (frame, sum) in sums.iter_mut().enumerate() {
            if update[frame] {
                add_frame(sum, &totals, frame, sign);
            }
        }
        return;
    }
    let (cells, reflect) = match folded {
        [false, false] => (net.pair.cells(), false),
        [true, true] => (net.pair.cells(), true),
        [true, false] => (net.pair.mixed_cells(), false),
        [false, true] => (net.pair.mixed_cells(), true),
    };
    let totals = if reflect {
        let mut buffer = [0 as FeatureIndex; MAX_ACTIVE_FEATURES];
        for (slot, &other) in buffer.iter_mut().zip(others) {
            *slot = mirror_row(other);
        }
        row_totals(cells, mirror_row(token), &buffer[..others.len()])
    } else {
        row_totals(cells, token, others)
    };
    for (frame, sum) in sums.iter_mut().enumerate() {
        if update[frame] {
            add_frame(sum, &totals, frame, sign);
        }
    }
}

/// [`row_totals`] over `hm`'s blocks (`[PAIR_ROWS][2][ROW]` halves): for each of `others`, the black frame's channels
/// from the block's row `folded[0]` and the white frame's from its row `folded[1]`, added up in the order
/// [`touch_pairs`] gives. The halves are those [`row_totals`] reads for each frame's fold, in the same order, so the
/// totals are its to the last bit.
#[inline]
pub(super) fn quad_totals(
    quad: &[u16],
    token: FeatureIndex,
    others: &[FeatureIndex],
    folded: [bool; 2],
) -> [f32; ROW] {
    let lines = folded.map(|folded| usize::from(folded) * ROW);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 and F16C for the CPU it runs on.
        unsafe { quad_totals_avx2(quad, token, others, lines) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    )))]
    {
        quad_totals_scalar(quad, token, others, lines)
    }
}

/// [`quad_totals`] in plain `f32`: the reference the AVX2 path is held to.
#[cfg(any(
    all(test, feature = "sym_hm"),
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))
))]
pub(super) fn quad_totals_scalar(
    quad: &[u16],
    token: FeatureIndex,
    others: &[FeatureIndex],
    lines: [usize; 2],
) -> [f32; ROW] {
    use super::reader::half_to_f32;

    let entry = |other: FeatureIndex, slot: usize| -> f32 {
        let line = lines[usize::from((PAIR..ROW).contains(&slot))];
        let half = quad[pair_row(token, other) * 2 * ROW + line + slot];
        half_to_f32(half).expect("the reader found every half finite")
    };
    let (mut even, mut odd) = ([0.0f32; ROW], [0.0f32; ROW]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        for (slot, total) in even.iter_mut().enumerate() {
            *total += entry(a, slot);
        }
        for (slot, total) in odd.iter_mut().enumerate() {
            *total += entry(b, slot);
        }
    }
    for &other in last {
        for (slot, total) in even.iter_mut().enumerate() {
            *total += entry(other, slot);
        }
    }
    std::array::from_fn(|slot| even[slot] + odd[slot])
}

/// [`quad_totals_scalar`] eight channels to a register, as [`row_totals_avx2`] adds a row.
///
/// # Safety
/// The running CPU must support AVX2 and F16C. The `cfg` that compiles this in names them, and the build is for
/// the host CPU, so it does.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn quad_totals_avx2(
    quad: &[u16],
    token: FeatureIndex,
    others: &[FeatureIndex],
    lines: [usize; 2],
) -> [f32; ROW] {
    use std::arch::x86_64::*;

    const GROUPS: usize = ROW / LANES;
    // Every block read is `pair_row(token, other)`, below `PAIR_ROWS`, and the table is that many blocks long.
    assert_eq!(quad.len(), PAIR_ROWS * 2 * ROW);
    assert!(lines.iter().all(|&line| line == 0 || line == ROW));
    let base = quad.as_ptr();
    let offsets = row_offsets(token, others, 2 * ROW);
    // Where in a block each group of eight is read: the black channels from one row, the white from one.
    let at: [usize; GROUPS] = std::array::from_fn(|group| {
        lines[usize::from((PAIR..ROW).contains(&(group * LANES)))] + group * LANES
    });
    let block = |row: usize| -> *const u16 {
        // SAFETY: the block starts `pair_row · 2 · ROW` in, and the whole block is in the table.
        unsafe { base.add(offsets[row] as usize) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    let add_row = |totals: &mut [__m256; GROUPS], block: *const u16| {
        for (total, &at) in totals.iter_mut().zip(&at) {
            // SAFETY: `at + 8 <= 2 · ROW`, inside the block; the load is unaligned.
            let eight = unsafe { _mm_loadu_si128(block.add(at).cast()) };
            *total = _mm256_add_ps(*total, _mm256_cvtph_ps(eight));
        }
    };
    let mut row = 0;
    while row + 2 <= others.len() {
        add_row(&mut even, block(row));
        add_row(&mut odd, block(row + 1));
        row += 2;
    }
    if row < others.len() {
        add_row(&mut even, block(row));
    }
    let mut out = [0.0f32; ROW];
    for ((chunk, e), o) in out.as_chunks_mut::<LANES>().0.iter_mut().zip(even).zip(odd) {
        // SAFETY: `chunk` is eight `f32`s long; the store is unaligned.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_add_ps(e, o)) };
    }
    out
}

/// The rows of `token` against each of `others`, both frames' channels, added up
/// in the order [`touch_pairs`] gives.
#[inline]
pub(super) fn row_totals(
    cells: &Cells,
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    match cells {
        Cells::Halves(halves) => half_totals(halves, token, others),
        // The two byte tables add their rows in the same order; see `pair_i8.rs`.
        Cells::Cubes { q, scale } => cube_totals(q, scale, token, others),
        Cells::Scaled { q } => scaled_totals(q, token, others),
    }
}

/// [`row_totals`] for the table as the file has it, halves.
#[inline]
fn half_totals(halves: &[u16], token: FeatureIndex, others: &[FeatureIndex]) -> [f32; ROW] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 and F16C for the CPU
        // it runs on.
        unsafe { row_totals_avx2(halves, token, others) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    )))]
    {
        row_totals_scalar(halves, token, others)
    }
}

/// [`row_totals`] in plain `f32`, a half widened by the reader's own conversion:
/// the reference the AVX2 path is held to.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "f16c"
    ))
))]
pub(super) fn row_totals_scalar(
    halves: &[u16],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use super::reader::half_to_f32;

    let row = |other: FeatureIndex| -> &[u16] {
        let at = pair_row(token, other) * ROW;
        &halves[at..at + ROW]
    };
    let widen = |half: u16| half_to_f32(half).expect("the reader found every half finite");
    let (mut even, mut odd) = ([0.0f32; ROW], [0.0f32; ROW]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        for (total, &half) in even.iter_mut().zip(row(a)) {
            *total += widen(half);
        }
        for (total, &half) in odd.iter_mut().zip(row(b)) {
            *total += widen(half);
        }
    }
    for &other in last {
        for (total, &half) in even.iter_mut().zip(row(other)) {
            *total += widen(half);
        }
    }
    std::array::from_fn(|channel| even[channel] + odd[channel])
}

/// [`row_totals_scalar`] with eight channels of a total in one register, each
/// group of eight halves widened by `vcvtph2ps`, which is exact as the scalar
/// widening is.
///
/// A pair16 row is thirty-two halves, sixty-four bytes, and the table is
/// [`Aligned64`](crate::Aligned64), so a row is one cache line. The loads are
/// unaligned all the same, so a table of another width is still read correctly.
///
/// # Safety
/// The running CPU must support AVX2 and F16C. The `cfg` that compiles this in
/// names both, and the build is for the host CPU, so it does.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn row_totals_avx2(
    halves: &[u16],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use std::arch::x86_64::*;

    const GROUPS: usize = ROW / LANES;
    // Every row read is `pair_row(token, other)`, which is below `PAIR_ROWS`, and
    // the table is that many rows long.
    assert_eq!(halves.len(), PAIR_ROWS * ROW);
    let base = halves.as_ptr();
    let offsets = row_offsets(token, others, ROW);
    let row = |at: usize| -> *const u16 {
        // SAFETY: the row starts `pair_row · ROW` in, and the whole row is in the
        // table (asserted above).
        unsafe { base.add(offsets[at] as usize) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    // Group by group of eight halves: `group · 8 + 8 <= ROW`, inside the row.
    let add_row = |totals: &mut [__m256; GROUPS], row: *const u16| {
        for (group, total) in totals.iter_mut().enumerate() {
            // SAFETY: see above; the row is in the table, and the load is unaligned.
            let eight = unsafe { _mm_loadu_si128(row.add(group * LANES).cast()) };
            *total = _mm256_add_ps(*total, _mm256_cvtph_ps(eight));
        }
    };
    let mut at = 0;
    while at + 2 <= others.len() {
        add_row(&mut even, row(at));
        add_row(&mut odd, row(at + 1));
        at += 2;
    }
    if at < others.len() {
        add_row(&mut even, row(at));
    }
    let mut out = [0.0f32; ROW];
    for ((chunk, e), o) in out.as_chunks_mut::<LANES>().0.iter_mut().zip(even).zip(odd) {
        // SAFETY: `chunk` is eight `f32`s long; the store is unaligned.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_add_ps(e, o)) };
    }
    out
}
