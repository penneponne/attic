//! The pair rows of a token against the others present, added up when the table
//! is held as bytes (see [`PairPrecision`](super::PairPrecision)).
//!
//! The order of the additions is `pair.rs`'s, to the letter: the rows go to two
//! totals in turn, even-numbered to one and odd to the other, a last odd row to the
//! first, and the two are added at the end. Every function here has a plain form
//! and an AVX2 one, and the AVX2 one is held to the plain one to the last bit.
//!
//! The wrappers are not inlined, so that the `touch_pairs` the default (halves)
//! table runs stays the one function it was, and the byte forms cost one call to a
//! token's rows and not a copy of themselves in every caller.
//!
//! A byte is widened to `f32` before it is added, `vpmovsxbd` and `vcvtdq2ps`, so
//! that the totals are floats as the halves' are and the caller sees no difference.
//! The last multiply and the add are one fused step (`vfmadd231ps`). For the cubes
//! that changes nothing, since the product is exact and the add rounds once either
//! way; for the scaled bytes the plain form uses [`f32::mul_add`], which rounds
//! once as the instruction does.

use crate::tokens::FeatureIndex;

use super::pair::pair_row;
use super::pair_table::{ROW, SCALED_ROW};
use super::{PAIR, PAIR_ROWS};

/// The channels the AVX2 path adds at once: a register of `f32`.
const LANES: usize = 8;

/// Registers of a row's channels.
const GROUPS: usize = ROW / LANES;

const _: () = assert!(
    PAIR.is_multiple_of(LANES),
    "pair channels come in whole groups of eight"
);

/// The totals for a table of cubes: `q` is `[PAIR_ROWS][ROW]` bytes, `scale` what
/// each channel's sum of `q³` is multiplied by at the end.
///
/// The cube of a level is exact in `f32` (127³ is below 2²⁴), so only the sums
/// round, and they round the same way in both forms.
#[inline(never)]
pub(super) fn cube_totals(
    q: &[u8],
    scale: &[f32; ROW],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { cube_totals_avx2(q, scale, token, others) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        cube_totals_scalar(q, scale, token, others)
    }
}

/// [`cube_totals`] in plain `f32`: the reference the AVX2 form is held to.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))
))]
pub(super) fn cube_totals_scalar(
    q: &[u8],
    scale: &[f32; ROW],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    let row = |other: FeatureIndex| -> &[u8] {
        let at = pair_row(token, other) * ROW;
        &q[at..at + ROW]
    };
    // `level²` and `level³` are exact in `f32`, so the product rounds in no way a
    // fused multiply-add would not: the AVX2 form adds the cube in one step.
    let cube = |byte: u8| {
        let level = f32::from(byte as i8);
        level * level * level
    };
    let (mut even, mut odd) = ([0.0f32; ROW], [0.0f32; ROW]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        for (total, &byte) in even.iter_mut().zip(row(a)) {
            *total += cube(byte);
        }
        for (total, &byte) in odd.iter_mut().zip(row(b)) {
            *total += cube(byte);
        }
    }
    for &other in last {
        for (total, &byte) in even.iter_mut().zip(row(other)) {
            *total += cube(byte);
        }
    }
    std::array::from_fn(|channel| (even[channel] + odd[channel]) * scale[channel])
}

/// [`cube_totals_scalar`] with eight channels of a total in one register.
///
/// # Safety
/// The running CPU must support AVX2 and FMA. The `cfg` that compiles this in names
/// both, and the build is for the host CPU, so it does.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn cube_totals_avx2(
    q: &[u8],
    scale: &[f32; ROW],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use std::arch::x86_64::*;

    // Every row read is `pair_row(token, other)`, which is below `PAIR_ROWS`, and
    // the table is that many rows long.
    assert_eq!(q.len(), PAIR_ROWS * ROW);
    let base = q.as_ptr();
    let row = |other: FeatureIndex| -> *const u8 {
        // SAFETY: the row starts `pair_row · ROW` in, and the whole row is in the
        // table (asserted above).
        unsafe { base.add(pair_row(token, other) * ROW) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    // Group by group of eight bytes: `group · 8 + 8 <= ROW`, inside the row.
    let add_row = |totals: &mut [__m256; GROUPS], row: *const u8| {
        for (group, total) in totals.iter_mut().enumerate() {
            // SAFETY: see above; the row is in the table, and the load is unaligned.
            let eight = unsafe { _mm_loadl_epi64(row.add(group * LANES).cast()) };
            let level = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eight));
            *total = _mm256_fmadd_ps(_mm256_mul_ps(level, level), level, *total);
        }
    };
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        add_row(&mut even, row(a));
        add_row(&mut odd, row(b));
    }
    for &other in last {
        add_row(&mut even, row(other));
    }
    let mut out = [0.0f32; ROW];
    for (group, ((chunk, e), o)) in out
        .as_chunks_mut::<LANES>()
        .0
        .iter_mut()
        .zip(even)
        .zip(odd)
        .enumerate()
    {
        // SAFETY: `chunk` is eight `f32`s long and so is the part of `scale` read;
        // both are unaligned.
        unsafe {
            let gain = _mm256_loadu_ps(scale.as_ptr().add(group * LANES));
            _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_mul_ps(_mm256_add_ps(e, o), gain));
        }
    }
    out
}

/// A row's two scales, black frame's then white frame's.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))
))]
#[inline]
fn row_scales(row: &[u8]) -> (f32, f32) {
    let read =
        |at: usize| f32::from_le_bytes(row[at..at + 4].try_into().expect("a scale is four bytes"));
    (read(ROW), read(ROW + 4))
}

/// The totals for a table with a scale to each half of a row: `q` is
/// `[PAIR_ROWS][SCALED_ROW]` bytes.
///
/// Each entry is widened and multiplied by its half's scale as it is added, in one
/// fused step, so the totals are sums of the entries themselves.
#[inline(never)]
pub(super) fn scaled_totals(q: &[u8], token: FeatureIndex, others: &[FeatureIndex]) -> [f32; ROW] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: the `cfg` above is the build enabling AVX2 for the CPU it runs on.
        unsafe { scaled_totals_avx2(q, token, others) }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        scaled_totals_scalar(q, token, others)
    }
}

/// [`scaled_totals`] in plain `f32`: the reference the AVX2 form is held to.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))
))]
pub(super) fn scaled_totals_scalar(
    q: &[u8],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    let row = |other: FeatureIndex| -> &[u8] {
        let at = pair_row(token, other) * SCALED_ROW;
        &q[at..at + SCALED_ROW]
    };
    let add = |totals: &mut [f32; ROW], row: &[u8]| {
        let (black, white) = row_scales(row);
        let (low, high) = totals.split_at_mut(PAIR);
        for (total, &byte) in low.iter_mut().zip(&row[..PAIR]) {
            *total = f32::from(byte as i8).mul_add(black, *total);
        }
        for (total, &byte) in high.iter_mut().zip(&row[PAIR..ROW]) {
            *total = f32::from(byte as i8).mul_add(white, *total);
        }
    };
    let (mut even, mut odd) = ([0.0f32; ROW], [0.0f32; ROW]);
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        add(&mut even, row(a));
        add(&mut odd, row(b));
    }
    for &other in last {
        add(&mut even, row(other));
    }
    std::array::from_fn(|channel| even[channel] + odd[channel])
}

/// [`scaled_totals_scalar`] with eight channels of a total in one register.
///
/// # Safety
/// The running CPU must support AVX2 and FMA. The `cfg` that compiles this in names
/// both, and the build is for the host CPU, so it does.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn scaled_totals_avx2(
    q: &[u8],
    token: FeatureIndex,
    others: &[FeatureIndex],
) -> [f32; ROW] {
    use std::arch::x86_64::*;

    assert_eq!(q.len(), PAIR_ROWS * SCALED_ROW);
    let base = q.as_ptr();
    let row = |other: FeatureIndex| -> *const u8 {
        // SAFETY: the row starts `pair_row · SCALED_ROW` in, and the whole row is in
        // the table (asserted above).
        unsafe { base.add(pair_row(token, other) * SCALED_ROW) }
    };
    let mut even = [_mm256_setzero_ps(); GROUPS];
    let mut odd = [_mm256_setzero_ps(); GROUPS];
    let add_row = |totals: &mut [__m256; GROUPS], row: *const u8| {
        // SAFETY: the two scales are the last eight bytes of the row; the reads are
        // unaligned.
        let (black, white) = unsafe {
            (
                _mm256_set1_ps(row.add(ROW).cast::<f32>().read_unaligned()),
                _mm256_set1_ps(row.add(ROW + 4).cast::<f32>().read_unaligned()),
            )
        };
        // The black frame's groups come first, then the white frame's.
        let (low, high) = totals.split_at_mut(PAIR / LANES);
        for (group, total) in low.iter_mut().enumerate() {
            // SAFETY: `group · 8 + 8 <= ROW`, inside the row; the load is unaligned.
            let eight = unsafe { _mm_loadl_epi64(row.add(group * LANES).cast()) };
            let level = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eight));
            *total = _mm256_fmadd_ps(level, black, *total);
        }
        for (group, total) in high.iter_mut().enumerate() {
            // SAFETY: as above, the group being `PAIR / 8` further on.
            let eight = unsafe { _mm_loadl_epi64(row.add((PAIR / LANES + group) * LANES).cast()) };
            let level = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(eight));
            *total = _mm256_fmadd_ps(level, white, *total);
        }
    };
    let (twos, last) = others.as_chunks::<2>();
    for &[a, b] in twos {
        add_row(&mut even, row(a));
        add_row(&mut odd, row(b));
    }
    for &other in last {
        add_row(&mut even, row(other));
    }
    let mut out = [0.0f32; ROW];
    for ((chunk, e), o) in out.as_chunks_mut::<LANES>().0.iter_mut().zip(even).zip(odd) {
        // SAFETY: `chunk` is eight `f32`s long; the store is unaligned.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_add_ps(e, o)) };
    }
    out
}
