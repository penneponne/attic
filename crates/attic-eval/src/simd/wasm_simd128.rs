//! WebAssembly `simd128` feature-transformer accumulate/update kernels.
//!
//! The same operations as `crate::simd::avx2`, on 128-bit vectors of eight `i16`. Selected at
//! compile time when the build enables `simd128` on a wasm target; every other build compiles
//! this file out.
//!
//! The kernels are pure element-wise `i16` add/sub over contiguous columns, so there is nothing
//! to emulate: `i16x8.add` and `i16x8.sub` wrap exactly as the scalar baseline's
//! `wrapping_add`/`wrapping_sub` do, and the integer sum commutes, so applying the columns in
//! any order gives the scalar bits. Nothing here uses the relaxed-simd proposal, whose
//! operations may differ between hosts.
//!
//! The work is tiled: [`TILE_REGS`] vectors (64 lanes, 128 bytes) are loaded from the source, every
//! column of the change goes over them while they stay in registers, and they are stored once.
//! A column's slice of a tile is two cache lines, so a column is read as whole lines, and the
//! accumulator takes one load and one store per tile however many features moved.
//!
//! The entry points are safe functions with the scalar kernels' signatures. They check the
//! lengths and the column indices once, up front, and the loops below load by pointer on that
//! account.

use core::arch::wasm32::*;

use crate::features::FeatureIndex;
use crate::types::HIDDEN_SIZE;

/// `i16` lanes in a `v128`.
const LANES: usize = 8;

/// Vectors a tile holds.
const TILE_REGS: usize = 8;

/// `i16` lanes in a tile.
const TILE: usize = TILE_REGS * LANES;

const _: () = assert!(
    HIDDEN_SIZE.is_multiple_of(TILE),
    "every shape's accumulator is whole tiles"
);

/// The one release-mode check the pointer loads rely on: every index names a column inside
/// `weights`. Compared as `index < columns` so that no product can wrap `usize` (32 bits on
/// wasm32).
#[inline]
fn assert_columns(weights: &[i16], lists: [&[FeatureIndex]; 3]) {
    let columns = weights.len() / HIDDEN_SIZE;
    for list in lists {
        assert!(
            list.iter().all(|&idx| (idx as usize) < columns),
            "a feature column is out of range"
        );
    }
}

/// `dst = src + sum(added) - sum(removed_a) - sum(removed_b)`, per lane, wrapping.
///
/// `dst` may be `src` (the in-place updates): a tile is loaded whole before it is stored, and
/// no other tile is touched.
///
/// # Safety
/// `dst` and `src` must each be valid for `HIDDEN_SIZE` `i16` (`dst` for writes), and every
/// index in the three lists must name a whole column of `weights` (see [`assert_columns`]).
#[inline(always)]
unsafe fn combine(
    dst: *mut i16,
    src: *const i16,
    weights: &[i16],
    added: &[FeatureIndex],
    removed_a: &[FeatureIndex],
    removed_b: &[FeatureIndex],
) {
    let w = weights.as_ptr();
    for tile in 0..HIDDEN_SIZE / TILE {
        let off = tile * TILE;
        // SAFETY: `off + TILE <= HIDDEN_SIZE`, the length of `src` and `dst`, and each column
        // slice starts at `idx * HIDDEN_SIZE + off` with `idx` a column of `weights`, so its
        // `TILE` lanes lie inside `weights`. `v128_load` and `v128_store` have no alignment
        // requirement.
        unsafe {
            let mut acc = [i16x8_splat(0); TILE_REGS];
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot = v128_load(src.add(off + k * LANES).cast::<v128>());
            }
            for &idx in added {
                let col = w.add(idx as usize * HIDDEN_SIZE + off);
                for (k, slot) in acc.iter_mut().enumerate() {
                    *slot = i16x8_add(*slot, v128_load(col.add(k * LANES).cast::<v128>()));
                }
            }
            for list in [removed_a, removed_b] {
                for &idx in list {
                    let col = w.add(idx as usize * HIDDEN_SIZE + off);
                    for (k, slot) in acc.iter_mut().enumerate() {
                        *slot = i16x8_sub(*slot, v128_load(col.add(k * LANES).cast::<v128>()));
                    }
                }
            }
            for (k, slot) in acc.iter().enumerate() {
                v128_store(dst.add(off + k * LANES).cast::<v128>(), *slot);
            }
        }
    }
}

/// [`combine`] on `out` in place.
#[inline]
fn combine_in_place(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed_a: &[FeatureIndex],
    removed_b: &[FeatureIndex],
) {
    assert_eq!(out.len(), HIDDEN_SIZE);
    assert_columns(weights, [added, removed_a, removed_b]);
    let p = out.as_mut_ptr();
    // SAFETY: `out` is `HIDDEN_SIZE` long and the columns are checked above; `p` is the only
    // pointer into `out` in use, read and written through in tile order.
    unsafe { combine(p, p, weights, added, removed_a, removed_b) }
}

/// Add each active feature's FT weight column into `out`.
pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    combine_in_place(out, weights, indices, &[], &[]);
}

/// Subtract each feature's FT weight column from `out`.
pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
    combine_in_place(out, weights, &[], indices, &[]);
}

/// Fused add / sub delta.
pub fn add_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    combine_in_place(out, weights, added, removed, &[]);
}

/// Fused add / double-sub delta (a capture).
pub fn add_sub_sub_features(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed_a: &[FeatureIndex],
    removed_b: &[FeatureIndex],
) {
    combine_in_place(out, weights, added, removed_a, removed_b);
}

/// One-pass derive: `dst = src + sum(added) - sum(removed)`, per lane, wrapping.
///
/// Each tile is loaded from `src` once, has every column applied in registers, and is stored
/// to `dst` once, where copy-then-apply would read and write the whole half twice.
pub fn derive_features(
    dst: &mut [i16],
    src: &[i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    assert_eq!(dst.len(), HIDDEN_SIZE);
    assert_eq!(src.len(), HIDDEN_SIZE);
    assert_columns(weights, [added, removed, &[]]);
    // SAFETY: both halves are `HIDDEN_SIZE` long and the columns are checked above; `dst` and
    // `src` are distinct borrows, so they do not overlap.
    unsafe { combine(dst.as_mut_ptr(), src.as_ptr(), weights, added, removed, &[]) }
}

/// Take the `removed` threat rows off `out` and put the `added` ones on, in place.
///
/// A threat row is `HIDDEN_SIZE` signed bytes, widened to `i16` lane by lane
/// (`i16x8.load8x8_s`, one instruction per eight lanes) and added unscaled, wrapping like the
/// feature columns. A tile's row slice is [`TILE`] bytes, one cache line.
#[cfg(feature = "sfnn_threat")]
pub fn apply_threat_rows(out: &mut [i16], weights: &[i8], removed: &[u32], added: &[u32]) {
    assert_eq!(out.len(), HIDDEN_SIZE);
    let rows = weights.len() / HIDDEN_SIZE;
    // Checked once here so that the loops below load with plain pointers.
    assert!(
        removed
            .iter()
            .chain(added)
            .all(|&row| (row as usize) < rows),
        "a threat row is out of range"
    );
    let out_ptr = out.as_mut_ptr();
    let w_ptr = weights.as_ptr();
    for tile in 0..HIDDEN_SIZE / TILE {
        let off = tile * TILE;
        // SAFETY: `off + TILE <= HIDDEN_SIZE = out.len()`, and each row slice starts at
        // `row * HIDDEN_SIZE + off` with `row < rows`, so its `TILE` bytes (eight per register)
        // lie inside `weights`.
        unsafe {
            let mut acc = [i16x8_splat(0); TILE_REGS];
            for (k, slot) in acc.iter_mut().enumerate() {
                *slot = v128_load(out_ptr.add(off + k * LANES).cast::<v128>());
            }
            for &row in removed {
                let base = w_ptr.add(row as usize * HIDDEN_SIZE + off);
                for (k, slot) in acc.iter_mut().enumerate() {
                    *slot = i16x8_sub(*slot, i16x8_load_extend_i8x8(base.add(k * LANES)));
                }
            }
            for &row in added {
                let base = w_ptr.add(row as usize * HIDDEN_SIZE + off);
                for (k, slot) in acc.iter_mut().enumerate() {
                    *slot = i16x8_add(*slot, i16x8_load_extend_i8x8(base.add(k * LANES)));
                }
            }
            for (k, slot) in acc.iter().enumerate() {
                v128_store(out_ptr.add(off + k * LANES).cast::<v128>(), *slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;

    /// A small deterministic xorshift64*, so the inputs are the same on every host.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn i16s(&mut self, n: usize) -> Vec<i16> {
            // Whole-range values, with the extremes over-represented, so that the wrap is
            // exercised and not only the sums that stay small.
            (0..n)
                .map(|_| match self.next() % 8 {
                    0 => i16::MAX,
                    1 => i16::MIN,
                    _ => (self.next() >> 24) as i16,
                })
                .collect()
        }

        fn indices(&mut self, n: usize, columns: usize) -> Vec<FeatureIndex> {
            (0..n)
                .map(|_| (self.next() % columns as u64) as FeatureIndex)
                .collect()
        }
    }

    const COLUMNS: usize = 12;

    fn corpus(seed: u64) -> (Rng, Vec<i16>) {
        let mut rng = Rng(seed);
        let weights = rng.i16s(COLUMNS * HIDDEN_SIZE);
        (rng, weights)
    }

    #[test]
    fn add_features_matches_scalar() {
        let (mut rng, weights) = corpus(0x1234_5678_9abc_def1);
        for n in [0usize, 1, 2, 5, 11] {
            let idx = rng.indices(n, COLUMNS);
            let mut want = rng.i16s(HIDDEN_SIZE);
            let mut got = want.clone();
            scalar::add_features(&mut want, &weights, &idx);
            add_features(&mut got, &weights, &idx);
            assert_eq!(want, got, "{n} columns");
        }
    }

    #[test]
    fn sub_features_matches_scalar() {
        let (mut rng, weights) = corpus(0x2468_ace0_1357_9bd3);
        for n in [0usize, 1, 2, 5, 11] {
            let idx = rng.indices(n, COLUMNS);
            let mut want = rng.i16s(HIDDEN_SIZE);
            let mut got = want.clone();
            scalar::sub_features(&mut want, &weights, &idx);
            sub_features(&mut got, &weights, &idx);
            assert_eq!(want, got, "{n} columns");
        }
    }

    #[test]
    fn add_sub_features_matches_scalar() {
        let (mut rng, weights) = corpus(0x0f0f_1e1e_2d2d_3c3c);
        for (na, nr) in [(0usize, 0usize), (1, 0), (0, 1), (1, 1), (2, 3), (5, 4)] {
            let added = rng.indices(na, COLUMNS);
            let removed = rng.indices(nr, COLUMNS);
            let mut want = rng.i16s(HIDDEN_SIZE);
            let mut got = want.clone();
            scalar::add_sub_features(&mut want, &weights, &added, &removed);
            add_sub_features(&mut got, &weights, &added, &removed);
            assert_eq!(want, got, "added {na} removed {nr}");
        }
    }

    #[test]
    fn add_sub_sub_features_matches_scalar() {
        let (mut rng, weights) = corpus(0x7777_aaaa_5555_cccc);
        for (na, nra, nrb) in [(0usize, 0usize, 0usize), (1, 1, 1), (1, 2, 0), (3, 2, 4)] {
            let added = rng.indices(na, COLUMNS);
            let removed_a = rng.indices(nra, COLUMNS);
            let removed_b = rng.indices(nrb, COLUMNS);
            let mut want = rng.i16s(HIDDEN_SIZE);
            let mut got = want.clone();
            scalar::add_sub_sub_features(&mut want, &weights, &added, &removed_a, &removed_b);
            add_sub_sub_features(&mut got, &weights, &added, &removed_a, &removed_b);
            assert_eq!(want, got, "added {na} removed {nra}+{nrb}");
        }
    }

    #[test]
    fn derive_features_matches_copy_then_apply_for_every_arity() {
        let (mut rng, weights) = corpus(0x0bad_cafe_f00d_d00d);
        for na in 0..=3usize {
            for nr in 0..=3usize {
                let added = rng.indices(na, COLUMNS);
                let removed = rng.indices(nr, COLUMNS);
                let src = rng.i16s(HIDDEN_SIZE);
                let mut want = vec![0i16; HIDDEN_SIZE];
                scalar::derive_features(&mut want, &src, &weights, &added, &removed);
                let mut got = rng.i16s(HIDDEN_SIZE);
                derive_features(&mut got, &src, &weights, &added, &removed);
                assert_eq!(want, got, "added {na} removed {nr}");
            }
        }
    }

    /// The same column on both sides must cancel exactly, wrapping included.
    #[test]
    fn derive_features_cancels_a_column_added_and_removed() {
        let (mut rng, weights) = corpus(0x5eed_5eed_5eed_5eed);
        let src = rng.i16s(HIDDEN_SIZE);
        let mut got = vec![0i16; HIDDEN_SIZE];
        derive_features(&mut got, &src, &weights, &[4], &[4]);
        assert_eq!(got, src);
    }

    #[test]
    fn add_then_sub_the_same_columns_is_the_identity() {
        let (mut rng, weights) = corpus(0x1111_2222_3333_4444);
        let idx = rng.indices(7, COLUMNS);
        let init = rng.i16s(HIDDEN_SIZE);
        let mut got = init.clone();
        add_features(&mut got, &weights, &idx);
        sub_features(&mut got, &weights, &idx);
        assert_eq!(got, init);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn a_column_past_the_block_is_refused() {
        let weights = vec![0i16; 2 * HIDDEN_SIZE];
        let mut out = vec![0i16; HIDDEN_SIZE];
        add_features(&mut out, &weights, &[2]);
    }

    /// The threat-row kernel against the scalar one, over rows with every byte value and an
    /// accumulator near the `i16` limits, so that the widening and the wrap both matter.
    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn apply_threat_rows_matches_scalar() {
        const ROWS: usize = 12;
        let weights: Vec<i8> = (0..ROWS * HIDDEN_SIZE)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) as i8)
            .collect();
        assert!(weights.contains(&i8::MIN) && weights.contains(&i8::MAX));
        let init: Vec<i16> = (0..HIDDEN_SIZE)
            .map(|i| match i % 4 {
                0 => i16::MAX - (i % 7) as i16,
                1 => i16::MIN + (i % 5) as i16,
                _ => (i as i16).wrapping_mul(97),
            })
            .collect();
        let shapes: [(&[u32], &[u32]); 5] = [
            (&[], &[]),
            (&[], &[0, 3, 11]),
            (&[2, 5], &[]),
            (&[1, 4, 9], &[0, 4, 10, 11]),
            (&[7, 7], &[7, 7, 7]),
        ];
        for (removed, added) in shapes {
            let mut want = init.clone();
            scalar::apply_threat_rows(&mut want, &weights, removed, added);
            let mut got = init.clone();
            apply_threat_rows(&mut got, &weights, removed, added);
            assert_eq!(want, got, "removed {removed:?} added {added:?}");
        }
    }

    /// A row put on and taken off again leaves the accumulator as it was.
    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn apply_threat_rows_undoes_itself() {
        let weights: Vec<i8> = (0..3 * HIDDEN_SIZE).map(|i| (i * 31) as i8).collect();
        let init = Rng(29).i16s(HIDDEN_SIZE);
        let mut got = init.clone();
        apply_threat_rows(&mut got, &weights, &[], &[0, 1, 2]);
        apply_threat_rows(&mut got, &weights, &[2, 0, 1], &[]);
        assert_eq!(got, init);
    }

    #[cfg(feature = "sfnn_threat")]
    #[test]
    #[should_panic(expected = "out of range")]
    fn apply_threat_rows_refuses_a_row_past_the_block() {
        let weights = vec![0i8; 2 * HIDDEN_SIZE];
        let mut out = vec![0i16; HIDDEN_SIZE];
        apply_threat_rows(&mut out, &weights, &[], &[2]);
    }
}
