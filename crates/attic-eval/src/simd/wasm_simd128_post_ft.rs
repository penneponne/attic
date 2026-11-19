//! WebAssembly `simd128` post-feature-transformer kernels: the output transform and the dense
//! layers.
//!
//! The 128-bit counterparts of `crate::simd::avx2_post_ft`, for a build that enables `simd128`
//! on a wasm target. Every kernel here equals [`crate::simd::scalar_post_ft`] bit for bit, and
//! the optional relaxed dot product is used only for unsigned 7-bit inputs, where its
//! result is defined and equal across hosts.
//!
//! ## The element-wise multiply
//!
//! `(clamp(a, 0, 254) * clamp(b, 0, 254)) >> 9`: the product is at most `254 * 254 = 64516`,
//! which fits an **unsigned** 16-bit lane, so `i16x8.mul` (the low 16 bits of the product) is
//! the exact product and a logical `u16x8.shr` finishes it, with no widening. The file-scale
//! form clamps to 127 and shifts by 7; its product, `16129`, fits either way.
//!
//! ## The dense layers
//!
//! Where AVX2 has `maddubs` (unsigned byte times signed byte, pairs added with saturation),
//! standard WASM SIMD widens both operands to `i16` and adds each pair of products into an
//! `i32` with `i32x4.dot_i16x8_s`. This computes the exact sum for every input byte. The standard SIMD path serves
//! `affine`, `affine_u7` and `affine_u7_wide` alike. The input is widened once per 16-byte chunk
//! and shared by the four rows a block processes. Relaxed SIMD can consume 7-bit activations
//! directly in each of those four rows.

use core::arch::wasm32::*;

use crate::simd::scalar_post_ft;
use crate::types::HIDDEN_SIZE;

const EWM_HALF: usize = HIDDEN_SIZE / 2;

/// `i16` lanes in a `v128`.
const I16_LANES: usize = 8;

/// Input bytes one 128-bit step of the affine kernels consumes.
const U8_LANES: usize = 16;

const _: () = assert!(
    EWM_HALF.is_multiple_of(2 * I16_LANES),
    "the output transform stores whole 16-byte vectors"
);

/// Right-shift applied to the affine output before the clipped ReLU (`kWeightScaleBits`).
const WEIGHT_SCALE_BITS: u32 = 6;

/// Right-shift applied to the squared affine output: the square carries `2 * 6` fractional bits
/// of the weight scale and 7 of the activation scale.
const SQR_SHIFT: u32 = 2 * WEIGHT_SCALE_BITS + 7;

/// A square at or above `2^26` shifts to 128 or more and clamps to 127, so the magnitude is
/// capped there and the square of what is left cannot leave an `i32`.
const SQR_CAP: i32 = 1 << ((SQR_SHIFT + 7) / 2);

const _: () = assert!(
    ((SQR_CAP as i64) * (SQR_CAP as i64)) >> SQR_SHIFT == 128,
    "the cap is the first magnitude that clamps"
);

/// The pairwise multiply of one perspective, shared by the two scales: lane `j` against lane
/// `j + HIDDEN_SIZE/2`, each clamped to `[0, cap]`, multiplied and shifted right by `shift`.
///
/// `cap` and `shift` must keep the product inside an unsigned 16-bit lane (`cap * cap < 2^16`),
/// which holds for the two callers (254 and 127).
#[inline(always)]
fn ewm<const CAP: i16, const SHIFT: u32>(half: &[i16], out: &mut [u8]) {
    assert_eq!(half.len(), HIDDEN_SIZE);
    assert_eq!(out.len(), EWM_HALF);
    const { assert!((CAP as i32) * (CAP as i32) < (1 << 16)) };

    let zero = i16x8_splat(0);
    let cap = i16x8_splat(CAP);
    let h = half.as_ptr();
    let o = out.as_mut_ptr();
    for chunk in 0..EWM_HALF / (2 * I16_LANES) {
        let off = chunk * 2 * I16_LANES;
        // SAFETY: `off + 16 <= EWM_HALF`, so the four loads at `off` and `off + 8` in each half
        // and the 16-byte store at `off` stay inside `half` (`2 * EWM_HALF` lanes) and `out`
        // (`EWM_HALF` bytes).
        unsafe {
            let mut prod = [zero; 2];
            for (i, slot) in prod.iter_mut().enumerate() {
                let at = off + i * I16_LANES;
                let a = v128_load(h.add(at).cast::<v128>());
                let b = v128_load(h.add(at + EWM_HALF).cast::<v128>());
                let a = i16x8_max(i16x8_min(a, cap), zero);
                let b = i16x8_max(i16x8_min(b, cap), zero);
                *slot = u16x8_shr(i16x8_mul(a, b), SHIFT);
            }
            // Both are in `[0, 126]`, so the unsigned saturating narrow keeps them as they are.
            v128_store(
                o.add(off).cast::<v128>(),
                u8x16_narrow_i16x8(prod[0], prod[1]),
            );
        }
    }
}

/// Pairwise element-wise multiply for one perspective.
///
/// Lane `j` pairs with lane `j + HIDDEN_SIZE/2`; both are clamped to `[0, 254]` (the post-×2
/// scale of the standard and progress shapes), multiplied, and shifted right by 9.
pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
    ewm::<{ 127 * 2 }, 9>(half, out);
}

/// [`ewm_one_perspective`] for an accumulator at the scale the file holds (the threat shape):
/// both lanes clamped to `[0, 127]`, the product shifted right by 7.
#[cfg(feature = "sfnn_threat")]
pub fn ewm_one_perspective_file_scale(half: &[i16], out: &mut [u8]) {
    ewm::<127, 7>(half, out);
}

/// The standard shapes' output transform for one perspective: each accumulator lane clamped to
/// `[0, 127]` and narrowed to a byte (`nnue_feature_transformer.h`'s clipped ReLU, with no
/// multiply and no shift).
///
/// The unsigned saturating narrow takes a negative lane to 0 and a lane above 255 to 255, and the
/// minimum with 127 finishes the clamp, so the two steps equal `clamp(v, 0, 127)` for every
/// `i16`. `HIDDEN_SIZE` is whole 16-lane steps for every standard shape (128, 256 and 768).
pub fn clip_to_u7(half: &[i16], out: &mut [u8]) {
    assert_eq!(half.len(), out.len());
    let (lanes, rest) = half.as_chunks::<16>();
    let (bytes, tail) = out.as_chunks_mut::<16>();
    let cap = u8x16_splat(127);
    for (inp, out) in lanes.iter().zip(bytes) {
        // SAFETY: `inp` is 16 `i16`, two vectors, and `out` is 16 bytes, one.
        unsafe {
            let a = v128_load(inp.as_ptr().cast::<v128>());
            let b = v128_load(inp.as_ptr().add(I16_LANES).cast::<v128>());
            v128_store(
                out.as_mut_ptr().cast::<v128>(),
                u8x16_min(u8x16_narrow_i16x8(a, b), cap),
            );
        }
    }
    for (slot, &sum) in tail.iter_mut().zip(rest) {
        *slot = sum.clamp(0, 127) as u8;
    }
}

/// Narrow sixteen `i32` (four vectors) to bytes: `i32 -> i16` saturating, then `i16 -> u8`
/// saturating (a negative lane goes to 0), then the cap at 127.
///
/// The three steps equal `clamp(v, 0, 127)` for every `i32`: a value above 127 stays above 127
/// through both saturations and meets the cap, a negative one reaches 0 at the unsigned narrow,
/// and one in between is untouched.
#[inline(always)]
fn narrow_to_u7(a: v128, b: v128, c: v128, d: v128) -> v128 {
    let lo = i16x8_narrow_i32x4(a, b);
    let hi = i16x8_narrow_i32x4(c, d);
    u8x16_min(u8x16_narrow_i16x8(lo, hi), u8x16_splat(127))
}

/// Runs `step` over `input` sixteen lanes at a time, and over a final zero-padded block for the
/// tail, so that every length — the SFNN chains' 15 among them — takes the vector path. Only
/// the tail's own bytes of that block's result are kept.
#[inline(always)]
fn for_blocks(input: &[i32], output: &mut [u8], step: impl Fn(&[i32; 16]) -> v128) {
    assert_eq!(input.len(), output.len());
    let (blocks, rest) = input.as_chunks::<16>();
    let (outs, tail) = output.as_chunks_mut::<16>();
    for (block, out) in blocks.iter().zip(outs) {
        let bytes = step(block);
        // SAFETY: `out` is 16 bytes.
        unsafe { v128_store(out.as_mut_ptr().cast::<v128>(), bytes) };
    }
    if !rest.is_empty() {
        let mut block = [0i32; 16];
        block[..rest.len()].copy_from_slice(rest);
        let bytes = step(&block);
        let mut buf = [0u8; 16];
        // SAFETY: `buf` is 16 bytes.
        unsafe { v128_store(buf.as_mut_ptr().cast::<v128>(), bytes) };
        tail.copy_from_slice(&buf[..tail.len()]);
    }
}

/// Four `i32` vectors from a 16-element block.
#[inline(always)]
fn load_block(block: &[i32; 16]) -> [v128; 4] {
    let p = block.as_ptr();
    // SAFETY: the block is 16 `i32`, four vectors.
    unsafe {
        [
            v128_load(p.cast::<v128>()),
            v128_load(p.add(4).cast::<v128>()),
            v128_load(p.add(8).cast::<v128>()),
            v128_load(p.add(12).cast::<v128>()),
        ]
    }
}

/// `output = clamp(input >> WEIGHT_SCALE_BITS, 0, 127)`.
pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
    for_blocks(input, output, |block| {
        let [a, b, c, d] = load_block(block);
        narrow_to_u7(
            i32x4_shr(a, WEIGHT_SCALE_BITS),
            i32x4_shr(b, WEIGHT_SCALE_BITS),
            i32x4_shr(c, WEIGHT_SCALE_BITS),
            i32x4_shr(d, WEIGHT_SCALE_BITS),
        )
    });
}

/// `output = clamp(input² >> SQR_SHIFT, 0, 127)`, the scalar kernel's `i64` square without the
/// `i64`.
///
/// The magnitude is taken (`i32::MIN` stays negative as an `i32` but is `2^31` as a `u32`, which
/// is why the cap is an unsigned minimum) and capped at [`SQR_CAP`], the first magnitude whose
/// square already clamps. What is left squares inside an `i32`, and the square is shifted
/// logically.
pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    let cap = i32x4_splat(SQR_CAP);
    let sq = |v: v128| {
        let m = u32x4_min(i32x4_abs(v), cap);
        u32x4_shr(i32x4_mul(m, m), SQR_SHIFT)
    };
    for_blocks(input, output, |block| {
        let [a, b, c, d] = load_block(block);
        narrow_to_u7(sq(a), sq(b), sq(c), sq(d))
    });
}

/// The one release-mode check the pointer loads in the affine kernel rely on: every slice is as
/// long as the shape says. One `assert!` over a combined condition, so the hot path has a single
/// not-taken branch; `checked_mul` also rules out an `output.len() * padded_in` that wraps around
/// to a matching `weights.len()`.
#[inline(always)]
fn assert_affine_shape(
    output_len: usize,
    biases_len: usize,
    weights_len: usize,
    input_len: usize,
    in_dims: usize,
    padded_in: usize,
) {
    assert!(
        output_len == biases_len
            && in_dims <= padded_in
            && input_len >= in_dims
            && output_len.checked_mul(padded_in) == Some(weights_len),
        "affine kernel: slice lengths do not match the layer shape"
    );
}

/// The input chunk at `off`, widened to two vectors of `i16` (zero-extended: exact for every
/// byte).
#[inline(always)]
fn widen_input(x: v128) -> (v128, v128) {
    (i16x8_extend_low_u8x16(x), i16x8_extend_high_u8x16(x))
}

/// `acc + dot(x, w)` over one 16-byte chunk: the weights sign-extended, each pair of products
/// added into an `i32` lane.
#[inline(always)]
fn dot_chunk(acc: v128, x: (v128, v128), w: v128) -> v128 {
    let lo = i32x4_dot_i16x8(x.0, i16x8_extend_low_i8x16(w));
    let hi = i32x4_dot_i16x8(x.1, i16x8_extend_high_i8x16(w));
    i32x4_add(acc, i32x4_add(lo, hi))
}

// The signed weights come first. With unsigned 7-bit activations, two products
// sum to at least -32512 and at most 32258, so an i16 intermediate cannot saturate.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn dot_u7_chunk(acc: v128, input: v128, weights: v128) -> v128 {
    i32x4_relaxed_dot_i8x16_i7x16_add(weights, input, acc)
}

/// The four `i32` lanes of `a` added together.
#[inline(always)]
fn hsum(a: v128) -> i32 {
    let s = i32x4_add(a, i32x4_shuffle::<2, 3, 0, 1>(a, a));
    let s = i32x4_add(s, i32x4_shuffle::<1, 0, 3, 2>(s, s));
    i32x4_extract_lane::<0>(s)
}

/// The lane sums of four accumulators as one vector `[sum(a0), sum(a1), sum(a2), sum(a3)]`.
#[inline(always)]
fn hsum4(a0: v128, a1: v128, a2: v128, a3: v128) -> v128 {
    let s01 = i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(a0, a1),
        i32x4_shuffle::<1, 3, 5, 7>(a0, a1),
    );
    let s23 = i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(a2, a3),
        i32x4_shuffle::<1, 3, 5, 7>(a2, a3),
    );
    i32x4_add(
        i32x4_shuffle::<0, 2, 4, 6>(s01, s23),
        i32x4_shuffle::<1, 3, 5, 7>(s01, s23),
    )
}

/// Integer affine transform: `output[j] = biases[j] + sum_i weights[j][i] * input[i]`, with
/// `i32` wrapping, weights row-major `[output][padded_in]`, for **any** `u8` input.
///
/// Four output rows are processed together so each 16-byte input chunk is widened once for four
/// weight rows; a trailing `output.len() % 4` rows go one at a time. An `in_dims` that is not a
/// multiple of 16 falls back to the scalar kernel (no current shape has one). `i32` addition
/// wraps and commutes, so summing in per-lane partials leaves the bits unchanged.
pub fn affine(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    affine_impl::<false>(output, biases, weights, input, in_dims, padded_in);
}

fn affine_impl<const U7: bool>(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    // A real assert, not a debug one: the loads below go through raw pointers and trust these
    // lengths.
    assert_affine_shape(
        output.len(),
        biases.len(),
        weights.len(),
        input.len(),
        in_dims,
        padded_in,
    );

    if !in_dims.is_multiple_of(U8_LANES) {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
        return;
    }

    let chunks = in_dims / U8_LANES;
    let rows = output.len();
    let in_ptr = input.as_ptr();
    let w_ptr = weights.as_ptr();
    let b_ptr = biases.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut j = 0;
    while j + 4 <= rows {
        // SAFETY: `j + 4 <= rows` and `weights.len() == rows * padded_in`, so the four row bases
        // are in range, and each 16-byte load at `off + 16 <= in_dims <= padded_in` is inside
        // the input and inside its weight row.
        unsafe {
            let r0 = w_ptr.add(j * padded_in);
            let r1 = w_ptr.add((j + 1) * padded_in);
            let r2 = w_ptr.add((j + 2) * padded_in);
            let r3 = w_ptr.add((j + 3) * padded_in);
            let mut a0 = i32x4_splat(0);
            let mut a1 = i32x4_splat(0);
            let mut a2 = i32x4_splat(0);
            let mut a3 = i32x4_splat(0);
            for c in 0..chunks {
                let off = c * U8_LANES;
                let input = v128_load(in_ptr.add(off).cast::<v128>());
                #[cfg(target_feature = "relaxed-simd")]
                if U7 {
                    a0 = dot_u7_chunk(a0, input, v128_load(r0.add(off).cast::<v128>()));
                    a1 = dot_u7_chunk(a1, input, v128_load(r1.add(off).cast::<v128>()));
                    a2 = dot_u7_chunk(a2, input, v128_load(r2.add(off).cast::<v128>()));
                    a3 = dot_u7_chunk(a3, input, v128_load(r3.add(off).cast::<v128>()));
                    continue;
                }
                let x = widen_input(input);
                a0 = dot_chunk(a0, x, v128_load(r0.add(off).cast::<v128>()));
                a1 = dot_chunk(a1, x, v128_load(r1.add(off).cast::<v128>()));
                a2 = dot_chunk(a2, x, v128_load(r2.add(off).cast::<v128>()));
                a3 = dot_chunk(a3, x, v128_load(r3.add(off).cast::<v128>()));
            }
            // `j + 4 <= rows == biases.len() == output.len()`.
            let bias = v128_load(b_ptr.add(j).cast::<v128>());
            v128_store(
                out_ptr.add(j).cast::<v128>(),
                i32x4_add(hsum4(a0, a1, a2, a3), bias),
            );
        }
        j += 4;
    }

    while j < rows {
        // SAFETY: `j < rows` and `weights.len() == rows * padded_in`; the loads are in range as
        // above.
        unsafe {
            let r = w_ptr.add(j * padded_in);
            let mut acc = i32x4_splat(0);
            for c in 0..chunks {
                let off = c * U8_LANES;
                let input = v128_load(in_ptr.add(off).cast::<v128>());
                let weights = v128_load(r.add(off).cast::<v128>());
                #[cfg(target_feature = "relaxed-simd")]
                if U7 {
                    acc = dot_u7_chunk(acc, input, weights);
                    continue;
                }
                acc = dot_chunk(acc, widen_input(input), weights);
            }
            *out_ptr.add(j) = (*b_ptr.add(j)).wrapping_add(hsum(acc));
        }
        j += 1;
    }
}

/// [`affine`] for inputs of at most 127, the domain of the relaxed dot product.
#[inline]
pub fn affine_u7(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    debug_assert!(input[..in_dims].iter().all(|&x| x <= 127));
    affine_impl::<true>(output, biases, weights, input, in_dims, padded_in);
}

/// [`affine_u7`] for weights that passed `scalar_post_ft::pair_sums_fit_i16`: nothing to exploit
/// here, since the sums are accumulated in `i32` throughout.
#[inline]
pub fn affine_u7_wide(
    output: &mut [i32],
    biases: &[i32],
    weights: &[i8],
    input: &[u8],
    in_dims: usize,
    padded_in: usize,
) {
    affine_u7(output, biases, weights, input, in_dims, padded_in);
}

#[cfg(test)]
mod tests {
    use super::*;

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

        fn i8s(&mut self, n: usize) -> Vec<i8> {
            // Every byte value, with the extremes over-represented.
            (0..n)
                .map(|_| match self.next() % 8 {
                    0 => i8::MIN,
                    1 => i8::MAX,
                    _ => (self.next() >> 32) as i8,
                })
                .collect()
        }

        fn u8s(&mut self, n: usize, max: u8) -> Vec<u8> {
            (0..n)
                .map(|_| match self.next() % 8 {
                    0 => max,
                    1 => 0,
                    _ => ((self.next() >> 32) % (max as u64 + 1)) as u8,
                })
                .collect()
        }

        fn i32s(&mut self, n: usize) -> Vec<i32> {
            (0..n)
                .map(|_| match self.next() % 8 {
                    0 => i32::MAX,
                    1 => i32::MIN,
                    2 => 0,
                    // Around every threshold of the two activations.
                    3 => ((self.next() >> 40) as i32 % 12_000) - 6_000,
                    _ => (self.next() >> 32) as i32 >> (self.next() % 28),
                })
                .collect()
        }

        fn i16s(&mut self, n: usize) -> Vec<i16> {
            (0..n)
                .map(|_| match self.next() % 6 {
                    0 => i16::MAX,
                    1 => i16::MIN,
                    // Around the clamps at 127 and 254.
                    2 => ((self.next() >> 40) % 300) as i16 - 20,
                    _ => (self.next() >> 24) as i16,
                })
                .collect()
        }
    }

    /// Every layer size a shape has, and some it does not: the 15 and 16 of the SFNN
    /// activations, a 64-wide fc_1, the 2304 fc_0 input, widths that are not a multiple of 4.
    const LENGTHS: [usize; 12] = [0, 1, 3, 4, 15, 16, 17, 31, 32, 33, 64, 4_097];

    #[test]
    fn clipped_relu_matches_scalar() {
        let mut rng = Rng(0xc11b_0001);
        for n in LENGTHS {
            let input = rng.i32s(n);
            let mut want = vec![0u8; n];
            let mut got = vec![0xAA; n];
            scalar_post_ft::clipped_relu(&input, &mut want);
            clipped_relu(&input, &mut got);
            assert_eq!(want, got, "length {n}");
        }
    }

    #[test]
    fn clipped_relu_matches_scalar_on_boundary_values() {
        let input = [
            -1_000_000i32,
            -64,
            -1,
            0,
            1,
            63,
            64,
            8_000,
            8_127,
            8_128,
            i32::MAX,
            i32::MIN,
            -63,
            8_191,
            8_192,
            1 << 20,
        ];
        let mut want = [0u8; 16];
        let mut got = [0xAA; 16];
        scalar_post_ft::clipped_relu(&input, &mut want);
        clipped_relu(&input, &mut got);
        assert_eq!(want, got);
        assert_eq!(&got[..11], &[0u8, 0, 0, 0, 0, 0, 1, 125, 126, 127, 127]);
    }

    #[test]
    fn sqr_clipped_relu_matches_scalar() {
        let mut rng = Rng(0x5c11_b002);
        for n in LENGTHS {
            let input = rng.i32s(n);
            let mut want = vec![0u8; n];
            let mut got = vec![0xAA; n];
            scalar_post_ft::sqr_clipped_relu(&input, &mut want);
            sqr_clipped_relu(&input, &mut got);
            assert_eq!(want, got, "length {n}");
        }
    }

    /// Every input in a window around each threshold of the square, both signs, plus the
    /// extremes: the cap and the `i32::MIN` magnitude are where an off-by-one would live.
    #[test]
    fn sqr_clipped_relu_matches_scalar_around_every_threshold() {
        let mut input = Vec::new();
        for edge in [
            0i32, 724, 725, 1_024, 4_096, 8_160, 8_161, 8_191, 8_192, 8_193,
        ] {
            for d in -40..=40 {
                input.push(edge + d);
                input.push(-(edge + d));
            }
        }
        input.extend([
            i32::MAX,
            i32::MIN,
            i32::MIN + 1,
            i32::MAX - 1,
            1 << 30,
            -(1 << 30),
        ]);
        let mut want = vec![0u8; input.len()];
        let mut got = vec![0xAA; input.len()];
        scalar_post_ft::sqr_clipped_relu(&input, &mut want);
        sqr_clipped_relu(&input, &mut got);
        assert_eq!(want, got);
    }

    /// The full squares table over the range in which the output is not clamped, so that every
    /// output value 0..=127 is checked against the `i64` reference.
    #[test]
    fn sqr_clipped_relu_matches_scalar_on_every_value_below_the_cap() {
        let input: Vec<i32> = (-9_000..=9_000).collect();
        let mut want = vec![0u8; input.len()];
        let mut got = vec![0xAA; input.len()];
        scalar_post_ft::sqr_clipped_relu(&input, &mut want);
        sqr_clipped_relu(&input, &mut got);
        assert_eq!(want, got);
    }

    /// The standard shapes' transform is a loop in `transformer.rs`; this is that loop's body,
    /// the kernel held to it over every `i16` and over every length.
    #[test]
    fn clip_to_u7_matches_the_scalar_clamp_on_every_i16() {
        let every: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let want: Vec<u8> = every.iter().map(|&v| v.clamp(0, 127) as u8).collect();
        let mut got = vec![0xAA; every.len()];
        clip_to_u7(&every, &mut got);
        assert_eq!(want, got);
    }

    #[test]
    fn clip_to_u7_matches_the_scalar_clamp_for_every_length() {
        let mut rng = Rng(0xc11b_0009);
        for n in LENGTHS.into_iter().chain([128, 256, 768]) {
            let half = rng.i16s(n);
            let want: Vec<u8> = half.iter().map(|&v| v.clamp(0, 127) as u8).collect();
            let mut got = vec![0xAA; n];
            clip_to_u7(&half, &mut got);
            assert_eq!(want, got, "length {n}");
        }
    }

    #[test]
    fn ewm_one_perspective_matches_scalar() {
        let mut rng = Rng(0xe301_0003);
        for _ in 0..64 {
            let half = rng.i16s(HIDDEN_SIZE);
            let mut want = vec![0u8; EWM_HALF];
            let mut got = vec![0xAA; EWM_HALF];
            scalar_post_ft::ewm_one_perspective(&half, &mut want);
            ewm_one_perspective(&half, &mut got);
            assert_eq!(want, got);
        }
    }

    /// Every pair of clamped operands: the whole domain of the product.
    #[test]
    fn ewm_one_perspective_matches_scalar_on_every_clamped_pair() {
        const HALF: usize = EWM_HALF;
        let mut want = vec![0u8; HALF];
        let mut got = vec![0xAA; HALF];
        let mut half = vec![0i16; HIDDEN_SIZE];
        let mut pairs = Vec::new();
        for a in -3..=260i16 {
            for b in -3..=260i16 {
                pairs.push((a, b));
            }
        }
        for batch in pairs.chunks(HALF) {
            for (j, &(a, b)) in batch.iter().enumerate() {
                half[j] = a;
                half[j + HALF] = b;
            }
            scalar_post_ft::ewm_one_perspective(&half, &mut want);
            ewm_one_perspective(&half, &mut got);
            assert_eq!(want[..batch.len()], got[..batch.len()]);
        }
    }

    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn ewm_file_scale_matches_scalar() {
        let mut rng = Rng(0xf11e_0004);
        for _ in 0..64 {
            let half = rng.i16s(HIDDEN_SIZE);
            let mut want = vec![0u8; EWM_HALF];
            let mut got = vec![0xAA; EWM_HALF];
            scalar_post_ft::ewm_one_perspective_file_scale(&half, &mut want);
            ewm_one_perspective_file_scale(&half, &mut got);
            assert_eq!(want, got);
        }
    }

    #[cfg(feature = "sfnn_threat")]
    #[test]
    fn ewm_file_scale_matches_scalar_on_every_clamped_pair() {
        const HALF: usize = EWM_HALF;
        let mut want = vec![0u8; HALF];
        let mut got = vec![0xAA; HALF];
        let mut half = vec![0i16; HIDDEN_SIZE];
        let mut pairs = Vec::new();
        for a in -3..=135i16 {
            for b in -3..=135i16 {
                pairs.push((a, b));
            }
        }
        for batch in pairs.chunks(HALF) {
            for (j, &(a, b)) in batch.iter().enumerate() {
                half[j] = a;
                half[j + HALF] = b;
            }
            scalar_post_ft::ewm_one_perspective_file_scale(&half, &mut want);
            ewm_one_perspective_file_scale(&half, &mut got);
            assert_eq!(want[..batch.len()], got[..batch.len()]);
        }
    }

    /// The layer shapes every network shape has: `(rows, in_dims, padded_in)`.
    ///
    /// fc_0 reads up to 2304 bytes (the 2304 shape) and has 16 rows (or 32 for the 256
    /// shapes); fc_1 has 32 or 64 rows over a 32-byte input; fc_2 has one row. The padded width
    /// is sometimes wider than the live one, and the odd row counts leave a 4-row remainder.
    const AFFINE_SHAPES: [(usize, usize, usize); 14] = [
        (16, 2_304, 2_304),
        (16, 1_024, 1_024),
        (16, 1_536, 1_536),
        (32, 512, 512),
        (32, 256, 256),
        (16, 768, 768),
        (32, 32, 32),
        (64, 32, 32),
        (1, 32, 32),
        (5, 48, 64),
        (7, 16, 32),
        (3, 64, 64),
        (4, 16, 16),
        (9, 32, 32),
    ];

    /// The signature `affine` and its two named forms share.
    type AffineFn = fn(&mut [i32], &[i32], &[i8], &[u8], usize, usize);

    fn check_affine(rng: &mut Rng, input_max: u8, f: AffineFn) {
        for (rows, in_dims, padded_in) in AFFINE_SHAPES {
            let weights = rng.i8s(rows * padded_in);
            let biases: Vec<i32> = rng.i32s(rows);
            let input = rng.u8s(padded_in, input_max);
            let mut want = vec![0i32; rows];
            let mut got = vec![0x5555_5555; rows];
            scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
            f(&mut got, &biases, &weights, &input, in_dims, padded_in);
            assert_eq!(want, got, "rows {rows} in {in_dims} padded {padded_in}");
        }
    }

    #[test]
    fn affine_u7_matches_scalar() {
        let mut rng = Rng(0xaff1_0005);
        for _ in 0..6 {
            check_affine(&mut rng, 127, affine_u7);
        }
    }

    #[test]
    fn affine_u7_wide_matches_scalar() {
        let mut rng = Rng(0xaff1_0006);
        for _ in 0..6 {
            check_affine(&mut rng, 127, affine_u7_wide);
        }
    }

    /// The general kernel is exact for every byte, where `maddubs` would saturate above 127.
    #[test]
    fn affine_matches_scalar_for_every_input_byte() {
        let mut rng = Rng(0xaff1_0007);
        for _ in 0..6 {
            check_affine(&mut rng, 255, affine);
        }
    }

    /// Worst case for any intermediate: all-maximum input against all-minimum weights and the
    /// reverse, so that a lane that saturated or narrowed anywhere would show.
    #[test]
    fn affine_matches_scalar_on_extreme_operands() {
        let (rows, dims) = (16, 2_304);
        for (w, x) in [
            (i8::MIN, 255u8),
            (i8::MAX, 255),
            (i8::MIN, 127),
            (i8::MAX, 127),
        ] {
            let weights = vec![w; rows * dims];
            let input = vec![x; dims];
            let biases = vec![i32::MAX - 5; rows];
            let mut want = vec![0i32; rows];
            let mut got = vec![0i32; rows];
            scalar_post_ft::affine(&mut want, &biases, &weights, &input, dims, dims);
            affine(&mut got, &biases, &weights, &input, dims, dims);
            assert_eq!(want, got, "weight {w} input {x}");
        }
    }

    /// An `in_dims` that is not whole chunks takes the scalar fallback, and still reads only
    /// `in_dims` of each padded row.
    #[test]
    fn affine_with_a_ragged_width_matches_scalar() {
        let mut rng = Rng(0xaff1_0008);
        let (rows, in_dims, padded_in) = (6, 40, 64);
        let weights = rng.i8s(rows * padded_in);
        let biases = rng.i32s(rows);
        let input = rng.u8s(padded_in, 127);
        let mut want = vec![0i32; rows];
        let mut got = vec![0i32; rows];
        scalar_post_ft::affine(&mut want, &biases, &weights, &input, in_dims, padded_in);
        affine_u7(&mut got, &biases, &weights, &input, in_dims, padded_in);
        assert_eq!(want, got);
    }

    #[test]
    #[should_panic(expected = "slice lengths")]
    fn affine_refuses_weights_of_the_wrong_length() {
        let mut out = [0i32; 4];
        affine(&mut out, &[0; 4], &[0i8; 4 * 32 - 1], &[0u8; 32], 32, 32);
    }
}
