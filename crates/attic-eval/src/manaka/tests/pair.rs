//! The token-pair helper against the definition: which row a pair is.

use super::super::pair::pair_row;
use super::super::{HAS_PAIR, PAIR_ROWS, TOKENS};

/// The pair rows are the upper triangle read row by row, so walking it in that
/// order must count 0, 1, 2, … without a gap or a repeat, to exactly `C(2344, 2)`,
/// whichever way round the two tokens are given.
#[test]
fn pair_rows_number_the_upper_triangle_in_order() {
    let mut next = 0;
    for lo in 0..TOKENS as u32 {
        for hi in lo + 1..TOKENS as u32 {
            assert_eq!(pair_row(lo, hi), next, "({lo}, {hi})");
            assert_eq!(pair_row(hi, lo), next, "({hi}, {lo})");
            next += 1;
        }
    }
    assert_eq!(next, 2_745_996);
    // The table's own size, in a shape that has one; the others have no rows.
    assert_eq!(PAIR_ROWS, if HAS_PAIR { next } else { 0 });
}

/// The offsets the kernels read their rows at are `pair_row · stride` for each of
/// the others, from the path the build runs, at every count from none to the most a
/// position has, for the strides of the tables there are, with the extreme tokens
/// (the first, the last, the two next to each other) in the draw.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn the_row_offsets_are_the_pair_rows_times_the_stride() {
    use super::super::pair::{OFFSETS, row_offsets, row_offsets_scalar};
    use super::super::pair_table::{ROW, SCALED_ROW};
    use crate::tokens::FeatureIndex;

    if !HAS_PAIR {
        return;
    }
    let last = TOKENS as FeatureIndex - 1;
    let mut state = 17u64;
    let mut pick = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        match (state >> 60) % 6 {
            0 => 0,
            1 => last,
            2 => last - 1,
            _ => ((state >> 33) as usize % TOKENS) as FeatureIndex,
        }
    };
    for stride in [ROW, SCALED_ROW] {
        for count in 0..=OFFSETS {
            for _ in 0..40 {
                let token = pick();
                let mut others = Vec::new();
                while others.len() < count {
                    let other = pick();
                    if other != token && !others.contains(&other) {
                        others.push(other);
                    }
                }
                let want = row_offsets_scalar(token, &others, stride);
                let got = row_offsets(token, &others, stride);
                for (at, &other) in others.iter().enumerate() {
                    assert_eq!(
                        got[at] as usize,
                        pair_row(token, other) * stride,
                        "stride {stride}, {count} others, entry {at}"
                    );
                    assert_eq!(got[at], want[at], "stride {stride}, entry {at}");
                }
            }
        }
    }
}

/// The totals the derive adds to the pair sums, from the path the build runs, are
/// the plain ones to the last bit: the same rows, in the same two totals, in the
/// same order. Every count of other tokens from none to thirty-nine, so that an
/// odd last row and an empty set are both met, against tokens drawn over the whole
/// table.
#[test]
fn the_row_totals_are_the_plain_ones_to_the_last_bit() {
    use super::super::pair::row_totals_scalar;
    use super::super::{PAIR, PairTable};
    use super::fixtures::synthetic;
    use crate::tokens::{FeatureIndex, other_frame};

    if !HAS_PAIR {
        return;
    }
    let net = synthetic();
    let table: &PairTable = &net.pair;
    let mut state = 5u64;
    let mut pick = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as usize % TOKENS) as FeatureIndex
    };
    for count in 0..40 {
        let token = pick();
        let mut others = Vec::new();
        while others.len() < count {
            let other = pick();
            if other != token && !others.contains(&other) {
                others.push(other);
            }
        }
        let want = row_totals_scalar(table, token, &others);
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "f16c"
        ))]
        {
            // SAFETY: the `cfg` is the build enabling AVX2 and F16C.
            let got = unsafe { super::super::pair::row_totals_avx2(table, token, &others) };
            for (channel, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{count} others, channel {channel}"
                );
            }
        }
        // Against the file's table read in any order, each frame from its own
        // numbering: within an f32 rounding of each row.
        let file = |frame: usize, other: FeatureIndex, channel: usize| -> f64 {
            let (a, b) = match frame {
                0 => (token, other),
                _ => (other_frame(token), other_frame(other)),
            };
            f64::from(table.value(pair_row(a, b), channel))
        };
        for (at, &w) in want.iter().enumerate() {
            let (frame, channel) = (at / PAIR, at % PAIR);
            let plain: f64 = others.iter().map(|&o| file(frame, o, channel)).sum();
            assert!(
                (f64::from(w) - plain).abs() <= 1e-5 * (count as f64).max(1.0),
                "{count} others, frame {frame}, channel {channel}: {w} against {plain}"
            );
        }
    }
}

/// The byte tables, rounded from the synthetic net's halves.
fn rounded(precision: super::super::PairPrecision) -> super::super::PairTable {
    let mut table = super::fixtures::synthetic().pair.replicate();
    table.use_precision(precision);
    table
}

#[test]
fn the_precision_names() {
    use super::super::PairPrecision;
    for (name, want) in [
        (None, PairPrecision::F16),
        (Some("f16"), PairPrecision::F16),
        (Some("i8"), PairPrecision::I8),
        (Some("I8"), PairPrecision::I8),
        (Some("i8row"), PairPrecision::I8Row),
        (Some("I8ROW"), PairPrecision::I8Row),
        (Some("u8"), PairPrecision::F16),
        (Some(""), PairPrecision::F16),
    ] {
        assert_eq!(PairPrecision::from_name(name), want, "{name:?}");
    }
}

/// A byte is a rounding of the half it came from, to within half a step of the
/// table's own: for a table of cubes half the gap to the next cube, in the units of
/// the channel; for a table with a scale to a half, half that scale. Every entry of
/// a sample of rows, both frames.
#[test]
fn a_byte_table_is_within_half_a_step_of_the_halves() {
    use super::super::{PAIR, PairPrecision};

    if !HAS_PAIR {
        return;
    }
    let net = super::fixtures::synthetic();
    for precision in [PairPrecision::I8, PairPrecision::I8Row] {
        let table = rounded(precision);
        assert_eq!(table.precision(), precision);
        assert_eq!(table.rows(), PAIR_ROWS);
        // A copy for another NUMA node is the same table.
        let copy = table.replicate();
        assert_eq!(copy.precision(), precision);
        for (row, channel) in [(0, 0), (1234, PAIR - 1), (PAIR_ROWS - 1, PAIR / 2)] {
            assert_eq!(
                copy.value(row, channel).to_bits(),
                table.value(row, channel).to_bits()
            );
        }
        // The channel's largest entry, over the whole table.
        let largest: [f32; PAIR] = std::array::from_fn(|channel| {
            (0..table.rows()).fold(0.0f32, |m, row| m.max(net.pair.value(row, channel).abs()))
        });
        let mut state = 11u64;
        for _ in 0..4000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let row = (state >> 33) as usize % table.rows();
            for (channel, &largest) in largest.iter().enumerate() {
                let (want, got) = (net.pair.value(row, channel), table.value(row, channel));
                let bound = match precision {
                    PairPrecision::I8 => {
                        // The cubes either side of the entry are `3q² + 3q + 1`
                        // units apart at `q`, a unit being `max / 127³`.
                        let unit = largest / (127.0 * 127.0 * 127.0);
                        let q = (got.abs() / unit).cbrt().round();
                        0.5 * (3.0 * q * q + 3.0 * q + 1.0) * unit * 1.001
                    }
                    _ => {
                        let row_largest =
                            (0..PAIR).fold(0.0f32, |m, c| m.max(net.pair.value(row, c).abs()));
                        0.5 * row_largest / 127.0 * 1.001
                    }
                };
                assert!(
                    (want - got).abs() <= bound + 1e-7,
                    "{precision:?} row {row} channel {channel}: {got} for {want}, bound {bound}"
                );
            }
        }
    }
}

/// The totals from the bytes, from the path the build runs, are the plain ones to
/// the last bit, for every count of other tokens from none to thirty-nine.
#[test]
fn the_byte_totals_are_the_plain_ones_to_the_last_bit() {
    use super::super::pair_i8::{cube_totals_scalar, scaled_totals_scalar};
    use super::super::pair_table::Cells;
    use super::super::{PAIR, PairPrecision};
    use crate::tokens::FeatureIndex;

    if !HAS_PAIR {
        return;
    }
    let cubes = rounded(PairPrecision::I8);
    let scaled = rounded(PairPrecision::I8Row);
    let mut state = 7u64;
    let mut pick = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as usize % TOKENS) as FeatureIndex
    };
    for count in 0..40 {
        let token = pick();
        let mut others = Vec::new();
        while others.len() < count {
            let other = pick();
            if other != token && !others.contains(&other) {
                others.push(other);
            }
        }
        let (Cells::Cubes { q, scale }, Cells::Scaled { q: scaled_q }) =
            (cubes.cells(), scaled.cells())
        else {
            panic!("the tables were rounded");
        };
        let plain_cubes = cube_totals_scalar(q, scale, token, &others);
        let plain_scaled = scaled_totals_scalar(scaled_q, token, &others);
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            use super::super::pair_i8::{cube_totals_avx2, scaled_totals_avx2};
            // SAFETY: the `cfg` is the build enabling AVX2.
            let (got_cubes, got_scaled) = unsafe {
                (
                    cube_totals_avx2(q, scale, token, &others),
                    scaled_totals_avx2(scaled_q, token, &others),
                )
            };
            for (channel, (g, w)) in got_cubes.iter().zip(&plain_cubes).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "cubes, {count} others, channel {channel}"
                );
            }
            for (channel, (g, w)) in got_scaled.iter().zip(&plain_scaled).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "scaled, {count} others, channel {channel}"
                );
            }
        }
        // And against the entries read one at a time, each frame from its own
        // numbering, within an f32 rounding of each row.
        for (table, totals) in [(&cubes, &plain_cubes), (&scaled, &plain_scaled)] {
            for (at, &w) in totals.iter().enumerate() {
                let (frame, channel) = (at / PAIR, at % PAIR);
                let plain: f64 = others
                    .iter()
                    .map(|&o| {
                        let (a, b) = match frame {
                            0 => (token, o),
                            _ => (
                                crate::tokens::other_frame(token),
                                crate::tokens::other_frame(o),
                            ),
                        };
                        f64::from(table.value(pair_row(a, b), channel))
                    })
                    .sum();
                assert!(
                    (f64::from(w) - plain).abs() <= 1e-4 * plain.abs().max(1.0),
                    "{count} others, frame {frame}, channel {channel}: {w} against {plain}"
                );
            }
        }
    }
}
