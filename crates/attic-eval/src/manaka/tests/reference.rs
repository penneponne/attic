//! The forward pass written a second way, in `f64`, from the definition.

use attic_state::Position;

use super::super::shape::SQUARES;
use super::super::{
    DIM, DUAL, FEATURE_LEN, FIELD, HAND_CUT, HAS_FIELD, HAS_MOMENT, HAS_PAIR, HEAD_AT, MOMENT,
    NOSQR, ORDER, PAIR, PAIR_RADIUS, SymNetwork, TOKENS,
};
use super::encode::column_of;
#[cfg(feature = "sym_hm")]
use super::fixtures::tokens_present_hm;
use super::fixtures::{tokens_present, upto};

/// The row of the pair `{a, b}` counted out: the pairs with a lower token below
/// `lo` come first, `TOKENS − 1 − i` of them for each `i`, and `hi` is
/// `hi − lo − 1` into its own. The accumulator's formula is the closed form of
/// this sum; the two should never need to agree by construction.
fn counted_pair_row(a: u32, b: u32) -> usize {
    let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
    (0..lo).map(|i| TOKENS - 1 - i).sum::<usize>() + (hi - lo - 1)
}

/// Whether the pair `{a, b}` has a row at all in a table cut at the shape's radius,
/// written out from the definition and not from the engine's tables.
///
/// A token is a board piece (`id < 2268`, `id = (kind · 2 + theirs) · 81 + square`,
/// kings are kind 7) or a held one. A pair of two held pieces is kept, and so is one
/// that touches a king; a held piece against any other board piece is kept unless the
/// shape cuts the hand; two other pieces are kept when their squares,
/// `file = square / 9` and `rank = square % 9`, are within the radius of each other
/// on both axes.
fn kept(a: u32, b: u32) -> bool {
    let board = |token: u32| token < 14 * 2 * 81;
    let king = |token: u32| board(token) && token / 81 / 2 == 7;
    match (board(a), board(b)) {
        (false, false) => return true,
        (false, true) => return !HAND_CUT || king(b),
        (true, false) => return !HAND_CUT || king(a),
        (true, true) => {}
    }
    if king(a) || king(b) {
        return true;
    }
    let (a, b) = (a % 81, b % 81);
    let near = |x: u32, y: u32| (x as i32 - y as i32).unsigned_abs() as usize <= PAIR_RADIUS;
    near(a / 9, b / 9) && near(a % 9, b % 9)
}

/// The forward pass written a second way, in `f64`, from the definition.
///
/// The elementary symmetric polynomial `e_k` is the coefficient of `x^k` in
/// `∏ (1 + v_i x)`, so this builds it by multiplying out the product rather
/// than by Newton's identities from power sums — a different route to the same
/// number, which is what makes agreement mean something. The pair sum is taken
/// over every pair of tokens present, one at a time, at a row number counted
/// out rather than computed. The second moment is built entry by entry from its
/// definition, `Σ_i v_i[a] · v_i[b]` for each `a ≤ b`, and the field slot by slot,
/// `Σ_i field[i][s·f + c]`, where the accumulator adds a token at a time.
///
/// The head is then the plain textbook form of the definition: each block is
/// `clamp(x · gamma, 0, 1)`, each hidden layer is `clamp(b + w · x, 0, 1)` and the
/// value is the row's output as it is. A dual head reads each layer's sum `z` as
/// `[clamp(z², 0, 1) ‖ clamp(z, 0, 1)]` and the value row reads both layers' reads
/// side by side.
pub(super) fn reference_value(net: &SymNetwork, pos: &Position) -> f64 {
    reference_value_with(net, pos, cfg!(feature = "sym_hm"))
}

/// [`reference_value`] with the fold named: `fold` reads each frame as an `hm` net does, and `false` reads the
/// tokens as they are, whatever the build. The second is what lets a test say what folding changed.
pub(super) fn reference_value_with(net: &SymNetwork, pos: &Position, fold: bool) -> f64 {
    let mut features = Vec::with_capacity(FEATURE_LEN);
    for color in [pos.side_to_move(), pos.side_to_move().flip()] {
        // A net read with `hm` is summed over the tokens of the frame's own fold.
        #[cfg(feature = "sym_hm")]
        let tokens = if fold {
            tokens_present_hm(pos, color)
        } else {
            tokens_present(pos, color)
        };
        #[cfg(not(feature = "sym_hm"))]
        let tokens = {
            let _ = fold;
            tokens_present(pos, color)
        };
        // What a block becomes on its way to the first dense layer, with the gain
        // of its `i`-th entry given by `gamma`: multiplied by it and clipped.
        let put = |features: &mut Vec<f64>, block: &[f64], gamma: &dyn Fn(usize) -> f64| {
            features.extend(
                block
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v * gamma(i)).clamp(0.0, 1.0)),
            );
        };

        let mut elementary = vec![[0.0f64; ORDER + 1]; DIM];
        for (channel, e) in elementary.iter_mut().enumerate() {
            e[0] = 1.0;
            for &token in &tokens {
                let v = f64::from(net.embed[token as usize * DIM + channel]);
                for k in (1..=ORDER).rev() {
                    e[k] += e[k - 1] * v;
                }
            }
        }
        for k in 1..=ORDER {
            let block: Vec<f64> = elementary.iter().map(|e| e[k]).collect();
            put(&mut features, &block, &|channel| {
                f64::from(net.gamma[(k - 1) * DIM + channel])
            });
        }

        if HAS_PAIR {
            // Under `nopairfold` the pairs are those of the board as it stands, whatever the fold.
            let unfolded;
            let pairs = if fold && super::super::HM_NOPAIRFOLD {
                unfolded = tokens_present(pos, color);
                &unfolded
            } else {
                &tokens
            };
            let mut sum = [0.0f64; PAIR];
            for (i, &a) in pairs.iter().enumerate() {
                for &b in &pairs[i + 1..] {
                    if !kept(a, b) {
                        continue;
                    }
                    let row = counted_pair_row(a, b);
                    for (channel, s) in sum.iter_mut().enumerate() {
                        *s += f64::from(net.pair.value(row, channel));
                    }
                }
            }
            put(&mut features, &sum, &|channel| {
                f64::from(net.pair_gamma[channel])
            });
        }

        if HAS_MOMENT {
            // Entry by entry, the upper triangle row by row: (0,0) (0,1) …
            // (0,m−1) (1,1) … (m−1,m−1). The sum over tokens is taken for each
            // entry on its own.
            let mut triangle = Vec::new();
            for a in upto(MOMENT) {
                for b in a..MOMENT {
                    let entry: f64 = tokens
                        .iter()
                        .map(|&token| {
                            let row = &net.moment_embed[token as usize * MOMENT..][..MOMENT];
                            f64::from(row[a]) * f64::from(row[b])
                        })
                        .sum();
                    triangle.push(entry);
                }
            }
            // A gain for each entry, not each row or channel.
            put(&mut features, &triangle, &|entry| {
                f64::from(net.moment_gamma[entry])
            });
        }

        if HAS_FIELD {
            // Slot by slot, square-major: the 81 squares in turn, each of `f`
            // channels.
            let mut slots = Vec::new();
            for square in upto(SQUARES) {
                for channel in upto(FIELD) {
                    let sum: f64 = tokens
                        .iter()
                        .map(|&token| {
                            let row = &net.field[token as usize * SQUARES * FIELD..];
                            f64::from(row[square * FIELD + channel])
                        })
                        .sum();
                    slots.push(sum);
                }
            }
            // The gain is one per channel and every square shares it.
            let channel_of = column_of(FIELD);
            put(&mut features, &slots, &|slot| {
                f64::from(net.field_gamma[channel_of(slot)])
            });
        }
    }

    // The sum of a layer, before any activation.
    let pre = |w: &[f32], b: &[f32], x: &[f64]| -> Vec<f64> {
        b.iter()
            .enumerate()
            .map(|(row, b)| {
                let dot: f64 = w[row * x.len()..(row + 1) * x.len()]
                    .iter()
                    .zip(x)
                    .map(|(w, x)| f64::from(*w) * x)
                    .sum();
                f64::from(*b) + dot
            })
            .collect()
    };
    let clip = |z: Vec<f64>| -> Vec<f64> { z.into_iter().map(|z| z.clamp(0.0, 1.0)).collect() };
    let head = if DUAL {
        // `nosqr` reads the plain clip in the square's place.
        let double = |z: &[f64]| -> Vec<f64> {
            z.iter()
                .map(|z| if NOSQR { *z } else { z * z }.clamp(0.0, 1.0))
                .chain(z.iter().map(|z| z.clamp(0.0, 1.0)))
                .collect()
        };
        let first = double(&pre(&net.w1, &net.b1, &features));
        let second = double(&pre(&net.w2, &net.b2, &first));
        // `noskip` leaves the first layer's half out of what the value row reads.
        [first, second].concat().split_off(HEAD_AT)
    } else {
        clip(pre(
            &net.w2,
            &net.b2,
            &clip(pre(&net.w1, &net.b1, &features)),
        ))
    };
    let dot: f64 = net
        .value_w
        .iter()
        .zip(&head)
        .map(|(w, h)| f64::from(*w) * h)
        .sum();
    f64::from(net.value_b) + dot
}
