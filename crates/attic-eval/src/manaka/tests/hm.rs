//! `hm`: a net read with each frame mirrored when its own king is on the sixth to ninth file.
//!
//! Run only by the build that has it (`synn_token_128x2_32_32_pair16_o1_hm`). The reference is
//! `reference_value`, which sums the tokens of a position whose board was *actually mirrored* for each
//! frame that folds (`tokens_present_hm`); nothing here asks the accumulator what it thinks a fold is.

use attic_state::Color;

use super::super::{Accumulator, FinnyCache, HM, evaluate_value, value_from_sums};
use super::fixtures::{assert_close, assert_close_relative, king_walk_game, synthetic};
use super::reference::{reference_value, reference_value_with};
use crate::MoveDelta;

/// Which of the cases a game has been through, so that a test cannot pass by never meeting the branch it is
/// about.
#[derive(Default, Debug)]
struct Met {
    /// `[black folds][white folds]` of the position after a move.
    folds: [[usize; 2]; 2],
    /// A derive in which exactly one frame's fold changed, and one in which both did. A move lifts one piece,
    /// so at most one king, and a king is the *own* king of only one frame: both can never change at once,
    /// and `both_crossed` is asserted to stay zero rather than expected to be met.
    one_crossed: usize,
    both_crossed: usize,
    /// Of the frames that crossed, black's and white's.
    black_crossed: usize,
    white_crossed: usize,
    /// A derive in which the frames stood in different folds (one folded, one not) and neither crossed.
    split_and_steady: usize,
}

/// A derive from the parent's accumulator is the accumulator the position after it would be refreshed to, in
/// every sum, over games that walk the kings across the fifth/sixth line and back: so every kind of move meets
/// every kind of fold, and a frame is built again whenever its king crosses.
#[test]
fn a_derive_is_a_refresh_when_the_kings_walk_across_the_line() {
    const { assert!(HM, "this test is for the hm build") };
    let net = synthetic();
    let mut cache = FinnyCache;
    let mut met = Met::default();
    for seed in 1..=24 {
        let game = king_walk_game(seed, 140, 0.30, 0.25);
        let mut acc = Accumulator::new();
        acc.refresh(net, &game[0].0);
        let mut derived = Accumulator::new();
        for (before, mv) in &game {
            let delta = MoveDelta::from_move(before, *mv);
            let mut after = before.clone();
            let _ = after.do_move(*mv);
            Accumulator::derive_into_cached(&acc, &mut derived, net, &after, &delta, &mut cache);

            let mut rebuilt = Accumulator::new();
            rebuilt.refresh(net, &after);

            // What the fold did, read from the accumulators themselves: a frame crossed when its fold
            // differs from its parent's.
            let (was, now) = (acc.folded_for_test(), derived.folded_for_test());
            assert_eq!(
                now,
                rebuilt.folded_for_test(),
                "seed {seed}: the fold a derive ends in"
            );
            let crossed = [was[0] != now[0], was[1] != now[1]];
            met.folds[usize::from(now[0])][usize::from(now[1])] += 1;
            match crossed {
                [true, true] => met.both_crossed += 1,
                [true, false] | [false, true] => met.one_crossed += 1,
                [false, false] if now[0] != now[1] => met.split_and_steady += 1,
                _ => {}
            }
            met.black_crossed += usize::from(crossed[0]);
            met.white_crossed += usize::from(crossed[1]);

            for color in [Color::Black, Color::White] {
                assert_close(
                    "power sums",
                    derived.perspective(color),
                    rebuilt.perspective(color),
                    1e-3,
                );
                assert_close_relative(
                    "pair",
                    &derived.pair(color),
                    &rebuilt.pair(color),
                    super::forward::PAIR_DRIFT,
                );
                assert_close_relative(
                    "moment",
                    &derived.moment(color),
                    &rebuilt.moment(color),
                    super::forward::SUM_DRIFT,
                );
                assert_close_relative(
                    "field",
                    &derived.field(color),
                    &rebuilt.field(color),
                    super::forward::SUM_DRIFT,
                );
            }
            let want = evaluate_value(net, &after);
            let got = value_from_sums(net, &derived, after.side_to_move());
            assert!(
                (want - got).abs() < 1e-4,
                "seed {seed}: value {got} against {want}"
            );
            acc.clone_from(&derived);
        }
    }
    println!("{met:?}");
    assert!(
        met.folds.iter().flatten().all(|&n| n > 0),
        "all four combinations of the frames' folds are met: {met:?}"
    );
    assert!(met.one_crossed > 20, "{met:?}");
    assert_eq!(
        met.both_crossed, 0,
        "one move cannot change both frames' own king: {met:?}"
    );
    assert!(
        met.black_crossed > 10 && met.white_crossed > 10,
        "each colour's king crosses: {met:?}"
    );
    assert!(
        met.split_and_steady > 20,
        "frames in different folds, steady: {met:?}"
    );
}

/// The accumulator's values are the independent reference's: an hm net's forward over a game that crosses the
/// line a lot, against `reference_value` (f64, over the tokens of the actually mirrored board).
#[test]
fn the_forward_pass_matches_the_mirrored_board_reference_across_the_line() {
    let net = synthetic();
    let mut worst = 0.0f64;
    let mut folded_positions = 0;
    for seed in 1..=8 {
        for (before, mv) in king_walk_game(seed, 120, 0.30, 0.25) {
            let mut after = before.clone();
            let _ = after.do_move(mv);
            let got = f64::from(evaluate_value(net, &after));
            worst = worst.max((got - reference_value(net, &after)).abs());
            let mut acc = Accumulator::new();
            acc.refresh(net, &after);
            folded_positions += usize::from(acc.folded_for_test() != [false, false]);
        }
    }
    assert!(
        folded_positions > 100,
        "{folded_positions} positions with a folded frame"
    );
    assert!(
        worst < 1e-4,
        "f32 forward is {worst} from the f64 reference"
    );
}

/// Folding changes the value: with the weights fixed, the value of a position some frame of which folds is
/// not the value its tokens as they stand would give, so the tests above are not comparing a constant, and
/// a position none of whose frames fold is the same either way.
#[test]
fn a_folded_frame_is_not_the_unfolded_one_and_an_unfolded_one_is() {
    let net = synthetic();
    let (mut moved, mut same) = (0, 0);
    for seed in 1..=6 {
        for (before, mv) in king_walk_game(seed, 120, 0.30, 0.25) {
            let mut after = before.clone();
            let _ = after.do_move(mv);
            let mut acc = Accumulator::new();
            acc.refresh(net, &after);
            let (folded, plain) = (
                reference_value_with(net, &after, true),
                reference_value_with(net, &after, false),
            );
            if acc.folded_for_test() == [false, false] {
                assert_eq!(folded.to_bits(), plain.to_bits(), "no frame folds");
                same += 1;
            } else if (folded - plain).abs() > 1e-4 {
                moved += 1;
            }
        }
    }
    assert!(moved > 50 && same > 50, "moved {moved}, same {same}");
}

/// What a touch adds to the pair sums, in every fold of the two frames and with either frame or both
/// updated, is each frame's own read of the first table to the last bit: the rows of its tokens as it
/// reflects them, in the order given, and that frame's half of them. Frames folded differently read the
/// second table and not two rows a pair, so this is what holds that table to the first. At every
/// precision the table can be held in, since the second table is rounded by itself.
#[test]
fn a_touch_in_any_fold_adds_what_each_frame_reads_alone() {
    use super::super::pair::{row_totals, touch_pairs_hm};
    use super::super::{PAIR, PairPrecision};
    use crate::tokens::{FeatureIndex, mirror_row};

    const { assert!(HM, "this test is for the hm build") };
    for precision in [PairPrecision::F16, PairPrecision::I8, PairPrecision::I8Row] {
        let mut net = synthetic().replicate();
        net.use_pair_precision(precision);
        let mut state = 11u64;
        let mut pick = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as usize % super::super::TOKENS) as FeatureIndex
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
            let reflected: Vec<FeatureIndex> = others.iter().map(|&o| mirror_row(o)).collect();
            for folded in [[false, false], [true, false], [false, true], [true, true]] {
                for update in [[true, true], [true, false], [false, true]] {
                    let mut sums = [vec![0.0; PAIR], vec![0.0; PAIR]];
                    touch_pairs_hm(&mut sums, &net, token, &others, 1.0, folded, update);
                    for frame in 0..2 {
                        let totals = if folded[frame] {
                            row_totals(net.pair.cells(), mirror_row(token), &reflected)
                        } else {
                            row_totals(net.pair.cells(), token, &others)
                        };
                        for channel in 0..PAIR {
                            let want = if update[frame] {
                                f64::from(totals[frame * PAIR + channel])
                            } else {
                                0.0
                            };
                            assert_eq!(
                                sums[frame][channel].to_bits(),
                                want.to_bits(),
                                "{precision:?}, {count} others, folds {folded:?}, updates {update:?}, \
                                 frame {frame}, channel {channel}"
                            );
                        }
                    }
                }
            }
        }
    }
}
