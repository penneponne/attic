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
