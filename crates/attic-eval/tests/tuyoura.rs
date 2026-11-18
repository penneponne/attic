#![cfg(feature = "sfnn_ka2")]

use attic_eval::{
    FC_0_OUTPUT_DIMS, LAYER_STACKS, MoveDelta, NUM_FEATURES, active_features, evaluate,
    load_network_with_warnings, set_fv_scale,
};
use attic_state::{Color, parse_sfen, parse_usi_move};
use std::path::Path;

#[test]
fn ka2_has_both_folded_and_side_specific_king_features() {
    let mut pos = parse_sfen("8k/9/9/9/9/9/9/4p4/4K4 b - 1").unwrap();
    assert_eq!(NUM_FEATURES, 1791);
    assert_eq!(LAYER_STACKS, 8);
    assert_eq!(FC_0_OUTPUT_DIMS, 8);
    let before = attic_eval::active_features_both(&pos);
    let black = &before[Color::Black.index()];
    let white = &before[Color::White.index()];
    assert_eq!(black.len(), 42);
    assert_eq!(white.len(), 42);
    // A2 folds the kings onto the same plane at 1548. K adds
    // separate friend and opponent planes starting at 1629 and 1710.
    for feature in [1548, 1592, 1673, 1710] {
        assert!(black.contains(&feature), "missing black feature {feature}");
    }
    for feature in [1584, 1628, 1709, 1746] {
        assert!(white.contains(&feature), "missing white feature {feature}");
    }
    let king_move = parse_usi_move("5i5h", &pos).unwrap();
    let delta = MoveDelta::from_move(&pos, king_move);
    pos.do_move(king_move);
    for color in [Color::Black, Color::White] {
        let half = delta
            .half(color)
            .expect("king capture updates both K-A2 halves");
        assert_eq!(half.removed().len(), 3);
        assert_eq!(half.added().len(), 3);
        let mut incremental = before[color.index()].clone();
        for index in half.removed() {
            let slot = incremental.iter().position(|item| item == index).unwrap();
            incremental.swap_remove(slot);
        }
        incremental.extend(half.added());
        incremental.sort_unstable();
        let mut refreshed = active_features(&pos, color);
        refreshed.sort_unstable();
        assert_eq!(incremental, refreshed);
    }
}

#[test]
fn reads_the_distributed_network() {
    let Ok(path) = std::env::var("ATTIC_TUYOURA_EVAL") else {
        eprintln!("set ATTIC_TUYOURA_EVAL to the distributed evaluation directory");
        return;
    };
    let path = Path::new(&path);
    let (net, warnings) = load_network_with_warnings(&path.join("nn.bin")).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    attic_eval::progress::load_parameters(&path.join("progress.bin")).unwrap();
    set_fv_scale(24);

    // The two score columns came from the corresponding YaneuraOu build's e command.
    let extra = std::env::var_os("ATTIC_TUYOURA_REFERENCE")
        .map(|path| std::fs::read_to_string(path).unwrap());
    let reference = extra
        .as_deref()
        .unwrap_or(include_str!("data/tuyoura_eval.tsv"));
    for row in reference.lines() {
        let mut fields = row.split('\t');
        let sfen = fields.next().unwrap();
        let column = usize::from(cfg!(feature = "sfnn_ka2_4096_8_64_progress8x"));
        let expected = fields.nth(column).unwrap().parse::<i32>().unwrap();
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(evaluate(&net, &pos), expected, "{sfen}");
    }
}
