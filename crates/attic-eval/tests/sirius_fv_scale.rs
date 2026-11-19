//! `FV_SCALE` and the score range of the threat shape.
//!
//! Sirius divides the network's output by `FV_SCALE` (26 unless set otherwise) and keeps the
//! result in `+-31753`. The fixtures are all taken at 26 and none of them reaches the clamp
//! (their largest score is 6,992), so this is where the other scales and the clamp are held to
//! what the module says.
//!
//! The only test in its binary: the scale is process-global, and a second test would race it.
//!
//! `ATTIC_SIRIUS_REF` names Sirius's reference directory (see `sirius_parity.rs`); without it
//! the test prints a notice and passes.

#![cfg(feature = "sfnn_threat")]

use std::path::PathBuf;

use attic_eval::{
    FV_SCALE_DEFAULT, SCORE_LIMIT, evaluate, fv_scale, load_network_with_warnings, set_fv_scale,
};
use attic_state::{Position, parse_sfen};

fn positions(dir: &std::path::Path) -> Vec<(String, i32)> {
    let sfens = std::fs::read_to_string(dir.join("eval-re/packed-500.tsv")).expect("positions");
    let oracle =
        std::fs::read_to_string(dir.join("eval-re/packed-500-static.tsv")).expect("oracle");
    sfens
        .lines()
        .zip(oracle.lines())
        .map(|(s, o)| {
            let (_, rest) = s.split_once('\t').unwrap();
            let (_, value) = o.split_once('\t').unwrap();
            (
                rest.strip_prefix("sfen ").unwrap().to_string(),
                value.parse().unwrap(),
            )
        })
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_scale_divides_the_raw_output_and_the_score_is_clamped() {
    let Some(dir) = std::env::var_os("ATTIC_SIRIUS_REF").map(PathBuf::from) else {
        eprintln!(
            "skipping: ATTIC_SIRIUS_REF is not set (Sirius's nn.bin is obtained out-of-band)"
        );
        return;
    };
    let (net, _) = load_network_with_warnings(&dir.join("eval/nn.bin")).expect("load");

    // The shape's default is Sirius's, with nothing set.
    assert_eq!(FV_SCALE_DEFAULT, 26);
    assert_eq!(fv_scale(), 26);

    let all = positions(&dir);
    let moderate: Vec<&(String, i32)> = all.iter().filter(|(_, v)| v.abs() < 1_000).collect();
    let large: Vec<&(String, i32)> = all.iter().filter(|(_, v)| v.abs() > 3_000).collect();
    assert!(
        moderate.len() >= 20 && large.len() >= 20,
        "need both kinds of position"
    );

    let eval_at = |scale: i32, pos: &Position| {
        set_fv_scale(scale);
        evaluate(&net, pos)
    };

    for (sfen, sirius) in moderate.iter().take(60).map(|p| (&p.0, p.1)) {
        let pos: Position = parse_sfen(sfen).unwrap();
        // At 26 the fixture's own value.
        assert_eq!(eval_at(26, &pos), sirius, "{sfen}");
        // At 1 the raw output, which stays inside the clamp for a score this small
        // (|raw| < 26 * 1000 + 26).
        let raw = eval_at(1, &pos);
        assert!(raw.abs() < SCORE_LIMIT, "{sfen}: raw {raw}");
        // Truncation toward zero, the way a C++ `/` does it.
        for scale in [2, 13, 24, 26, 28, 40, 128] {
            assert_eq!(eval_at(scale, &pos), raw / scale, "{sfen} at {scale}");
        }
        assert_eq!(
            raw / 26,
            sirius,
            "{sfen}: the raw output and the fixture disagree"
        );
    }

    // A score that would pass the limit is held to it, either sign.
    let mut clamped = [0usize; 2];
    for (sfen, sirius) in large.iter().take(60).map(|p| (&p.0, p.1)) {
        let pos: Position = parse_sfen(sfen).unwrap();
        let at_one = eval_at(1, &pos);
        assert_eq!(
            at_one,
            sirius.signum() * SCORE_LIMIT,
            "{sfen}: {sirius} at scale 1"
        );
        clamped[usize::from(sirius < 0)] += 1;
        // Where the clamp does not bite, the division is the plain one.
        assert_eq!(eval_at(26, &pos), sirius, "{sfen}");
    }
    assert!(
        clamped[0] > 0 && clamped[1] > 0,
        "the clamp should be seen on both signs: {clamped:?}"
    );

    set_fv_scale(FV_SCALE_DEFAULT);
}
