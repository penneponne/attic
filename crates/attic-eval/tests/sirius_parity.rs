//! Sirius's static evaluation, reproduced exactly.
//!
//! `ATTIC_SIRIUS_REF` names a directory holding Sirius's `nn.bin` (`eval/nn.bin`) and the
//! four fixture sets taken from the real binary's `e` command (side-to-move perspective):
//!
//! | set | positions | what it covers |
//! |---|---|---|
//! | `fixture/sfens-16384.tsv` + `fixture/static-16384.tsv` | 16,384 | ordinary positions |
//! | `eval-re/probe-3000.tsv` + `probe-3000-static.tsv` | 3,000 | entering kings, mutations, dense piles; every stack, king codes to 104 |
//! | `eval-re/packed-500.tsv` + `packed-500-static.tsv` | 500 | 38 packed pieces: threat lists at the 128 cap |
//! | `eval-re/moves-2000.tsv` + `moves-2000-static.tsv` | 2,000 | `sfen X moves m1..mk`, evaluated after the moves |
//!
//! Every line is `index <TAB> sfen <board> <side> <hand> <ply> [moves ...]` (or, in the
//! oracle files, `index <TAB> eval`). The network and the fixtures are not in the
//! repository, so without the variable the tests print a notice and pass.
//!
//! The kernels are chosen when the crate is compiled, from the target features, so the
//! backend a run exercises is the build's: `-C target-cpu=x86-64` runs the scalar kernels,
//! `x86-64-v3` the AVX2 ones, and `native` whatever the host has. The test prints which it
//! is. Run it once per build to cover each:
//!
//! ```text
//! ATTIC_SIRIUS_REF=/path/to/sirius-ref RUSTFLAGS="-C target-cpu=x86-64"    cargo test --release -p attic-eval --features sfnn_halfkahm2_threat_1024_15_64_8gp --test sirius_parity -- --nocapture
//! ATTIC_SIRIUS_REF=/path/to/sirius-ref RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test ...
//! ATTIC_SIRIUS_REF=/path/to/sirius-ref                                     cargo test ...   # native
//! ```
//!
//! On an AVX-512 host the native build compiles the same AVX2 kernels the `x86-64-v3` one
//! does (the threat shape has no AVX-512 kernels, so the arms that select them do not apply
//! to it), which is why the test says so rather than assuming the feature set decides.

#![cfg(feature = "sfnn_threat")]

use std::collections::HashMap;
use std::path::PathBuf;

use attic_eval::{
    Accumulator, FinnyCache, MoveDelta, NnueNetwork, SCORE_LIMIT, evaluate, evaluate_with,
    fv_scale, layer_stack_index, load_network_with_warnings,
};
use attic_state::{Position, parse_sfen, parse_usi_move};

fn reference_dir() -> Option<PathBuf> {
    std::env::var_os("ATTIC_SIRIUS_REF").map(PathBuf::from)
}

/// Which kernels this build compiled in, spelled from the target features it was given, and
/// what the CPU running it reports. The first decides the kernels (the choice is made when
/// the crate is compiled); the second is there so that a log from another machine says what
/// that machine had, and in particular whether an AVX-512 host ran the AVX2 kernels.
fn backend_description() -> String {
    let mut flags = Vec::new();
    if cfg!(target_feature = "avx512f") && cfg!(target_feature = "avx512bw") {
        flags.push("avx512f+avx512bw (present in the target features)");
    }
    if cfg!(target_feature = "avx2") {
        flags.push("avx2");
    }
    if cfg!(target_feature = "avxvnni") {
        flags.push("avxvnni");
    }
    if flags.is_empty() {
        flags.push("none of avx2/avx512: the portable scalar kernels");
    }
    let kernels = if cfg!(target_feature = "avx2") {
        "AVX2 transformer, AVX2 ewm, AVX2 dense kernels"
    } else {
        "scalar everywhere"
    };
    #[cfg(target_arch = "x86_64")]
    let host = format!(
        "host CPU at run time: avx2 {}, avx512f {}, avx512bw {}, avx512vnni {}, avxvnni {}",
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx512bw"),
        std::arch::is_x86_feature_detected!("avx512vnni"),
        std::arch::is_x86_feature_detected!("avxvnni"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let host = "host CPU at run time: not x86-64".to_string();
    format!(
        "target features: {}; kernels in use: {kernels}; the threat shape has no AVX-512 kernels, \
         so an AVX-512 host runs the same AVX2 ones; {host}",
        flags.join(", ")
    )
}

/// One position of a fixture: its id, the SFEN, and the moves to play from it.
struct Case {
    id: String,
    sfen: String,
    moves: Vec<String>,
}

fn read_cases(path: &std::path::Path) -> Vec<Case> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (id, rest) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("no tab in `{line}`"));
            let rest = rest
                .strip_prefix("sfen ")
                .unwrap_or_else(|| panic!("`{line}` does not start with `sfen `"));
            let (sfen, moves) = match rest.split_once(" moves ") {
                Some((sfen, moves)) => (sfen, moves.split(' ').map(str::to_string).collect()),
                None => (rest, Vec::new()),
            };
            Case {
                id: id.to_string(),
                sfen: sfen.to_string(),
                moves,
            }
        })
        .collect()
}

fn read_oracle(path: &std::path::Path) -> HashMap<String, i32> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.is_empty())
        .map(|line| {
            let (id, value) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("no tab in `{line}`"));
            (id.to_string(), value.trim().parse().expect("an integer"))
        })
        .collect()
}

fn load(dir: &std::path::Path) -> NnueNetwork {
    let path = dir.join("eval/nn.bin");
    let (net, warnings) = load_network_with_warnings(&path)
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
    assert!(warnings.is_empty(), "loader warnings: {warnings:?}");
    net
}

/// `(positions file, oracle file, label, expected size)` of the four sets.
fn sets(dir: &std::path::Path) -> Vec<(PathBuf, PathBuf, &'static str, usize)> {
    vec![
        (
            dir.join("fixture/sfens-16384.tsv"),
            dir.join("fixture/static-16384.tsv"),
            "fixture",
            16_384,
        ),
        (
            dir.join("eval-re/probe-3000.tsv"),
            dir.join("eval-re/probe-3000-static.tsv"),
            "probe",
            3_000,
        ),
        (
            dir.join("eval-re/packed-500.tsv"),
            dir.join("eval-re/packed-500-static.tsv"),
            "packed",
            500,
        ),
        (
            dir.join("eval-re/moves-2000.tsv"),
            dir.join("eval-re/moves-2000-static.tsv"),
            "moves",
            2_000,
        ),
    ]
}

fn position_after(case: &Case) -> Position {
    let mut pos = parse_sfen(&case.sfen)
        .unwrap_or_else(|e| panic!("{}: bad sfen `{}`: {e:?}", case.id, case.sfen));
    for usi in &case.moves {
        let mv = parse_usi_move(usi, &pos)
            .unwrap_or_else(|e| panic!("{}: bad move `{usi}`: {e:?}", case.id));
        pos.do_move(mv);
    }
    pos
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_static_evaluation_matches_sirius_on_every_fixture() {
    let Some(dir) = reference_dir() else {
        eprintln!(
            "skipping: ATTIC_SIRIUS_REF is not set (Sirius's nn.bin and fixtures are obtained out-of-band)"
        );
        return;
    };
    let net = load(&dir);
    assert_eq!(fv_scale(), 26, "the shape's default scale is Sirius's");
    eprintln!("sirius parity, {}", backend_description());

    let mut stacks_seen = [0usize; 8];
    let mut summary = Vec::new();
    for (positions, oracle, label, size) in sets(&dir) {
        let cases = read_cases(&positions);
        let want = read_oracle(&oracle);
        assert_eq!(cases.len(), size, "{label}: positions");
        assert_eq!(want.len(), size, "{label}: oracle");

        let mut mismatches = Vec::new();
        for case in &cases {
            let pos = position_after(case);
            let got = evaluate(&net, &pos);
            stacks_seen[layer_stack_index(&pos)] += 1;
            let expected = want[&case.id];
            if got != expected {
                mismatches.push(format!(
                    "  {}: Sirius {expected}, here {got}   {} moves {:?}",
                    case.id, case.sfen, case.moves
                ));
            }
        }
        summary.push(format!("{label} {}/{size}", size - mismatches.len()));
        assert!(
            mismatches.is_empty(),
            "{label}: {} of {size} differ, first ten:\n{}",
            mismatches.len(),
            mismatches
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    eprintln!(
        "sirius parity: {} (stack histogram {stacks_seen:?})",
        summary.join(", ")
    );
    assert!(
        stacks_seen.iter().all(|&n| n > 0),
        "the fixtures should reach every stack: {stacks_seen:?}"
    );
}

/// The `moves` set again, but each position reached the way the search reaches it: a root
/// refresh, then one derive per move with the delta and the finny table. Sirius's value for
/// the position after the moves is the oracle, so this checks the incremental update against
/// the real network, not only against this crate's own refresh.
#[test]
#[cfg_attr(miri, ignore)]
fn incremental_updates_reach_sirius_values_on_the_moves_set() {
    let Some(dir) = reference_dir() else {
        eprintln!("skipping: ATTIC_SIRIUS_REF is not set");
        return;
    };
    let net = load(&dir);
    let cases = read_cases(&dir.join("eval-re/moves-2000.tsv"));
    let want = read_oracle(&dir.join("eval-re/moves-2000-static.tsv"));
    assert_eq!(cases.len(), 2_000);

    let mut cache = FinnyCache::new();
    let mut matches = 0usize;
    let mut plies = 0usize;
    for case in &cases {
        let mut pos = parse_sfen(&case.sfen).expect("sfen");
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);
        for usi in &case.moves {
            let mv = parse_usi_move(usi, &pos).expect("move");
            let delta = MoveDelta::from_move(&pos, mv);
            pos.do_move(mv);
            let mut child = Accumulator::new();
            Accumulator::derive_into_cached(&acc, &mut child, &net, &pos, &delta, &mut cache);
            acc = child;
            plies += 1;
        }
        let got = evaluate_with(&net, &acc, &pos);
        let expected = want[&case.id];
        assert_eq!(
            got, expected,
            "{}: {} moves {:?}",
            case.id, case.sfen, case.moves
        );
        assert_eq!(
            got,
            evaluate(&net, &pos),
            "{}: incremental != refresh",
            case.id
        );
        matches += 1;
    }
    eprintln!("incremental parity: {matches}/2000 positions, {plies} derived plies");
    assert_eq!(matches, 2_000);
}

#[test]
fn the_score_stays_inside_sirius_range() {
    assert_eq!(SCORE_LIMIT, 31_753);
}
