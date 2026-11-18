//! Session tests for the `clipstats` command.
//!
//! What is counted is `attic-eval`'s business and is tested there against a network built in
//! memory. These say what the command answers, with and without the counters in the build, and
//! that asking for the counts is not an unknown command in either.

mod common;

use common::drive;

/// The `info string` lines of a transcript that start with `clipstats`.
fn clipstats_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("info string "))
        .filter(|l| l.starts_with("clipstats"))
        .collect()
}

#[cfg(not(feature = "clip_stats"))]
#[test]
#[cfg_attr(miri, ignore)]
fn a_build_without_the_counters_says_how_to_get_one() {
    let out = drive("clipstats\nquit\n");
    assert_eq!(
        clipstats_lines(&out),
        ["clipstats: this build does not count; build with --features clip_stats"],
        "{out}"
    );
    assert!(!out.contains("unknown command"), "{out}");
}

#[cfg(not(feature = "clip_stats"))]
#[test]
#[cfg_attr(miri, ignore)]
fn a_build_without_the_counters_answers_reset_the_same_way() {
    let out = drive("clipstats reset\nquit\n");
    assert_eq!(clipstats_lines(&out).len(), 1, "{out}");
    assert!(!out.contains("unknown command"), "{out}");
}

#[cfg(feature = "clip_stats")]
#[test]
#[cfg_attr(miri, ignore)]
fn a_counting_build_prints_the_net_line_and_one_line_a_layer() {
    let out = drive("clipstats\nquit\n");
    let lines = clipstats_lines(&out);
    // The net's line, then power sums, then (per shape) the optional blocks, then both hidden
    // layers: never fewer than four.
    assert!(lines.len() >= 4, "{out}");
    assert!(lines[0].starts_with("clipstats first_layer="), "{out}");
    assert!(lines.iter().any(|l| l.contains("layer=\"power sums\"")));
    assert!(lines.iter().any(|l| l.contains("layer=\"hidden 1\"")));
    assert!(lines.iter().any(|l| l.contains("layer=\"hidden 2\"")));
}

#[cfg(feature = "clip_stats")]
#[test]
#[cfg_attr(miri, ignore)]
fn reset_is_answered_and_a_bad_word_is_told_the_usage() {
    let out = drive("clipstats reset\nclipstats nonsense\nquit\n");
    let lines = clipstats_lines(&out);
    assert_eq!(
        lines,
        ["clipstats: reset", "clipstats: usage: clipstats [reset]"],
        "{out}"
    );
}
