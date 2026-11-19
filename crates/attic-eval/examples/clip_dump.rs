//! Evaluate a file of positions once each, with the counters of a `clip_stats` build on, and print
//! how often each layer of the head sat on its floor or ceiling.
//!
//! ```text
//! cargo run --release -p attic-eval --features synn_token_128x2_32_32_pair16_o1,clip_stats \
//!     --example clip_dump -- --net nn.bin --sfens positions.sfen [--limit N]
//! ```
//!
//! This is the engine's side of `tideborn-train activations --sfen FILE`: that command counts the
//! f32 net over a directory of positions and writes them out, and this one counts the engine's own
//! arithmetic (the f16 tables, the `i16` or `i8` first layer set by `ATTIC_SYNN_W1`) over the same
//! file, one evaluation a position and no search, so that the two lines of a layer are about the
//! same positions and differ only in the arithmetic.
//!
//! The lines are the ones the engine's `clipstats` command prints, with the same fields as the
//! trainer's `activations`.

use std::io::BufRead;
use std::path::PathBuf;

use attic_eval::manaka::{clip_stats, evaluate_value, load_network};
use attic_state::parse_sfen;

fn main() {
    let mut net = None;
    let mut sfens = None;
    let mut limit = usize::MAX;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--net" => net = args.next().map(PathBuf::from),
            "--sfens" => sfens = args.next().map(PathBuf::from),
            "--limit" => limit = args.next().and_then(|n| n.parse().ok()).unwrap_or(limit),
            other => {
                eprintln!("unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }
    let (Some(net), Some(sfens)) = (net, sfens) else {
        eprintln!("usage: clip_dump --net FILE --sfens FILE [--limit N]");
        std::process::exit(2);
    };

    let network = load_network(&net).unwrap_or_else(|e| {
        eprintln!("{}: {e}", net.display());
        std::process::exit(1);
    });
    let file = std::fs::File::open(&sfens).unwrap_or_else(|e| {
        eprintln!("{}: {e}", sfens.display());
        std::process::exit(1);
    });

    for (index, line) in std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .enumerate()
    {
        let pos = parse_sfen(line.trim()).unwrap_or_else(|e| {
            eprintln!("line {index}: {e:?}");
            std::process::exit(1);
        });
        evaluate_value(&network, &pos);
    }
    for line in clip_stats::summary().lines() {
        println!("{line}");
    }
}
