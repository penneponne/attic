//! The incremental update of the threat shape equals a from-scratch refresh at every ply of
//! random legal games.
//!
//! The games start from the Sirius fixture positions (ordinary, advanced kings, dense piles,
//! the 128-edge cap) and from hand-made ones whose kings stand on the mirror boundary, and
//! are steered toward the moves that make the update hard: king moves (a half is rebuilt, and
//! the king's file can cross the mirror line), captures, drops, promotions, and moves that
//! open a line for a slider that did not move. At every ply three ways of getting the child
//! accumulator are held against a refresh of the position:
//!
//! * `update_after_move`, which does its own `do_move`;
//! * `derive_into`, the search's form without the finny table;
//! * `derive_into_cached`, with one finny table kept for the whole run.
//!
//! "Held against" means both halves equal, both threat lists equal (rows and order, which the
//! cap makes part of the answer), and the evaluation equal. The lists are also compared with
//! an enumeration written here from the rules in `SPEC.md` (a ray walk over a mailbox, the
//! cap spelled the way the reference's code spells it), which shares nothing with the bitboard
//! one in the crate, so that a fault they would both make is not what the test trusts.
//!
//! `ATTIC_SIRIUS_REF` names Sirius's reference directory (see `sirius_parity.rs`); without it
//! the test prints a notice and passes. `ATTIC_SIRIUS_GAMES` scales the number of games
//! (default 600) and `ATTIC_SIRIUS_PLIES` their length (default 90).

#![cfg(feature = "sfnn_threat")]

use std::collections::HashSet;
use std::path::PathBuf;

use attic_eval::threat::{MAX_EDGES, ThreatList, build_lists};
use attic_eval::{
    Accumulator, FinnyCache, MoveDelta, NnueNetwork, evaluate, evaluate_with,
    load_network_with_warnings,
};
use attic_state::{
    Color, Move, Piece, PieceKind, Position, format_sfen, format_usi_move, parse_sfen,
};

fn reference_dir() -> Option<PathBuf> {
    std::env::var_os("ATTIC_SIRIUS_REF").map(PathBuf::from)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A small deterministic xorshift64*.
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

    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---- The independent enumeration -------------------------------------------------------

/// What the oracle needs to know about a piece: its owner (0 Black), and the moves it makes.
#[derive(Clone, Copy)]
struct Cell {
    color: usize,
    is_king: bool,
    piece: Piece,
}

fn mailbox(pos: &Position) -> [Option<Cell>; 81] {
    std::array::from_fn(|i| {
        let sq = attic_state::Square::from_index(i as u8).unwrap();
        pos.board().get(sq).map(|piece| Cell {
            color: piece.color.index(),
            is_king: piece.kind == PieceKind::King,
            piece,
        })
    })
}

/// The squares the piece on `s` attacks, ascending, rays cut at (and including) the first
/// piece. Black moves toward rank 0, White toward rank 8.
fn oracle_attacks(board: &[Option<Cell>; 81], s: usize) -> Vec<usize> {
    let cell = board[s].expect("an attacker");
    let (file, rank) = ((s / 9) as i32, (s % 9) as i32);
    let fwd = if cell.color == 0 { -1 } else { 1 };
    let mut steps: Vec<(i32, i32)> = Vec::new();
    let mut rays: Vec<(i32, i32)> = Vec::new();
    let gold = [(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)];
    let diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
    let orthogonals = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    match (cell.piece.kind, cell.piece.promoted) {
        (PieceKind::Pawn, false) => steps.push((0, fwd)),
        (PieceKind::Lance, false) => rays.push((0, fwd)),
        (PieceKind::Knight, false) => steps.extend([(-1, 2 * fwd), (1, 2 * fwd)]),
        (PieceKind::Silver, false) => {
            steps.extend([(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)])
        }
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            steps.extend(gold)
        }
        (PieceKind::Bishop, false) => rays.extend(diagonals),
        (PieceKind::Rook, false) => rays.extend(orthogonals),
        (PieceKind::Bishop, true) => {
            rays.extend(diagonals);
            steps.extend(orthogonals);
        }
        (PieceKind::Rook, true) => {
            rays.extend(orthogonals);
            steps.extend(diagonals);
        }
        (PieceKind::King, _) => unreachable!("a king attacks nothing here"),
    }
    let mut out = Vec::new();
    for (df, dr) in steps {
        let (f, r) = (file + df, rank + dr);
        if (0..9).contains(&f) && (0..9).contains(&r) {
            out.push((f * 9 + r) as usize);
        }
    }
    for (df, dr) in rays {
        let (mut f, mut r) = (file + df, rank + dr);
        while (0..9).contains(&f) && (0..9).contains(&r) {
            let t = (f * 9 + r) as usize;
            out.push(t);
            if board[t].is_some() {
                break;
            }
            f += df;
            r += dr;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Each perspective's list of `(attacker, target)`, as the reference code builds it
/// (`threat_lists` in `sirius_eval.cpp`): one pass per perspective when the two mirror
/// differently, one pass for both otherwise with the cap checked on both lists.
fn oracle_edges(pos: &Position) -> [Vec<(usize, usize)>; 2] {
    let board = mailbox(pos);
    let king = |color: Color| pos.king_square(color).unwrap().index() as usize;
    let (black_king, white_king) = (king(Color::Black), king(Color::White));
    let flip = [black_king > 44, white_king < 36];
    let separate = flip[0] != flip[1];

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for s in 0..81 {
        let Some(a) = board[s] else { continue };
        if a.is_king {
            continue;
        }
        for t in oracle_attacks(&board, s) {
            match board[t] {
                Some(target) if !target.is_king => edges.push((s, t)),
                _ => {}
            }
        }
    }

    let colors = |&(s, t): &(usize, usize)| (board[s].unwrap().color, board[t].unwrap().color);
    let mut lists: [Vec<(usize, usize)>; 2] = [Vec::new(), Vec::new()];
    if separate {
        for (p, list) in lists.iter_mut().enumerate() {
            for edge in &edges {
                let (aw, tw) = colors(edge);
                let keep = tw != aw || p == aw;
                if list.len() < MAX_EDGES && keep {
                    list.push(*edge);
                }
            }
        }
    } else {
        for edge in &edges {
            if lists[0].len() < MAX_EDGES && lists[1].len() < MAX_EDGES {
                let (aw, tw) = colors(edge);
                lists[0].push(*edge);
                lists[1].push(*edge);
                if aw == tw {
                    // The side that does not own both pieces does not see the edge.
                    lists[1 - aw].pop();
                }
            }
        }
    }
    lists
}

/// Whether the side to move attacks the other king: a position no game reaches. Judged with
/// the oracle's own attack walk, so it needs nothing from the crate under test.
fn side_not_to_move_is_in_check(pos: &Position) -> bool {
    let board = mailbox(pos);
    let stm = pos.side_to_move().index();
    let victim = pos.king_square(pos.side_to_move().flip()).unwrap().index() as usize;
    // Adjacent kings attack each other; the walk below leaves kings out.
    let mine = pos.king_square(pos.side_to_move()).unwrap().index() as usize;
    let adjacent = (mine / 9).abs_diff(victim / 9) <= 1 && (mine % 9).abs_diff(victim % 9) <= 1;
    adjacent
        || (0..81).any(|s| {
            board[s].is_some_and(|c| c.color == stm && !c.is_king)
                && oracle_attacks(&board, s).contains(&victim)
        })
}

fn assert_lists_match_the_oracle(pos: &Position, lists: &[ThreatList; 2], ctx: &str) {
    let want = oracle_edges(pos);
    for p in 0..2 {
        let got: Vec<(usize, usize)> = lists[p].edges().map(|(a, t, _)| (a, t)).collect();
        assert_eq!(
            got,
            want[p],
            "{ctx}: perspective {p} edges differ from the independent enumeration at {}",
            format_sfen(pos)
        );
    }
}

// ---- Coverage --------------------------------------------------------------------------

#[derive(Default, Debug)]
struct Coverage {
    skipped_starts: usize,
    plies: usize,
    games: usize,
    captures: usize,
    drops: usize,
    promotions: usize,
    king_moves: usize,
    /// A king move after which its perspective mirrors differently than before.
    mirror_crossings: usize,
    /// A move after which a slider that did not move attacks a square beyond the one the
    /// mover vacated.
    slider_unblockings: usize,
    /// A ply after which some list holds `MAX_EDGES` edges.
    capped_lists: usize,
    /// A ply whose two perspectives mirror differently (two enumeration passes).
    separate_passes: usize,
}

fn mirrors(pos: &Position) -> [bool; 2] {
    [
        pos.king_square(Color::Black).unwrap().file() >= 5,
        pos.king_square(Color::White).unwrap().file() <= 3,
    ]
}

/// Whether `f` lies strictly between `a` and `t` on a straight or diagonal line.
fn strictly_between(a: usize, t: usize, f: usize) -> bool {
    let (af, ar) = ((a / 9) as i32, (a % 9) as i32);
    let (tf, tr) = ((t / 9) as i32, (t % 9) as i32);
    let (ff, fr) = ((f / 9) as i32, (f % 9) as i32);
    let (dx, dy) = ((tf - af).signum(), (tr - ar).signum());
    let dist = (tf - af).abs().max((tr - ar).abs());
    if (tf - af != 0 && tr - ar != 0) && (tf - af).abs() != (tr - ar).abs() {
        return false;
    }
    (1..dist).any(|k| af + k * dx == ff && ar + k * dy == fr)
}

fn is_slider(piece: Piece) -> bool {
    matches!(
        (piece.kind, piece.promoted),
        (PieceKind::Lance, false) | (PieceKind::Bishop, _) | (PieceKind::Rook, _)
    )
}

// ---- Checking one ply ------------------------------------------------------------------

fn assert_same_as_refresh(
    net: &NnueNetwork,
    acc: &Accumulator,
    pos: &Position,
    fresh_lists: &[ThreatList; 2],
    ctx: &str,
) {
    let mut fresh = Accumulator::new();
    fresh.refresh(net, pos);
    for color in [Color::Black, Color::White] {
        assert!(
            acc.perspective(color) == fresh.perspective(color),
            "{ctx}: {color:?} half diverged from a refresh at {}",
            format_sfen(pos)
        );
        assert!(
            acc.threat_list(color) == &fresh_lists[color.index()],
            "{ctx}: {color:?} threat list diverged from a fresh enumeration at {}",
            format_sfen(pos)
        );
        assert!(
            fresh.threat_list(color) == &fresh_lists[color.index()],
            "{ctx}: refresh stored another list than build_lists at {}",
            format_sfen(pos)
        );
    }
    assert_eq!(
        evaluate_with(net, acc, pos),
        evaluate(net, pos),
        "{ctx}: evaluate paths disagree at {}",
        format_sfen(pos)
    );
}

/// One frame of a game: the move, its undo token and the three accumulators.
struct Frame {
    mv: Move,
    undo: attic_state::Undo,
    updated: Accumulator,
    derived: Accumulator,
    cached: Accumulator,
}

fn play_game(
    net: &NnueNetwork,
    cache: &mut FinnyCache,
    start: &str,
    seed: u64,
    max_plies: usize,
    cov: &mut Coverage,
) {
    let mut pos = parse_sfen(start).unwrap_or_else(|e| panic!("bad start `{start}`: {e:?}"));
    // The fixtures are static positions and some (the mutated ones) are not reachable in a
    // game: the side that just moved is in check, so the side to move could take its king.
    // A search never meets such a position, and the feature code rightly refuses to move a
    // king into a hand; those starts are skipped.
    if side_not_to_move_is_in_check(&pos) {
        cov.skipped_starts += 1;
        return;
    }
    let mut rng = Rng(seed | 1);
    let root_lists = build_lists(&pos);
    assert_lists_match_the_oracle(&pos, &root_lists, "root");
    let mut root = Accumulator::new();
    root.refresh(net, &pos);
    assert_same_as_refresh(net, &root, &pos, &root_lists, "root");

    let mut stack: Vec<Frame> = Vec::new();
    let mut legal: Vec<Move> = Vec::with_capacity(256);
    cov.games += 1;

    for ply in 0..max_plies {
        legal.clear();
        pos.generate_legal_all(&mut legal);
        if legal.is_empty() {
            break;
        }

        // Steer. Uniform play almost never moves the king, and the king's moves are the
        // ones that rebuild a half and flip its mirror.
        let preference = rng.pick(8);
        let is_king_move = |m: &Move| !m.is_drop() && m.moved_piece_after().kind == PieceKind::King;
        let is_capture = |m: &Move| !m.is_drop() && pos.board().get(m.to_sq()).is_some();
        let pool: Vec<Move> = match preference {
            0 | 1 => legal.iter().copied().filter(is_king_move).collect(),
            2 | 3 => legal.iter().copied().filter(is_capture).collect(),
            4 => legal.iter().copied().filter(|m| m.is_promote()).collect(),
            5 => legal.iter().copied().filter(|m| m.is_drop()).collect(),
            _ => Vec::new(),
        };
        let mv = if pool.is_empty() {
            legal[rng.pick(legal.len())]
        } else {
            pool[rng.pick(pool.len())]
        };
        let ctx = format!("game from `{start}` ply {ply} `{}`", format_usi_move(mv));

        if let Some(victim) = (!mv.is_drop())
            .then(|| pos.board().get(mv.to_sq()))
            .flatten()
        {
            assert!(
                victim.kind != PieceKind::King,
                "{ctx}: the generator offers a king capture at {} (start `{start}`)",
                format_sfen(&pos)
            );
        }
        let before_mirrors = mirrors(&pos);
        let before_lists = build_lists(&pos);
        let from = (!mv.is_drop()).then(|| mv.from_sq().index() as usize);
        let delta = MoveDelta::from_move(&pos, mv);
        let parent = stack.last();
        let parent_updated = parent.map_or(&root, |f| &f.updated);
        let parent_derived = parent.map_or(&root, |f| &f.derived);
        let parent_cached = parent.map_or(&root, |f| &f.cached);

        let captured = is_capture(&mv);
        let king_move = !mv.is_drop() && mv.moved_piece_after().kind == PieceKind::King;
        let updated = parent_updated.update_after_move(net, &mut pos, mv);
        let undo = pos.do_move(mv);
        let lists = build_lists(&pos);
        assert_lists_match_the_oracle(&pos, &lists, &ctx);

        let mut derived = Accumulator::new();
        Accumulator::derive_into(parent_derived, &mut derived, net, &pos, &delta);
        let mut cached = Accumulator::new();
        Accumulator::derive_into_cached(parent_cached, &mut cached, net, &pos, &delta, cache);

        assert_same_as_refresh(
            net,
            &updated,
            &pos,
            &lists,
            &format!("{ctx} [update_after_move]"),
        );
        assert_same_as_refresh(net, &derived, &pos, &lists, &format!("{ctx} [derive_into]"));
        assert_same_as_refresh(
            net,
            &cached,
            &pos,
            &lists,
            &format!("{ctx} [derive_into_cached]"),
        );

        // Coverage.
        cov.plies += 1;
        cov.captures += usize::from(captured);
        cov.drops += usize::from(mv.is_drop());
        cov.promotions += usize::from(mv.is_promote());
        cov.king_moves += usize::from(king_move);
        let after_mirrors = mirrors(&pos);
        if king_move && before_mirrors != after_mirrors {
            cov.mirror_crossings += 1;
        }
        cov.capped_lists += usize::from(lists.iter().any(|l| l.edge_count() == MAX_EDGES));
        cov.separate_passes += usize::from(after_mirrors[0] != after_mirrors[1]);
        if let Some(from) = from {
            let to = mv.to_sq().index() as usize;
            let opened = (0..2).any(|p| {
                let old: HashSet<(usize, usize)> =
                    before_lists[p].edges().map(|(a, t, _)| (a, t)).collect();
                lists[p].edges().any(|(a, t, _)| {
                    a != to
                        && !old.contains(&(a, t))
                        && strictly_between(a, t, from)
                        && pos
                            .board()
                            .get(attic_state::Square::from_index(a as u8).unwrap())
                            .is_some_and(is_slider)
                })
            });
            cov.slider_unblockings += usize::from(opened);
        }

        stack.push(Frame {
            mv,
            undo,
            updated,
            derived,
            cached,
        });
    }

    // Back out, and check that each parent still describes the restored position.
    while let Some(frame) = stack.pop() {
        pos.undo_move(frame.mv, frame.undo);
        let parent = stack.last();
        let lists = build_lists(&pos);
        for (name, acc) in [
            ("updated", parent.map_or(&root, |f| &f.updated)),
            ("derived", parent.map_or(&root, |f| &f.derived)),
            ("cached", parent.map_or(&root, |f| &f.cached)),
        ] {
            assert_same_as_refresh(
                net,
                acc,
                &pos,
                &lists,
                &format!("unwind `{}` [{name}]", format_usi_move(frame.mv)),
            );
        }
    }
}

/// Starting positions: every `stride`-th line of a fixture file, then hand-made ones.
fn starting_positions(dir: &std::path::Path, games: usize) -> Vec<String> {
    let per_set = games / 5;
    let mut out = Vec::new();
    for (file, count) in [
        ("fixture/sfens-16384.tsv", per_set * 2),
        ("eval-re/probe-3000.tsv", per_set),
        ("eval-re/packed-500.tsv", per_set),
        ("eval-re/moves-2000.tsv", per_set),
    ] {
        let text = std::fs::read_to_string(dir.join(file)).expect("fixture file");
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        let stride = (lines.len() / count.max(1)).max(1);
        for line in lines.iter().step_by(stride).take(count) {
            let rest = line
                .split_once('\t')
                .unwrap()
                .1
                .strip_prefix("sfen ")
                .unwrap();
            out.push(rest.split(" moves ").next().unwrap().to_string());
        }
    }
    out.extend(
        [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            // Kings on the middle file, one step from the mirror line, with sliders about.
            "4k4/9/4r4/9/9/9/4B4/9/4K4 b RG2gs2n3p 1",
            "4k4/9/9/4r4/9/4R4/9/9/4K4 w 2G 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
            // Both kings on file 5 with long open files, and on file 6/4 across it.
            "l3k3l/9/9/9/9/9/9/9/L3K3L b 2R2B4G4S4N18P 1",
            "l4k2l/9/9/9/9/9/9/9/L2K4L w 2R2B4G4S4N18P 1",
        ]
        .map(str::to_string),
    );
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn incremental_updates_equal_a_refresh_along_random_games() {
    let Some(dir) = reference_dir() else {
        eprintln!(
            "skipping: ATTIC_SIRIUS_REF is not set (Sirius's nn.bin is obtained out-of-band)"
        );
        return;
    };
    let path = dir.join("eval/nn.bin");
    let (net, warnings) = load_network_with_warnings(&path)
        .unwrap_or_else(|e| panic!("cannot load {}: {e}", path.display()));
    assert!(warnings.is_empty(), "{warnings:?}");

    let games = env_usize("ATTIC_SIRIUS_GAMES", 600);
    let plies = env_usize("ATTIC_SIRIUS_PLIES", 90);
    let starts = starting_positions(&dir, games);
    let mut cache = FinnyCache::new();
    let mut cov = Coverage::default();
    for (i, start) in starts.iter().enumerate() {
        play_game(
            &net,
            &mut cache,
            start,
            0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1),
            plies,
            &mut cov,
        );
    }
    eprintln!("random games: {cov:?}");
    assert!(cov.plies > 0);
    assert!(
        cov.skipped_starts * 4 < starts.len(),
        "too many starting positions were unreachable: {cov:?}"
    );
    assert!(cov.captures > 0, "no capture was played");
    assert!(cov.drops > 0, "no drop was played");
    assert!(cov.promotions > 0, "no promotion was played");
    assert!(cov.king_moves > 0, "no king move was played");
    assert!(cov.mirror_crossings > 0, "no king crossed the mirror line");
    assert!(cov.slider_unblockings > 0, "no slider was unblocked");
    assert!(cov.capped_lists > 0, "no list reached the 128-edge cap");
    assert!(
        cov.separate_passes > 0,
        "no position needed two enumeration passes"
    );
}

/// The mirror line, by hand: each king walks across it and back, with the other pieces
/// still, and every step is held against a refresh.
#[test]
#[cfg_attr(miri, ignore)]
fn a_king_walking_across_the_mirror_line_stays_exact() {
    let Some(dir) = reference_dir() else {
        eprintln!("skipping: ATTIC_SIRIUS_REF is not set");
        return;
    };
    let (net, _) = load_network_with_warnings(&dir.join("eval/nn.bin")).expect("load");
    let mut cache = FinnyCache::new();

    // Black's king crosses 5->6 (file index 4 -> 5) and White's 5->4 (4 -> 3).
    let mut pos = parse_sfen("l3k3l/9/p3p3p/9/9/9/P3P3P/9/L3K3L b R2B3G4S4N 1").unwrap();
    let mut acc = Accumulator::new();
    acc.refresh(&net, &pos);
    let line = [
        "5i6i", "5a4a", "6i7i", "4a5a", "7i6i", "5a4a", "6i5i", "4a3a",
    ];
    let mut seen = HashSet::new();
    for usi in line {
        let mv = attic_state::parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("{usi}: {e:?}"));
        seen.insert(mirrors(&pos));
        let delta = MoveDelta::from_move(&pos, mv);
        pos.do_move(mv);
        let mut child = Accumulator::new();
        Accumulator::derive_into_cached(&acc, &mut child, &net, &pos, &delta, &mut cache);
        let lists = build_lists(&pos);
        assert_same_as_refresh(&net, &child, &pos, &lists, usi);
        acc = child;
    }
    seen.insert(mirrors(&pos));
    assert!(
        seen.len() >= 3,
        "the walk should visit several mirror states: {seen:?}"
    );
}
