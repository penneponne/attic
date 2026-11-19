#!/usr/bin/env python3
"""Search the Sirius fixture positions with the threat shape, twice, and diff the answers.

    sirius_search_check.py --ref DIR --binary A [--binary B ...] [--count 400] [--depth 9]

The incremental update of the threat shape is checked three ways in the tests (against a
refresh, position by position). This is the fourth, end to end: every position is searched
by each `--binary` and the output must be identical line for line. Build the two binaries
from the same source, one plain and one with `threat_force_refresh`, which rebuilds the
accumulator at every move instead of deriving it:

    cargo build --release -p attic --features sfnn_halfkahm2_threat_1024_15_64_8gp
    cargo build --release -p attic --features sfnn_halfkahm2_threat_1024_15_64_8gp,threat_force_refresh

A search that comes out the same with and without the incremental update has no position in
its tree where the two differed. Pass the plain binary twice first (`--binary A --binary A`)
to see that the search itself repeats: if it does not, comparing two builds says nothing.

What is compared, per position: every `info` line with `time`, `nps` and `hashfull` taken out
(the first two are the clock and the third depends on the table's fill, which a fixed hash makes
repeat but which is not what is being checked), and the `bestmove` line. The engine is driven
one position at a time, waiting for `bestmove` before the next command, with `Threads 1`, a
fixed hash, `PvInterval 0` so that every iteration prints, and `usinewgame` + `isready` between
positions so that nothing carries over.

The positions are every k-th line of Sirius's fixture sets (see `sirius_parity.rs` for the
files). A fixture position is a static one and some are not reachable in a game: the side to
move attacks the other king, so a king could be captured, which no search meets. Those are
left out, judged here with a mailbox walk of the attack rules.

For each binary the script also prints nodes per second over the whole run, summed from the
`nodes` and `time` of each position's last `info` line (`time` is in milliseconds, so this is a
figure for the run and not for any one position). It is a measurement of a loaded machine if
the machine is loaded; run it alone to compare binaries by speed.

`--eval-dir` loads another network than Sirius's. The positions are Sirius's either way, so
this is how to confirm that a change to the shared code left another shape's search alone:
build that shape before and after, and compare the two binaries on one of its networks.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import time
from pathlib import Path

DROP_FIELDS = re.compile(r" (?:time|nps|hashfull) \d+")


def parse_board(board: str) -> dict[int, tuple[int, str, bool]]:
    """`square -> (colour, kind, promoted)`; square = file * 9 + rank, file 0 being the 1 file."""
    out: dict[int, tuple[int, str, bool]] = {}
    rank, file = 0, 8
    promoted = False
    for ch in board:
        if ch == "/":
            rank, file = rank + 1, 8
        elif ch.isdigit():
            file -= int(ch)
        elif ch == "+":
            promoted = True
        else:
            out[file * 9 + rank] = (1 if ch.islower() else 0, ch.upper(), promoted)
            promoted = False
            file -= 1
    return out


def attacks(board: dict[int, tuple[int, str, bool]], sq: int) -> set[int]:
    colour, kind, promoted = board[sq]
    file, rank = divmod(sq, 9)
    fwd = 1 if colour else -1
    gold = [(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    steps: list[tuple[int, int]] = []
    rays: list[tuple[int, int]] = []
    if promoted and kind in "PLNS":
        steps = gold
    elif kind == "P":
        steps = [(0, fwd)]
    elif kind == "L":
        rays = [(0, fwd)]
    elif kind == "N":
        steps = [(-1, 2 * fwd), (1, 2 * fwd)]
    elif kind == "S":
        steps = [(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)]
    elif kind == "G":
        steps = gold
    elif kind == "B":
        rays, steps = diag, (orth if promoted else [])
    elif kind == "R":
        rays, steps = orth, (diag if promoted else [])
    out: set[int] = set()
    for df, dr in steps:
        f, r = file + df, rank + dr
        if 0 <= f < 9 and 0 <= r < 9:
            out.add(f * 9 + r)
    for df, dr in rays:
        f, r = file + df, rank + dr
        while 0 <= f < 9 and 0 <= r < 9:
            out.add(f * 9 + r)
            if f * 9 + r in board:
                break
            f, r = f + df, r + dr
    return out


def reachable(sfen: str) -> bool:
    """False when the side to move could take the other king (no game gets there)."""
    board_s, stm, *_ = sfen.split()
    board = parse_board(board_s)
    me = 1 if stm == "w" else 0
    kings = {c: sq for sq, (c, k, _) in board.items() if k == "K"}
    if len(kings) != 2:
        return False
    mine, theirs = kings[me], kings[1 - me]
    if abs(mine // 9 - theirs // 9) <= 1 and abs(mine % 9 - theirs % 9) <= 1:
        return False
    return not any(
        c == me and k != "K" and theirs in attacks(board, sq) for sq, (c, k, _) in board.items()
    )


def positions(ref: Path, count: int) -> list[str]:
    """`count` positions, spread over the four fixture sets, as `sfen ... [moves ...]` strings."""
    sets = [
        ("fixture/sfens-16384.tsv", 0.50),
        ("eval-re/moves-2000.tsv", 0.25),
        ("eval-re/probe-3000.tsv", 0.15),
        ("eval-re/packed-500.tsv", 0.10),
    ]
    out: list[str] = []
    for name, share in sets:
        lines = [l for l in (ref / name).read_text().splitlines() if l]
        want = max(1, round(count * share))
        stride = max(1, len(lines) // (want * 2))
        taken = 0
        for line in lines[::stride]:
            sfen = line.split("\t", 1)[1].removeprefix("sfen ")
            if reachable(sfen.split(" moves ")[0]):
                out.append(sfen)
                taken += 1
                if taken == want:
                    break
    return out


class Engine:
    def __init__(self, binary: str, eval_dir: str, hash_mb: int) -> None:
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("usi")
        self.read_until("usiok")
        for name, value in [
            ("EvalDir", eval_dir),
            ("Threads", "1"),
            ("USI_Hash", str(hash_mb)),
            ("PvInterval", "0"),
        ]:
            self.send(f"setoption name {name} value {value}")
        self.notices = self.sync()

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited before `{prefix}`; got {lines[-5:]}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def sync(self) -> list[str]:
        """isready -> readyok; returns the `info string` lines seen on the way."""
        self.send("isready")
        return [l for l in self.read_until("readyok") if l.startswith("info string")]

    def search(self, position: str, depth: int) -> list[str]:
        self.send("usinewgame")
        self.sync()
        self.send(f"position sfen {position}")
        self.send(f"go depth {depth}")
        return self.read_until("bestmove")

    def close(self) -> None:
        self.send("quit")
        self.proc.wait(timeout=30)


def normalise(lines: list[str]) -> list[str]:
    return [DROP_FIELDS.sub("", l) for l in lines if l.startswith(("info depth", "bestmove"))]


def last_info(lines: list[str]) -> tuple[int, int]:
    """`(nodes, time_ms)` of the last `info depth` line."""
    nodes = time_ms = 0
    for l in lines:
        if l.startswith("info depth"):
            m = re.search(r" nodes (\d+)", l)
            t = re.search(r" time (\d+)", l)
            if m and t:
                nodes, time_ms = int(m.group(1)), int(t.group(1))
    return nodes, time_ms


def run(binary: str, args: argparse.Namespace, sfens: list[str]) -> tuple[list[list[str]], list[str], float, tuple[int, int]]:
    engine = Engine(binary, str(args.eval_dir or args.ref / "eval"), args.hash)
    results: list[list[str]] = []
    nodes = ms = 0
    started = time.time()
    for i, sfen in enumerate(sfens):
        raw = engine.search(sfen, args.depth)
        results.append(normalise(raw))
        n, t = last_info(raw)
        nodes, ms = nodes + n, ms + t
        if (i + 1) % 50 == 0:
            print(f"  {binary}: {i + 1}/{len(sfens)}", file=sys.stderr, flush=True)
    notices = engine.notices
    engine.close()
    return results, notices, time.time() - started, (nodes, ms)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="Sirius reference directory (eval/nn.bin and the fixtures)")
    ap.add_argument("--binary", action="append", required=True, help="an attic binary (repeat to compare several)")
    ap.add_argument(
        "--eval-dir",
        type=Path,
        help="EvalDir to load (default REF/eval). Another network works too, to compare two builds "
        "of a shape that is not the threat one: the positions are Sirius's either way",
    )
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--hash", type=int, default=16, help="USI_Hash in MB")
    ap.add_argument("--out", type=Path, help="write each binary's normalised output here")
    args = ap.parse_args()
    if len(args.binary) < 2:
        ap.error("give at least two --binary to compare")

    sfens = positions(args.ref, args.count)
    print(f"{len(sfens)} positions, depth {args.depth}, Threads 1, USI_Hash {args.hash}", file=sys.stderr)

    runs = []
    for binary in args.binary:
        results, notices, seconds, totals = run(binary, args, sfens)
        runs.append((binary, results, notices, seconds, totals))
        lines = sum(len(r) for r in results)
        nodes, ms = totals
        print(
            f"{binary}: {len(results)} positions, {lines} lines, {seconds:.1f} s wall, "
            f"{nodes} nodes in {ms} ms of search = {nodes * 1000 // max(ms, 1)} nps; "
            f"load notices: {notices or 'none'}"
        )
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            tag = f"{len(runs) - 1}-{Path(binary).name}"
            (args.out / f"{tag}.txt").write_text(
                "\n".join(f"## {s}\n" + "\n".join(r) for s, r in zip(sfens, results)) + "\n"
            )

    base_binary, base, *_ = runs[0]
    status = 0
    for binary, results, *_ in runs[1:]:
        different = [i for i, (a, b) in enumerate(zip(base, results)) if a != b]
        compared = sum(len(a) for a in base)
        print(
            f"{base_binary} vs {binary}: {len(sfens) - len(different)}/{len(sfens)} positions identical, "
            f"{compared} lines compared"
        )
        for i in different[:3]:
            status = 1
            print(f"--- position {i}: {sfens[i]}")
            for l in list(difflib.unified_diff(base[i], results[i], "first", "second", lineterm="", n=0))[:12]:
                print(l)
        if different:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
