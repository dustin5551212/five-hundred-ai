"""Play a match at the terminal: `python -m fivehundred [--seed N]`.

A scorecard, not a UI — this exists so the port can be watched doing the whole
thing: deal, auction, kitty, ten tricks, score, next hand.
"""

from __future__ import annotations

import argparse
import random
import sys

from .game import MatchResult, cpu_table, play_match
from .scoring import apply_score


def play(seed: int, target: int) -> MatchResult:
    bidders, players = cpu_table()
    return play_match(random.Random(seed), bidders, players, target=target)


def render(match: MatchResult) -> str:
    """The scorecard, with the running totals beside each hand."""
    lines = [
        "   #  contract                    tricks       result              us / them",
        "  " + "-" * 74,
    ]
    totals = {"us": 0, "them": 0}
    for record in match.hands:
        if record.thrown_in or record.score is None:
            lines.append(f"  {record.index:>2}. {'passed out':<26} {'':<12} {'':<18}"
                         f" {totals['us']:>4} / {totals['them']:>4}")
            continue
        totals = apply_score(totals, record.score)
        tricks = " ".join(f"{team} {count}" for team, count in sorted(record.tricks.items()))
        lines.append(
            f"  {record.index:>2}. {record.label:<26} {tricks:<12} {record.score.note:<18}"
            f" {totals['us']:>4} / {totals['them']:>4}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fivehundred", description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="deal seed (default: random)")
    parser.add_argument("--target", type=int, default=500, help="match target (default 500)")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(1, 1_000_000)
    match = play(seed, args.target)

    print(f"500 — four players, seed {seed}, first to {args.target}\n")
    print(render(match))
    print()
    print(f"  us {match.us}   them {match.them}"
          f"   -> {match.winner} wins in {len(match.hands)} hands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
