"""A paired head-to-head benchmark over seeded deals.

The naive version of this — hand agent A one team, agent B the other, over N
deals — is quietly wrong in two ways, both of which showed up here:

1. **Deal luck.** One team's cards are fixed per deal, so a 2,000-deal run is
   really one sample of 2,000 card distributions. Comparing two runs that use
   different hands measures the cards as much as the players.
2. **Ties.** Ten tricks split 5–5 often (~20% of hands). A "win rate" that
   counts a tie as a loss understates both sides and does not mirror when the
   seats are swapped.

So each deal is played **twice**, once with A on the north–south seats and
once with B there, on the identical deal, contract and contractor. The pair is
the unit of analysis: the cards cancel inside it, and the reported interval is
a bootstrap over deals, not over individual plays.

    python -m fivehundred.benchmark --deals 2000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agents import HeuristicAgent, LowestAgent, RandomAgent
from .cards import deal
from .env import SEATS, TEAM_OF, Agent, play_hand
from .rules import DENOMS

# Nullo is excluded: it is a 3-card-trick game where the contractor's partner
# sits out, and mixing it into a trick-count average would compare different
# games. It gets its own benchmark (see README roadmap).
BENCH_CONTRACTS: tuple[str, ...] = ("C", "D", "H", "S", "NT")


@dataclass
class Result:
    label: str
    a: str
    b: str
    deals: int
    a_tricks: float          # mean tricks per hand, over both seatings
    b_tricks: float
    diff: float              # a_tricks - b_tricks, ~ the effect size
    ci95: tuple[float, float]
    a_win_rate: float        # share of hands won outright
    tie_rate: float
    by_contract: dict[str, float] = field(default_factory=dict)  # denom -> a - b


def _agent(kind: str, rng: random.Random) -> Agent:
    if kind == "random":
        return RandomAgent(rng)
    if kind == "lowest":
        return LowestAgent()
    if kind == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"unknown agent kind: {kind}")


def _bootstrap_ci(values: list[float], rng: random.Random, draws: int = 4000) -> tuple[float, float]:
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)
    )
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def run(label: str, a_kind: str, b_kind: str, deals: int, seed: int) -> Result:
    master = random.Random(seed)
    diffs: list[float] = []
    a_total = b_total = 0.0
    a_wins = ties = 0
    by_contract: dict[str, list[float]] = {d: [] for d in BENCH_CONTRACTS}

    for i in range(deals):
        hands, _kitty = deal(master)
        contract = BENCH_CONTRACTS[i % len(BENCH_CONTRACTS)]
        contractor = SEATS[i % 4]  # every side leads a quarter of the deals
        dealt = {seat: hands[j] for j, seat in enumerate(SEATS)}
        arm_seed = master.getrandbits(63)

        a_tricks: list[float] = []
        for flip in (0, 1):
            a_team, b_team = ("NS", "WE") if flip == 0 else ("WE", "NS")
            rng = random.Random(arm_seed + flip)  # same stochastic draws per arm
            agents = {
                seat: _agent(a_kind if TEAM_OF[seat] == a_team else b_kind, rng)
                for seat in SEATS
            }
            out = play_hand(dealt, contract, contractor, agents)
            a_tricks.append(out.tricks[a_team])
            if flip == 0:
                if out.tricks["NS"] > out.tricks["WE"]:
                    a_wins += 1 if a_team == "NS" else 0
                elif out.tricks["NS"] == out.tricks["WE"]:
                    ties += 1

        mean_a = sum(a_tricks) / 2
        diffs.append(mean_a - (10 - mean_a))
        a_total += mean_a
        b_total += 10 - mean_a
        by_contract[contract].append(mean_a - (10 - mean_a))

    lo, hi = _bootstrap_ci(diffs, master)
    return Result(
        label=label,
        a=a_kind,
        b=b_kind,
        deals=deals,
        a_tricks=a_total / deals,
        b_tricks=b_total / deals,
        diff=a_total / deals - b_total / deals,
        ci95=(lo, hi),
        a_win_rate=a_wins / deals,
        tie_rate=ties / deals,
        by_contract={d: sum(v) / len(v) for d, v in by_contract.items() if v},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    matchups = [
        ("heuristic (A) vs random (B)", "heuristic", "random"),
        ("heuristic (A) vs lowest-card (B)", "heuristic", "lowest"),
        ("heuristic (A) vs heuristic (B)", "heuristic", "heuristic"),
    ]

    print(f"{'matchup':34s} {'deals':>6s} {'A tricks':>9s} {'B tricks':>9s} {'A-B':>7s}  {'95% CI':>16s} {'A wins':>7s} {'tie':>6s}")
    results = []
    for label, a, b in matchups:
        r = run(label, a, b, args.deals, args.seed)
        results.append(asdict(r) | {"ci95": list(r.ci95)})
        print(
            f"{label:34s} {r.deals:6d} {r.a_tricks:9.2f} {r.b_tricks:9.2f} {r.diff:+7.2f}"
            f"  [{r.ci95[0]:+.2f}, {r.ci95[1]:+.2f}] {100 * r.a_win_rate:6.1f}% {100 * r.tie_rate:5.1f}%"
        )

    print("\nby contract (A tricks minus B tricks):")
    print(f"{'matchup':34s} " + " ".join(f"{d:>7s}" for d in BENCH_CONTRACTS))
    for r in results:
        print(f"{r['label']:34s} " + " ".join(f"{r['by_contract'][d]:+7.2f}" for d in BENCH_CONTRACTS))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"seed": args.seed, "results": results}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
