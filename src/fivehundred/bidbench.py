"""What a bidder is worth: paired benchmarks for the auction.

The play benchmark asks how many tricks an agent wins. This asks a different
question — **how many points does a bidding system earn** — and it needs its own
discipline, because a contract decision and the cards that justify it are not the
same measurement.

Two benchmarks, both paired by deal so card luck cancels:

**Contract value** (`contract_benchmark`). Every bidder is shown the same hand
and asked for a contract; the contract is then played out by the same fixed
players. One bidder's number is its mean score per hand, and because every arm
sees the identical deal, the difference between two arms is a paired quantity
with no deal-luck term in it. This isolates the decision.

**The auction** (`pair_benchmark`). Two bidding systems sit opposite each other
at a real auction, each deal is played twice with the systems swapping sides, and
the pair's net is what A earned over the deal. This measures the system as it is
actually used, including being outbid — at the cost of a wider interval, because
sometimes the side you are measuring never gets to hold the contract.

Both hold **play** fixed at the shipped heuristic. The question here is the
auction, so the card play is a constant, not a variable; the play benchmark is
where card play gets measured.

That constant is also the benchmark's blind spot: a defence that never tries to
break a contract flatters a bidder that overreaches. So `--sweep` runs the whole
question again with the *searcher* defending (`pimc:12`, twelve sampled worlds),
sweeping the bidder's safety margin. The contractor's side keeps the shipped
heuristic in that run too — the only thing that changes is how hard the defence
tries.

    python -m fivehundred.bidbench --deals 2000
    python -m fivehundred.bidbench --sweep --sweep-deals 250
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .agents import HeuristicAgent, choose_bid, choose_discards
from .auction import SEATS, TEAM, Auction
from .bid import Bid
from .cards import Card, deal
from .env import TEAM_OF, Agent, play_hand
from .game import LETTER, SCORING_TEAM, cpu_bidder, exchange_kitty
from .scoring import HandScore, score_contract

LETTERS: tuple[str, ...] = ("S", "W", "N", "E")


# ── the arms ─────────────────────────────────────────────────────────────────

def pass_bidder(_seat: str, _hand: Sequence[Card], _current: Bid | None) -> Bid | None:
    """Never bids: the hand is thrown in and nobody scores. The zero line."""
    return None


def fixed_bidder(tricks: int = 6, denom: str = "S") -> Callable[..., Bid | None]:
    """Bids the same contract on the opening turn, passes otherwise. No judgement.

    Six spades is the bottom of the ladder, so there is nothing it could legally
    bid over — which is exactly the point of the arm: a bidder with no opinion.
    """
    fixed = Bid(tricks, denom)

    def bid(_seat: str, _hand: Sequence[Card], current: Bid | None) -> Bid | None:
        return fixed if current is None else None

    return bid


def no_safety_bidder(seat: str, hand: Sequence[Card], current: Bid | None) -> Bid | None:
    """The shipped estimate with the safety margin removed: bids at the edge."""
    return choose_bid(hand, current, safety=0.0)


def gambler_bidder(seat: str, hand: Sequence[Card], current: Bid | None) -> Bid | None:
    """The shipped estimate, but willing to bid one trick past what it counts."""
    return choose_bid(hand, current, partner_help=2.2, kitty_help=1.3, safety=-1.0)


# ── the defence ─────────────────────────────────────────────────────────────

def make_defence(kind: str, seed: int = 0) -> Agent:
    """A card-play policy for the defending side.

    `heuristic` is the shipped CPU, which is what the rest of this file has
    always used. `pimc:N` is the searcher from `search.py`: it samples N worlds,
    plays each candidate card out in all of them, and picks the card that takes
    the most tricks for its side. That is a defence *trying* — the thing the
    first version of this benchmark was missing, and the reason the aggressive
    bidding arms looked better than they were.
    """
    if kind in ("heuristic", "none", ""):
        return HeuristicAgent()
    if kind.startswith("pimc"):
        from .search import PIMCAgent

        worlds = int(kind.split(":", 1)[1]) if ":" in kind else 4
        return PIMCAgent(worlds=worlds, seed=seed)
    raise ValueError(f"unknown defence: {kind}")


def policy_for_contract(
    seat: str,
    defence: str = "heuristic",
    seed: int = 0,
) -> dict[str, Agent]:
    """The contractor's side plays the shipped heuristic; the other side defends.

    Holding card play fixed for the side being measured is the point: the only
    thing that varies between arms of the bidding benchmark is the contract. What
    varies here is the quality of the opposition.
    """
    contracting_team = TEAM_OF[LETTER[seat]]
    return {
        letter: HeuristicAgent()
        if TEAM_OF[letter] == contracting_team
        else make_defence(defence, seed)
        for letter in LETTERS
    }


# ── one hand, evaluated ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DealOutcome:
    seat: str
    bid: Bid | None
    tricks_by_team: dict[str, int]
    score: HandScore | None

    @property
    def thrown_in(self) -> bool:
        return self.score is None

    @property
    def points_for_bidder(self) -> int:
        return 0 if self.score is None else self.score.contractor_points

    @property
    def points_against_bidder(self) -> int:
        return 0 if self.score is None else self.score.opponent_points


def evaluate_contract(
    hands: dict[str, list[Card]],
    kitty: Sequence[Card],
    seat: str,
    bid: Bid | None,
    players: dict[str, Agent] | None = None,
    discarder: Callable[[Sequence[Card], str, int], list[Card]] = choose_discards,
) -> DealOutcome:
    """Deal a hand to one bidder and score whatever it decides.

    The hands are copied, so every arm in a comparison starts from the identical
    deal. `players` is the fixed card-play policy for all four seats.
    """
    table: dict[str, list[Card]] = {s: list(cards) for s, cards in hands.items()}
    if bid is None:
        return DealOutcome(seat, None, {}, None)

    exchange_kitty(table, kitty, seat, bid, discarder)

    policy = players or policy_for_contract(seat)
    outcome = play_hand(table, bid.denom, LETTER[seat], policy)  # type: ignore[arg-type]
    tricks_by_team = {SCORING_TEAM[team]: count for team, count in outcome.tricks.items()}
    score = score_contract(bid, TEAM[seat], tricks_by_team)
    return DealOutcome(seat, bid, tricks_by_team, score)


# ── aggregates ──────────────────────────────────────────────────────────────

def _bootstrap_ci(values: list[float], rng: random.Random, draws: int = 4000) -> tuple[float, float]:
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


@dataclass
class BidderResult:
    label: str
    deals: int
    points: float            # mean points the bidding side earned per hand
    ci95: tuple[float, float]
    conceded: float          # mean points the defenders earned per hand
    made_rate: float         # share of contracts made
    pass_rate: float         # share of hands thrown in
    mean_bid_tricks: float   # average level of the contracts bid
    mean_tricks: float       # what the contractor actually took
    diff: float = 0.0        # mean paired difference against the reference arm
    diff_ci95: tuple[float, float] = (0.0, 0.0)
    points_per_deal: list[float] = field(default_factory=list, repr=False)

    @property
    def overreach(self) -> float:
        """Tricks bid minus tricks taken: positive is a bidder that asks too much."""
        return self.mean_bid_tricks - self.mean_tricks


def contract_benchmark(
    arms: dict[str, Callable[[str, Sequence[Card], Bid | None], Bid | None]],
    deals: int,
    seed: int,
    reference: str | None = None,
) -> dict[str, BidderResult]:
    """Score each bidder on the identical deals, with the seat rotating.

    Rotating the bidding seat matters: a bidder that only ever sees south's hands
    is being measured partly on south's luck. Every arm sees the same seat on the
    same deal, so the comparison stays paired.
    """
    master = random.Random(seed)
    results: dict[str, BidderResult] = {}
    accumulated: dict[str, dict[str, list[float]]] = {
        label: {"points": [], "conceded": [], "made": [], "pass": [], "bid": [], "tricks": []}
        for label in arms
    }

    for i in range(deals):
        hands, kitty = deal(master)
        dealt = {LETTER[seat]: hands[j] for j, seat in enumerate(SEATS)}
        seat = SEATS[i % len(SEATS)]
        hand = dealt[LETTER[seat]]
        policy: dict[str, Agent] = {LETTER[s]: HeuristicAgent() for s in SEATS}

        for label, bidder in arms.items():
            bid = bidder(seat, hand, None)
            outcome = evaluate_contract(dealt, kitty, seat, bid, policy)
            bucket = accumulated[label]
            bucket["points"].append(outcome.points_for_bidder)
            bucket["conceded"].append(outcome.points_against_bidder)
            if outcome.thrown_in:
                bucket["pass"].append(1.0)
            else:
                bucket["pass"].append(0.0)
                bucket["made"].append(1.0 if outcome.score and outcome.score.made else 0.0)
                bucket["bid"].append(outcome.bid.tricks if outcome.bid else 0.0)
                bucket["tricks"].append(float(outcome.score.contractor_tricks if outcome.score else 0))

    for label, bucket in accumulated.items():
        points = bucket["points"]
        made = bucket["made"] or [0.0]
        bids = bucket["bid"] or [0.0]
        tricks = bucket["tricks"] or [0.0]
        results[label] = BidderResult(
            label=label,
            deals=deals,
            points=sum(points) / deals,
            ci95=_bootstrap_ci(points, master),
            conceded=sum(bucket["conceded"]) / deals,
            made_rate=sum(made) / len(made) if bucket["made"] else 0.0,
            pass_rate=sum(bucket["pass"]) / deals,
            mean_bid_tricks=sum(bids) / len(bids) if bucket["bid"] else 0.0,
            mean_tricks=sum(tricks) / len(tricks) if bucket["tricks"] else 0.0,
            points_per_deal=points,
        )

    if reference is not None:
        base = results[reference].points_per_deal
        for result in results.values():
            if result.label == reference:
                continue
            diffs = [a - b for a, b in zip(result.points_per_deal, base)]
            result.diff = sum(diffs) / deals
            result.diff_ci95 = _bootstrap_ci(diffs, master)
    return results


# ── the sweep: how much safety margin, against a defence that tries ─────────

@dataclass
class SweepRow:
    """One safety margin, measured against each defence on the same deals."""

    safety: float
    deals: int
    points: dict[str, float]                  # defence -> mean points per hand
    made: dict[str, float]                    # defence -> share of contracts made
    passed: float
    cost: float                               # strong defence minus weak, paired
    cost_ci95: tuple[float, float]
    #: defence -> (paired difference against the shipped arm, lo, hi)
    vs_shipped: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "shipped (0.2)" if self.safety == 0.2 else f"safety {self.safety:+.1f}"


def safety_sweep(
    safeties: Sequence[float],
    deals: int,
    seed: int,
    weak: str = "heuristic",
    strong: str = "pimc:4",
) -> list[SweepRow]:
    """Sweep the bidder's safety margin under a weak and a strong defence.

    Every arm sees the identical deals under both defences, so `cost` — what the
    strong defence takes away — is itself a paired quantity.
    """
    master = random.Random(seed)
    rows: list[SweepRow] = []
    per_arm: list[tuple[float, dict[str, list[float]], dict[str, list[float]], int]] = []

    # The deals are drawn once and reused for every arm and every defence, or the
    # comparison between arms would be measuring the cards again.
    tables: list[tuple[dict[str, list[Card]], list[Card], str, int]] = []
    for i in range(deals):
        hands, kitty = deal(master)
        dealt = {LETTER[seat]: hands[j] for j, seat in enumerate(SEATS)}
        # The search seed belongs to the deal, not to the arm: otherwise two arms
        # would be compared on different random defences as well as different bids.
        tables.append((dealt, kitty, SEATS[i % len(SEATS)], master.getrandbits(31)))

    for safety in safeties:
        arm = {
            "safety": safety,
        }
        bidder = (lambda safety: (lambda seat, hand, current: choose_bid(hand, current, safety=safety)))(safety)
        totals: dict[str, list[float]] = {weak: [], strong: []}
        made: dict[str, list[float]] = {weak: [], strong: []}
        passes = 0

        for dealt, kitty, seat, defence_seed in tables:
            bid = bidder(seat, dealt[LETTER[seat]], None)
            if bid is None:
                passes += 1
                for kind in (weak, strong):
                    totals[kind].append(0.0)
                continue
            for kind in (weak, strong):
                policy = policy_for_contract(seat, kind, seed=defence_seed)
                outcome = evaluate_contract(dealt, kitty, seat, bid, policy)
                totals[kind].append(float(outcome.points_for_bidder))
                if outcome.score is not None:
                    made[kind].append(1.0 if outcome.score.made else 0.0)

        diffs = [s - w for s, w in zip(totals[strong], totals[weak])]
        per_arm.append((safety, totals, made, passes))
        rows.append(
            SweepRow(
                safety=safety,
                deals=deals,
                points={k: sum(v) / deals for k, v in totals.items()},
                made={k: (sum(v) / len(v) if v else 0.0) for k, v in made.items()},
                passed=passes / deals,
                cost=sum(diffs) / deals,
                cost_ci95=_bootstrap_ci(diffs, master),
            )
        )

    # Now the number a reader wants: each arm against the shipped one, under each
    # defence. Every arm saw the same deals and the same defence randomness, so
    # these differences are paired too.
    reference = next((r for r in rows if abs(r.safety - 0.2) < 1e-9), rows[0])
    base_totals = next(t for s, t, _m, _p in per_arm if s == reference.safety)
    for row in rows:
        totals = next(t for s, t, _m, _p in per_arm if s == row.safety)
        for kind in (weak, strong):
            series = [a - b for a, b in zip(totals[kind], base_totals[kind])]
            lo, hi = _bootstrap_ci(series, master)
            row.vs_shipped[kind] = (sum(series) / deals, lo, hi)
    return rows


# ── the auction, with two systems at the table ───────────────────────────────

@dataclass
class PairResult:
    label: str
    deals: int
    a_points: float          # mean points per hand for side A across both seatings
    b_points: float
    diff: float
    ci95: tuple[float, float]
    a_contracts: float       # share of hands where A's side held the contract
    thrown_in: float         # share of hands nobody bid on
    by_deal: list[float] = field(default_factory=list, repr=False)


def pair_benchmark(
    label: str,
    a: Callable[[str, Sequence[Card], Bid | None], Bid | None],
    b: Callable[[str, Sequence[Card], Bid | None], Bid | None],
    deals: int,
    seed: int,
    defence: str = "heuristic",
) -> PairResult:
    """Two bidding systems at one table, each deal played twice with sides swapped.

    Play is the shipped heuristic for both sides in both runs, so what differs
    between the two runs of a deal is only which system sits where.
    """
    master = random.Random(seed)
    per_deal: list[float] = []
    a_total = b_total = 0.0
    a_held = thrown = 0

    for i in range(deals):
        hands, kitty = deal(master)
        dealt = {LETTER[seat]: hands[j] for j, seat in enumerate(SEATS)}
        dealer = SEATS[i % len(SEATS)]
        defence_seed = master.getrandbits(31)

        a_net = b_net = 0.0
        for flip in (0, 1):
            a_team = "us" if flip == 0 else "them"
            bidders = {seat: (a if TEAM[seat] == a_team else b) for seat in SEATS}
            auction = Auction(first=SEATS[(SEATS.index(dealer) + 1) % len(SEATS)])
            while not auction.is_over:
                seat = auction.turn
                assert seat is not None
                bid = bidders[seat](seat, dealt[LETTER[seat]], auction.high_bid)
                auction.pass_turn(seat) if bid is None else auction.place(seat, bid)

            result = auction.result
            assert result is not None
            if result.type == "passedOut" or result.contract is None or result.seat is None:
                if flip == 0:
                    thrown += 1
                continue

            table = {s: list(cards) for s, cards in dealt.items()}
            exchange_kitty(table, kitty, result.seat, result.contract)
            # Whoever is not holding the contract defends — and in this benchmark the
            # defence is allowed to be the searcher, symmetrically for both arms.
            policy = policy_for_contract(result.seat, defence, seed=defence_seed)
            outcome = play_hand(table, result.contract.denom, LETTER[result.seat], policy)
            tricks_by_team = {
                SCORING_TEAM[team]: count for team, count in outcome.tricks.items()
            }
            score = score_contract(result.contract, TEAM[result.seat], tricks_by_team)
            # A hand pays both sides — the contractors their contract, the
            # defenders ten a trick — so a run is not zero-sum and both are summed.
            if score.contractor_team == a_team:
                a_net += score.contractor_points
                b_net += score.opponent_points
                if flip == 0:
                    a_held += 1
            else:
                b_net += score.contractor_points
                a_net += score.opponent_points

        per_deal.append(a_net - b_net)
        a_total += a_net
        b_total += b_net

    lo, hi = _bootstrap_ci(per_deal, master)
    return PairResult(
        label=label,
        deals=deals,
        a_points=a_total / deals,
        b_points=b_total / deals,
        diff=a_total / deals - b_total / deals,
        ci95=(lo, hi),
        a_contracts=a_held / deals,
        thrown_in=thrown / deals,
        by_deal=per_deal,
    )


# ── cli ─────────────────────────────────────────────────────────────────────

def run_sweep(args) -> int:
    """How much safety margin, and how much of it survives a defence that tries."""
    safeties = [0.6, 0.2, 0.0, -0.5, -1.0, -1.5]
    print(f"retuning the safety margin — {args.sweep_deals} deals, seed {args.seed}")
    print(f"weak defence: shipped heuristic   |   strong defence: {args.strong}\n")

    rows = safety_sweep(safeties, args.sweep_deals, args.seed, strong=args.strong)
    print(f"{'arm':16s} {'pts (weak)':>10s} {'pts (strong)':>12s} "
          f"{'vs shipped (strong)':>20s} {'95% CI':>17s} {'made w/s':>10s} {'passed':>7s}")
    for row in rows:
        weak = row.points["heuristic"]
        strong = row.points[args.strong]
        diff, lo, hi = row.vs_shipped[args.strong]
        marker = "     —      " if row.safety == 0.2 else f"{diff:+9.2f}"
        ci = "        —       " if row.safety == 0.2 else f"[{lo:+7.2f}, {hi:+7.2f}]"
        print(f"{row.label:16s} {weak:+10.2f} {strong:+12.2f} {marker:>20s} {ci:>17s} "
              f"{100 * row.made['heuristic']:4.1f}/{100 * row.made[args.strong]:4.1f}% "
              f"{100 * row.passed:6.1f}%")

    best_weak = max(rows, key=lambda r: r.points["heuristic"])
    best_strong = max(rows, key=lambda r: r.points[args.strong])
    print(f"\n  best against the weak defence  : {best_weak.label:16s} "
          f"{best_weak.points['heuristic']:+.2f} points/hand")
    print(f"  best against the strong defence: {best_strong.label:16s} "
          f"{best_strong.points[args.strong]:+.2f} points/hand")

    print(f"\nthe auction, both systems at a table, sides swapped, defence = {args.strong}\n")
    print(f"{'matchup':34s} {'A pts':>7s} {'B pts':>7s} {'A-B':>7s} {'95% CI':>17s}")
    auction_rows = []
    for label, a, b in [
        ("shipped vs shipped (control)", cpu_bidder, cpu_bidder),
        ("no safety margin vs shipped", no_safety_bidder, cpu_bidder),
        ("one trick higher vs shipped", gambler_bidder, cpu_bidder),
    ]:
        r = pair_benchmark(label, a, b, args.sweep_deals, args.seed, defence=args.strong)
        auction_rows.append(asdict(r) | {"ci95": list(r.ci95)})
        print(f"{label:34s} {r.a_points:+7.2f} {r.b_points:+7.2f} {r.diff:+7.2f} "
              f"[{r.ci95[0]:+7.2f}, {r.ci95[1]:+7.2f}]")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "seed": args.seed,
            "deals": args.sweep_deals,
            "strong_defence": args.strong,
            "sweep": [
                {**asdict(row), "cost_ci95": list(row.cost_ci95), "label": row.label}
                for row in rows
            ],
            "auction": auction_rows,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fivehundred.bidbench", description=__doc__)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="retune the safety margin under a weak and a strong defence")
    parser.add_argument("--sweep-deals", type=int, default=250)
    parser.add_argument("--strong", default="pimc:12", help="the defence that tries (default pimc:12)")
    args = parser.parse_args(argv)

    if args.sweep:
        return run_sweep(args)

    arms = {
        "pass (never bids)": pass_bidder,
        "always 6 Spades": fixed_bidder(6, "S"),
        "shipped CPU": cpu_bidder,
        "shipped, no safety margin": no_safety_bidder,
        "shipped, 1 trick higher": gambler_bidder,
    }

    print(f"contract value — {args.deals} deals, seed {args.seed}, play fixed at the shipped heuristic\n")
    results = contract_benchmark(arms, args.deals, args.seed, reference="shipped CPU")
    print(f"{'bidder':26s} {'points':>8s} {'95% CI':>17s} {'vs shipped':>11s} {'95% CI':>17s} "
          f"{'conceded':>9s} {'made':>6s} {'passed':>7s} {'bid':>5s} {'took':>5s}")
    for label, r in results.items():
        diff = "     —    " if label == "shipped CPU" else f"{r.diff:+11.2f}"
        diff_ci = "        —       " if label == "shipped CPU" else f"[{r.diff_ci95[0]:+7.2f}, {r.diff_ci95[1]:+7.2f}]"
        print(f"{label:26s} {r.points:+8.2f} [{r.ci95[0]:+7.2f}, {r.ci95[1]:+7.2f}] {diff} {diff_ci} "
              f"{r.conceded:9.2f} {100 * r.made_rate:5.1f}% {100 * r.pass_rate:6.1f}% "
              f"{r.mean_bid_tricks:5.2f} {r.mean_tricks:5.2f}")
    print("\n  points = what the bidding side scored per hand; conceded = what it handed the")
    print("  defenders per hand. A bidder that passes every hand scores 0 and concedes 0.")

    pairs = [
        ("shipped vs pass", cpu_bidder, pass_bidder),
        ("shipped vs always 6 Spades", cpu_bidder, fixed_bidder(6, "S")),
        ("shipped vs shipped (control)", cpu_bidder, cpu_bidder),
        ("no safety margin vs shipped", no_safety_bidder, cpu_bidder),
        ("1 trick higher vs shipped", gambler_bidder, cpu_bidder),
    ]
    print(f"\nthe auction — two systems at one table, sides swapped each deal\n")
    print(f"{'matchup':32s} {'A pts':>7s} {'B pts':>7s} {'A-B':>7s} {'95% CI':>17s} "
          f"{'A held':>7s} {'passed':>7s}")
    pair_rows = []
    for label, a, b in pairs:
        r = pair_benchmark(label, a, b, args.deals, args.seed)
        pair_rows.append(asdict(r) | {"ci95": list(r.ci95)})
        print(f"{label:32s} {r.a_points:+7.2f} {r.b_points:+7.2f} {r.diff:+7.2f} "
              f"[{r.ci95[0]:+7.2f}, {r.ci95[1]:+7.2f}] {100 * r.a_contracts:6.1f}% "
              f"{100 * r.thrown_in:6.1f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "seed": args.seed,
            "deals": args.deals,
            "contract_value": {
                label: {k: v for k, v in asdict(r).items() if k != "points_per_deal"}
                | {"ci95": list(r.ci95)}
                for label, r in results.items()
            },
            "auction": pair_rows,
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
