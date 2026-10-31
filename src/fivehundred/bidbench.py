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

    python -m fivehundred.bidbench --deals 2000
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
from .env import Agent, play_hand
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

    policy = players or {letter: HeuristicAgent() for letter in LETTERS}
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
            policy: dict[str, Agent] = {LETTER[s]: HeuristicAgent() for s in SEATS}
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

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fivehundred.bidbench", description=__doc__)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

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
