"""The bidding benchmark has to be trustworthy before its numbers mean anything.

What is checked here is the harness, not the bidders: that every arm sees the
same cards, that the control comes out at exactly zero, that a passing table
scores nothing, that the seat rotates, and that a nullo contract is scored
through the path where the partner sits out.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.bid import NULLO, Bid
from fivehundred.bidbench import (
    contract_benchmark,
    evaluate_contract,
    fixed_bidder,
    pair_benchmark,
    pass_bidder,
)
from fivehundred.cards import Card, deal
from fivehundred.game import LETTER, cpu_bidder, deal_hand
from fivehundred.auction import SEATS


def test_the_control_arm_differs_from_itself_by_exactly_nothing():
    arms = {"cpu": cpu_bidder, "cpu again": cpu_bidder}
    results = contract_benchmark(arms, deals=40, seed=3, reference="cpu")
    control = results["cpu again"]
    assert control.diff == 0.0
    assert control.diff_ci95 == (0.0, 0.0)
    assert control.points == results["cpu"].points


def test_a_table_that_never_bids_scores_nothing_at_all():
    results = contract_benchmark({"pass": pass_bidder}, deals=30, seed=5)
    pass_arm = results["pass"]
    assert pass_arm.points == 0.0
    assert pass_arm.conceded == 0.0
    assert pass_arm.pass_rate == 1.0
    assert pass_arm.made_rate == 0.0


def test_every_arm_is_measured_on_the_same_deals():
    """A spy bidder records the hands it is shown; two arms must see one list."""
    seen: dict[str, list[str]] = {"a": [], "b": []}

    def spy(key: str):
        def bidder(_seat: str, hand, _current) -> Bid | None:
            seen[key].append(",".join(c.id for c in hand))
            return None

        return bidder

    contract_benchmark({"a": spy("a"), "b": spy("b")}, deals=25, seed=9)
    assert seen["a"] == seen["b"]
    assert len(seen["a"]) == 25


def test_the_bidding_seat_rotates_through_the_table():
    seats: list[str] = []

    def record(_seat: str, hand, _current) -> Bid | None:
        seats.append(_seat)
        return None

    contract_benchmark({"record": record}, deals=8, seed=1)
    assert set(seats) == set(SEATS)
    assert seats[0] == "south" and seats[4] == "south", "four deals per full turn"


def test_the_same_seed_gives_the_same_numbers():
    first = contract_benchmark({"cpu": cpu_bidder}, deals=20, seed=11)
    second = contract_benchmark({"cpu": cpu_bidder}, deals=20, seed=11)
    assert first["cpu"].points == second["cpu"].points
    assert first["cpu"].ci95 == second["cpu"].ci95


def test_a_nullo_contract_is_scored_with_the_partner_sitting_out():
    rng = random.Random(4)
    hands, kitty = deal_hand(rng)
    outcome = evaluate_contract(hands, kitty, "south", Bid(0, NULLO))
    assert not outcome.thrown_in
    assert outcome.bid == Bid(0, NULLO)
    assert sum(outcome.tricks_by_team.values()) == 10, "three players still play ten tricks"
    assert outcome.score is not None
    assert outcome.score.note in ("Nullo made", "Nullo defeated")


def test_passing_a_hand_outscores_nobody_but_is_never_negative():
    results = contract_benchmark({"pass": pass_bidder, "cpu": cpu_bidder}, deals=60, seed=13)
    pass_arm = results["pass"]
    assert pass_arm.points >= 0.0
    assert pass_arm.points_per_deal and all(p == 0.0 for p in pass_arm.points_per_deal)


def test_the_auction_control_is_exactly_zero_in_both_seatings():
    control = pair_benchmark("control", cpu_bidder, cpu_bidder, deals=40, seed=17)
    assert control.diff == 0.0
    assert control.ci95 == (0.0, 0.0)
    assert control.a_points == control.b_points


def test_beating_a_side_that_never_bids_is_easy_and_large():
    result = pair_benchmark("cpu vs pass", cpu_bidder, pass_bidder, deals=120, seed=19)
    assert result.diff > 0
    assert result.ci95[0] > 0, "the interval should clear zero on this much signal"
    assert result.thrown_in < 0.6, "the side that bids takes most of the contracts"


def test_a_fixed_bid_is_worse_than_the_shipped_bidder():
    result = pair_benchmark("cpu vs always 6S", cpu_bidder, fixed_bidder(6, "S"), deals=120, seed=23)
    assert result.diff > 0
    assert result.ci95[0] > 0


def test_the_pairs_are_counted_as_deals_not_hands():
    result = pair_benchmark("cpu vs pass", cpu_bidder, pass_bidder, deals=30, seed=29)
    assert result.deals == 30
    assert len(result.by_deal) == 30
