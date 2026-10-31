"""The retune has to be sound before its answer is worth having.

The point of these tests is the machinery: that the searching defence is applied
to the defending side and only the defending side, that it really is a searcher,
that the arm comparison is paired on identical deals *and* identical defence
randomness, and that the control still comes out at exactly zero.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.agents import HeuristicAgent, choose_bid
from fivehundred.auction import SEATS
from fivehundred.bid import Bid
from fivehundred.bidbench import (
    contract_benchmark,
    evaluate_contract,
    make_defence,
    pair_benchmark,
    policy_for_contract,
    safety_sweep,
    gambler_bidder,
)
from fivehundred.cards import deal
from fivehundred.game import LETTER, cpu_bidder
from fivehundred.search import PIMCAgent


def test_a_heuristic_defence_is_the_shipped_cpu():
    assert isinstance(make_defence("heuristic"), HeuristicAgent)
    assert isinstance(make_defence("pimc:4"), PIMCAgent)


def test_an_unknown_defence_is_refused_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown defence"):
        make_defence("stockfish")


def test_only_the_defending_side_searches():
    policy = policy_for_contract("south", "pimc:2", seed=1)
    # South and North hold the contract; West and East defend.
    assert isinstance(policy["S"], HeuristicAgent)
    assert isinstance(policy["N"], HeuristicAgent)
    assert isinstance(policy["W"], PIMCAgent)
    assert isinstance(policy["E"], PIMCAgent)

    # And it flips with the contracting seat, not with the bidding side's label.
    other = policy_for_contract("west", "pimc:2", seed=1)
    assert isinstance(other["W"], HeuristicAgent)
    assert isinstance(other["E"], HeuristicAgent)
    assert isinstance(other["S"], PIMCAgent)
    assert isinstance(other["N"], PIMCAgent)


def test_the_search_defence_actually_changes_the_play():
    """A defence that never differs from the heuristic would make this benchmark a lie."""
    rng = random.Random(31)
    hands, kitty = deal(rng)
    dealt = {LETTER[seat]: hands[j] for j, seat in enumerate(SEATS)}

    same = 0
    trials = 6
    for i in range(trials):
        seat = SEATS[i % 4]
        bid = Bid(6, "S")
        weak = evaluate_contract(dealt, kitty, seat, bid, policy_for_contract(seat, "heuristic", 5))
        strong = evaluate_contract(dealt, kitty, seat, bid, policy_for_contract(seat, "pimc:6", 5))
        same += weak.tricks_by_team == strong.tricks_by_team
    assert same < trials, "the searcher played exactly like the heuristic in every hand"


def test_arms_are_paired_on_deals_and_on_defence_randomness():
    rows = safety_sweep([0.2, -1.0], deals=6, seed=41, strong="pimc:2")
    assert len(rows) == 2
    for row in rows:
        assert row.deals == 6
        assert set(row.points) == {"heuristic", "pimc:2"}


def test_the_sweep_is_reproducible():
    first = safety_sweep([0.2], deals=4, seed=43, strong="pimc:2")
    second = safety_sweep([0.2], deals=4, seed=43, strong="pimc:2")
    assert first[0].points == second[0].points
    assert first[0].cost == second[0].cost


def test_the_control_is_still_exactly_zero_with_a_searching_defence():
    """Same bidder both sides, sides swapped: the pair has to cancel exactly."""
    control = pair_benchmark("control", cpu_bidder, cpu_bidder, deals=8, seed=47, defence="pimc:2")
    assert control.diff == 0.0
    assert control.ci95 == (0.0, 0.0)


def test_a_swept_arm_still_reports_a_made_rate_and_a_pass_rate():
    row = safety_sweep([0.2], deals=6, seed=53, strong="pimc:2")[0]
    assert 0.0 <= row.made["heuristic"] <= 1.0
    assert 0.0 <= row.passed <= 1.0
    assert row.label.startswith("safety") or row.label.startswith("shipped")
