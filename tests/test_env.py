"""The hand simulator: every card is played once, tricks add up, and the
Nullo trio really is a trio.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.agents import HeuristicAgent, LowestAgent, RandomAgent
from fivehundred.cards import deal
from fivehundred.env import PARTNER_OF, SEATS, TEAM_OF, play_hand, play_order


def agents_for(rng: random.Random) -> dict:
    return {seat: RandomAgent(random.Random(rng.random())) for seat in SEATS}


@pytest.mark.parametrize("contract", ["C", "D", "H", "S", "NT"])
def test_four_player_hand_plays_all_40_cards_and_takes_10_tricks(contract):
    rng = random.Random(3)
    hands, kitty = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    out = play_hand(dealt, contract, contractor="S", agents=agents_for(rng))
    assert sum(out.tricks.values()) == 10
    assert len(out.plays) == 40
    assert len({card.id for _, card in out.plays}) == 40
    assert len(kitty) == 5  # untouched: the kitty is not part of the play phase


def test_contractor_leads_the_first_trick():
    rng = random.Random(11)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    out = play_hand(dealt, "H", contractor="W", agents=agents_for(rng))
    assert out.plays[0][0] == "W"


def test_nullo_partner_sits_out_and_tricks_are_three_cards():
    rng = random.Random(5)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    out = play_hand(dealt, "NULLO", contractor="S", agents={s: LowestAgent() for s in SEATS})
    assert sum(out.tricks.values()) == 10
    assert len(out.plays) == 30
    assert PARTNER_OF["S"] not in {seat for seat, _ in out.plays}


def test_play_order_is_clockwise_from_the_contractor():
    assert play_order("H", "S") == ["S", "W", "N", "E"]
    assert play_order("H", "N") == ["N", "E", "S", "W"]
    assert play_order("NULLO", "S") == ["S", "W", "E"]  # north sits out


def test_teams_partition_the_table():
    assert TEAM_OF["S"] == TEAM_OF["N"] == "NS"
    assert TEAM_OF["W"] == TEAM_OF["E"] == "WE"


def test_an_illegal_play_is_refused_rather_than_silently_accepted():
    class Cheat:
        def choose(self, view):
            off_suit = [c for c in view.hand if c not in view.legal]
            return off_suit[0] if off_suit else view.legal[0]

    rng = random.Random(2)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    with pytest.raises(ValueError, match="illegal"):
        play_hand(dealt, "H", contractor="S", agents={s: Cheat() for s in SEATS})


def test_an_empty_or_partial_trick_count_is_not_an_error():
    """The first lead of a hand has no tricks yet, and `serve.py` accepts an empty list.

    Replacing the score table with the caller's dict made `tricks={}` (or only
    one team) raise KeyError on the first winner — a 500 on a legal request.
    """
    from fivehundred.agents import HeuristicAgent

    rng = random.Random(4)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}

    for tricks in ({}, {"NS": 0, "WE": 0}, {"NS": 3}):
        out = play_hand(dealt, "H", "S", {s: HeuristicAgent() for s in SEATS}, tricks=dict(tricks))
        # The hand plays ten more tricks on top of whatever was already counted.
        assert sum(out.tricks.values()) == 10 + sum(tricks.values()), f"tricks={tricks} lost accounting"
        assert set(out.tricks) == {"NS", "WE"}


def test_identical_agents_are_unbiased_in_the_paired_benchmark():
    """The harness's own unit test: A vs A must come out at zero.

    Deal luck is cancelled inside each pair, so a same-agents run is the
    cleanest check that nothing in the seating, dealing or scoring favours a
    side. If this drifts off zero, every other number in the benchmark is
    suspect.
    """
    from fivehundred.benchmark import run

    result = run("control", "heuristic", "heuristic", deals=400, seed=99)
    assert abs(result.diff) < 0.05, f"paired control drifted: {result.diff:+.3f}"
    assert result.ci95[0] <= 0.0 <= result.ci95[1]


def test_heuristic_beats_random_and_the_interval_excludes_zero():
    """A sanity check on the harness, not a headline result."""
    from fivehundred.benchmark import run

    result = run("h vs r", "heuristic", "random", deals=400, seed=99)
    assert result.diff > 1.0
    assert result.ci95[0] > 0.0

