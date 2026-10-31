"""Search: a hand resumed mid-trick must finish the same way it would have.

`play_hand` can start from a position that already has cards on the table,
which is how PIMC asks "what is this hand worth if I play this card?". If
resuming were subtly wrong — a skipped seat, a lost trick, a mis-read led
suit — every search result would be quietly meaningless. So: play a hand
straight through with a deterministic policy, then replay it from a cut point
and require the identical outcome.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.agents import HeuristicAgent
from fivehundred.cards import Card, deal
from fivehundred.env import SEATS, TEAM_OF, play_hand
from fivehundred.search import PIMCAgent
from fivehundred.env import View


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 9, 13])
def test_resuming_mid_hand_reproduces_the_original_outcome(cut):
    rng = random.Random(17)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    agents = {seat: HeuristicAgent() for seat in SEATS}

    full = play_hand(dealt, "H", "S", agents)
    log = list(full.plays)

    played = {card.id for _seat, card in log[:cut]}
    remaining = {
        seat: [c for c in dealt[seat] if c.id not in played] for seat in SEATS
    }
    # Rebuild the position: which trick, what is on the table, who has won what.
    in_trick = cut % 4
    trick = log[cut - in_trick:cut] if in_trick else []
    completed = (cut - in_trick) // 4
    winners = 0  # recompute from the log rather than trusting arithmetic
    from fivehundred.rules import effective_suit, trick_winner

    totals = {"NS": 0, "WE": 0}
    for i in range(0, cut - in_trick, 4):
        chunk = log[i:i + 4]
        led = effective_suit(chunk[0][1], "H")
        w = trick_winner(chunk, "H", led)
        totals[TEAM_OF[chunk[w][0]]] += 1

    resumed = play_hand(
        remaining, "H", "S", {seat: HeuristicAgent() for seat in SEATS},
        trick=trick, tricks=totals, log=log[:cut],
    )
    assert resumed.tricks == full.tricks
    assert resumed.plays[cut:] == tuple(log[cut:])


def test_pimc_returns_a_legal_card_and_is_deterministic_per_seed():
    rng = random.Random(4)
    hands, _ = deal(rng)
    dealt = {seat: hands[i] for i, seat in enumerate(SEATS)}
    captured: list[View] = []

    class Capture(HeuristicAgent):
        def choose(self, view):
            if not captured:
                captured.append(view)
            return super().choose(view)

    play_hand(dealt, "S", "S", {seat: Capture() for seat in SEATS})
    view = captured[0]
    assert len(view.legal) > 1

    a = PIMCAgent(worlds=4, seed=3).choose(view)
    b = PIMCAgent(worlds=4, seed=3).choose(view)
    assert a.id == b.id
    assert a in view.legal
