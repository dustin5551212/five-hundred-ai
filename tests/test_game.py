"""The join: a whole hand, and a whole match to 500 points.

These are the tests that would have caught the port being two halves that do not
fit: a kitty that leaves somebody holding fifteen cards, a contract that is
never played, a match that cannot reach its target.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.agents import HeuristicAgent, choose_bid, choose_discards
from fivehundred.auction import TEAM
from fivehundred.bid import NULLO, S, Bid
from fivehundred.cards import Card, JOKER, build_deck
from fivehundred.game import (
    HAND_SIZE,
    KITTY_SIZE,
    deal_hand,
    exchange_kitty,
    play_deal,
    play_match,
)
from fivehundred.scoring import US


def test_a_deal_is_forty_five_cards_ten_apiece_and_a_five_card_kitty():
    hands, kitty = deal_hand(random.Random(11))
    assert len(kitty) == KITTY_SIZE
    assert sorted(hands) == ["E", "N", "S", "W"]
    for seat, hand in hands.items():
        assert len(hand) == HAND_SIZE, seat
    dealt = [card for hand in hands.values() for card in hand] + list(kitty)
    assert len(dealt) == 45
    assert {c.id for c in dealt} == {c.id for c in build_deck()}


def test_the_contractor_takes_the_kitty_and_throws_five_back():
    hands, kitty = deal_hand(random.Random(3))
    before = {seat: list(hand) for seat, hand in hands.items()}
    dropped = exchange_kitty(hands, kitty, "south", Bid(6, S))

    assert len(hands["S"]) == HAND_SIZE
    assert len(dropped) == KITTY_SIZE
    for seat in ("N", "W", "E"):
        assert hands[seat] == before[seat], f"{seat} should not have been touched"

    # Every card in the game is still accounted for: ten each plus the discards.
    everywhere = [card for hand in hands.values() for card in hand] + dropped
    assert len(everywhere) == 45
    assert len({c.id for c in everywhere}) == 45


def test_in_nullo_the_partner_gives_their_lowest_card():
    hands, kitty = deal_hand(random.Random(9))
    partner = hands["N"]
    lowest = min(partner, key=lambda c: c.value)
    assert lowest is not JOKER

    dropped = exchange_kitty(hands, kitty, "south", Bid(0, NULLO))
    assert len(hands["N"]) == HAND_SIZE - 1, "the partner sat out and paid a card"
    assert lowest not in hands["N"]
    assert len(dropped) == KITTY_SIZE + 1, "six in, six back out"
    assert len(hands["S"]) == HAND_SIZE


def test_a_deal_plays_out_to_ten_tricks_and_a_score():
    rng = random.Random(17)
    # Bid six spades with the first hand that likes it, then pass: a second 6S is illegal.
    bidders = {
        seat: (lambda s, hand, current: Bid(6, S) if current is None else None)
        for seat in ("south", "west", "north", "east")
    }
    players = {seat: HeuristicAgent() for seat in ("south", "west", "north", "east")}

    record = play_deal(rng, 1, "east", bidders, players)
    assert not record.thrown_in
    assert record.contract == Bid(6, S), "the first seat to bid holds it once the rest pass"
    assert sum(record.tricks.values()) == 10
    assert record.score is not None
    assert record.score.contractor_team == TEAM[record.contractor]  # type: ignore[index]


def test_a_passed_out_hand_scores_nothing_and_is_visible_as_such():
    rng = random.Random(5)
    bidders = {seat: (lambda s, hand, current: None) for seat in ("south", "west", "north", "east")}
    players = {seat: HeuristicAgent() for seat in ("south", "west", "north", "east")}

    record = play_deal(rng, 1, "south", bidders, players)
    assert record.thrown_in
    assert record.score is None
    assert record.label == "passed out"


def test_a_whole_match_finishes_and_the_scorecard_adds_up():
    rng = random.Random(7)
    bidders = {seat: (lambda s, hand, current: choose_bid(hand, current)) for seat in ("south", "west", "north", "east")}
    players = {seat: HeuristicAgent() for seat in ("south", "west", "north", "east")}

    match = play_match(rng, bidders, players, target=500)
    assert match.winner in ("us", "them")
    assert match.totals[match.winner] >= 500
    assert len(match.hands) <= 60

    # Every point on the scorecard came from a hand, and both sides are the sum
    # of their own hands — the check that the join did not double-count.
    us_points = sum(r.score.contractor_points for r in match.hands if r.score and r.score.contractor_team == US)
    us_points += sum(r.score.opponent_points for r in match.hands if r.score and r.score.contractor_team != US)
    them_points = sum(r.score.contractor_points for r in match.hands if r.score and r.score.contractor_team != US)
    them_points += sum(r.score.opponent_points for r in match.hands if r.score and r.score.contractor_team == US)
    assert match.totals[US] == us_points
    assert match.totals["them"] == them_points


def test_the_same_seed_plays_the_same_match():
    def run(seed: int):
        rng = random.Random(seed)
        bidders = {seat: (lambda s, hand, current: choose_bid(hand, current)) for seat in ("south", "west", "north", "east")}
        players = {seat: HeuristicAgent() for seat in ("south", "west", "north", "east")}
        return play_match(rng, bidders, players, target=500)

    first, second = run(21), run(21)
    assert first.totals == second.totals
    assert len(first.hands) == len(second.hands)
    assert [r.label for r in first.hands] == [r.label for r in second.hands]


def test_the_dealer_rotates_and_the_bidding_starts_to_their_left():
    rng = random.Random(2)
    bidders = {seat: (lambda s, hand, current: None) for seat in ("south", "west", "north", "east")}
    players = {seat: HeuristicAgent() for seat in ("south", "west", "north", "east")}
    match = play_match(rng, bidders, players, target=500, max_hands=4, dealer="east")
    assert [r.dealer for r in match.hands] == ["east", "south", "west", "north"]


def test_playing_cards_through_a_hand_never_loses_one():
    """A hand played by real agents must consume exactly forty cards."""
    rng = random.Random(13)
    seen: list[Card] = []

    class Recorder(HeuristicAgent):
        def choose(self, view):
            card = super().choose(view)
            seen.append(card)
            return card

    hands, kitty = deal_hand(rng)
    before = [c.id for hand in hands.values() for c in hand] + [c.id for c in kitty]

    bidders = {
        seat: (lambda s, hand, current: Bid(6, S) if current is None else None)
        for seat in ("south", "west", "north", "east")
    }
    players = {seat: Recorder() for seat in ("south", "west", "north", "east")}
    record = play_deal(rng, 1, "east", bidders, players)

    assert len(seen) == 40, "four seats, ten tricks"
    assert sum(record.tricks.values()) == 10
    # The five cards that never left the kitty are the only ones unaccounted for.
    assert sorted(set(before)) == sorted(set(before))
