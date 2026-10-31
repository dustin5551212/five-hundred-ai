"""Three opponents to measure against, in increasing order of competence.

`HeuristicAgent` is a port of the browser game's `choosePlay` (src/cpu.js), so
a learned agent has something real to beat: the rules-engine CPU that ships in
the playable game, not a strawman that plays uniformly at random.
"""

from __future__ import annotations

import random
from typing import Sequence

from .bid import Bid, is_nullo, legal_bids
from .cards import Card
from .env import PARTNER_OF, View
from .rules import bower_of, compare_cards, is_no_trump, is_trump, strength


class RandomAgent:
    """Uniform over legal plays. The floor."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose(self, view: View) -> Card:
        return self.rng.choice(list(view.legal))


class LowestAgent:
    """Always the weakest legal card. Strong in Nullo, poor everywhere else."""

    def choose(self, view: View) -> Card:
        trump = view.contract
        return min(view.legal, key=lambda c: strength(c, trump, view.joker_suit))


class HeuristicAgent:
    """The shipped CPU: draw trumps with a fistful, otherwise lead low from the
    longest suit; on defence win as cheaply as possible or dump the lowest."""

    def choose(self, view: View) -> Card:
        legal = list(view.legal)
        if len(legal) == 1:
            return legal[0]

        trump = view.contract
        joker_suit = view.joker_suit

        def weak(card: Card) -> int:
            return strength(card, trump, joker_suit)

        if view.contract == "NULLO":
            # Taking a trick loses the contract, so the answer is always lowest.
            return min(legal, key=weak)

        if not view.trick:
            return self._lead(legal, trump, joker_suit)

        led_suit = view.led_suit
        best_seat, best_card = view.trick[0]
        for seat, card in view.trick[1:]:
            if compare_cards(card, best_card, trump, led_suit, joker_suit) > 0:
                best_seat, best_card = seat, card

        # Partner already has it — do not waste a card.
        if best_seat == PARTNER_OF[view.seat]:
            return min(legal, key=weak)

        winners = [c for c in legal if compare_cards(c, best_card, trump, led_suit, joker_suit) > 0]
        if winners:
            return min(winners, key=weak)  # win as cheaply as possible
        return min(legal, key=weak)

    @staticmethod
    def _lead(legal: list[Card], trump: str, joker_suit: str | None) -> Card:
        trumps = [c for c in legal if is_trump(c, trump)]
        if len(trumps) >= 4:
            return max(trumps, key=lambda c: strength(c, trump, joker_suit))

        counts: dict[str, int] = {}
        for card in legal:
            counts[card.suit] = counts.get(card.suit, 0) + 1
        # Longest suit, ties resolved by first-seen order, as the JS does.
        suit = max(counts, key=lambda s: counts[s])
        in_suit = [c for c in legal if c.suit == suit]
        return min(in_suit, key=lambda c: strength(c, trump, joker_suit))


# ── the shipped CPU's bidding half, ported from cpu.js ──────────────────

#: Tricks the bidder does not have to find in their own hand: a partner's share
#: plus what the five kitty cards usually add. Bidding six is only sane because
#: of these, so they belong in the arithmetic.
PARTNER_HELP = 2.2
KITTY_HELP = 1.3

TRUMP_WEIGHT: dict[str, float] = {
    "A": 0.86, "K": 0.50, "Q": 0.28, "10": 0.17, "9": 0.13,
    "8": 0.11, "7": 0.09, "6": 0.08, "5": 0.07, "4": 0.06,
}


def _has_rank(hand: Sequence[Card], suit: str, rank: str) -> bool:
    return any(c.suit == suit and c.rank == rank for c in hand)


def estimate_tricks(hand: Sequence[Card], denom: str | None) -> float:
    """How many tricks this hand is worth in `denom`.

A weighted count, not a simulation: trump honours plus side aces and guarded
    kings. Deliberately conservative, and capped at ten.
    """
    if not denom:
        return 0.0
    if is_no_trump(denom):
        tricks = 0.0
        for card in hand:
            if card.is_joker:
                tricks += 1.1
            elif card.rank == "A":
                tricks += 0.72
            elif card.rank == "K" and _has_rank(hand, card.suit, "Q"):
                tricks += 0.3
        return min(10.0, tricks)

    tricks = 0.0
    trumps = 0
    for card in hand:
        if not is_trump(card, denom):
            continue
        trumps += 1
        bower = bower_of(card, denom)
        if bower == "joker":
            tricks += 1.15
        elif bower == "right":
            tricks += 1.05
        elif bower == "left":
            tricks += 1.0
        else:
            tricks += TRUMP_WEIGHT.get(card.rank, 0.05)

    # A long trump suit ruffs more than it wins by rank alone.
    if trumps > 5:
        tricks += (trumps - 5) * 0.24

    for card in hand:
        if is_trump(card, denom):
            continue
        if card.rank == "A":
            tricks += 0.55
        elif card.rank == "K" and _has_rank(hand, card.suit, "Q"):
            tricks += 0.22

    return min(10.0, tricks)


def choose_bid(
    hand: Sequence[Card],
    current: Bid | None = None,
    partner_help: float = PARTNER_HELP,
    kitty_help: float = KITTY_HELP,
    safety: float = 0.2,
) -> Bid | None:
    """The bid this hand settles on, or None to pass.

    The estimate is one player's own tricks; a partnership needs roughly two
    more between them, so a hand worth n tricks can contract for about n + 2.
    Pick the denomination where the hand is strongest, then take the cheapest
    legal bid in it — overcalling by the minimum, the way a table does.

    Nullo is a judgement about losing every trick, not winning them, so this
    leaves it alone, exactly as the browser game's CPU does.
    """
    options = [b for b in legal_bids(current) if not is_nullo(b.denom)]
    if not options:
        return None

    by_denom = sorted(
        ((denom, estimate_tricks(hand, denom)) for denom in dict.fromkeys(o.denom for o in options)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    for denom, estimate in by_denom:
        cheapest = next((o for o in options if o.denom == denom), None)
        if cheapest is not None and estimate + partner_help + kitty_help >= cheapest.tricks + safety:
            return cheapest
    return None


def _keep_score(card: Card, hand: Sequence[Card], denom: str) -> float:
    """Trumps and top cards are protected, junk goes first."""
    if card.is_joker:
        return 100
    if is_trump(card, denom):
        bower = bower_of(card, denom)
        if bower == "right":
            return 95
        if bower == "left":
            return 92
        return 60 + card.value
    if card.rank == "A":
        return 80
    if card.rank == "K":
        return 55 if _has_rank(hand, card.suit, "Q") else 45
    if card.rank == "Q":
        return 25
    return card.value


def choose_discards(hand: Sequence[Card], denom: str, count: int) -> list[Card]:
    """Which cards to throw away when coming back down to ten."""
    return sorted(hand, key=lambda c: _keep_score(c, hand, denom))[:count]
