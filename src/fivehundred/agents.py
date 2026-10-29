"""Three opponents to measure against, in increasing order of competence.

`HeuristicAgent` is a port of the browser game's `choosePlay` (src/cpu.js), so
a learned agent has something real to beat: the rules-engine CPU that ships in
the playable game, not a strawman that plays uniformly at random.
"""

from __future__ import annotations

import random

from .cards import Card
from .env import PARTNER_OF, View
from .rules import compare_cards, is_trump, strength


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
