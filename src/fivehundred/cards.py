"""The 45-card pack and a deal that is reproducible from a seed.

The deck contract is deliberately the same one the browser game enforces
(`src/deck.js` in professional-500), because the point of this package is to
be the same game, not an approximate one: 54 cards minus all 2s and 3s minus
one joker is 45, dealt ten apiece with a five-card kitty left over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

SUITS: tuple[str, ...] = ("C", "D", "H", "S")
RANKS: tuple[str, ...] = ("4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
RANK_VALUE: dict[str, int] = {rank: i + 4 for i, rank in enumerate(RANKS)}

JOKER_SUIT = "JOKER"
JOKER_RANK = "JOKER"
JOKER_VALUE = 15


@dataclass(frozen=True, slots=True)
class Card:
    """A card is its suit and rank; its value is natural rank only.

    Trump order is *not* `value` — a left bower is worth 13 but beats an ace
    of trump. Anything that decides a trick goes through `rules.compare_cards`.
    """

    suit: str
    rank: str

    @property
    def value(self) -> int:
        return JOKER_VALUE if self.suit == JOKER_SUIT else RANK_VALUE[self.rank]

    @property
    def is_joker(self) -> bool:
        return self.suit == JOKER_SUIT

    @property
    def id(self) -> str:
        return JOKER_SUIT if self.is_joker else f"{self.suit}-{self.rank}"

    def __repr__(self) -> str:  # keeps test output readable
        return self.id

    def __str__(self) -> str:
        return self.id


JOKER = Card(JOKER_SUIT, JOKER_RANK)


def build_deck() -> list[Card]:
    """44 suited cards plus the joker."""
    deck = [Card(suit, rank) for suit in SUITS for rank in RANKS]
    deck.append(JOKER)
    return deck


def deal(
    rng: random.Random,
    players: int = 4,
    per_player: int = 10,
) -> tuple[list[list[Card]], list[Card]]:
    """Round-robin deal, then the remainder is the kitty.

    Round-robin matters: it is how a real table deals and it is what makes a
    seeded deal reproducible across the JS game and this engine.
    """
    deck = build_deck()
    rng.shuffle(deck)
    hands: list[list[Card]] = [[] for _ in range(players)]
    for i in range(players * per_player):
        hands[i % players].append(deck[i])
    kitty = deck[players * per_player:]
    return hands, kitty


def sort_hand(cards: Sequence[Card]) -> list[Card]:
    """Human order: joker, then suits by their SUITS order, high to low."""
    return sorted(cards, key=lambda c: (c.suit == JOKER_SUIT, c.suit, c.value), reverse=False)
