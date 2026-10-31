"""Turning a seat's view into a fixed-length vector.

Everything here is information the seat is allowed to have: its own hand, the
cards everyone has already played, the trick in progress, the contract, the
score and its own role. Nothing peeks at another hand.

148 numbers per decision:

    0..44    the hand, as a multi-hot over the 45 cards
    45..89   the trick in progress
    90..134  every card played so far this hand
    135..139 contract: C, D, H, S, NT
    140..144 led suit: none, C, D, H, S
    145..146 role: is contractor, is the contractor's partner
    147      tricks won by my side, normalised to [-1, 1]
"""

from __future__ import annotations

from .cards import Card, SUITS, RANKS, JOKER_SUIT
from .env import PARTNER_OF, TEAM_OF, View
from .rules import NT

N_CARDS = 45
CONTRACT_ORDER = ("C", "D", "H", "S", NT)
LED_ORDER = (None, "C", "D", "H", "S")
FEATURE_DIM = N_CARDS * 3 + len(CONTRACT_ORDER) + len(LED_ORDER) + 2 + 1

# Stable index for every card id: C-4..C-A, D-*, H-*, S-*, JOKER.
CARD_INDEX: dict[str, int] = {
    f"{suit}-{rank}": i for i, (suit, rank) in enumerate(
        (s, r) for s in SUITS for r in RANKS
    )
}
CARD_INDEX[JOKER_SUIT] = N_CARDS - 1
CARDS: tuple[Card, ...] = tuple(
    Card(suit, rank) for suit in SUITS for rank in RANKS
) + (Card(JOKER_SUIT, "JOKER"),)


def index_of(card: Card) -> int:
    return CARD_INDEX[card.id]


def card_at(index: int) -> Card:
    return CARDS[index]


def encode(view: View) -> list[float]:
    x = [0.0] * FEATURE_DIM
    for card in view.hand:
        x[index_of(card)] = 1.0
    for _seat, card in view.trick:
        x[N_CARDS + index_of(card)] = 1.0
    for _seat, card in view.played:
        x[2 * N_CARDS + index_of(card)] = 1.0

    base = 3 * N_CARDS
    if view.contract in CONTRACT_ORDER:
        x[base + CONTRACT_ORDER.index(view.contract)] = 1.0
    base += len(CONTRACT_ORDER)
    if view.led_suit in LED_ORDER:
        x[base + LED_ORDER.index(view.led_suit)] = 1.0
    base += len(LED_ORDER)

    x[base] = 1.0 if view.seat == view.contractor else 0.0
    x[base + 1] = 1.0 if view.seat == PARTNER_OF.get(view.contractor, "") else 0.0

    mine = dict(view.tricks).get(TEAM_OF[view.seat], 0)
    theirs = dict(view.tricks).get("WE" if TEAM_OF[view.seat] == "NS" else "NS", 0)
    x[base + 2] = (mine - theirs) / 5.0
    return x


def legal_mask(view: View) -> list[bool]:
    mask = [False] * N_CARDS
    for card in view.legal:
        mask[index_of(card)] = True
    return mask
