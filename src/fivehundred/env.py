"""The play phase of a 500 hand: seats, turn order, tricks, and a result.

    S (south) and N (north) are one partnership; W (west) and E (east) the other.
    Play runs clockwise: S -> W -> N -> E.

Scope, stated plainly: this module plays the cards after the contract is
settled. The auction, the kitty exchange and the partner's gift are not here
yet — see the README's roadmap. Nothing in the scoring benchmark depends on
them, because every comparison is tricks taken against tricks taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .cards import Card
from .rules import NULLO, effective_suit, legal_plays, trick_winner

SEATS: tuple[str, ...] = ("S", "W", "N", "E")
TEAM_OF: dict[str, str] = {"S": "NS", "N": "NS", "W": "WE", "E": "WE"}
PARTNER_OF: dict[str, str] = {"S": "N", "N": "S", "W": "E", "E": "W"}


@dataclass(frozen=True, slots=True)
class View:
    """What a seat knows when it is its turn. Everything else stays hidden."""

    hand: tuple[Card, ...]
    legal: tuple[Card, ...]
    trick: tuple[tuple[str, Card], ...]
    led_suit: str | None
    contract: str
    seat: str
    joker_suit: str | None


class Agent(Protocol):
    def choose(self, view: View) -> Card: ...


@dataclass(frozen=True, slots=True)
class Outcome:
    tricks: dict[str, int]
    tricks_by_contractor: int
    plays: tuple[tuple[str, Card], ...] = field(default=())

    @property
    def winner_team(self) -> str:
        return "NS" if self.tricks["NS"] > self.tricks["WE"] else "WE"


def play_order(contract: str, contractor: str) -> list[str]:
    """Clockwise from the contractor; the Nullo partner sits the hand out."""
    start = SEATS.index(contractor)
    order = [SEATS[(start + i) % 4] for i in range(4)]
    if contract == NULLO:
        order = [seat for seat in order if seat != PARTNER_OF[contractor]]
    return order


def play_hand(
    hands: dict[str, Sequence[Card]],
    contract: str,
    contractor: str,
    agents: dict[str, Agent],
    joker_suit: str | None = None,
) -> Outcome:
    """Play one hand out and return the tricks each side took."""
    remaining: dict[str, list[Card]] = {seat: list(cards) for seat, cards in hands.items()}
    order = play_order(contract, contractor)
    trick_size = len(order)
    tricks = {"NS": 0, "WE": 0}
    log: list[tuple[str, Card]] = []

    leader = contractor
    while any(remaining[seat] for seat in order):
        idx = order.index(leader)
        play_order_this_trick = (order[idx:] + order[:idx])[:trick_size]

        trick: list[tuple[str, Card]] = []
        led_suit: str | None = None
        for seat in play_order_this_trick:
            hand = remaining[seat]
            if not hand:
                raise RuntimeError(f"{seat} was asked to play with no cards left")
            legal = legal_plays(hand, led_suit, contract, joker_suit)
            view = View(
                hand=tuple(hand),
                legal=tuple(legal),
                trick=tuple(trick),
                led_suit=led_suit,
                contract=contract,
                seat=seat,
                joker_suit=joker_suit,
            )
            card = agents[seat].choose(view)
            if card not in legal:
                raise ValueError(f"{seat} tried to play {card}, illegal in {[c.id for c in legal]}")
            hand.remove(card)
            if led_suit is None:
                led_suit = effective_suit(card, contract, joker_suit)
            trick.append((seat, card))
            log.append((seat, card))

        winner_idx = trick_winner(trick, contract, led_suit, joker_suit)
        leader = trick[winner_idx][0]
        tricks[TEAM_OF[leader]] += 1

    contractor_team = TEAM_OF[contractor]
    return Outcome(tricks=tricks, tricks_by_contractor=tricks[contractor_team], plays=tuple(log))
