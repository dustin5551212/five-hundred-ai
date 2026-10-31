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
    """What a seat knows when it is its turn.

    Public information only beyond the hand: every card played so far, the
    trick in progress, the score, and who holds the contract. A seat never sees
    another seat's hand, which is what makes this a game of imperfect
    information rather than a puzzle.
    """

    hand: tuple[Card, ...]
    legal: tuple[Card, ...]
    trick: tuple[tuple[str, Card], ...]
    led_suit: str | None
    contract: str
    seat: str
    joker_suit: str | None
    played: tuple[tuple[str, Card], ...] = ()   # every card played this hand
    tricks: tuple[tuple[str, int], ...] = ()    # (team, tricks) so far
    contractor: str = ""


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
    trick: Sequence[tuple[str, Card]] | None = None,
    tricks: dict[str, int] | None = None,
    log: Sequence[tuple[str, Card]] | None = None,
) -> Outcome:
    """Play one hand out and return the tricks each side took.

    `trick`, `tricks` and `log` exist so a hand can be **resumed mid-trick**
    from a position reached elsewhere — which is what search needs: put a
    candidate card on the table, then play the rest out with a base policy and
    see what the hand was worth. Left empty, it deals with a fresh hand led by
    the contractor.
    """
    remaining: dict[str, list[Card]] = {seat: list(cards) for seat, cards in hands.items()}
    order = play_order(contract, contractor)
    trick_size = len(order)
    totals = {"NS": 0, "WE": 0} if tricks is None else {k: int(v) for k, v in tricks.items()}
    played_log: list[tuple[str, Card]] = list(log or ())
    current: list[tuple[str, Card]] = list(trick or ())

    leader = order[0]  # the contractor, for a fresh hand
    if current:
        leader = current[0][0]
    elif played_log:
        # Resuming on a trick boundary: whoever took the last completed trick leads.
        tail = played_log[-trick_size:]
        led = effective_suit(tail[0][1], contract, joker_suit)
        leader = tail[trick_winner(tail, contract, led, joker_suit)][0]

    while current or any(remaining[seat] for seat in order):
        if current:
            # Resume: the seats still to play in this trick follow the last player.
            start = order.index(current[-1][0])
            seats = (order[start + 1:] + order[: start + 1])[: trick_size - len(current)]
        else:
            start = order.index(leader)
            seats = (order[start:] + order[:start])[:trick_size]

        led_suit: str | None = (
            effective_suit(current[0][1], contract, joker_suit) if current else None
        )

        for seat in seats:
            hand = remaining[seat]
            if not hand:
                raise RuntimeError(f"{seat} was asked to play with no cards left")
            legal = legal_plays(hand, led_suit, contract, joker_suit)
            view = View(
                hand=tuple(hand),
                legal=tuple(legal),
                trick=tuple(current),
                led_suit=led_suit,
                contract=contract,
                seat=seat,
                joker_suit=joker_suit,
                played=tuple(played_log),
                tricks=tuple(totals.items()),
                contractor=contractor,
            )
            card = agents[seat].choose(view)
            if card not in legal:
                raise ValueError(f"{seat} tried to play {card}, illegal in {[c.id for c in legal]}")
            hand.remove(card)
            if led_suit is None:
                led_suit = effective_suit(card, contract, joker_suit)
            current.append((seat, card))
            played_log.append((seat, card))

        winner_idx = trick_winner(current, contract, led_suit, joker_suit)
        leader = current[winner_idx][0]
        totals[TEAM_OF[leader]] += 1
        current = []

    contractor_team = TEAM_OF[contractor]
    return Outcome(
        tricks=totals,
        tricks_by_contractor=totals[contractor_team],
        plays=tuple(played_log),
    )
