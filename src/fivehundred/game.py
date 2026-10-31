"""A whole hand of 500, and a whole match to 500 points.

This is the join that was missing: `cards` deals, `auction` decides the
contract, the contractor takes the kitty, `env` plays the tricks and `scoring`
pays out. Ported from the browser game's `src/app.js`, which keeps the same
order of events.

Two scope notes, stated so nobody has to guess:

* The kitty rule is the browser game's: the contractor picks up all five cards
  and discards back down to ten, and in a Nullo contract the partner also gives
  their lowest card — they sit the hand out, so their hand size does not matter.
* The joker is never nominated to a suit here. In No Trumps and Nullo it is a
  free card of no suit, which is what `rules.legal_plays` already models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from .agents import HeuristicAgent, choose_bid, choose_discards
from .auction import PARTNER, SEATS, TEAM, Auction
from .bid import Bid, is_nullo
from .cards import Card, deal
from .env import TEAM_OF, Agent, play_hand
from .rules import is_nullo as rules_is_nullo
from .rules import lowest_card
from .scoring import US, HandScore, apply_score, match_winner, score_contract

#: The auction names seats in words; the play phase uses letters.
LETTER: dict[str, str] = {"south": "S", "west": "W", "north": "N", "east": "E"}
SEAT_OF: dict[str, str] = {v: k for k, v in LETTER.items()}

#: env talks about "NS"/"WE"; scoring talks about "us"/"them".
SCORING_TEAM: dict[str, str] = {"NS": US, "WE": "them"}

HAND_SIZE = 10
KITTY_SIZE = 5


class Bidder(Protocol):
    """A seat's bidding judgement: a bid, or None to pass."""

    def __call__(self, seat: str, hand: Sequence[Card], current: Bid | None) -> Bid | None: ...


#: A seat's play judgement. The same `choose(view)` protocol the benchmark's
#: random / lowest / heuristic agents implement, so any of them can sit down.
Player = Agent


@dataclass(frozen=True, slots=True)
class HandRecord:
    """What happened on one deal, in the terms a scorecard would use."""

    index: int
    dealer: str
    contract: Bid | None
    contractor: str | None
    tricks: dict[str, int]
    score: HandScore | None
    thrown_in: bool

    @property
    def label(self) -> str:
        if self.thrown_in:
            return "passed out"
        assert self.contract is not None
        return f"{self.contract.label} by {self.contractor}"


@dataclass(slots=True)
class MatchResult:
    totals: dict[str, int]
    hands: list[HandRecord] = field(default_factory=list)
    winner: str | None = None

    @property
    def us(self) -> int:
        return self.totals.get(US, 0)

    @property
    def them(self) -> int:
        return self.totals.get("them", 0)


def cpu_bidder(_seat: str, hand: Sequence[Card], current: Bid | None) -> Bid | None:
    """The shipped CPU's judgement, in the shape the auction wants."""
    return choose_bid(hand, current)


def cpu_table() -> tuple[dict[str, Bidder], dict[str, Player]]:
    """Four seats of the same opponent: any table the benchmark can measure."""
    bidders: dict[str, Bidder] = {seat: cpu_bidder for seat in SEATS}
    players: dict[str, Player] = {seat: HeuristicAgent() for seat in SEATS}
    return bidders, players


def deal_hand(rng: random.Random) -> tuple[dict[str, list[Card]], list[Card]]:
    """Ten cards to each of four seats, five to the kitty."""
    hands, kitty = deal(rng, players=4, per_player=HAND_SIZE)
    return {LETTER[seat]: hand for seat, hand in zip(SEATS, hands)}, kitty


def exchange_kitty(
    hands: dict[str, list[Card]],
    kitty: Sequence[Card],
    contractor: str,
    contract: Bid,
    discarder: Callable[[Sequence[Card], str, int], list[Card]] = choose_discards,
) -> list[Card]:
    """The contractor takes the kitty, then throws cards away until ten remain.

    Returns the discarded cards so a scorecard can show what went back.
    """
    seat = LETTER[contractor]
    gained = list(kitty)
    hands[seat] = list(hands[seat]) + gained

    if rules_is_nullo(contract.denom):
        # The partner sits the hand out; their lowest card is a free upgrade.
        partner = LETTER[PARTNER[contractor]]
        gift = lowest_card(hands[partner])
        hands[partner].remove(gift)
        hands[seat].append(gift)

    excess = len(hands[seat]) - HAND_SIZE
    if excess <= 0:
        return []
    dropped = discarder(hands[seat], contract.denom, excess)
    for card in dropped:
        hands[seat].remove(card)
    return list(dropped)


def play_deal(
    rng: random.Random,
    index: int,
    dealer: str,
    bidders: dict[str, Bidder],
    players: dict[str, Player],
    discarder: Callable[[Sequence[Card], str, int], list[Card]] = choose_discards,
) -> HandRecord:
    """One deal, start to finish."""
    hands, kitty = deal_hand(rng)
    first = SEATS[(SEATS.index(dealer) + 1) % len(SEATS)]
    auction = Auction(first=first)

    while not auction.is_over:
        seat = auction.turn
        assert seat is not None
        bid = bidders[seat](seat, hands[LETTER[seat]], auction.high_bid)
        if bid is None:
            auction.pass_turn(seat)
        else:
            auction.place(seat, bid)

    result = auction.result
    assert result is not None
    if result.type == "passedOut" or result.contract is None or result.seat is None:
        return HandRecord(index, dealer, None, None, {}, None, thrown_in=True)

    contract = result.contract
    contractor = result.seat
    exchange_kitty(hands, kitty, contractor, contract, discarder)

    agents = {LETTER[seat]: players[seat] for seat in SEATS}
    outcome = play_hand(
        {seat: hand for seat, hand in hands.items()},
        contract.denom,
        LETTER[contractor],
        agents,
    )

    tricks_by_team = {
        SCORING_TEAM[team]: count for team, count in outcome.tricks.items()
    }
    score = score_contract(contract, TEAM[contractor], tricks_by_team)
    return HandRecord(index, dealer, contract, contractor, tricks_by_team, score, thrown_in=False)


def play_match(
    rng: random.Random,
    bidders: dict[str, Bidder],
    players: dict[str, Player],
    target: int = 500,
    max_hands: int = 60,
    dealer: str = "east",
) -> MatchResult:
    """Play hands until one side reaches the target.

    The dealer rotates, as at a table. `max_hands` is a guard rather than a
    rule: a match that cannot finish in sixty hands is a bug, not a bad night.
    """
    totals = {US: 0, "them": 0}
    match = MatchResult(totals=totals)
    current_dealer = dealer

    for index in range(1, max_hands + 1):
        record = play_deal(rng, index, current_dealer, bidders, players)
        match.hands.append(record)
        if record.score is not None:
            totals = apply_score(totals, record.score)
            match.totals = totals
            # A tie at the top goes to the side that just won the hand.
            tie_break = record.score.contractor_team
            winner = match_winner(totals, target=target, tie_break=tie_break)
            if winner is not None:
                match.winner = winner
                return match
        current_dealer = SEATS[(SEATS.index(current_dealer) + 1) % len(SEATS)]

    return match
