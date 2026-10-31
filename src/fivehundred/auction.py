"""The bidding round.

Ported from the browser game's `src/auction.js`. Clockwise from the dealer's
left, a seat either outbids the current high bid or passes and is out. The
round ends when everyone but the high bidder has passed — that bid is the
contract — or when all four pass, in which case the hand is thrown in.

One rename: `pass` is a Python keyword, so the method is `pass_turn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bid import Bid, DENOM_ORDER, is_legal_bid, legal_bids

SEATS: tuple[str, ...] = ("south", "west", "north", "east")

PARTNER: dict[str, str] = {"south": "north", "north": "south", "west": "east", "east": "west"}

TEAM: dict[str, str] = {"south": "us", "north": "us", "west": "them", "east": "them"}

ResultType = Literal["contract", "passedOut"]


@dataclass(frozen=True, slots=True)
class AuctionResult:
    type: ResultType
    seat: str | None
    team: str | None
    contract: Bid | None


@dataclass(frozen=True, slots=True)
class AuctionEntry:
    seat: str
    bid: Bid | None  # None is a pass


class Auction:
    """A bidding round. Mutating, like the table it models."""

    def __init__(self, first: str = "south") -> None:
        if first not in SEATS:
            raise ValueError(f"Unknown seat: {first}")
        self.turn_seat = first
        self.passed: set[str] = set()
        self.current: AuctionEntry | None = None
        self.history: list[AuctionEntry] = []
        self.finished = False

    # ── queries ──────────────────────────────────────────────────────────

    @property
    def turn(self) -> str | None:
        return None if self.finished else self.turn_seat

    @property
    def is_over(self) -> bool:
        return self.finished

    @property
    def high_bid(self) -> Bid | None:
        return self.current.bid if self.current else None

    def can_bid(self, seat: str, bid: Bid) -> bool:
        if self.finished or seat != self.turn_seat or seat in self.passed:
            return False
        return is_legal_bid(bid, self.high_bid)

    def can_pass(self, seat: str) -> bool:
        return not self.finished and seat == self.turn_seat and seat not in self.passed

    def options(self, seat: str) -> list[Bid]:
        """Bids this seat could legally make right now."""
        if not self.can_pass(seat):
            return []
        return legal_bids(self.high_bid)

    def contract(self) -> Bid | None:
        """The contract once the round is over (None if it was passed out)."""
        result = self.result
        return result.contract if result else None

    # ── actions ──────────────────────────────────────────────────────────

    def place(self, seat: str, bid: Bid) -> "Auction":
        self._assert_turn(seat)
        if seat in self.passed:
            raise ValueError(f"{seat} has already passed")
        if not is_legal_bid(bid, self.high_bid):
            raise ValueError(f"{seat} cannot bid {bid.tricks}{bid.denom} here")
        self.current = AuctionEntry(seat, bid)
        self.history.append(self.current)
        self._advance()
        return self

    def pass_turn(self, seat: str) -> "Auction":
        self._assert_turn(seat)
        self.passed.add(seat)
        self.history.append(AuctionEntry(seat, None))
        self._advance()
        return self

    def _assert_turn(self, seat: str) -> None:
        if self.finished:
            raise ValueError("Auction is already over")
        if seat != self.turn_seat:
            raise ValueError(f"Out of turn: {seat}, expecting {self.turn_seat}")

    def _advance(self) -> None:
        # Everyone passed without a bid: the hand is thrown in.
        if self.current is None and len(self.passed) == len(SEATS):
            self.finished = True
            return
        # Only the high bidder is still live: that bid is the contract.
        if self.current is not None and len(self.passed) >= len(SEATS) - 1:
            self.finished = True
            return
        i = SEATS.index(self.turn_seat)
        for _ in range(len(SEATS)):
            i = (i + 1) % len(SEATS)
            if SEATS[i] not in self.passed:
                self.turn_seat = SEATS[i]
                return
        raise RuntimeError("no seat left to act")  # unreachable: the checks above catch it

    # ── result ───────────────────────────────────────────────────────────

    @property
    def result(self) -> AuctionResult | None:
        if not self.finished:
            return None
        if self.current is None:
            return AuctionResult("passedOut", None, None, None)
        return AuctionResult(
            "contract",
            self.current.seat,
            TEAM[self.current.seat],
            self.current.bid,
        )

    def notes(self) -> dict[str, Bid | str]:
        """Last action each seat took, for the table bubbles."""
        out: dict[str, Bid | str] = {}
        for entry in self.history:
            out[entry.seat] = entry.bid if entry.bid is not None else "Pass"
        return out


def run_auction(actions: list[Bid | None], first: str = "south") -> AuctionResult:
    """Play a scripted auction out. A `None` in the list is a pass.

    Used by the tests and by the parity fixture: a transcript in, a result out.
    """
    auction = Auction(first=first)
    for action in actions:
        seat = auction.turn
        if seat is None:
            break
        auction.pass_turn(seat) if action is None else auction.place(seat, action)
    return auction.result  # type: ignore[return-value]


def openers_denoms() -> tuple[str, ...]:
    """Denominations a legal first bid may be made in — all of them."""
    return DENOM_ORDER
