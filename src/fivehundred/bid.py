"""500 bids: tricks + denomination, their values and their precedence.

Ported from the browser game's `src/bid.js`, value for value. The ladder is
6S 6C 6D 6H 6NT 7S ... 10NT, one step up in tricks outranking any suit change.

Nullo (misère) is not on that ladder: it is an ordinary 250-point bid, so it
sits between 8S (240) and 8C (260) and can be bid over anything cheaper. Open
Nullo is worth 500 — the same as ten hearts — and loses that tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

S = "S"
C = "C"
D = "D"
H = "H"
NT = "NT"
NULLO = "NULLO"
OPEN_NULLO = "OPEN_NULLO"

DENOMS: tuple[str, ...] = (S, C, D, H, NT, NULLO, OPEN_NULLO)

#: Suit / no-trump precedence, lowest first.
DENOM_ORDER: tuple[str, ...] = (S, C, D, H, NT)

DENOM_NAME: dict[str, str] = {
    S: "Spades",
    C: "Clubs",
    D: "Diamonds",
    H: "Hearts",
    NT: "No Trumps",
    NULLO: "Nullo",
    OPEN_NULLO: "Open Nullo",
}

DENOM_SHORT: dict[str, str] = {
    S: "\u2660",
    C: "\u2663",
    D: "\u2666",
    H: "\u2665",
    NT: "NT",
    NULLO: "Nullo",
    OPEN_NULLO: "Open Nullo",
}

MIN_TRICKS = 6
MAX_TRICKS = 10

_BASE: dict[str, int] = {S: 40, C: 60, D: 80, H: 100, NT: 120}
_TRICK_STEP = 100

NULLO_VALUE = 250
OPEN_NULLO_VALUE = 500
OPEN_NULLO_TIEBREAK = 0.5


def is_nullo(denom: str) -> bool:
    return denom in (NULLO, OPEN_NULLO)


def is_no_trump(denom: str) -> bool:
    return denom == NT or is_nullo(denom)


def is_suit(denom: str) -> bool:
    return denom in (S, C, D, H)


@dataclass(frozen=True, slots=True)
class Bid:
    """A contract offer. Nullo bids carry tricks = 0 because they win nothing."""

    tricks: int
    denom: str

    def __post_init__(self) -> None:
        if self.denom not in DENOMS:
            raise ValueError(f"Unknown denomination: {self.denom}")
        if is_nullo(self.denom):
            if self.tricks != 0:
                raise ValueError(f"{self.denom} must be created with tricks = 0")
            return
        if not isinstance(self.tricks, int) or not MIN_TRICKS <= self.tricks <= MAX_TRICKS:
            raise ValueError(
                f"Tricks must be an integer {MIN_TRICKS}-{MAX_TRICKS}, got {self.tricks}"
            )

    @property
    def value(self) -> int:
        """Points the contract is worth if made."""
        return bid_value(self)

    @property
    def rank(self) -> float:
        return bid_rank(self)

    @property
    def label(self) -> str:
        return contract_label(self)

    @property
    def short(self) -> str:
        return contract_short(self)

    def __repr__(self) -> str:  # keeps test output readable
        return f"Bid({self.tricks}, {self.denom})"


def make_bid(tricks: int, denom: str) -> Bid:
    return Bid(tricks, denom)


def bid_value(bid: Bid) -> int:
    if bid.denom == NULLO:
        return NULLO_VALUE
    if bid.denom == OPEN_NULLO:
        return OPEN_NULLO_VALUE
    return _BASE[bid.denom] + (bid.tricks - MIN_TRICKS) * _TRICK_STEP


def bid_rank(bid: Bid) -> float:
    """Position on the ladder — a bid is worth what it scores.

    Nullo at 250 therefore sits between 8S (240) and 8C (260). Open Nullo loses
    the tie with ten hearts by half a point, exactly as the JS does.
    """
    value = bid_value(bid)
    return value - OPEN_NULLO_TIEBREAK if bid.denom == OPEN_NULLO else float(value)


def compare_bids(a: Bid | None, b: Bid | None) -> int:
    """1 if `a` outranks `b`, -1 if it does not, 0 if they are the same rank."""
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    ra, rb = bid_rank(a), bid_rank(b)
    return (ra > rb) - (ra < rb)


def outbids(bid: Bid | None, previous: Bid | None) -> bool:
    return compare_bids(bid, previous) > 0


def legal_bids(previous: Bid | None = None) -> list[Bid]:
    """Every bid that legally outranks `previous`, lowest first."""
    bids = [
        Bid(tricks, denom)
        for tricks in range(MIN_TRICKS, MAX_TRICKS + 1)
        for denom in DENOM_ORDER
    ]
    bids.extend(Bid(0, denom) for denom in (NULLO, OPEN_NULLO))
    return sorted((b for b in bids if outbids(b, previous)), key=bid_rank)


def is_legal_bid(bid: Bid, previous: Bid | None = None) -> bool:
    return any(b.tricks == bid.tricks and b.denom == bid.denom for b in legal_bids(previous))


def contract_label(bid: Bid) -> str:
    if bid.denom == NULLO:
        return "Nullo"
    if bid.denom == OPEN_NULLO:
        return "Open Nullo"
    return f"{bid.tricks} {DENOM_NAME[bid.denom]}"


def contract_short(bid: Bid) -> str:
    """Short label for chips and buttons, e.g. "6\u2665", "7NT", "Nullo"."""
    if bid.denom == NULLO:
        return "Nullo"
    if bid.denom == OPEN_NULLO:
        return "Open Nullo"
    return f"{bid.tricks}{DENOM_SHORT[bid.denom]}"


def highest(bids: Iterable[Bid]) -> Bid | None:
    """Top of a collection — the auction's current high bid."""
    return max(bids, key=bid_rank, default=None)


def as_sequence(bids: Sequence[Bid]) -> list[tuple[int, str]]:
    """Serialisable form, for fixtures and logs."""
    return [(b.tricks, b.denom) for b in bids]
