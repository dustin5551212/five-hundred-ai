"""Trump order, follow-suit and trick resolution for every denomination.

This is the part every home-grown 500 engine gets wrong, so it is ported rule
by rule from the browser game's `src/trump.js` (which has 26 tests on it) and
then tested again here against the same expectations.

    suit contract:  Joker > right bower > left bower > A K Q 10 9 8 7 6 5 4

The left bower is the jack of the suit of the same colour, and it plays as
trump — so follow-suit must be judged on `effective_suit`, never on
`card.suit`. In No Trumps and Nullo there is no trump suit and no bowers, the
jacks revert to their printed rank, and an unnominated joker belongs to no
suit and may only be played when void in the led suit.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .cards import Card, JOKER_VALUE

SUIT_DENOMS: tuple[str, ...] = ("C", "D", "H", "S")
NT = "NT"
NULLO = "NULLO"
OPEN_NULLO = "OPEN_NULLO"
DENOMS: tuple[str, ...] = SUIT_DENOMS + (NT, NULLO, OPEN_NULLO)

# Suit of the same colour — the left-bower suit.
SAME_COLOUR: dict[str, str] = {"C": "S", "S": "C", "D": "H", "H": "D"}

BOWER_JOKER = "joker"
BOWER_RIGHT = "right"
BOWER_LEFT = "left"

TRUMP_RANK_STRENGTH: dict[str, int] = {
    "A": 12, "K": 11, "Q": 10, "10": 9, "9": 8, "8": 7, "7": 6, "6": 5, "5": 4, "4": 3,
}
JOKER_STRENGTH = 15
RIGHT_BOWER_STRENGTH = 14
LEFT_BOWER_STRENGTH = 13
NOMINATED_JOKER_STRENGTH = 99


def is_no_trump(denom: str) -> bool:
    return denom in (NT, NULLO, OPEN_NULLO)


def is_nullo(denom: str) -> bool:
    """Both nullo contracts are played to lose every trick, so both count."""
    return denom in (NULLO, OPEN_NULLO)


def joker_mode(denom: str, joker_suit: str | None = None) -> str:
    """How the joker behaves: 'trump', 'nominated', 'free' or 'plain'."""
    if not denom:
        return "plain"
    if not is_no_trump(denom):
        return "trump"
    return "nominated" if joker_suit else "free"


def bower_of(card: Card, trump: str | None) -> str | None:
    if not trump or is_no_trump(trump):
        return None
    if card.is_joker:
        return BOWER_JOKER
    if card.rank != "J":
        return None
    if card.suit == trump:
        return BOWER_RIGHT
    if card.suit == SAME_COLOUR[trump]:
        return BOWER_LEFT
    return None


def is_trump(card: Card, trump: str | None) -> bool:
    """False throughout No Trumps and Nullo — the joker there is not trump."""
    if not trump or is_no_trump(trump):
        return False
    if card.is_joker:
        return True
    if card.suit == trump:
        return True
    return bower_of(card, trump) == BOWER_LEFT


def effective_suit(card: Card, trump: str | None, joker_suit: str | None = None) -> str | None:
    if is_no_trump(trump or ""):
        return joker_suit if card.is_joker else card.suit
    return (trump or "") if is_trump(card, trump) else card.suit


def strength(card: Card, trump: str | None, joker_suit: str | None = None) -> int:
    """Strength *within* a category. Only comparable inside one category."""
    if not trump:
        return card.value
    if card.is_joker:
        return NOMINATED_JOKER_STRENGTH if joker_mode(trump, joker_suit) == "nominated" else JOKER_STRENGTH
    if is_trump(card, trump):
        bower = bower_of(card, trump)
        if bower == BOWER_RIGHT:
            return RIGHT_BOWER_STRENGTH
        if bower == BOWER_LEFT:
            return LEFT_BOWER_STRENGTH
        return TRUMP_RANK_STRENGTH[card.rank]
    return card.value


def _rank_of(card: Card) -> int:
    return NOMINATED_JOKER_STRENGTH if card.is_joker else card.value


def compare_cards(
    a: Card,
    b: Card,
    trump: str | None,
    led_suit: str | None,
    joker_suit: str | None = None,
) -> int:
    """1 if `a` wins the trick, -1 if `b` does, 0 if neither can."""
    if a.id == b.id:
        return 0

    # An unnominated joker in NT/Nullo is the highest card in the pack.
    if joker_mode(trump or "", joker_suit) == "free":
        if a.is_joker:
            return 1
        if b.is_joker:
            return -1

    a_trump = is_trump(a, trump)
    b_trump = is_trump(b, trump)
    if a_trump != b_trump:
        return 1 if a_trump else -1
    if a_trump:
        return (strength(a, trump, joker_suit) > strength(b, trump, joker_suit)) - (
            strength(a, trump, joker_suit) < strength(b, trump, joker_suit)
        )

    a_led = effective_suit(a, trump, joker_suit) == led_suit
    b_led = effective_suit(b, trump, joker_suit) == led_suit
    if a_led != b_led:
        return 1 if a_led else -1
    if not a_led:
        return 0  # both off suit — neither can take the trick
    return (_rank_of(a) > _rank_of(b)) - (_rank_of(a) < _rank_of(b))


def trick_winner(
    plays: Sequence[tuple[str, Card]],
    trump: str | None,
    led_suit: str | None,
    joker_suit: str | None = None,
) -> int:
    """Index of the winning play. `plays` is in play order."""
    if not plays:
        raise ValueError("trick_winner needs at least one play")
    best = 0
    for i in range(1, len(plays)):
        if compare_cards(plays[i][1], plays[best][1], trump, led_suit, joker_suit) > 0:
            best = i
    return best


def legal_plays(
    hand: Sequence[Card],
    led_suit: str | None,
    trump: str | None,
    joker_suit: str | None = None,
) -> list[Card]:
    """Must follow the led suit, counting bowers and the joker as trump.

    In Nullo a void player holding the unnominated joker must play it — that is
    the trap the game is built around, not a quirk.
    """
    if led_suit is None:
        return list(hand)

    following = [c for c in hand if effective_suit(c, trump, joker_suit) == led_suit]
    if following:
        return following

    if joker_mode(trump or "", joker_suit) == "free" and is_nullo(trump or ""):
        jokers = [c for c in hand if c.is_joker]
        if jokers:
            return jokers
    return list(hand)


def lowest_card(cards: Iterable[Card]) -> Card:
    """Lowest by plain rank — the Nullo partner's gift, joker never lowest."""
    return min(cards, key=lambda c: c.value)


def suit_counts(hand: Sequence[Card]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in hand:
        counts[card.suit] = counts.get(card.suit, 0) + 1
    return counts
