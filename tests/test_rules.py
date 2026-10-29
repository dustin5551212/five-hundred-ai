"""The rules that decide tricks, tested against the same expectations the
browser game's 26-test trump suite encodes. If these drift, the benchmark is
measuring a different game.
"""

from __future__ import annotations

import random

import pytest

from fivehundred.cards import Card, JOKER, build_deck, deal
from fivehundred.rules import (
    LEFT_BOWER_STRENGTH,
    RIGHT_BOWER_STRENGTH,
    bower_of,
    compare_cards,
    effective_suit,
    is_trump,
    legal_plays,
    lowest_card,
    trick_winner,
)


def test_deck_is_45_cards_with_one_joker():
    deck = build_deck()
    assert len(deck) == 45
    assert sum(1 for c in deck if c.is_joker) == 1
    assert len({c.id for c in deck}) == 45
    # all 2s and 3s are out
    assert not any(c.rank in ("2", "3") for c in deck)


def test_deal_gives_ten_each_and_a_five_card_kitty():
    hands, kitty = deal(random.Random(1))
    assert [len(h) for h in hands] == [10, 10, 10, 10]
    assert len(kitty) == 5
    all_cards = [c for h in hands for c in h] + kitty
    assert len({c.id for c in all_cards}) == 45


def test_a_seeded_deal_is_reproducible():
    a, _ = deal(random.Random(42))
    b, _ = deal(random.Random(42))
    assert [[c.id for c in h] for h in a] == [[c.id for c in h] for h in b]


def test_bowers_are_the_jacks():
    hearts_trump_js = Card("H", "J")
    diamonds_jack = Card("D", "J")
    clubs_jack = Card("C", "J")

    assert bower_of(hearts_trump_js, "H") == "right"
    assert bower_of(diamonds_jack, "H") == "left"  # D<->H same colour
    assert bower_of(clubs_jack, "H") is None
    assert bower_of(JOKER, "H") == "joker"


def test_trump_order_joker_right_left_ace():
    trump = "S"
    order = [JOKER, Card("S", "J"), Card("C", "J"), Card("S", "A")]
    for i in range(len(order) - 1):
        assert (
            compare_cards(order[i], order[i + 1], trump, "S") == 1
        ), f"{order[i]} should beat {order[i + 1]}"
    # the left bower is below the right bower but above the ace of trump
    assert compare_cards(Card("S", "J"), Card("C", "J"), trump, "S") == 1
    assert compare_cards(Card("C", "J"), Card("S", "A"), trump, "S") == 1


def test_left_bower_is_trump_not_its_printed_suit():
    # hearts trump: the diamond jack is trump, so it cannot follow a diamond lead
    hand = [Card("D", "J"), Card("D", "4"), Card("S", "9")]
    legal = legal_plays(hand, led_suit="D", trump="H")
    assert [c.id for c in legal] == ["D-4"]
    assert effective_suit(Card("D", "J"), "H") == "H"
    assert is_trump(Card("D", "J"), "H")


def test_trump_led_must_be_followed_with_trump_including_bowers():
    hand = [Card("H", "4"), Card("D", "J"), Card("H", "K"), Card("S", "A")]
    legal = legal_plays(hand, led_suit="H", trump="H")
    assert {c.id for c in legal} == {"H-4", "D-J", "H-K"}


def test_cannot_follow_suit_means_any_card():
    hand = [Card("C", "4"), Card("S", "9")]
    legal = legal_plays(hand, led_suit="D", trump="H")
    assert len(legal) == 2


def test_highest_trump_takes_the_trick_over_the_led_suit():
    plays = [("S", Card("D", "A")), ("W", Card("H", "4")), ("N", Card("D", "K")), ("E", Card("D", "5"))]
    assert trick_winner(plays, "H", "D") == 1  # the ruff


def test_off_suit_card_cannot_win_without_trump():
    plays = [("S", Card("D", "A")), ("W", Card("C", "K"))]
    assert trick_winner(plays, "H", "D") == 0


def test_no_trumps_has_no_bowers_and_the_jacks_revert():
    assert bower_of(Card("S", "J"), "NT") is None
    assert not is_trump(Card("S", "J"), "NT")
    assert compare_cards(Card("S", "A"), Card("S", "J"), "NT", "S") == 1


def test_unnominated_joker_in_no_trumps_wins_anything():
    assert compare_cards(JOKER, Card("S", "A"), "NT", "S") == 1
    # and it cannot follow a suit at all
    hand = [JOKER, Card("S", "4")]
    assert [c.id for c in legal_plays(hand, led_suit="S", trump="NT")] == ["S-4"]
    # but it may be played when void
    hand = [JOKER, Card("H", "4")]
    assert len(legal_plays(hand, led_suit="S", trump="NT")) == 2


def test_nullo_void_player_holding_the_joker_must_play_it():
    hand = [JOKER, Card("H", "4"), Card("H", "9")]
    legal = legal_plays(hand, led_suit="S", trump="NULLO")
    assert [c.id for c in legal] == ["JOKER"]


def test_lowest_card_is_by_plain_rank_and_never_the_joker():
    assert lowest_card([Card("S", "A"), Card("H", "4"), JOKER]).id == "H-4"


def test_strength_of_bowers_is_above_ace_but_below_joker():
    assert RIGHT_BOWER_STRENGTH < 15 and LEFT_BOWER_STRENGTH < RIGHT_BOWER_STRENGTH
