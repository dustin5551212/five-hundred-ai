"""Bids: the ladder, the nullo contracts, and what may be bid over what."""

from __future__ import annotations

import pytest

from fivehundred.bid import (
    C,
    D,
    H,
    NT,
    NULLO,
    OPEN_NULLO,
    S,
    Bid,
    bid_rank,
    bid_value,
    compare_bids,
    contract_label,
    contract_short,
    is_no_trump,
    is_nullo,
    legal_bids,
    outbids,
)


def test_a_step_in_tricks_outranks_any_change_of_denomination():
    ladder = [Bid(6, denom) for denom in (S, C, D, H, NT)] + [Bid(7, S)]
    ranks = [bid_rank(b) for b in ladder]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)
    assert outbids(Bid(7, S), Bid(6, NT))


def test_contract_values_are_the_avondale_schedule():
    assert bid_value(Bid(6, S)) == 40
    assert bid_value(Bid(6, NT)) == 120
    assert bid_value(Bid(7, S)) == 140
    assert bid_value(Bid(10, NT)) == 520
    assert bid_value(Bid(0, NULLO)) == 250
    assert bid_value(Bid(0, OPEN_NULLO)) == 500


def test_nullo_is_an_ordinary_250_point_bid_on_the_ladder():
    # 8S is 240, 8C is 260: nullo slots exactly between them.
    assert outbids(Bid(0, NULLO), Bid(8, S))
    assert not outbids(Bid(0, NULLO), Bid(8, C))
    assert bid_rank(Bid(0, NULLO)) == 250


def test_open_nullo_is_worth_ten_hearts_but_loses_the_tie():
    assert bid_value(Bid(0, OPEN_NULLO)) == bid_value(Bid(10, H)) == 500
    assert outbids(Bid(10, H), Bid(0, OPEN_NULLO))
    assert not outbids(Bid(0, OPEN_NULLO), Bid(10, H))


def test_compare_bids_handles_the_empty_seat():
    assert compare_bids(None, None) == 0
    assert compare_bids(None, Bid(6, S)) == -1
    assert compare_bids(Bid(6, S), None) == 1
    assert compare_bids(Bid(6, S), Bid(6, S)) == 0


def test_bids_outside_the_contract_space_are_refused():
    for tricks, denom in [(5, S), (11, S), (6, "XX"), (0, S)]:
        with pytest.raises(ValueError):
            Bid(tricks, denom)
    with pytest.raises(ValueError):
        Bid(6, NULLO)  # nullo wins nothing, so it has no trick count


def test_the_legal_ladder_starts_at_the_floor_and_ends_at_ten_no_trumps():
    opening = legal_bids()
    assert opening[0] == Bid(6, S)
    assert opening[-1] == Bid(10, NT)
    assert len(opening) == 27  # 5 denominations x 5 levels, plus both nullos

    after_seven_nt = legal_bids(Bid(7, NT))
    assert after_seven_nt[0] == Bid(8, S), "nullo is already below seven no trumps"
    assert after_seven_nt[-1] == Bid(10, NT)

    after_eight_spades = legal_bids(Bid(8, S))
    assert Bid(0, NULLO) in after_eight_spades, "nullo at 250 beats 8S at 240"


def test_everything_bid_is_labelled_the_way_a_table_says_it():
    assert contract_label(Bid(6, H)) == "6 Hearts"
    assert contract_short(Bid(6, H)) == "6\u2665"
    assert contract_short(Bid(7, NT)) == "7NT"
    assert contract_label(Bid(0, NULLO)) == "Nullo"
    assert contract_short(Bid(0, OPEN_NULLO)) == "Open Nullo"


def test_nullo_and_no_trumps_are_distinguished_by_the_rule_helpers():
    assert is_nullo(NULLO) and is_nullo(OPEN_NULLO)
    assert not is_nullo(NT)
    assert is_no_trump(NT) and is_no_trump(NULLO) and is_no_trump(OPEN_NULLO)
    assert not is_no_trump(H)


def test_the_ladder_after_ten_diamonds_is_the_top_of_the_scale():
    top = legal_bids(Bid(10, D))
    # 10D is 480, so nullo (250) is already behind; Open Nullo at 500 sits between
    # ten hearts (580) and ten no trumps (620).
    assert [b.denom for b in top] == [OPEN_NULLO, H, NT]

    # Ten no trumps is the top of the game; Open Nullo at 500 loses to ten hearts.
    assert legal_bids(Bid(10, NT)) == []
    assert legal_bids(Bid(10, H)) == [Bid(10, NT)]
    assert legal_bids(Bid(0, OPEN_NULLO)) == [Bid(10, H), Bid(10, NT)]
