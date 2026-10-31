"""The auction: turn order, what is legal, and when the round is over."""

from __future__ import annotations

import pytest

from fivehundred.auction import PARTNER, SEATS, TEAM, Auction
from fivehundred.bid import C, H, NT, NULLO, S, Bid

def fresh(first: str = "south") -> Auction:
    return Auction(first=first)


def test_the_bidding_goes_clockwise_from_the_first_seat():
    auction = fresh("west")
    assert auction.turn == "west"
    auction.pass_turn("west")
    assert auction.turn == "north"
    auction.pass_turn("north")
    assert auction.turn == "east"


def test_four_passes_throw_the_hand_in():
    auction = fresh()
    for seat in list(SEATS):
        auction.pass_turn(seat)
    assert auction.is_over
    result = auction.result
    assert result is not None and result.type == "passedOut"
    assert result.contract is None and result.seat is None


def test_three_passes_leave_the_bidder_with_the_contract():
    auction = fresh("south")
    auction.place("south", Bid(6, H))
    auction.pass_turn("west")
    auction.pass_turn("north")
    auction.pass_turn("east")
    assert auction.is_over
    result = auction.result
    assert result is not None
    assert result.type == "contract"
    assert result.seat == "south"
    assert result.team == "us"
    assert result.contract == Bid(6, H)


def test_a_bidder_who_is_outbid_gets_another_turn():
    auction = fresh("south")
    auction.place("south", Bid(6, H))
    auction.place("west", Bid(6, NT))
    auction.pass_turn("north")
    auction.pass_turn("east")
    assert auction.turn == "south", "south is still live and may bid again"
    auction.pass_turn("south")
    assert auction.is_over
    result = auction.result
    assert result is not None
    assert result.contract == Bid(6, NT)


def test_partnerships_are_across_the_table():
    assert PARTNER["south"] == "north"
    assert PARTNER["west"] == "east"
    assert TEAM["south"] == TEAM["north"] == "us"
    assert TEAM["west"] == TEAM["east"] == "them"


def test_you_may_not_bid_out_of_turn_or_lower_than_the_table():
    auction = fresh("south")
    with pytest.raises(ValueError, match="Out of turn"):
        auction.place("west", Bid(6, S))

    auction.place("south", Bid(8, S))
    with pytest.raises(ValueError):
        auction.place("west", Bid(6, NT))  # below 8S
    with pytest.raises(ValueError):
        auction.place("west", Bid(8, S))  # the same bid again


def test_passing_twice_or_acting_after_the_round_is_refused():
    auction = fresh("south")
    auction.pass_turn("south")
    with pytest.raises(ValueError):
        auction.pass_turn("south")  # not your turn: it is west's
    for seat in ("west", "north", "east"):
        auction.pass_turn(seat)
    assert auction.is_over
    with pytest.raises(ValueError, match="already over"):
        auction.pass_turn("south")


def test_nullo_may_be_bid_once_the_table_has_reached_seven():
    auction = fresh("south")
    auction.place("south", Bid(6, S))
    assert auction.can_bid("west", Bid(0, NULLO)), "nullo at 250 beats six spades"
    assert auction.can_bid("west", Bid(7, S)), "and so does seven spades"
    assert all(not auction.can_bid("north", b) for b in auction.options("west")), "not your turn"


def test_options_list_only_what_the_table_permits():
    auction = fresh("south")
    auction.place("south", Bid(9, NT))
    options = auction.options("west")
    assert options[0] == Bid(10, S), "the cheapest step above nine no trumps"
    assert options[-1] == Bid(10, NT), "the top of the ladder"
    assert Bid(0, "OPEN_NULLO") in options, "Open Nullo at 500 clears 420 in the middle"
    assert auction.options("north") == []


def test_notes_record_each_seat_s_last_action():
    auction = fresh("south")
    auction.pass_turn("south")
    auction.place("west", Bid(6, C))
    auction.pass_turn("north")
    notes = auction.notes()
    assert notes["south"] == "Pass"
    assert notes["west"] == Bid(6, C)
    assert notes["north"] == "Pass"
    assert "east" not in notes
