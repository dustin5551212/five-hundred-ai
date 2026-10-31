"""What a hand pays, and when the match is over."""

from __future__ import annotations

from fivehundred.bid import H, NT, NULLO, OPEN_NULLO, S, Bid
from fivehundred.scoring import (
    MATCH_TARGET,
    SLAM_VALUE,
    TRICK_POINTS,
    apply_score,
    match_winner,
    score_contract,
)


def test_a_made_contract_pays_its_value_and_the_defenders_get_ten_a_trick():
    result = score_contract(Bid(7, H), "us", {"us": 7, "them": 3})
    assert result.made
    assert result.value == 200
    assert result.contractor_points == 200
    assert result.opponent_points == 3 * TRICK_POINTS
    assert result.note == "Contract made"


def test_a_defeated_contract_costs_its_value_and_the_defenders_still_score():
    result = score_contract(Bid(8, S), "them", {"us": 5, "them": 5})
    assert not result.made
    assert result.contractor_points == -240  # 8S is 40 + 2 step-ups of 100
    assert result.opponent_points == 50
    assert result.note == "Contract defeated"


def test_taking_every_trick_is_a_slam_worth_at_least_250():
    cheap = score_contract(Bid(6, S), "us", {"us": 10, "them": 0})
    assert cheap.made and cheap.value == SLAM_VALUE
    assert cheap.note == "Slam"

    # A bid already worth more than the slam keeps its own value — and the browser
    # game only calls it a slam when the bonus is what changed the price.
    rich = score_contract(Bid(9, NT), "us", {"us": 10, "them": 0})
    assert rich.value == 420
    assert rich.note == "Contract made"


def test_nullo_is_all_or_nothing_and_the_defenders_score_nothing():
    won = score_contract(Bid(0, NULLO), "us", {"us": 0, "them": 10})
    assert won.made and won.contractor_points == 250 and won.opponent_points == 0
    assert won.note == "Nullo made"

    lost = score_contract(Bid(0, OPEN_NULLO), "them", {"us": 9, "them": 1})
    assert not lost.made and lost.contractor_points == -500
    assert lost.opponent_points == 0
    assert lost.note == "Nullo defeated"


def test_scores_accumulate_on_both_sides_of_the_scorecard():
    totals = {"us": 100, "them": 60}
    after_us = apply_score(totals, score_contract(Bid(6, NT), "us", {"us": 7, "them": 3}))
    assert after_us == {"us": 220, "them": 90}

    # Same contract, the other side bidding it and failing at it: they lose the value
    # and we take ten a trick for the nine we made.
    after_them = apply_score(after_us, score_contract(Bid(6, NT), "them", {"us": 9, "them": 1}))
    assert after_them == {"us": 310, "them": -30}


def test_apply_score_does_not_mutate_the_scorecard_it_was_given():
    totals = {"us": 100, "them": 60}
    apply_score(totals, score_contract(Bid(6, S), "us", {"us": 6, "them": 4}))
    assert totals == {"us": 100, "them": 60}


def test_the_match_ends_at_the_target():
    assert match_winner({"us": 499, "them": 500}) == "them"
    assert match_winner({"us": 500, "them": 499}) == "us"
    assert match_winner({"us": 499, "them": 499}) is None
    assert match_winner({"us": 0, "them": 0}) is None


def test_if_both_sides_pass_the_target_the_higher_total_wins():
    assert match_winner({"us": 620, "them": 540}) == "us"
    assert match_winner({"us": 500, "them": 550}) == "them"


def test_a_tie_at_the_top_goes_to_the_side_that_just_won_the_hand():
    assert match_winner({"us": 500, "them": 500}, tie_break="us") == "us"
    assert match_winner({"us": 500, "them": 500}, tie_break="them") == "them"
    assert match_winner({"us": 500, "them": 500}) == "us", "default is the side that won the hand"


def test_the_target_is_five_hundred_points():
    assert MATCH_TARGET == 500
    assert match_winner({"us": 499, "them": 0}) is None
    assert match_winner({"us": 500, "them": 0}) == "us"
