"""The port checked against the original, not against my reading of it.

`tests/fixtures/js_parity.json` is produced by `scripts/dump_js_parity.mjs`, which
imports the browser game's own modules and answers a matrix of questions: every
bid's value/rank/label, the legal-bid ladder after eight different previous
bids, six scripted auctions, 594 scoring cases and eight match-end cases.

If this file fails, the Python game and the game people actually play have
drifted apart, and any number measured with the Python engine is measuring a
different game.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fivehundred.auction import Auction
from fivehundred.bid import Bid, contract_label, contract_short, legal_bids
from fivehundred.scoring import match_winner, score_contract

FIXTURE = Path(__file__).parent / "fixtures" / "js_parity.json"


@pytest.fixture(scope="module")
def reference():
    return json.loads(FIXTURE.read_text())


def pairs(bids) -> list[tuple[int, str]]:
    return [(b.tricks, b.denom) for b in bids]


def test_every_bid_has_the_same_value_rank_and_label(reference):
    mismatches = []
    for entry in reference["bids"]:
        bid = Bid(entry["tricks"], entry["denom"])
        got = {
            "value": bid.value,
            "rank": bid.rank,
            "label": contract_label(bid),
            "short": contract_short(bid),
        }
        want = {k: entry[k] for k in got}
        if got != want:
            mismatches.append((entry["denom"], entry["tricks"], want, got))
    assert not mismatches, f"{len(mismatches)} bids differ, first: {mismatches[:3]}"


def test_the_ladder_of_legal_bids_matches_after_every_previous_bid(reference):
    for case in reference["legal"]:
        previous = Bid(*case["previous"]) if case["previous"] else None
        assert pairs(legal_bids(previous)) == [tuple(b) for b in case["bids"]], (
            f"legal bids after {case['previous']} differ"
        )


def test_scripted_auctions_end_the_same_way(reference):
    for case in reference["auctions"]:
        auction = Auction(first=case["first"])
        turns = []
        for action in case["actions"]:
            if auction.is_over:
                break
            turns.append(auction.turn)
            if action == "pass":
                auction.pass_turn(auction.turn)
            else:
                _, tricks, denom = action.split(":")
                auction.place(auction.turn, Bid(int(tricks), denom))

        assert turns == case["turns"], f"turn order differs in {case['actions']}"
        assert auction.is_over == case["finished"]

        result = auction.result
        want = case["result"]
        assert result is not None
        assert result.type == want["type"]
        assert result.seat == want["seat"]
        assert result.team == want["team"]
        got_contract = (result.contract.tricks, result.contract.denom) if result.contract else None
        assert got_contract == (tuple(want["contract"]) if want["contract"] else None)

        notes = {
            seat: (value if isinstance(value, str) else [value.tricks, value.denom])
            for seat, value in auction.notes().items()
        }
        assert notes == case["notes"], f"notes differ in {case['actions']}"


def test_every_scoring_case_pays_out_the_same(reference):
    mismatches = []
    for case in reference["scores"]:
        contract = Bid(*case["contract"])
        result = score_contract(contract, case["contractorTeam"], case["tricks"])
        got = {
            "made": result.made,
            "contractorTricks": result.contractor_tricks,
            "opponentTricks": result.opponent_tricks,
            "value": result.value,
            "contractorPoints": result.contractor_points,
            "opponentPoints": result.opponent_points,
            "note": result.note,
        }
        if got != case["result"]:
            mismatches.append((case["contract"], case["contractorTeam"], case["tricks"], case["result"], got))
    assert not mismatches, f"{len(mismatches)} of {len(reference['scores'])} cases differ, first: {mismatches[:2]}"


def test_the_match_ends_with_the_same_winner(reference):
    for case in reference["matches"]:
        got = match_winner(case["totals"], target=500, tie_break=case["tieBreak"])
        assert got == case["winner"], f"totals {case['totals']} tie-break {case['tieBreak']}"


def test_the_fixture_is_not_empty(reference):
    """A fixture that quietly shrank would make every test above vacuous."""
    assert len(reference["bids"]) == 27
    assert len(reference["legal"]) >= 6
    assert len(reference["scores"]) > 500
    assert len(reference["matches"]) >= 6
