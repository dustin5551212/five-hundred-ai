"""What a completed hand pays, and who has won the match.

Ported from the browser game's `src/scoring.js`.

A suit or no-trump contract is made when the contractors take at least the
tricks they bid: they score the contract's value, or minus it if they fail.
Either way the defenders take ten a trick they won.

Taking every trick is a slam — worth 250 if the contract was worth less than
that, never more than a bid that was already worth more. Nullo is all or
nothing and the defenders score nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bid import Bid, bid_value, is_nullo

TRICK_POINTS = 10
MATCH_TARGET = 500
SLAM_VALUE = 250
TOTAL_TRICKS = 10

US = "us"
THEM = "them"
TEAMS: tuple[str, str] = (US, THEM)


@dataclass(frozen=True, slots=True)
class HandScore:
    """One hand's result, both sides' points separated out."""

    made: bool
    contractor_team: str
    contractor_tricks: int
    opponent_tricks: int
    value: int
    contractor_points: int
    opponent_points: int
    note: str

    @property
    def opponent_team(self) -> str:
        return THEM if self.contractor_team == US else US


def score_contract(
    contract: Bid,
    contractor_team: str,
    tricks_by_team: dict[str, int],
) -> HandScore:
    """Score one completed hand.

    `tricks_by_team` is {"us": n, "them": m} — the contractor's side and the
    defenders, in the seats' own terms rather than the contractor's.
    """
    if contractor_team not in TEAMS:
        raise ValueError(f"Unknown team: {contractor_team}")
    opponent_team = THEM if contractor_team == US else US
    made_tricks = int(tricks_by_team.get(contractor_team, 0))
    conceded = int(tricks_by_team.get(opponent_team, 0))

    if is_nullo(contract.denom):
        success = made_tricks == 0
        value = bid_value(contract)
        return HandScore(
            made=success,
            contractor_team=contractor_team,
            contractor_tricks=made_tricks,
            opponent_tricks=conceded,
            value=value,
            contractor_points=value if success else -value,
            opponent_points=0,
            note="Nullo made" if success else "Nullo defeated",
        )

    success = made_tricks >= contract.tricks
    value = bid_value(contract)
    note = "Contract made" if success else "Contract defeated"

    if success and made_tricks == TOTAL_TRICKS and value < SLAM_VALUE:
        value = SLAM_VALUE
        note = "Slam"

    return HandScore(
        made=success,
        contractor_team=contractor_team,
        contractor_tricks=made_tricks,
        opponent_tricks=conceded,
        value=value,
        contractor_points=value if success else -value,
        opponent_points=conceded * TRICK_POINTS,
        note=note,
    )


def apply_score(totals: dict[str, int], result: HandScore) -> dict[str, int]:
    """Apply a hand's score to the running match totals, returning a new dict."""
    next_totals = {US: int(totals.get(US, 0)), THEM: int(totals.get(THEM, 0))}
    next_totals[result.contractor_team] += result.contractor_points
    next_totals[result.opponent_team] += result.opponent_points
    return next_totals


def match_winner(
    totals: dict[str, int],
    target: int = MATCH_TARGET,
    tie_break: str = US,
) -> str | None:
    """Who has won, if anyone.

    First past the target wins; if a hand carries both sides past it, the
    higher total takes it, and a genuine tie goes to `tie_break` — by default
    the side that has just won the hand.
    """
    us = int(totals.get(US, 0))
    them = int(totals.get(THEM, 0))
    us_in = us >= target
    them_in = them >= target

    if not us_in and not them_in:
        return None
    if us_in and them_in and us == them:
        return THEM if tie_break == THEM else US
    return US if us > them else THEM
