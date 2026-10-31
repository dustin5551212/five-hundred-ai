"""A searcher: determinized Monte Carlo over the hidden cards.

The problem a card player actually has is that three hands are unknown. The
old answer, and still a good one, is to guess: sample a *world* — a legal
assignment of the unseen cards to the other seats — play the candidate card
out in that world with a base policy, and average the result over many worlds.
That is perfect-information Monte Carlo (PIMC), and in trick-taking games it
reliably beats a greedy play policy because it looks at what the rest of the
hand is worth rather than at the trick in front of it.

Two details that matter:

- **Common random numbers.** Every candidate card is tried in the *same*
  sampled worlds. Scoring candidates in different worlds compares the worlds
  as much as the cards — the same mistake the benchmark made before it was
  paired.
- **The base policy rolls out all four seats**, including the searcher's own
  future turns. It is not a perfect player; it is a consistent one, which is
  what makes the average meaningful.

This is search, not learning. The learned policy is a separate agent
(`policy.LearnedAgent`), and the README reports both separately because they
are not the same claim.
"""

from __future__ import annotations

import random
from typing import Sequence

from .agents import HeuristicAgent
from .cards import Card, build_deck
from .env import SEATS, TEAM_OF, View, play_hand
from .rules import NULLO


class PIMCAgent:
    def __init__(self, worlds: int = 10, base: type = HeuristicAgent, seed: int = 0) -> None:
        self.worlds = worlds
        self.base = base
        self.rng = random.Random(seed)
        self._deck = build_deck()
        self._base_agents = {seat: base() for seat in SEATS}

    def choose(self, view: View) -> Card:
        legal = list(view.legal)
        if len(legal) == 1:
            return legal[0]
        if view.contract == NULLO:
            # Nullo is a 3-card-trick game with a partner sitting out; the
            # world-sampling arithmetic below assumes four seats.
            return self._base_agents[view.seat].choose(view)

        sampled = [self._sample_world(view) for _ in range(self.worlds)]
        team = TEAM_OF[view.seat]

        best_card, best_score = None, float("-inf")
        for card in legal:
            total = 0
            for world in sampled:
                hands = dict(world)
                hands[view.seat] = [c for c in view.hand if c.id != card.id]
                outcome = play_hand(
                    hands,
                    view.contract,
                    view.contractor,
                    self._base_agents,
                    trick=[*view.trick, (view.seat, card)],
                    tricks=dict(view.tricks),
                    log=[*view.played, (view.seat, card)],
                )
                total += outcome.tricks[team]
            score = total / len(sampled)
            if score > best_score:
                best_card, best_score = card, score
        return best_card or legal[0]

    def _sample_world(self, view: View) -> dict[str, Sequence[Card]]:
        """Deal the unseen cards into the other seats' remaining hand sizes."""
        mine = {c.id for c in view.hand}
        seen = {card.id for _seat, card in view.played}
        pool = [c for c in self._deck if c.id not in mine and c.id not in seen]

        played_by_seat: dict[str, int] = {seat: 0 for seat in SEATS}
        for seat, _card in view.played:
            played_by_seat[seat] += 1

        needs = {
            seat: 10 - played_by_seat[seat]
            for seat in SEATS
            if seat != view.seat
        }
        self.rng.shuffle(pool)
        world: dict[str, Sequence[Card]] = {}
        i = 0
        for seat, need in needs.items():
            world[seat] = pool[i:i + need]
            i += need
        return world
