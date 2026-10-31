# Five Hundred AI

A Python implementation of **500** — the 45-card trick-taking game — with a benchmark harness and
three agents to measure: a learned policy, a searcher, and the hand-tuned CPU from my browser game.

The game itself is complete: deal, auction, kitty exchange, ten tricks, contract scoring and a match
to 500 points. `python -m fivehundred --seed 7` plays one out at the terminal. What is *not* yet
measured is bidding — the benchmark still compares agents on tricks taken, with the contract fixed.

This is the companion to [professional-500](https://github.com/dustin5551212/professional-500)
(zero dependencies, 112 tests, 4-player networked). That game ships a rule-based opponent in
`src/cpu.js`. This repo exists to find out how good it is, and to beat it.

## The result

The shipped heuristic is worth **+1.56 tricks a hand over random play**. Search beats it by
**+0.98**. The learned policy, on its own, does not beat it yet.

| agent | vs heuristic (tricks/hand) | 95% CI | deals | what it is |
|---|---|---|---|---|
| random | −1.56 | ±0.06 | 2,000 | the floor |
| lowest legal card | −1.88 | ±0.06 | 2,000 | greedy in the wrong direction |
| behaviour cloning | **−0.45** | [−0.57, −0.34] | 400 | learned; matches the heuristic on 89.9% of held-out decisions |
| self-play REINFORCE | **−1.04** | [−1.18, −0.90] | 400 | learned; *worse* than the cloning it started from |
| **PIMC search, 8 worlds** | **+0.49** | [+0.39, +0.59] | 600 | samples the hidden hands, plays each out |
| **PIMC search, 24 worlds** | **+0.98** | [+0.83, +1.13] | 300 | same, more samples |
| heuristic vs itself | 0.00 | exactly 0 | 2,000 | the harness's control |

Numbers use seeds 7 (baselines) and 21 (agents), on different deal counts, so compare intervals
rather than the last decimal. `pytest` — **36 passed**.

## Benchmark design

Two mistakes are easy to make here, and this harness made both before it was fixed:

1. **Deal luck.** Dealt one way, each deal gives a fixed pair of hands to each side, so a 2,000-deal
   run is one sample of 2,000 card distributions. Two runs that use different hands measure the
   cards as much as the players. Fix: play every deal **twice**, once with A on the north–south
   seats and once with B there, on the identical deal, contract and contractor. The pair is the unit
   of analysis, so hand strength cancels inside it.
2. **Ties counted as losses.** Ten tricks split 5–5 in ~19% of hands. A win-rate column that folds
   ties into losses understates both sides and does not mirror when the seats swap. The first
   version of this file reported 57.7% and 24.5% for the same two agents with their seats exchanged.
   That asymmetry was the tell. Fix: report mean tricks with a bootstrap interval, and publish the
   tie rate beside the win rate.

The `heuristic vs itself` row is the control: sides swapped and pairs cancelled, it must come out at
exactly 0.00. If it ever doesn't, no other number here is trustworthy.

The benchmark samples the five non-Nullo denominations and rotates the contractor through all four
seats, so each side leads a quarter of the hands. Throughput: ~600 hands/second on one core, so the
2,000-deal baseline comparison takes 6.6 seconds.

## The learned policy, honestly

`train.py bc` records the heuristic's decisions over 5,000 deals and fits a small MLP (148 features
→ 256 → 256 → 45 card logits, illegal cards masked). The validation split is made **by hand**, not
by decision: decisions inside one hand share cards and opponents, so a decision-level split would
report accuracy the model does not have on unseen deals. Training agreement and held-out agreement
are printed side by side for the same reason.

Held-out agreement plateaus at **89.9%** while training cross-entropy keeps falling to 0.049 and
training agreement reaches 93.8%. That gap is memorisation, and it is the reason the headline number
is the held-out one. More epochs do not help; the first version of this model (12 epochs, 3,000
hands) sat at 88.2% held out, and the extra training moved it one and a half points.

Disagreement is not spread evenly (`scripts/diagnose_bc.py`):

| decision type | agreement |
|---|---|
| **leading** (`n=5,000`) | **80.1%** |
| following (`n=15,000`) | 92.7% |
| as contractor | 88.6% |
| as defender | 89.8% |

Following a suit is nearly solved: the legal set is small and the rule is local. Choosing a **lead**
is where three quarters of the errors live, because the heuristic's rule — draw trumps with four or
more, otherwise lead low from the longest suit — depends on the *shape* of the hand, and a flat
multi-hot encodes which cards are held but not how they combine. That is a feature-engineering
problem before it is a learning problem, and it is the first thing to fix.

It still plays **0.45 tricks worse** than the teacher it copied, which is the honest cost of
imitation: a policy that agrees 90% of the time compounds the other 10% into positions its teacher
never visited.

`train.py selfplay` then takes the north–south seats against the heuristic on west–east and applies
REINFORCE to the side's trick share with a running baseline. Over 20,000 hands the mean reward
**did not move** (−0.07 to −0.12, flat), and the resulting policy is **worse than the cloning it
started from** — twice, from two different starting checkpoints (−0.95, then −1.04). That is a
negative result and it stays in the table.

Diagnosis, so the next attempt is not a guess: one gradient step per hand, with the advantage taken
against a 0.99-decay baseline, is an extremely noisy estimate. The whole twenty-decision trajectory
is credited with the same return, so credit is smeared across every decision and variance dominates
the update. The fix is not "more hands" — it is **batched updates with mean-normalised advantages**,
per-decision baselines from a critic, and PPO-style ratio clipping so one bad batch cannot walk the
policy away from a working one. Until that exists, the learned-policy row stays negative.

## Why search works, and why that is a different claim

A card player's real problem is that three hands are unknown. PIMC samples a *world* — a legal
assignment of the unseen cards to the other seats, consistent with everything played so far — plays
each candidate card out in that world with the heuristic, and averages the hand's outcome over many
worlds. Two details that matter:

- **Common random numbers.** Every candidate is tried in the *same* sampled worlds, because scoring
  candidates in different worlds compares the worlds as much as the cards. It is the same pairing
  idea as the benchmark, for the same reason.
- **The rollout policy is the heuristic**, including for the searcher's own future turns. It is not
  optimal; it is consistent, which is what makes the average mean something.

It scales the way Monte Carlo should: +0.49 at 8 worlds, +0.98 at 24. Cost is the honest limitation —
about 0.07 s per hand at 8 worlds, so a 300-deal paired comparison takes 3–4 minutes.

**This is search, not learning.** It gets its own row because "I wrote a search" and "I trained a
model" are different claims, and blending them into one impressive sentence would be dishonest.

## Rules fidelity

The rules are ported from the browser game's `src/trump.js`, which carries 26 tests, because the
parts that break home-grown 500 engines are subtle:

- **left bower** — the jack of the suit of the same colour (C↔S, D↔H) is *trump*, so follow-suit is
  judged on `effective_suit`, never on `card.suit`. With hearts trump, a diamond lead cannot be
  followed with the ♦J.
- **trump order** — joker > right bower > left bower > A K Q 10 9 8 7 6 5 4.
- **No Trumps / Nullo** — no trump suit, no bowers, jacks revert to printed rank. An unnominated
  joker belongs to no suit: highest card in the pack, playable only when void in the led suit.
- **Nullo** — a void player holding the joker *must* play it, the contractor's partner sits the hand
  out, and tricks are three cards.

`play_hand` can also **resume a hand mid-trick**, which is what search needs. A test plays a hand
straight through, then replays it from eight different cut points and requires the identical
outcome. It caught a real bug: resuming exactly on a trick boundary started the wrong seat leading.

## Scope, stated plainly

This plays the cards **after** the contract is settled. There is no auction, no kitty exchange and
no partner's gift, so a contractor here holds ten cards rather than sixteen. Nullo is implemented
and tested but excluded from the trick averages — a 3-card-trick game averaged against a 4-card one
would be a category error.

## Run it

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pytest numpy
.venv/bin/python -m pytest -q                                     # 36 passed

PYTHONPATH=src .venv/bin/python -m fivehundred.benchmark --deals 2000 --seed 7 \
    --json benchmarks/baseline.json

# the learned policy, then the searches
PYTHONPATH=src .venv/bin/python -m fivehundred.train bc --hands 3000 --epochs 12 --out models/bc.pt
PYTHONPATH=src .venv/bin/python -c "
from fivehundred.benchmark import run
print(run('pimc24', 'pimc:24', 'heuristic', 300, 21))"
```

PyTorch picks the GPU up as `cuda:0` under ROCm. On this machine `/dev/kfd` needs `render` group
membership, so training runs as `sg render -c '...'` in any session that predates the group change.

## Layout

```
src/fivehundred/cards.py      Card, the 45-card pack, seeded round-robin deal
src/fivehundred/rules.py      trump order, bowers, effective suit, legal plays, trick winner
src/fivehundred/bid.py        contracts, the 27-bid ladder, nullo and open nullo, labels
src/fivehundred/auction.py    the bidding round: turn order, legality, contract or thrown in
src/fivehundred/scoring.py    contract made/defeated, slams, nullo, match to 500
src/fivehundred/game.py       a whole hand (kitty exchange included) and a whole match
src/fivehundred/env.py        play-phase simulator; resumable mid-trick
src/fivehundred/agents.py     random / lowest-card / the shipped heuristic (ported from cpu.js)
src/fivehundred/features.py   148-number encoding of a seat's view (no peeking)
src/fivehundred/policy.py     the MLP, masked logits, and the agent that plays it
src/fivehundred/train.py      behaviour cloning, then self-play REINFORCE
src/fivehundred/search.py     PIMC: sampled worlds, common random numbers, rollout
src/fivehundred/benchmark.py  paired head-to-head with bootstrap intervals
scripts/diagnose_bc.py        where the cloned policy disagrees with its teacher, by decision type
scripts/dump_js_parity.mjs    asks the browser game the same questions, for the parity fixture
tests/                        81 tests — rules, bidding, auction, scoring, a whole match, search,
                              and the parity suite
```

## Is it the same game? Two proofs, not a promise

**The parity fixture.** `scripts/dump_js_parity.mjs` imports the browser game's own `bid.js`,
`auction.js` and `scoring.js` and answers 27 bids, eight legal-bid ladders, six scripted auctions,
594 scoring cases and eight match-end cases into `tests/fixtures/js_parity.json`. `tests/test_parity.py`
replays every one of those questions against the Python modules and demands identical answers. It is
committed, so `pytest` needs neither node nor the other repo; regenerate it if the JS rules change.
If that suite fails, the Python engine and the game people actually play have drifted apart.

**A whole match, end to end.** `tests/test_game.py` plays matches to 500 with four seats of the
shipped CPU: 45 cards dealt, ten each and a five-card kitty, the contractor back down to ten, ten
tricks, the contract scored, the dealer rotating. The tests assert the card accounting (45 in, 45
out, never a duplicate), that a passing table throws the hand in, that a Nullo contractor gets the
partner's lowest card, and that the scorecard equals the sum of its hands — which is the check that
the join did not double-count.

## Roadmap

1. **Fix the learner** — batched, mean-normalised advantages, a critic for per-decision baselines,
   PPO clipping. Target: beat the cloning policy first, then the heuristic. Anything less is not
   progress.
2. **Search with a learned prior** — let the policy bias world sampling and rollout choices, cutting
   the worlds needed for the same strength. That is where learning and search stop competing.
3. ~~**Auction + kitty**~~ — done: `bid.py`, `auction.py`, `game.py`. What remains is a *benchmark*
   for bidding: the shipped CPU's `choose_bid` is ported and a whole match runs, but no paired
   comparison yet says whether it is any good at choosing contracts, the way the play benchmark says
   what it is worth at choosing cards.
4. **Ship it** — export a policy and give it a seat in the browser game, so the agent is playable at
   the table instead of a row in a table.
