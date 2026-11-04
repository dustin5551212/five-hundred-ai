# Five Hundred AI

A Python implementation of **500** — the 45-card trick-taking game — with a benchmark harness and
three agents to measure: a learned policy, a searcher, and the hand-tuned CPU from my browser game.

**Start here:** `pip install pytest && pytest -q` — 118 tests, no PyTorch needed, about 6 seconds.
The learned-policy and search sections below say what each one is worth, including the two
experiments that failed.

The game itself is complete: deal, auction, kitty exchange, ten tricks, contract scoring and a match
to 500 points. `python -m fivehundred --seed 7` plays one out at the terminal. Both halves are
measured too — card play in tricks, and bidding in points (see *Is the CPU any good at bidding?*).

This is the companion to a browser game I wrote — 4-player networked, zero dependencies, 112 tests
in its own suite, currently a private repo. That game ships a rule-based opponent in `src/cpu.js`.
This repo exists to find out how good it is, and to beat it.

## The result

The shipped heuristic is worth **+1.56 tricks a hand over random play**. Search beats it by
**+0.98**. The learned policy, on its own, does not beat it yet.

Its *bidding* is worth **+17 points a hand over never bidding**, and it beats a no-judgement 6♠
bidder by **+60 points a hand at a real table**. It also *looks* like it underbids: bidding at the
edge is worth **+10.68** points a hand in contract value. That gain **does not survive a defence
that fights** — at a table against a searching defence it is **+1.32**, with an interval 28 points
wide — so the shipped margin did not move. The full argument is in *Is the CPU any good at bidding?*

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
rather than the last decimal. `pytest` — **118 passed**.

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
.venv/bin/python -m pytest -q                                     # 118 passed

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

## Play against it in the browser

The searcher is worth +0.98 tricks a hand over the heuristic, which is more useful at a table than
in a table. `serve.py` puts it behind HTTP and `professional-500` seats it at the computer chairs:

```sh
# 1. the agent (stdlib only, no dependencies)
PYTHONPATH=src .venv/bin/python -m fivehundred.serve --port 5200 --worlds 24

# 2. the table, with the agent in the CPU seats
#    open http://localhost:5173/?ai=1        (or ?ai=http://host:port)
```

`GET /health` → `{"ok": true, "agent": "pimc", "worlds": 24}`. `POST /move` takes one view and
returns one card id:

```jsonc
{"view": {"seat": "N", "contract": "S", "contractor": "W", "joker_suit": null,
          "hand": ["H-Q", "S-4"], "legal": ["H-Q", "S-4"],
          "trick": [["S", "H-4"]], "led_suit": "H",
          "played": [["S", "H-4"]], "tricks": [["NS", 0], ["WE", 0]]}}
-> {"card": "H-Q", "worlds": 24, "ms": 14.2}
```

The view is the same object `env.View` gives an agent: that seat's hand, the legal cards, the trick
in progress, every card played this hand, and the tricks so far. **Nothing else crosses the wire**,
so a sidecar bug cannot leak another seat's hand — there is nothing to leak.

Measured on this machine, 120 decisions against the heuristic's play:

| worlds | p50 | p95 | max |
|---|---|---|---|
| 8 | 2.6 ms | 16.8 ms | 24.4 ms |
| 24 | 6.8 ms | 49.7 ms | 65.9 ms |

The browser CPU's think time is 620 ms, so there is roughly 10× headroom at 24 worlds.

**Two failure modes the wiring exists to absorb.** A dead sidecar must not stall a table: the page
falls back to its own heuristic on any error, timeout, 500, malformed body, or a card that is not
legal for the seat. And the POST is cross-origin (the table is on 5173, the agent on 5200, and over a
tunnel the table's origin is the tunnel's), so it is always preflighted — without `OPTIONS` handling
and `Access-Control-Allow-Origin` the browser blocks it and the table falls back *silently*, which
looks exactly like "the agent is not working". Both paths have tests: `tests/test_serve.py` covers
the preflight and the validation, `src/ai/agent.test.js` in `professional-500` covers the fallbacks.

## Layout

```
src/fivehundred/cards.py      Card, the 45-card pack, seeded round-robin deal
src/fivehundred/rules.py      trump order, bowers, effective suit, legal plays, trick winner
src/fivehundred/bid.py        contracts, the 27-bid ladder, nullo and open nullo, labels
src/fivehundred/auction.py    the bidding round: turn order, legality, contract or thrown in
src/fivehundred/scoring.py    contract made/defeated, slams, nullo, match to 500
src/fivehundred/game.py       a whole hand (kitty exchange included) and a whole match
src/fivehundred/serve.py      the search agent behind HTTP, for the browser table
src/fivehundred/env.py        play-phase simulator; resumable mid-trick
src/fivehundred/agents.py     random / lowest-card / the shipped heuristic (ported from cpu.js)
src/fivehundred/features.py   148-number encoding of a seat's view (no peeking)
src/fivehundred/policy.py     the MLP, masked logits, and the agent that plays it
src/fivehundred/train.py      behaviour cloning, then self-play REINFORCE
src/fivehundred/search.py     PIMC: sampled worlds, common random numbers, rollout
src/fivehundred/benchmark.py  paired head-to-head with bootstrap intervals
scripts/diagnose_bc.py        where the cloned policy disagrees with its teacher, by decision type
scripts/dump_js_parity.mjs    asks the browser game the same questions, for the parity fixture
tests/                       118 tests — rules, bidding, auction, scoring, a whole match, the bidding
                              benchmark and its defence harness, search, and the parity suite
```

## Is the CPU any good at *bidding*?

The play benchmark measures card play with the contract fixed. That leaves the other half of the
game unmeasured, so `bidbench.py` measures it — two ways, both paired by deal, both with card play
held fixed at the shipped heuristic so the only variable is the auction.

**Contract value**: every bidder is shown the same hand, picks a contract, and plays it out. Reported
as points per hand for the bidding side, with the difference against the shipped bidder and a
bootstrap interval over the 2,000 deals (seed 21).

| bidder | points/hand | 95% CI | vs shipped | 95% CI | conceded | made | passed |
|---|---|---|---|---|---|---|---|
| pass (never bids) | 0.00 | — | −17.12 | [−19.13, −15.15] | 0.00 | — | 100% |
| always 6♠, no judgement | +7.75 | [+5.58, +9.92] | −9.38 | [−11.97, −6.77] | 42.28 | 55.4% | 0% |
| **shipped CPU** | **+17.12** | [+15.10, +19.23] | — | — | 6.22 | 87.8% | 74.9% |
| shipped, safety margin removed | +21.27 | [+19.09, +23.43] | +4.14 | [+3.13, +5.19] | 8.34 | 86.9% | 67.9% |
| shipped, bids one trick higher | +35.22 | [+32.24, +38.09] | +18.09 | [+15.68, +20.59] | 25.29 | 77.2% | 22.1% |

**The auction**: two systems sit opposite each other, and each deal is played twice with the systems
swapping sides, so what differs is only who sits where. This counts *both* sides' points, which is
what a match actually does — a bid you fail pays the defenders ten a trick.

| matchup | A pts/hand | B pts/hand | A − B | 95% CI | A held the contract |
|---|---|---|---|---|---|
| shipped vs pass | +67.33 | +24.67 | **+42.66** | [+39.09, +46.11] | 49.9% |
| shipped vs always 6♠ | +105.97 | +45.72 | **+60.25** | [+55.94, +64.17] | 47.3% |
| shipped vs shipped (control) | +81.80 | +81.80 | 0.00 | exactly 0 | 42.0% |
| no safety margin vs shipped | +90.72 | +84.32 | **+6.40** | [+4.11, +8.72] | 50.7% |
| one trick higher vs shipped | +111.05 | +85.75 | **+25.30** | [+18.21, +32.05] | 83.5% |

The control row is the harness's own honesty check: identical bidders, sides swapped, must come out
at exactly zero. If it ever doesn't, none of the other rows mean anything.

**What the numbers say.** The shipped bidder is worth about **+17 points a hand over never bidding**,
and at a real table it beats a no-judgement 6♠ bidder by **+60 points a hand** — so it is doing real
work. But it **underbids**: removing its safety margin is worth another +4 points a hand, and letting
it bid one trick higher is worth **+18 points a hand** in contract value and **+25 at a real table**.
Both intervals clear zero, on 2,000 deals.

One caveat that turned into the next section: in every number above, the *defence* is the shipped
heuristic — a side that plays each trick on its own merits and does not go looking for a contract to
break. Seating a defence that searches changes the answer for the aggressive arm, so read the +25
with the retune below.

Two caveats, because the first draft of this section overstated it. The aggressive arms bid far more
often (83.5% of contracts held against 42%) and concede far more to the defenders (25.3 points a hand
against 6.2); the net column still favours them, but the edge is "bid more, and mostly make it",
not a better auction by any subtle measure. And card play is held fixed, so a defence that punishes
overbidding — which the shipped heuristic does not try to do — would eat into that margin.

`python -m fivehundred.bidbench --deals 2000 --json benchmarks/bidding.json` reproduces every number
above in about 27 seconds.

## Does the margin survive a defence that tries?

Every number above was measured against the shipped heuristic defending — a side that plays each trick
on its own merits and never goes looking for a contract to break. That flatters overbidding, so the
same sweep was run twice: once against that defence, once against the searcher from `search.py`
(`PIMCAgent`, 12 sampled worlds) seated at the defending side. The contractor's own side stays on the
shipped heuristic, so the bidder is still the only thing being measured — only the quality of the
opposition changes.

250 deals, seed 21, both defences on the identical deals, and the same search seed per deal for every
arm, so the difference between arms is the bid and nothing else:

| safety margin | points/hand, weak defence | points/hand, searching defence | vs shipped (searching) | 95% CI | made w/s | passed |
|---|---|---|---|---|---|---|
| +0.6 (very timid) | +11.96 | +11.48 | −8.80 | [−13.32, −4.76] | 100 / 97% | 86.4% |
| **+0.2 (shipped)** | +22.24 | +20.28 | — | — | 94 / 93% | 72.0% |
| 0.0 | +25.72 | +24.24 | **+3.96** | [+0.80, +7.32] | 90 / 89% | 64.0% |
| −0.5 | +32.12 | +30.96 | **+10.68** | [+5.68, +15.76] | 83 / 83% | 43.2% |
| −1.0 (one trick higher) | +35.44 | +32.36 | **+12.08** | [+5.52, +18.96] | 78 / 76% | 23.2% |
| −1.5 | +35.96 | +31.60 | +11.32 | [+3.40, +19.32] | 73 / 70% | 7.6% |

And at a real table, both sides' points counted. Every cell below is one setting — 250 paired deals,
seed 21, sides swapped — so the only thing varying across a row is the bid margin. The
heuristic-defence column was re-measured with the edge arm; the earlier version of it did not
reproduce and has been replaced rather than kept as decoration.

| matchup | A − B, searching defence | 95% CI | A − B, heuristic defence | 95% CI |
|---|---|---|---|---|
| shipped vs shipped (control) | 0.00 | exactly 0 | 0.00 | exactly 0 |
| no safety margin vs shipped | **+3.68** | [−2.08, +9.16] | +8.32 | [+3.96, +12.76] |
| bids at the edge (−0.5) vs shipped | **+1.32** | [−12.76, +16.00] | +12.72 | [+0.00, +25.48] |
| one trick higher vs shipped | **−7.24** | [−29.08, +14.48] | +16.76 | [−3.20, +36.32] |

Read the two columns together and the whole finding falls out: **aggressive bidding pays only while
the defenders are not trying.** Against the shipped heuristic every aggressive arm gains; against a
defence that searches, the edge arm collapses to +1.32 and one-trick-higher goes negative. The
point estimate does not even order the arms the same way in both columns, which is what noise looks
like when it is mistaken for a result.

**What survives, and what doesn't.** The searching defence costs every arm points, and it costs the
aggressive ones more (−0.5 at the timid end, −4.4 at the far end) — the shape you would expect if the
defence is doing its job. Three verdicts come out of the retune:

- **"Remove the safety margin" survives.** Bidding at the edge is worth **+3.96** a hand in contract
  value with the interval clear of zero, and **+10.68** if you take the −0.5 setting, which is the
  best arm on both defences. At a real table the edge shrinks to +3.68 and its interval now includes
  zero — real, but no longer certain.
- **"Bid a trick higher" does not.** Against the heuristic defence it looked worth **+25.30** a hand
  at the table; against a defence that searches it is **−7.24**, interval spanning zero. That arm was
  not finding better contracts, it was farming a defence that does not fight, and 83% of the hands it
  won the auction for were hands it then had to make.

So the honest version of the first finding is narrower than "the CPU underbids by a trick" — and, now
that the edge arm has actually been played at a table, narrower again:

- **"Remove the safety margin" is real in contract value and unproven at a table.** At 0.0 it is
  **+3.96** a hand with the interval clear of zero; at −0.5 it is **+10.68**, the best arm against
  both defences. But a table pays the defenders ten a trick for every contract you fail, and there
  the same arms read **+3.68** [−2.08, +9.16] and **+1.32** [−12.76, +16.00]. The −0.5 arm has the
  higher contract value *and* the worse table row, which is the clearest possible statement that the
  two metrics are not interchangeable.
- **"Bid a trick higher" does not.** Against the heuristic defence it looked worth **+16 to +25** a
  hand at the table; against a defence that searches it is **−7.24**, interval spanning zero. That
  arm was not finding better contracts, it was farming a defence that does not fight, and 83% of the
  hands it won the auction for were hands it then had to make.
- **So no margin ships.** The shipped 0.2 stays, and a test pins it with this reasoning attached, so
  moving it later has to be a deliberate act backed by a table-level result rather than a
  contract-value one. Reopening the question is cheap: `edge_bidder` and the rest of the arms are
  still wired into the harness.

The gap the earlier version of this section reported — "about 10 points a hand, about 4 at a table" —
was written before the edge arm had ever defended against anything but the heuristic. It was the
contract-value number wearing a table's clothes. What the evidence supports is smaller and duller:
**the CPU bids conservatively, and conservative bidding costs little against opposition that is
actually trying to beat it.**

Caveats worth keeping: 250 deals rather than 2,000, because the searcher costs ~120 ms a hand; PIMC-12
is a competent defender but its rollouts still use the shipped heuristic as the base policy; and it
defends every hand, which a real pair would not manage for a whole session. The sweep reproduces with
`python -m fivehundred.bidbench --sweep --sweep-deals 250 --json benchmarks/bidding-sweep.json`.

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
3. ~~**Auction + kitty**~~ — done: `bid.py`, `auction.py`, `game.py`, and `bidbench.py` to measure
   the result. The benchmark's first finding is that the shipped CPU underbids by about a trick.
4. ~~**Ship it**~~ — done for search: `serve.py` exposes `/move` over HTTP and the browser game seats
   it with `?ai=1`, or `?ai=http://host:port` for the served agent, falling back to the built-in CPU
   if the agent offers an illegal card. Still open for a *learned* policy, which has nothing worth
   exporting yet.
