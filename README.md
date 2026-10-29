# Five Hundred AI

A Python implementation of the play phase of **500** — the 45-card trick-taking game — plus the
benchmark harness for beating its hand-tuned CPU.

This is the companion to my browser game ([professional-500](https://github.com/dustin5551212/professional-500),
zero dependencies, 112 tests, 4-player networked). The game ships a rule-based opponent in
`src/cpu.js`. That opponent is the thing to beat, and this repo is how "beat" gets measured.

## Status

| milestone | state |
|---|---|
| Faithful rules port (deck, trump, bowers, follow-suit, trick resolution) | **done** — 15 tests |
| Play-phase simulator (seats, teams, turn order, Nullo trio) | **done** — 12 tests |
| Baselines (random, lowest-card, the shipped heuristic) | **done** |
| Paired head-to-head benchmark with bootstrap intervals | **done** |
| Auction + kitty exchange | not started |
| Learned agent (self-play) | not started |
| Ship the agent back into the browser game | not started |

`pytest` — **27 passed**.

## Baseline results

2,000 deals, seed 7, every deal played twice with the sides swapped (see *Benchmark design*).
`A − B` is the mean tricks-per-hand advantage; the interval is a percentile bootstrap over deals.

| matchup | A tricks | B tricks | A − B | 95% CI | A wins | tied |
|---|---|---|---|---|---|---|
| heuristic vs random | 5.78 | 4.22 | **+1.56** | [+1.50, +1.62] | 57.6% | 19.1% |
| heuristic vs lowest-card | 5.94 | 4.06 | **+1.88** | [+1.82, +1.94] | 62.9% | 17.5% |
| heuristic vs heuristic (control) | 5.00 | 5.00 | **+0.00** | [+0.00, +0.00] | 40.0% | 21.5% |

By contract (A − B): clubs +1.51, diamonds +1.49, hearts +1.66, spades +1.45, no trumps +1.70.

So the opponent a learned agent has to clear is worth **+1.56 tricks a hand over random play**, and
roughly **19% of hands are drawn 5–5** — which is why the metric below is tricks, not wins.

Throughput: ~600 hands/second on one core, so a million-hand training run is about 30 minutes of
CPU. The whole comparison above takes 6.6 seconds.

## Benchmark design

Two mistakes are easy to make here, and this harness made both before it was fixed:

1. **Deal luck.** Dealt one way, each deal gives a fixed pair of hands to each side, so a 2,000-deal
   run is one sample of 2,000 card distributions. Two runs that use different hands measure the
   cards as much as the players. Fix: play every deal **twice**, once with A on the north–south
   seats and once with B there, on the identical deal, contract and contractor. The pair is the
   unit of analysis, so hand strength cancels inside it.
2. **Ties counted as losses.** Ten tricks split 5–5 in ~19% of hands. A win-rate column that folds
   ties into losses understates both sides and does not mirror when the seats swap. Fix: report
   mean tricks with a bootstrap interval, and publish the tie rate next to the win rate.

The `heuristic vs heuristic` row is the harness's own control: with the sides swapped and the pairs
cancelled, it must come out at exactly 0.00. If it ever doesn't, every other number here is suspect.

The benchmark samples the five non-Nullo denominations and rotates the contractor through all four
seats, so each side leads a quarter of the hands.

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

## Scope, stated plainly

This plays the cards **after** the contract is settled. There is no auction, no kitty exchange and
no partner's gift yet, so a contractor here still holds ten cards rather than sixteen. Nullo is
implemented in the engine (and tested) but excluded from the trick-count average, because a
3-card-trick game would be averaged against a 4-card one. Those are the next two pieces of work,
and the reasons are in the roadmap rather than hidden.

## Run it

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -q                                  # 27 passed

PYTHONPATH=src .venv/bin/python -m fivehundred.benchmark --deals 2000 --seed 7 \
    --json benchmarks/baseline.json
```

## Layout

```
src/fivehundred/cards.py      Card, the 45-card pack, seeded round-robin deal
src/fivehundred/rules.py      trump order, bowers, effective suit, legal plays, trick winner
src/fivehundred/env.py        play-phase simulator: seats, teams, turn order, Nullo trio
src/fivehundred/agents.py     random / lowest-card / the shipped heuristic (ported from cpu.js)
src/fivehundred/benchmark.py  paired head-to-head with bootstrap intervals
tests/test_rules.py           15 tests — bowers, follow-suit, NT + Nullo joker rules
tests/test_env.py             12 tests — card accounting, Nullo trio, illegal-play refusal, controls
```

## Roadmap

1. **Auction + kitty** — 15 cards to the contractor, six discards (five plus the partner's gift),
   bid legality and the Nullo variant, so the engine is the whole game rather than its second half.
2. **Nullo benchmark** — its own metric (contract made / defeated), not a trick average.
3. **Learned agent** — starting from the heuristic as a demonstration policy, then self-play
   improvement, judged only by head-to-head `A − B` against it with the interval excluding zero.
4. **Ship it** — export the trained policy and let it take a seat in the browser game, so the agent
   is playable at the table rather than a number in a table.
