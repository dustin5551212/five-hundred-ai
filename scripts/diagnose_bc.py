"""Where does the cloned policy disagree with its teacher?

Held-out agreement plateaus near 90%. That 10% is not noise, so it is worth
knowing which decisions it sits in: leading, following, contracting, defending.
"""
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "src")
from fivehundred.agents import HeuristicAgent
from fivehundred.benchmark import BENCH_CONTRACTS, run
from fivehundred.cards import deal
from fivehundred.env import SEATS, play_hand
from fivehundred.features import encode, index_of, legal_mask
from fivehundred.policy import load, masked_logits


class Meta(HeuristicAgent):
    def __init__(self):
        self.samples = []

    def choose(self, view):
        card = super().choose(view)
        self.samples.append((
            encode(view), legal_mask(view), index_of(card),
            not view.trick,
            view.seat == view.contractor,
            view.contract,
        ))
        return card


rng = random.Random(777)
meta = Meta()
for i in range(500):
    hands, _ = deal(rng)
    play_hand({s: hands[j] for j, s in enumerate(SEATS)},
              BENCH_CONTRACTS[i % 5], SEATS[i % 4], {s: meta for s in SEATS})

net = load("models/bc.pt")
x = torch.tensor([s[0] for s in meta.samples])
m = torch.tensor([s[1] for s in meta.samples])
t = torch.tensor([s[2] for s in meta.samples])
with torch.no_grad():
    pred = masked_logits(net(x), m).argmax(-1)

groups = defaultdict(lambda: [0, 0])
for (s, p, target, lead, contractor, contract) in zip(
        meta.samples, pred.tolist(), t.tolist(),
        [s[3] for s in meta.samples], [s[4] for s in meta.samples],
        [s[5] for s in meta.samples]):
    for key in (("leading" if lead else "following"),
                ("contractor" if contractor else "defender"),
                contract):
        groups[key][0] += 1
        groups[key][1] += int(p == target)

print("disagreement by decision type (500 hands, 20,000 decisions):")
for key, (n, correct) in sorted(groups.items(), key=lambda kv: kv[1][1] / kv[1][0]):
    print(f"  {key:12s} n={n:6d}  agree {100 * correct / n:5.1f}%  disagree {100 * (1 - correct / n):5.1f}%")

r = run("bc", "learned:models/bc.pt", "heuristic", 400, 21)
print(f"\nbc vs heuristic: A {r.a_tricks:.3f}  B {r.b_tricks:.3f}  diff {r.diff:+.3f}  "
      f"CI [{r.ci95[0]:+.2f},{r.ci95[1]:+.2f}]")
