"""Train the policy: imitate the heuristic, then improve on it by playing.

    python -m fivehundred.train bc      --hands 3000 --epochs 12
    python -m fivehundred.train selfplay --hands 30000 --init models/bc.pt

Two stages, because a randomly initialised policy in a 45-way action space
learns nothing from sparse hand outcomes:

1. **Behaviour cloning.** Record the shipped heuristic's decisions over a few
   thousand deals and fit the network to them by cross-entropy. This gives a
   policy that plays about as well as the heuristic — the starting point, not
   the result.
2. **Self-play REINFORCE.** The learner takes the north–south seats against the
   heuristic on west–east, and the return is the side's trick share,
   `(tricks - 5) / 5`, with a running baseline to cut variance. The gradient
   pushes up the log-probability of decisions made in hands that went well.

The only result that counts is the paired benchmark in `benchmark.py` run with
`--a learned:models/selfplay.pt --b heuristic`, where the interval on the
difference must exclude zero.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

from .agents import HeuristicAgent
from .benchmark import BENCH_CONTRACTS
from .cards import Card, deal
from .env import SEATS, TEAM_OF, View, play_hand
from .features import encode, legal_mask
from .policy import LearnedAgent, PolicyNet, masked_logits, save, load


class Recorder(HeuristicAgent):
    """The heuristic, plus a note of every decision it made."""

    def __init__(self) -> None:
        self.samples: list[tuple[list[float], list[bool], int]] = []

    def choose(self, view: View) -> Card:
        card = super().choose(view)
        self.samples.append((encode(view), legal_mask(view), _legal_index(view, card)))
        return card


def _legal_index(view: View, card: Card) -> int:
    from .features import index_of

    return index_of(card)


def _tensors(samples: list[tuple[list[float], list[bool], int]], device: str):
    x = torch.tensor([s[0] for s in samples], dtype=torch.float32, device=device)
    mask = torch.tensor([s[1] for s in samples], dtype=torch.bool, device=device)
    target = torch.tensor([s[2] for s in samples], dtype=torch.long, device=device)
    return x, mask, target


def collect_behaviour(n_hands: int, seed: int) -> list[list[tuple[list[float], list[bool], int]]]:
    """One sample list per hand, so a validation split can be made by hand.

    Splitting by *decision* would leak: decisions inside one hand are correlated
    (same cards, same opponents), so a decision-level split reports accuracy the
    model does not have on unseen deals.
    """
    rng = random.Random(seed)
    per_hand: list[list[tuple[list[float], list[bool], int]]] = []
    for i in range(n_hands):
        rec = Recorder()
        hands, _ = deal(rng)
        contract = BENCH_CONTRACTS[i % len(BENCH_CONTRACTS)]
        contractor = SEATS[i % 4]
        play_hand({seat: hands[j] for j, seat in enumerate(SEATS)}, contract, contractor,
                  {seat: rec for seat in SEATS})
        per_hand.append(rec.samples)
    return per_hand


def train_bc(n_hands: int, epochs: int, out: Path, device: str, seed: int = 1) -> PolicyNet:
    torch.manual_seed(seed)
    per_hand = collect_behaviour(n_hands, seed)
    n_val = max(1, len(per_hand) // 10)
    train_samples = [s for hand in per_hand[n_val:] for s in hand]
    val_samples = [s for hand in per_hand[:n_val] for s in hand]
    print(f"behaviour cloning: {len(train_samples)} train decisions, "
          f"{len(val_samples)} held-out from {n_val} unseen hands")

    x, mask, target = _tensors(train_samples, device)
    vx, vmask, vtarget = _tensors(val_samples, device)

    net = PolicyNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(train_samples)
    best_val, best_state = -1.0, None
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, 512):
            idx = perm[start:start + 512]
            logp = torch.log_softmax(masked_logits(net(x[idx]), mask[idx]), dim=-1)
            loss = -logp.gather(1, target[idx].unsqueeze(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()) * len(idx)
        val_acc = _accuracy(net, vx, vmask, vtarget)
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            flag = "  *"
        print(f"  epoch {epoch + 1:2d}  train ce {total / n:.4f}  held-out agreement {100 * val_acc:.1f}%{flag}")

    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"kept the best epoch: {100 * best_val:.1f}% held-out agreement")
    save(net, out)
    print(f"wrote {out}")
    return net


def _accuracy(net: PolicyNet, x, mask, target) -> float:
    with torch.no_grad():
        correct = 0
        for start in range(0, len(target), 4096):
            sl = slice(start, start + 4096)
            pred = masked_logits(net(x[sl]), mask[sl]).argmax(dim=-1)
            correct += int((pred == target[sl]).sum())
    return correct / len(target)


def train_selfplay(n_hands: int, init: Path | None, out: Path, device: str,
                   batch_hands: int = 128, lr: float = 3e-4, seed: int = 2,
                   entropy_beta: float = 0.01) -> PolicyNet:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = load(init, device) if init else PolicyNet()
    net = net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    baseline = 0.0
    done = 0
    while done < n_hands:
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        for _ in range(batch_hands):
            hands, _ = deal(rng)
            contract = BENCH_CONTRACTS[done % len(BENCH_CONTRACTS)]
            contractor = SEATS[done % 4]
            agents = {seat: (LearnedAgent(net, device, sample=True, trace=True)
                             if TEAM_OF[seat] == "NS" else HeuristicAgent())
                      for seat in SEATS}
            out_come = play_hand({seat: hands[j] for j, seat in enumerate(SEATS)},
                                 contract, contractor, agents)
            reward = (out_come.tricks["NS"] - 5) / 5.0
            # every decision the learner made this hand, from both its seats
            xs, masks, idxs, logps = [], [], [], []
            for seat in ("S", "N"):
                for x, mask, idx, logp in agents[seat].trace:
                    xs.append(x); masks.append(mask); idxs.append(idx); logps.append(logp)
            if not xs:
                continue
            batch.append((
                torch.cat(xs), torch.cat(masks),
                torch.tensor(idxs, device=device), torch.tensor(logps, device=device,
                                                                 dtype=torch.float32),
                reward,
            ))
            done += 1
            baseline = 0.0 if done == 1 else 0.99 * baseline + 0.01 * reward

        losses = []
        for x, mask, idx, logp_old, reward in batch:
            logp = torch.log_softmax(masked_logits(net(x), mask), dim=-1)
            picked = logp.gather(1, idx.unsqueeze(1)).squeeze(1)
            advantage = reward - baseline
            entropy = -(logp.exp() * logp).sum(dim=-1).mean()
            loss = -(picked.sum() * advantage) / len(idx) - entropy_beta * entropy
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        if (done // batch_hands) % 10 == 0:
            print(f"  hands {done:6d}  mean reward {baseline:+.3f}  loss {sum(losses)/len(losses):+.4f}")

    save(net, out)
    print(f"wrote {out}")
    return net


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["bc", "selfplay"])
    p.add_argument("--hands", type=int, default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--init", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    print(f"device: {args.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if args.device.startswith("cuda") else ""))
    if args.stage == "bc":
        hands = args.hands or 3000
        train_bc(hands, args.epochs, args.out or Path("models/bc.pt"), args.device, args.seed)
    else:
        hands = args.hands or 30000
        train_selfplay(hands, args.init, args.out or Path("models/selfplay.pt"), args.device, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
