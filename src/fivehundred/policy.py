"""The policy network, and an agent that plays it.

A small MLP over `features.encode`: 148 -> 256 -> 256 -> 45 logits, one per
card, masked so an illegal card can never be chosen. 45 logits rather than
"one of the legal ones" keeps the output space fixed, which is what lets one
network play any position in any trick.

`sample=False` is how it plays in a benchmark (deterministic, always the best
legal card). `sample=True` is how it plays while training, and it records the
log-probability of each choice, because that is the number REINFORCE needs.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .cards import Card
from .env import View
from .features import FEATURE_DIM, N_CARDS, card_at, encode, legal_mask

MASK_VALUE = -1e9


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256, out_dim: int = N_CARDS) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, MASK_VALUE)


def to_tensors(view: View, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([encode(view)], dtype=torch.float32, device=device)
    mask = torch.tensor([legal_mask(view)], dtype=torch.bool, device=device)
    return x, mask


class LearnedAgent:
    """Plays the network. Optionally samples, and then remembers what it did."""

    def __init__(self, net: PolicyNet, device: str = "cpu", sample: bool = False, trace: bool = False) -> None:
        self.net = net.to(device).eval()
        self.device = device
        self.sample = sample
        self.trace: list[tuple[torch.Tensor, torch.Tensor, int, float]] = []
        self.record = trace

    def choose(self, view: View) -> Card:
        x, mask = to_tensors(view, self.device)
        with torch.no_grad():
            logits = masked_logits(self.net(x), mask)
            logp = torch.log_softmax(logits, dim=-1)
            if self.sample:
                idx = int(torch.multinomial(logp.exp(), 1).item())
            else:
                idx = int(logits.argmax(dim=-1).item())
        if self.record:
            self.trace.append((x, mask, idx, float(logp[0, idx].item())))
        return card_at(idx)

    def reset_trace(self) -> None:
        self.trace = []


def save(net: PolicyNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "config": {"hidden": net.body[0].out_features}}, path)


def load(path: Path | str, device: str = "cpu") -> PolicyNet:
    blob = torch.load(path, map_location=device, weights_only=True)
    net = PolicyNet(hidden=blob["config"]["hidden"])
    net.load_state_dict(blob["state_dict"])
    return net.eval()
