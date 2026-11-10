"""Re-run the agent matchups and store them in benchmarks/agents.json.

Why this exists: the baseline matchups were always written to JSON by
`fivehundred.benchmark`, but the agent rows (cloning, self-play, the two
searches) were run ad hoc in a REPL and only ever landed in the README. That is
how a stale -0.58 survived in six documents while the README said -0.45.

Run from the package root:

    PYTHONPATH=src .venv/bin/python scripts/store_agents.py

Takes a few minutes: the 24-world search dominates.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fivehundred.benchmark import run

SEED = 21
OUT = Path("benchmarks/agents.json")

MATCHUPS = [
    ("behaviour cloning", "learned:models/bc.pt", "heuristic", 400),
    ("self-play REINFORCE", "learned:models/selfplay.pt", "heuristic", 400),
    ("PIMC search, 8 worlds", "pimc:8", "heuristic", 600),
    ("PIMC search, 24 worlds", "pimc:24", "heuristic", 300),
]


def main() -> int:
    rows = []
    for label, a, b, deals in MATCHUPS:
        print(f"running {label} over {deals} deals…", flush=True)
        r = run(label, a, b, deals, SEED)
        rows.append(asdict(r) | {"ci95": list(r.ci95)})
        print(f"  {label}: {r.diff:+.2f}  [{r.ci95[0]:+.2f}, {r.ci95[1]:+.2f}]", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "results": rows}, indent=2) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
