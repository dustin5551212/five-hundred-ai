"""serve.py — the search agent behind HTTP, so the browser game can seat it.

The table in ``professional-500`` is JavaScript and runs the whole game in the
host page. This process answers exactly one question at a time — *given this
view, which card do I play?* — and it is deliberately stateless: it keeps
nothing between requests, and it knows no more than the view it is handed. Same
discipline the relay follows, for the same reason: a service that accumulates
state is a service you have to reason about on every failure.

Wire contract (JSON in, JSON out):

    GET  /health                     -> {"ok": true, "agent": "pimc", "worlds": 24}
    POST /move   {"view": {...}}     -> {"card": "S-A", "worlds": 24, "ms": 3.1}

A *view* is the fields of :class:`fivehundred.env.View`, with every card written
as its id (``"S-A"``, ``"JOKER"``) and every play as a two-element list:

    {
      "seat": "W", "contract": "H", "contractor": "S", "joker_suit": null,
      "hand": ["S-A", "H-K", ...],          # this seat's cards
      "legal": ["S-A", ...],               # what it may play right now
      "trick": [["S", "H-4"], ["W", "H-9"]],# the trick in progress
      "led_suit": "H",
      "played": [["S", "C-3"], ...],       # every card played this hand
      "tricks": [["NS", 3], ["WE", 2]]
    }

A malformed view is a 400 with ``{"error": …}``. The caller is expected to fall
back to its own heuristic on any non-200, timeout, or error — a table that
stalls because a sidecar died is worse than one that plays a slightly weaker
card, and ``professional-500`` does exactly that.

Run it::

    python -m fivehundred.serve --port 5200 --worlds 24

No third-party dependencies: this is ``http.server``, matching the rest of the
package. At 24 worlds a decision costs p50 ~7 ms and p95 ~50 ms on this machine,
against the browser CPU's 620 ms think time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence

from .cards import JOKER_SUIT, Card, build_deck
from .env import SEATS, TEAM_OF, View
from .search import PIMCAgent


class BadView(ValueError):
    """The payload was not a usable view. Always a 400, never a 500."""


_RANKS = {"4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
_SUITS = {"C", "D", "H", "S"}
_CONTRACTS = _SUITS | {"NT", "NULLO", "OPEN_NULLO"}


def card_from_id(card_id: Any) -> Card:
    """``"S-A"`` -> Card("S", "A"); ``"JOKER"`` -> the joker. Anything else is a bad view."""
    if not isinstance(card_id, str):
        raise BadView(f"card must be a string id, got {type(card_id).__name__}")
    if card_id == JOKER_SUIT:
        return Card(JOKER_SUIT, JOKER_SUIT)
    suit, _, rank = card_id.partition("-")
    if suit not in _SUITS or rank not in _RANKS:
        raise BadView(f"not a card id: {card_id!r}")
    return Card(suit, rank)


def _cards(raw: Any, field: str) -> tuple[Card, ...]:
    if not isinstance(raw, list):
        raise BadView(f"{field} must be a list of card ids")
    return tuple(card_from_id(c) for c in raw)


def _plays(raw: Any, field: str) -> tuple[tuple[str, Card], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BadView(f"{field} must be a list of [seat, card] pairs")
    out: list[tuple[str, Card]] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise BadView(f"{field} entries must be [seat, card] pairs")
        seat, card = entry
        if seat not in SEATS:
            raise BadView(f"unknown seat in {field}: {seat!r}")
        out.append((seat, card_from_id(card)))
    return tuple(out)


def view_from_json(raw: Any) -> View:
    """Build a :class:`View` from the wire form, or raise :class:`BadView`.

    Every field is checked because this input crosses a process boundary and is
    the one place a bug here would look like a bug in the agent.
    """
    if not isinstance(raw, dict):
        raise BadView("view must be an object")

    seat = raw.get("seat")
    if seat not in SEATS:
        raise BadView(f"seat must be one of {list(SEATS)}, got {seat!r}")

    contract = raw.get("contract")
    if contract not in _CONTRACTS:
        raise BadView(f"contract must be one of {sorted(_CONTRACTS)}, got {contract!r}")

    contractor = raw.get("contractor") or seat
    if contractor not in SEATS:
        raise BadView(f"unknown contractor: {contractor!r}")

    joker_suit = raw.get("joker_suit")
    if joker_suit is not None and joker_suit not in _SUITS:
        raise BadView(f"joker_suit must be a suit or null, got {joker_suit!r}")

    led_suit = raw.get("led_suit")
    if led_suit is not None and led_suit != JOKER_SUIT and led_suit not in _SUITS:
        raise BadView(f"led_suit must be a suit, 'JOKER' or null, got {led_suit!r}")

    hand = _cards(raw.get("hand"), "hand")
    legal = _cards(raw.get("legal"), "legal")

    hand_ids = {c.id for c in hand}
    stray = [c.id for c in legal if c.id not in hand_ids]
    if stray:
        raise BadView(f"legal cards not in hand: {stray}")

    tricks_raw = raw.get("tricks") or []
    if not isinstance(tricks_raw, list):
        raise BadView("tricks must be a list of [team, count] pairs")
    tricks: list[tuple[str, int]] = []
    for entry in tricks_raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise BadView("tricks entries must be [team, count] pairs")
        team, count = entry
        if team not in ("NS", "WE"):
            raise BadView(f"unknown team in tricks: {team!r}")
        if not isinstance(count, int) or count < 0:
            raise BadView(f"tricks count must be a non-negative int, got {count!r}")
        tricks.append((team, count))

    return View(
        hand=hand,
        legal=legal,
        trick=_plays(raw.get("trick"), "trick"),
        led_suit=led_suit,
        contract=contract,
        seat=seat,
        joker_suit=joker_suit,
        played=_plays(raw.get("played"), "played"),
        tricks=tuple(tricks),
        contractor=contractor,
    )


def choose_card(view: View, worlds: int, seed: int) -> Card:
    """One decision. A fresh agent per call: no state to leak between requests."""
    agent = PIMCAgent(worlds=worlds, seed=seed)
    return agent.choose(view)


def _log(record: dict[str, Any]) -> None:
    """One JSON line per request — greppable, and cheap enough to leave on."""
    print(json.dumps(record, separators=(",", ":")), file=sys.stdout, flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "fivehundred-pimc/1.0"
    worlds = 24
    seed = 21
    _lock = threading.Lock()

    # ── plumbing ────────────────────────────────────────────────

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib hook
        """Silence the default Common Log Format; we emit our own JSON instead."""

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The table is served from a different origin than this port — and may be
        # served from a tunnel while the agent stays local — so a JSON POST from the
        # page is always cross-origin and always preflighted. Without these headers
        # the browser blocks the request and the table silently falls back to the
        # heuristic, which looks exactly like "the agent is not working".
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib hook
        """The preflight a cross-origin JSON POST needs. Answer it or nothing works."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise BadView("empty body")
        if length > 1_000_000:
            raise BadView("body too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BadView(f"body is not JSON: {exc}") from exc

    # ── routes ──────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        if self.path.split("?")[0] != "/health":
            self._send(404, {"error": "unknown endpoint"})
            return
        self._send(200, {"ok": True, "agent": "pimc", "worlds": self.worlds})

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        route = self.path.split("?")[0]
        if route != "/move":
            self._send(404, {"error": "unknown endpoint"})
            return

        started = time.perf_counter()
        try:
            body = self._read_json()
            view = view_from_json(body.get("view") if isinstance(body, dict) else None)
            worlds = int(body.get("worlds", self.worlds)) if isinstance(body, dict) else self.worlds
            if not 1 <= worlds <= 512:
                raise BadView(f"worlds must be 1..512, got {worlds}")
            seed = int(body.get("seed", self.seed)) if isinstance(body, dict) else self.seed
        except BadView as exc:
            _log({"t": round(time.time(), 3), "route": "move", "status": 400, "error": str(exc)})
            self._send(400, {"error": str(exc)})
            return

        try:
            with self._lock:  # one decision at a time: this box has 6 cores and a table has 4 seats
                card = choose_card(view, worlds, seed)
        except Exception as exc:  # noqa: BLE001 - a bug here must not kill the server
            _log({"t": round(time.time(), 3), "route": "move", "status": 500, "error": repr(exc)})
            self._send(500, {"error": f"agent failed: {exc!r}"})
            return

        ms = (time.perf_counter() - started) * 1000
        _log({
            "t": round(time.time(), 3), "route": "move", "status": 200, "seat": view.seat,
            "contract": view.contract, "worlds": worlds, "ms": round(ms, 1),
            "card": card.id, "legal": len(view.legal), "trick_len": len(view.trick),
        })
        self._send(200, {"card": card.id, "worlds": worlds, "ms": round(ms, 1)})


def serve(host: str, port: int, worlds: int, seed: int) -> ThreadingHTTPServer:
    """Bind and return the server; the caller runs ``serve_forever``."""
    Handler.worlds = worlds
    Handler.seed = seed
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the 500 search agent over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5200)
    parser.add_argument("--worlds", type=int, default=24, help="sample worlds per decision")
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args(argv)

    httpd = serve(args.host, args.port, args.worlds, args.seed)
    host, port = httpd.server_address[:2]
    _log({"t": round(time.time(), 3), "event": "listening", "host": host, "port": port, "worlds": args.worlds})
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _log({"t": round(time.time(), 3), "event": "stopped"})
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    raise SystemExit(main())
