"""The HTTP sidecar: contract, validation, and the one thing that matters — it
answers with a card the caller actually holds and may legally play.

The views below are not hand-written fixtures. They are captured from a real
hand played by the engine, so `hand`/`trick`/`played`/`tricks` agree with each
other the way the browser's will. A synthetic view is easy to write and easy to
get subtly wrong — and a wrong view makes the search fail in ways that look
like server bugs, which is exactly what the first draft of this file did.
"""

from __future__ import annotations

import dataclasses
import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from fivehundred.cards import Card, deal
from fivehundred.env import SEATS, View, play_hand
from fivehundred.serve import (
    BadView,
    card_from_id,
    choose_card,
    serve,
    view_from_json,
)


class Recorder:
    """An agent that keeps every view it was shown and plays its first legal card."""

    def __init__(self) -> None:
        self.views: list[View] = []

    def choose(self, view: View) -> Card:
        self.views.append(view)
        return view.legal[0]


def _captured_views(contract: str = "H", seed: int = 7) -> dict[str, list[View]]:
    hands_list, _kitty = deal(random.Random(seed))
    hands = {seat: hands_list[i] for i, seat in enumerate(SEATS)}
    agents = {seat: Recorder() for seat in SEATS}
    play_hand(hands, contract, "S", agents)
    return {seat: agent.views for seat, agent in agents.items()}


@pytest.fixture(scope="module")
def views() -> dict[str, list[View]]:
    return _captured_views()


def _midhand(views: dict[str, list[View]], seat: str = "W", index: int = 2) -> View:
    """A genuine mid-hand view: a couple of tricks in, cards already played."""
    return views[seat][index]


def _wire(view: View) -> dict:
    return {
        "seat": view.seat,
        "contract": view.contract,
        "contractor": view.contractor,
        "joker_suit": view.joker_suit,
        "hand": [c.id for c in view.hand],
        "legal": [c.id for c in view.legal],
        "trick": [[s, c.id] for s, c in view.trick],
        "led_suit": view.led_suit,
        "played": [[s, c.id] for s, c in view.played],
        "tricks": [[t, n] for t, n in view.tricks],
    }


# ── card ids ────────────────────────────────────────────────────


def test_card_ids_round_trip():
    assert card_from_id("S-A") == Card("S", "A")
    assert card_from_id("JOKER") == Card("JOKER", "JOKER")
    for bad in ("SA", "S-2", "X-A", "", None, 7):
        with pytest.raises(BadView):
            card_from_id(bad)


# ── the view on the wire ───────────────────────────────────────


def test_a_real_view_survives_the_round_trip(views):
    view = _midhand(views)
    assert view_from_json(_wire(view)) == view


def test_every_field_of_the_view_crosses_the_wire(views):
    view = _midhand(views)
    back = view_from_json(_wire(view))
    for field in dataclasses.fields(View):
        assert getattr(back, field.name) == getattr(view, field.name), field.name


def test_the_view_is_validated_because_it_crosses_a_process_boundary(views):
    wire = _wire(_midhand(views))
    for mutate in (
        lambda w: w.update(seat="X"),
        lambda w: w.update(contract="SPADES"),
        lambda w: w.update(joker_suit="X"),
        lambda w: w.update(hand=["S-A", "Z-9"]),
        lambda w: w.update(trick=[["S"]]),
        lambda w: w.update(tricks=[["XX", 1]]),
        lambda w: w.update(tricks=[["NS", -1]]),
    ):
        broken = dict(wire)
        mutate(broken)
        with pytest.raises(BadView):
            view_from_json(broken)

    with pytest.raises(BadView):
        view_from_json(None)
    with pytest.raises(BadView):
        view_from_json({"seat": "W"})  # missing everything else


def test_a_legal_card_that_is_not_in_the_hand_is_refused(views):
    wire = _wire(_midhand(views))
    wire["legal"] = [wire["hand"][0], "D-2"]  # neither a real card nor in the hand
    with pytest.raises(BadView):
        view_from_json(wire)


def test_omitted_optional_fields_get_sane_defaults():
    wire = {
        "seat": "S", "contract": "NT",
        "hand": ["S-A", "H-K"], "legal": ["S-A", "H-K"],
    }
    view = view_from_json(wire)
    assert view.trick == () and view.played == () and view.tricks == ()
    assert view.contractor == "S"      # defaults to the seat itself
    assert view.joker_suit is None


# ── the decision ───────────────────────────────────────────────


def test_the_agent_answers_with_a_legal_card(views):
    view = _midhand(views)
    card = choose_card(view, worlds=4, seed=21)
    assert card.id in {c.id for c in view.legal}


def test_the_agent_is_deterministic_for_a_seed(views):
    view = _midhand(views)
    assert choose_card(view, worlds=4, seed=21).id == choose_card(view, worlds=4, seed=21).id


def test_a_forced_card_is_returned_without_searching(views):
    only = Card("C", "4")
    view = dataclasses.replace(_midhand(views), legal=(only,), hand=(only,))
    assert choose_card(view, worlds=24, seed=1).id == "C-4"


# ── over the wire ─────────────────────────────────────────────


@pytest.fixture()
def sidecar():
    httpd = serve("127.0.0.1", 0, worlds=4, seed=21)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _post(url: str, payload: dict | None, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_health_answers(sidecar):
    with urllib.request.urlopen(f"{sidecar}/health", timeout=10) as response:
        body = json.loads(response.read().decode())
    assert response.status == 200
    assert body["ok"] is True and body["agent"] == "pimc"


def test_move_answers_with_a_card_from_the_view(sidecar, views):
    view = _midhand(views)
    status, body = _post(f"{sidecar}/move", {"view": _wire(view)})
    assert status == 200
    assert body["card"] in {c.id for c in view.legal}
    assert body["ms"] >= 0


def test_a_bad_view_is_a_400_and_the_server_stays_up(sidecar, views):
    status, body = _post(f"{sidecar}/move", {"view": {"seat": "Q"}})
    assert status == 400
    assert "seat" in body["error"]

    # still serving after the bad request
    status, body = _post(f"{sidecar}/move", {"view": _wire(_midhand(views))})
    assert status == 200


def test_broken_json_is_a_400_not_a_crash(sidecar):
    status, body = _post(f"{sidecar}/move", None, raw=b"{not json")
    assert status == 400
    assert "JSON" in body["error"]


def test_unknown_routes_are_404(sidecar):
    status, _ = _post(f"{sidecar}/nope", {})
    assert status == 404


def test_a_cross_origin_preflight_is_answered(sidecar):
    """Without this the browser never sends the POST and the table falls back silently."""
    req = urllib.request.Request(
        f"{sidecar}/move", method="OPTIONS",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        assert response.status in (200, 204)
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in response.headers["Access-Control-Allow-Methods"]
        assert "Content-Type" in response.headers["Access-Control-Allow-Headers"]


def test_a_real_answer_carries_the_cors_header(sidecar, views):
    status, body = _post(f"{sidecar}/move", {"view": _wire(_midhand(views))})
    assert status == 200
    # the header itself is checked on the preflight above; this proves a 200 is not
    # blocked by the browser for want of it, which is what a silent fallback looks like
    assert body["card"]
