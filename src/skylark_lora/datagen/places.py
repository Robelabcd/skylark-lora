"""Airports, metro areas and ambiguous city names, and how users refer to them.

Each way of naming a place maps to exactly one correct code:
"Dulles" -> IAD, "Chicago" -> CHI (metro), "O'Hare" -> ORD.
Held-out airports are never used in training data.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_DATA = json.loads(
    (Path(__file__).parent / "airports.json").read_text(encoding="utf-8")
)

HELD_OUT: frozenset[str] = frozenset(_DATA["held_out"])
AMBIGUOUS: list[dict] = _DATA["ambiguous"]


@dataclass(frozen=True)
class Place:
    code: str  # the correct answer: an airport code or a metro code
    refs: tuple[str, ...]  # ways a user may name it
    city: str  # a readable name, used in replies and booking summaries
    metro: str | None = None  # the metro area this airport belongs to, if any


def _build() -> list[Place]:
    metro_of = {a: m["code"] for m in _DATA["metros"] for a in m["airports"]}
    places = [
        Place(a["code"], tuple(a["refs"]), a["city"], metro_of.get(a["code"]))
        for a in _DATA["airports"]
    ]
    places += [
        Place(m["code"], tuple(m["refs"]), m["refs"][0], m["code"])
        for m in _DATA["metros"]
    ]
    return places


ALL_PLACES: list[Place] = _build()
TRAIN_PLACES: list[Place] = [p for p in ALL_PLACES if p.code not in HELD_OUT]
HELD_OUT_NAMES: frozenset[str] = frozenset(
    ref for p in ALL_PLACES if p.code in HELD_OUT for ref in p.refs
)


def pick_pair(
    rng: random.Random, places: list[Place] = TRAIN_PLACES
) -> tuple[Place, Place]:
    """Two places that are different and not in the same metro area."""
    while True:
        a, b = rng.sample(places, 2)
        if (a.metro or a.code) != (b.metro or b.code):
            return a, b


def name(rng: random.Random, place: Place, allow_code: bool = True) -> str:
    """One way a user might write this place."""
    refs = [r for r in place.refs if allow_code or not (len(r) == 3 and r.isupper())]
    return rng.choice(refs or list(place.refs))


def is_metro(place: Place) -> bool:
    return place.metro == place.code
