"""Small text helpers: sentence assembly and the "messiness" filters people's typing adds."""

from __future__ import annotations

import random
import re

from skylark_lora.datagen.dates import MONTHS, WEEKDAYS

_NEEDS_ON = tuple(MONTHS) + tuple(m[:3] for m in MONTHS) + tuple(WEEKDAYS)


def when(phrase: str) -> str:
    """Add "on" where English needs it: "on December 5", but "tomorrow", "this Friday"."""
    if phrase[0].isdigit() or phrase.startswith(_NEEDS_ON):
        return f"on {phrase}"
    if phrase.startswith("the ") and not phrase.startswith("the day"):
        return f"on {phrase}"
    return phrase


def sentence(text: str) -> str:
    """Capitalize the first letter and end with punctuation if there is none."""
    text = text.strip()
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!" else text + _end_mark(text)


def _end_mark(text: str) -> str:
    question_words = (
        "Can ",
        "Could ",
        "Any ",
        "What",
        "Which",
        "When",
        "Is ",
        "Do ",
        "Does ",
        "How",
        "Will",
    )
    return "?" if text.startswith(question_words) else ""


def join_and(items: list[str]) -> str:
    """["A"] -> "A"; ["A", "B"] -> "A and B"; ["A", "B", "C"] -> "A, B and C"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def join_or(items: list[str]) -> str:
    """["A", "B", "C"] -> "A, B or C"."""
    return ", ".join(items[:-1]) + " or " + items[-1]


def add_details(base: str, extras: list[str]) -> str:
    """Attach option phrases to the request part, before any follow-up question.

    "I'm flying A to B on May 2. What's available?" + ["nonstop"]
    -> "I'm flying A to B on May 2, nonstop. What's available?"
    """
    if not extras:
        return base
    head, sep, tail = base.partition(". ")
    return head.rstrip(".?!") + ", " + ", ".join(extras) + (sep + tail if sep else "")


def casual(rng: random.Random, text: str) -> str:
    """Light, common sloppiness: lowercase and no final punctuation."""
    return text.lower().rstrip(".?!") if rng.random() < 0.5 else text.rstrip(".")


_SHORTHAND = [
    (r"\bplease\b", "pls"),
    (r"\btomorrow\b", "tmrw"),
    (r"\btickets\b", "tix"),
    (r"\bflights\b", "flts"),
    (r"\bpeople\b", "ppl"),
    (r"\byou\b", "u"),
    (r"\band\b", "&"),
]


def typo(rng: random.Random, word: str) -> str:
    """Drop or swap one inner letter: "Boston" -> "Bostn" or "Botson"."""
    if len(word) < 6 or not word.isalpha():
        return word
    i = rng.randint(1, len(word) - 3)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def messy(rng: random.Random, text: str, place_words: list[str]) -> str:
    """Heavier mess for the "messy" category: shorthand, arrows, a typo, case changes.

    place_words = [origin name, destination name] as written in the text.
    """
    out = text
    for pattern, short in rng.sample(_SHORTHAND, k=3):
        out = re.sub(pattern, short, out, flags=re.IGNORECASE)
    origin, destination = place_words
    route = f"{origin} to {destination}"
    if route in out and rng.random() < 0.6:
        out = out.replace(
            route, f"{origin} {rng.choice(['->', '>', '2'])} {destination}", 1
        )
    words = [w for w in place_words if " " not in w and w.isalpha() and len(w) >= 6]
    if words and rng.random() < 0.7:
        target = rng.choice(words)
        out = out.replace(target, typo(rng, target), 1)
    out = out.rstrip(".?!")
    return out.upper() if rng.random() < 0.15 else out.lower()
