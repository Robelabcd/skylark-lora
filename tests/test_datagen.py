import datetime as dt
import json
import random
import re
from collections import Counter

import pytest

from skylark_lora.datagen import dates
from skylark_lora.datagen.build import (
    TrainCase,
    exam_user_messages,
    generate,
    normalize,
)
from skylark_lora.datagen.places import (
    ALL_PLACES,
    HELD_OUT,
    HELD_OUT_NAMES,
    TRAIN_PLACES,
)
from skylark_lora.datagen.scenarios import MIX

N = 600
EXAMPLES, STATS = generate(N, seed=1, split="train", forbidden=exam_user_messages())


# ---------- Dates ----------


def test_month_day_rolls_over_to_next_year():
    assert dates.next_month_day(dt.date(2026, 12, 15), 1, 5) == dt.date(2027, 1, 5)
    assert dates.next_month_day(dt.date(2026, 12, 15), 12, 20) == dt.date(2026, 12, 20)


def test_day_of_month_rolls_over_to_next_month():
    assert dates.next_day_of_month(dt.date(2027, 2, 25), 3) == dt.date(2027, 3, 3)
    assert dates.next_day_of_month(dt.date(2027, 1, 30), 31) == dt.date(2027, 1, 31)
    assert dates.next_day_of_month(dt.date(2027, 2, 1), 30) == dt.date(2027, 3, 30)


def test_weekday_is_always_in_the_future():
    tuesday = dt.date(2026, 12, 1)
    assert dates.next_weekday(tuesday, 4) == dt.date(2026, 12, 4)  # Friday
    assert dates.next_weekday(tuesday, 1) == dt.date(2026, 12, 8)  # next Tuesday


@pytest.mark.parametrize("seed", range(200))
def test_relative_phrases_mean_what_they_say(seed):
    rng = random.Random(seed)
    today = dt.date(2026, 11, 1) + dt.timedelta(days=rng.randint(0, 240))
    phrase, date = dates.departure(rng, today, relative=True)
    assert date >= today
    for i, day in enumerate(dates.WEEKDAYS):
        if phrase.endswith(day):
            assert date.weekday() == i and today < date <= today + dt.timedelta(days=7)
    if phrase.startswith("tomorrow"):
        assert date == today + dt.timedelta(days=1)


# ---------- Places ----------


def test_every_name_points_to_one_place():
    owners: dict[str, set[str]] = {}
    for place in ALL_PLACES:
        for ref in place.refs:
            owners.setdefault(ref, set()).add(place.code)
    shared = {ref: codes for ref, codes in owners.items() if len(codes) > 1}
    assert not shared


def test_held_out_airports_are_not_training_places():
    assert len(HELD_OUT) == 10
    assert not HELD_OUT & {p.code for p in TRAIN_PLACES}


# ---------- Generated data ----------


def test_every_example_passes_the_test_set_checks():
    for item in EXAMPLES:
        TrainCase.model_validate(item)


def test_category_mix_matches_plan():
    counts = Counter(item["category"] for item in EXAMPLES)
    for category, share in MIX.items():
        assert abs(counts[category] - share * N) <= 1


def _codes_in(value):
    """Every airport code in structured data: arguments, routes like "IAD-MIA", "via"."""
    if isinstance(value, dict):
        for v in value.values():
            yield from _codes_in(v)
    elif isinstance(value, list):
        for v in value:
            yield from _codes_in(v)
    elif isinstance(value, str):
        yield from re.findall(r"\b[A-Z]{3}\b", value)


def test_no_held_out_airport_appears_anywhere():
    for item in EXAMPLES:
        structured = [item["expected"]] + [
            m.get("tool_call") or m.get("content")
            for m in item["messages"]
            if m["role"] != "user"
        ]
        assert not set(_codes_in(structured)) & HELD_OUT, item["id"]
    text = json.dumps(
        [m["content"] for i in EXAMPLES for m in i["messages"] if m["role"] == "user"]
    )
    for ref in HELD_OUT_NAMES - HELD_OUT:
        assert not re.search(rf"\b{re.escape(ref)}\b", text, flags=re.IGNORECASE), ref


def test_no_test_set_message_is_reused():
    forbidden = exam_user_messages()
    for item in EXAMPLES:
        for m in item["messages"]:
            if m["role"] == "user":
                assert normalize(m["content"]) not in forbidden


def test_no_duplicate_conversations():
    keys = [json.dumps(item["messages"], sort_keys=True) for item in EXAMPLES]
    assert len(keys) == len(set(keys))


def test_same_seed_same_data():
    again, _ = generate(N, seed=1, split="train", forbidden=exam_user_messages())
    assert again == EXAMPLES


def test_cheapest_reference_picks_the_cheapest_offer():
    checked = 0
    for item in EXAMPLES:
        last = item["messages"][-1]["content"].lower()
        if item["category"] == "follow_up" and "cheapest one" in last:
            offers = item["messages"][2]["content"]["offers"]
            cheapest = min(offers, key=lambda o: o["price_usd"])
            assert item["expected"]["arguments"]["offer_id"] == cheapest["offer_id"]
            checked += 1
    assert checked > 0
