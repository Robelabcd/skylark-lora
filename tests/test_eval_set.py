import copy
from collections import Counter

import pytest
from pydantic import ValidationError

from skylark_lora.eval_set import CATEGORY_TARGETS, EvalCase, load_eval_set

CASES = load_eval_set()


# ---------- Checks on the real test set ----------


def test_ids_are_unique():
    ids = [case.id for case in CASES]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"duplicate ids: {duplicates}"


def test_id_prefix_matches_category():
    wrong = [case.id for case in CASES if not case.id.startswith(case.category + "-")]
    assert not wrong, f"id prefix does not match category: {wrong}"


def test_no_category_over_target():
    counts = Counter(case.category for case in CASES)
    over = {c: n for c, n in counts.items() if n > CATEGORY_TARGETS[c]}
    assert not over, f"categories over target: {over}"


def test_set_is_complete():
    counts = Counter(case.category for case in CASES)
    total, target = len(CASES), sum(CATEGORY_TARGETS.values())
    if total < target:
        progress = ", ".join(
            f"{c} {counts[c]}/{n}" for c, n in CATEGORY_TARGETS.items()
        )
        pytest.skip(f"test set in progress: {total}/{target} ({progress})")
    assert dict(counts) == CATEGORY_TARGETS


# ---------- Checks that the checker itself catches mistakes ----------

GOOD = {
    "id": "follow_up-999",
    "category": "follow_up",
    "today": "2026-12-01",
    "messages": [
        {"role": "user", "content": "Flights IAD to SEA on Dec 18"},
        {
            "role": "assistant",
            "tool_call": {
                "name": "search_flights",
                "arguments": {
                    "origin": "IAD",
                    "destination": "SEA",
                    "departure_date": "2026-12-18",
                },
            },
        },
        {
            "role": "tool",
            "name": "search_flights",
            "content": {"offers": [{"offer_id": "OF-1"}]},
        },
        {"role": "assistant", "content": "I found one flight."},
        {"role": "user", "content": "Book it, main fare"},
    ],
    "expected": {
        "name": "prepare_booking",
        "arguments": {"offer_id": "OF-1", "fare": "main"},
    },
}


def broken(change):
    case = copy.deepcopy(GOOD)
    change(case)
    return case


def test_good_case_passes():
    EvalCase.model_validate(GOOD)


@pytest.mark.parametrize(
    "label,case",
    [
        (
            "invalid expected call",
            broken(lambda c: c["expected"]["arguments"].pop("fare")),
        ),
        (
            "offer id not in context",
            broken(lambda c: c["expected"]["arguments"].update(offer_id="OF-9")),
        ),
        (
            "date before today",
            broken(
                lambda c: c.update(
                    today="2026-12-30",
                    messages=c["messages"][:1],
                    expected={
                        "name": "search_flights",
                        "arguments": {
                            "origin": "IAD",
                            "destination": "SEA",
                            "departure_date": "2026-12-18",
                        },
                    },
                )
            ),
        ),
        ("last message not from user", broken(lambda c: c["messages"].pop())),
        ("unknown category", broken(lambda c: c.update(category="other"))),
        ("bad id format", broken(lambda c: c.update(id="Follow Up 1"))),
        (
            "invalid call in context",
            broken(
                lambda c: c["messages"][1]["tool_call"]["arguments"].update(
                    origin="Dulles"
                )
            ),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_mistakes_are_caught(label, case):
    with pytest.raises(ValidationError):
        EvalCase.model_validate(case)
