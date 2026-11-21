import json

import pytest

from skylark_lora.eval_set import load_eval_set
from skylark_lora.scoring import (
    load_predictions,
    parse_output,
    score_all,
    score_case,
    summarize,
)

CASES = {case.id: case for case in load_eval_set()}

# relative_date-001: Dulles to Seattle this Friday -> IAD, SEA, 2026-12-04
SEARCH = CASES["relative_date-001"]
GOOD_SEARCH = {
    "name": "search_flights",
    "arguments": {
        "origin": "IAD",
        "destination": "SEA",
        "departure_date": "2026-12-04",
    },
}


def as_text(call):
    return json.dumps(call)


# ---------- Parsing ----------


@pytest.mark.parametrize(
    "label,output,format_ok",
    [
        ("bare JSON", as_text(GOOD_SEARCH), True),
        ("surrounding whitespace", "\n  " + as_text(GOOD_SEARCH) + "\n", True),
        (
            "tool_call tags",
            "<tool_call>\n" + as_text(GOOD_SEARCH) + "\n</tool_call>",
            True,
        ),
        ("code fence", "```json\n" + as_text(GOOD_SEARCH) + "\n```", False),
        (
            "wrapped in prose",
            "Sure! Here is the call: " + as_text(GOOD_SEARCH) + " Done.",
            False,
        ),
    ],
)
def test_call_is_found(label, output, format_ok):
    call, ok = parse_output(output)
    assert call == GOOD_SEARCH
    assert ok is format_ok


def test_string_arguments_are_decoded_but_not_clean_format():
    output = json.dumps(
        {"name": "search_flights", "arguments": json.dumps(GOOD_SEARCH["arguments"])}
    )
    call, ok = parse_output(output)
    assert call == GOOD_SEARCH
    assert ok is False


@pytest.mark.parametrize(
    "output", ["", "I can't help with that.", "{not json}", '["a", "list"]']
)
def test_no_call_found(output):
    assert parse_output(output) == (None, False)


# ---------- Grading one case ----------


def test_perfect_answer():
    r = score_case(SEARCH, as_text(GOOD_SEARCH))
    assert r.correct and r.format_ok and r.valid and r.correct_action
    assert r.error_type is None


def test_default_written_out_still_counts_as_correct():
    call = {
        **GOOD_SEARCH,
        "arguments": {**GOOD_SEARCH["arguments"], "adults": 1, "cabin": "economy"},
    }
    assert score_case(SEARCH, as_text(call)).correct


def test_wrong_date():
    call = {
        **GOOD_SEARCH,
        "arguments": {**GOOD_SEARCH["arguments"], "departure_date": "2026-12-11"},
    }
    r = score_case(SEARCH, as_text(call))
    assert not r.correct and r.correct_action
    assert r.error_type == "wrong_arguments"
    assert r.wrong_arguments == ["departure_date"]


def test_extra_non_default_argument_is_wrong():
    call = {**GOOD_SEARCH, "arguments": {**GOOD_SEARCH["arguments"], "max_stops": 0}}
    r = score_case(SEARCH, as_text(call))
    assert r.wrong_arguments == ["max_stops"]


def test_invalid_arguments():
    call = {
        **GOOD_SEARCH,
        "arguments": {**GOOD_SEARCH["arguments"], "departure_date": "12/04/2026"},
    }
    r = score_case(SEARCH, as_text(call))
    assert not r.valid and r.correct_action
    assert r.error_type == "invalid_arguments"


def test_unknown_tool():
    r = score_case(SEARCH, as_text({"name": "find_flights", "arguments": {}}))
    assert r.error_type == "unknown_name"


def test_malformed_output():
    r = score_case(SEARCH, "Flights from IAD to SEA on Friday.")
    assert r.error_type == "malformed" and r.predicted_name is None


def test_wrong_tool():
    call = {"name": "clarify", "arguments": {"question": "Which date?"}}
    r = score_case(SEARCH, as_text(call))
    assert r.error_type == "wrong_action" and r.valid


def test_outcome_text_is_not_scored_exactly():
    case = CASES["clarify-003"]
    call = {
        "name": "clarify",
        "arguments": {"question": "Which city are you leaving from?"},
    }
    assert score_case(case, as_text(call)).correct


# ---------- Whole runs ----------


def test_answer_key_scores_100_percent():
    """Sanity check: feeding in the answer key itself must score perfectly."""
    outputs = {case_id: as_text(case.expected) for case_id, case in CASES.items()}
    summary = summarize(score_all(list(CASES.values()), outputs))
    assert summary["accuracy"] == 1.0
    assert summary["format_compliance"] == 1.0
    assert summary["over_calling"] == 0.0
    assert summary["errors"] == {}
    assert set(summary["by_argument"].values()) == {1.0}


def test_summary_numbers():
    clarify = CASES["clarify-003"]
    outputs = {
        SEARCH.id: "Sure! " + as_text(GOOD_SEARCH),
        clarify.id: as_text(
            {
                "name": "search_flights",
                "arguments": {
                    "origin": "IAD",
                    "destination": "SEA",
                    "departure_date": "2026-12-20",
                },
            }
        ),
    }
    summary = summarize(score_all([SEARCH, clarify], outputs))
    assert summary["accuracy"] == 0.5
    assert summary["format_compliance"] == 0.5
    assert summary["over_calling"] == 1.0
    assert summary["under_calling"] == 0.0
    assert summary["errors"] == {"wrong_action": 1}


def test_missing_prediction_counts_as_malformed():
    results = score_all([SEARCH], {})
    assert results[0].error_type == "malformed"


def test_prediction_for_unknown_case_is_an_error():
    with pytest.raises(ValueError, match="unknown"):
        score_all([SEARCH], {"made-up-999": "{}"})


def test_load_predictions_rejects_duplicates(tmp_path):
    path = tmp_path / "preds.jsonl"
    row = json.dumps({"id": SEARCH.id, "output": "x"})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(path)
