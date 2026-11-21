"""Grade model outputs against the test set.

Every model is graded by this same code. A model's answers are saved first,
one line per test case:

    {"id": "clarify-002", "output": "<whatever the model wrote>"}

Each answer goes through four checks: parse, validate, right action, right
arguments. The rules are explained in docs/scoring.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skylark_lora.eval_set import CATEGORY_TARGETS, EvalCase, load_eval_set
from skylark_lora.tools import OUTCOMES, TOOLS, validate_call

_TOOL_CALL_TAG = re.compile(r"^<tool_call>\s*(.*?)\s*</tool_call>$", re.DOTALL)


# ---------- 1. Parse ----------


def parse_output(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Find the call in a model's raw output.

    Returns (call, format_ok). format_ok is True only when the output is
    exactly one JSON call, optionally inside <tool_call> tags. Otherwise the
    first JSON object in the text is used, and format_ok is False.
    """
    stripped = text.strip()
    tagged = _TOOL_CALL_TAG.match(stripped)
    body = tagged.group(1) if tagged else stripped
    try:
        call = json.loads(body)
        if isinstance(call, dict) and "name" in call:
            return _decode_string_arguments(call)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            call, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict) and "name" in call:
            decoded, _ = _decode_string_arguments(call)
            return decoded, False
    return None, False


def _decode_string_arguments(call: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Some models write "arguments" as a JSON string. Accept it, but not as clean format."""
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            return {**call, "arguments": json.loads(arguments)}, False
        except json.JSONDecodeError:
            return call, False
    return call, True


# ---------- 2-4. Validate, compare ----------


@dataclass
class CaseResult:
    id: str
    category: str
    expected_name: str
    predicted_name: str | None
    format_ok: bool
    valid: bool
    correct_action: bool
    correct: bool
    error_type: str | None
    stated_arguments: list[str] = field(default_factory=list)
    wrong_arguments: list[str] = field(default_factory=list)
    output: str = ""


def score_case(case: EvalCase, output: str) -> CaseResult:
    """Grade one output. error_type is the first check that failed."""
    expected = validate_call(case.expected)
    call, format_ok = parse_output(output)

    def result(**kw: Any) -> CaseResult:
        base = {
            "id": case.id,
            "category": case.category,
            "expected_name": expected.name,
            "predicted_name": None,
            "format_ok": format_ok,
            "valid": False,
            "correct_action": False,
            "correct": False,
            "error_type": None,
            "stated_arguments": sorted(case.expected["arguments"]),
            "output": output,
        }
        return CaseResult(**{**base, **kw})

    if call is None:
        return result(error_type="malformed")

    predicted = validate_call(call)
    name = call.get("name") if isinstance(call.get("name"), str) else None
    correct_action = name == expected.name

    if not predicted.ok:
        return result(
            predicted_name=name,
            correct_action=correct_action,
            error_type=predicted.error_type,
        )
    if not correct_action:
        return result(predicted_name=name, valid=True, error_type="wrong_action")

    # Outcomes (clarify, decline, respond): only the name is scored exactly.
    if expected.name in OUTCOMES:
        return result(
            predicted_name=name, valid=True, correct_action=True, correct=True
        )

    # Compare meaning, not spelling: both sides with defaults filled in.
    want = expected.arguments.model_dump()
    got = predicted.arguments.model_dump()
    wrong = sorted(key for key in want if want[key] != got[key])
    return result(
        predicted_name=name,
        valid=True,
        correct_action=True,
        correct=not wrong,
        error_type="wrong_arguments" if wrong else None,
        wrong_arguments=wrong,
    )


def score_all(cases: list[EvalCase], outputs: dict[str, str]) -> list[CaseResult]:
    """Grade every case. A missing prediction counts as an empty (malformed) output."""
    unknown = set(outputs) - {case.id for case in cases}
    if unknown:
        raise ValueError(f"predictions for unknown test cases: {sorted(unknown)}")
    return [score_case(case, outputs.get(case.id, "")) for case in cases]


# ---------- Summary ----------


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    n = len(results)
    expects_outcome = [r for r in results if r.expected_name in OUTCOMES]
    expects_tool = [r for r in results if r.expected_name in TOOLS]

    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    # Per-argument accuracy, over cases where the right tool was called validly.
    # An argument counts when the answer key states it or the model got it wrong.
    arg_seen: Counter[str] = Counter()
    arg_wrong: Counter[str] = Counter()
    for r in expects_tool:
        if r.correct_action and r.valid:
            arg_seen.update(set(r.stated_arguments) | set(r.wrong_arguments))
            arg_wrong.update(r.wrong_arguments)

    return {
        "cases": n,
        "accuracy": _rate(sum(r.correct for r in results), n),
        "format_compliance": _rate(sum(r.format_ok for r in results), n),
        "valid_rate": _rate(sum(r.valid for r in results), n),
        "action_accuracy": _rate(sum(r.correct_action for r in results), n),
        "over_calling": _rate(
            sum(r.predicted_name in TOOLS for r in expects_outcome),
            len(expects_outcome),
        ),
        "under_calling": _rate(
            sum(r.predicted_name in OUTCOMES for r in expects_tool), len(expects_tool)
        ),
        "by_category": {
            c: _rate(sum(r.correct for r in by_category[c]), len(by_category[c]))
            for c in CATEGORY_TARGETS
            if by_category[c]
        },
        "errors": dict(Counter(r.error_type for r in results if r.error_type)),
        "by_argument": {
            key: _rate(arg_seen[key] - arg_wrong[key], arg_seen[key])
            for key in sorted(arg_seen)
        },
    }


# ---------- Files and command line ----------


def load_predictions(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in outputs:
            raise ValueError(f"line {line_no}: duplicate prediction for {row['id']}")
        outputs[row["id"]] = row["output"]
    return outputs


def score_file(path: Path) -> tuple[list[CaseResult], dict[str, Any]]:
    results = score_all(load_eval_set(), load_predictions(path))
    return results, summarize(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a predictions file.")
    parser.add_argument(
        "predictions", type=Path, help="JSONL file: {id, output} per line"
    )
    parser.add_argument(
        "--details", type=Path, help="Write per-case results (JSONL) here"
    )
    args = parser.parse_args(argv)

    results, summary = score_file(args.predictions)
    print(json.dumps(summary, indent=2))
    if args.details:
        lines = (json.dumps(asdict(r), ensure_ascii=False) for r in results)
        args.details.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
