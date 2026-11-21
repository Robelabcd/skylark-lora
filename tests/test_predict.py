import json
from types import SimpleNamespace

import pytest

from skylark_lora.eval_set import load_eval_set
from skylark_lora.predict import run_predictions
from skylark_lora.run_openai import make_generator
from skylark_lora.scoring import load_predictions

CASES = load_eval_set()[:5]


def echo_model(messages):
    """A fake model: always answers list_bookings. No GPU or API key needed."""
    return json.dumps({"name": "list_bookings", "arguments": {}}), {"prompt_tokens": 10}


def test_writes_one_line_per_case(tmp_path):
    out = tmp_path / "preds.jsonl"
    assert run_predictions(CASES, echo_model, out) == 5
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in rows] == [c.id for c in CASES]
    assert all("latency_ms" in r and r["prompt_tokens"] == 10 for r in rows)
    assert set(load_predictions(out)) == {c.id for c in CASES}


def test_interrupted_run_resumes_without_repeating(tmp_path):
    out = tmp_path / "preds.jsonl"
    run_predictions(CASES[:2], echo_model, out)
    calls = []

    def counting_model(messages):
        calls.append(1)
        return echo_model(messages)

    assert run_predictions(CASES, counting_model, out) == 3
    assert len(calls) == 3
    assert len(out.read_text().splitlines()) == 5


def fake_openai_client(replies):
    """Mimics the parts of the OpenAI client that make_generator uses."""

    def create(model, messages):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15),
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_openai_generator_records_tokens():
    generate = make_generator(fake_openai_client(['{"name": "x"}']), "some-model")
    text, extra = generate([{"role": "user", "content": "hi"}])
    assert text == '{"name": "x"}'
    assert extra == {"prompt_tokens": 120, "output_tokens": 15}


def test_openai_generator_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("skylark_lora.run_openai.time.sleep", lambda s: None)
    generate = make_generator(fake_openai_client([TimeoutError(), "ok"]), "some-model")
    assert generate([])[0] == "ok"


def test_openai_generator_gives_up_after_three_tries(monkeypatch):
    monkeypatch.setattr("skylark_lora.run_openai.time.sleep", lambda s: None)
    errors = [TimeoutError(), TimeoutError(), TimeoutError()]
    generate = make_generator(fake_openai_client(errors), "some-model")
    with pytest.raises(TimeoutError):
        generate([])
