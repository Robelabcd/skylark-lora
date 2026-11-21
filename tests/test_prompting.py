import json

import pytest

from skylark_lora.eval_set import load_eval_set
from skylark_lora.few_shot import FEW_SHOT_EXAMPLES
from skylark_lora.prompting import system_prompt, to_chat_messages
from skylark_lora.tools import ALL_ACTIONS, validate_call

CASES = load_eval_set()
BY_ID = {case.id: case for case in CASES}


def test_system_prompt_states_today_with_weekday():
    assert "Today is Tuesday, 2026-12-01." in system_prompt("2026-12-01")


def test_system_prompt_lists_every_action():
    prompt = system_prompt("2026-12-01")
    for name in ALL_ACTIONS:
        assert f'"name": "{name}"' in prompt


def test_few_shot_examples_only_when_asked():
    assert "Examples:" not in system_prompt("2026-12-01")
    assert "Examples:" in system_prompt("2026-12-01", few_shot=True)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_case_becomes_valid_chat(case):
    messages = to_chat_messages(case)
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": case.messages[-1].content}
    assert {m["role"] for m in messages} <= {"system", "user", "assistant"}
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


def test_tool_calls_and_results_are_rendered():
    messages = to_chat_messages(BY_ID["follow_up-001"])
    assistant_call = json.loads(messages[2]["content"])
    assert assistant_call["name"] == "search_flights"
    assert messages[3]["content"].startswith("[Tool result: search_flights]")
    assert "OF-1022" in messages[3]["content"]


def test_answer_is_never_in_the_prompt():
    """For single-turn cases, the expected arguments must not appear in the prompt."""
    for case in CASES:
        arguments = case.expected["arguments"]
        if len(case.messages) == 1 and arguments:
            prompt = json.dumps(to_chat_messages(case, few_shot=True))
            assert json.dumps(arguments) not in prompt, case.id


@pytest.mark.parametrize(
    "example", FEW_SHOT_EXAMPLES, ids=lambda e: e["expected"]["name"]
)
def test_few_shot_answers_are_valid(example):
    assert validate_call(example["expected"]).ok


def test_few_shot_examples_are_not_test_cases():
    test_requests = {
        m.content.lower() for case in CASES for m in case.messages if m.role == "user"
    }
    for example in FEW_SHOT_EXAMPLES:
        assert example["user"].lower() not in test_requests
