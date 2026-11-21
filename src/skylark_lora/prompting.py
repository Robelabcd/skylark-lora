"""One prompt builder shared by every model, so every model takes the same exam.

A test case becomes a list of chat messages using only three roles
(system, user, assistant), which every chat model and API accepts:

- earlier assistant tool calls are written as the JSON call itself;
- earlier tool results are shown as a user message starting "[Tool result: ...]".
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from skylark_lora.eval_set import EvalCase
from skylark_lora.few_shot import FEW_SHOT_EXAMPLES
from skylark_lora.tools import tool_schemas

ROLE = (
    "You are Skylark, an airline assistant. For each user message, choose exactly one "
    "action: call a tool, or answer with clarify, decline or respond."
)

RULES = """\
Rules:
- Airports: use 3-letter IATA codes in capitals. A city with one main airport uses that \
airport (Seattle -> SEA). A multi-airport city named without a specific airport uses its \
metro code: WAS (Washington DC), NYC (New York), CHI (Chicago), LON (London), PAR (Paris), \
TYO (Tokyo). A named airport uses its own code (Dulles -> IAD).
- Dates: YYYY-MM-DD, worked out from today's date. A date that has already passed this \
month or year means the next one.
- Defaults: 1 adult, economy, any number of stops, one-way. Leave out arguments the user \
did not ask for.
- Use "clarify" when the origin, destination, date, return date of a round trip, booking \
fare, or the flight or booking meant is missing or ambiguous.
- Use "decline" for anything that is not about flights (hotels, cars, visas, weather) and \
for any payment details. Payment happens in the app, never in chat.
- Use "respond" when the answer is already in the conversation.
- To act on a booking whose ID you do not know, call list_bookings first.
- Bookings and cancellations are only prepared; the user confirms them in the app."""

OUTPUT_FORMAT = """\
Reply with exactly one JSON object and nothing else, in this form:
{"name": "<tool or outcome name>", "arguments": {...}}"""


def system_prompt(today: str, few_shot: bool = False) -> str:
    """The instructions every model sees. `today` is YYYY-MM-DD."""
    weekday = dt.date.fromisoformat(today).strftime("%A")
    tools = "\n".join(json.dumps(_compact(t["function"])) for t in tool_schemas())
    parts = [
        ROLE,
        f"Today is {weekday}, {today}.",
        f"Available actions:\n{tools}",
        RULES,
        OUTPUT_FORMAT,
    ]
    if few_shot:
        parts.append(_render_examples())
    return "\n\n".join(parts)


def _compact(schema: Any) -> Any:
    """Drop schema keys that add tokens but no information (titles, repeated descriptions)."""
    if isinstance(schema, dict):
        out = {k: _compact(v) for k, v in schema.items() if k != "title"}
        params = out.get("parameters")
        if isinstance(params, dict):
            params.pop("description", None)
        return out
    if isinstance(schema, list):
        return [_compact(v) for v in schema]
    return schema


def _render_examples() -> str:
    lines = ["Examples:"]
    for example in FEW_SHOT_EXAMPLES:
        lines.append(f"Today: {example['today']}")
        if example.get("context"):
            lines.append(f"Earlier in the conversation: {example['context']}")
        lines.append(f"User: {example['user']}")
        lines.append(f"Reply: {json.dumps(example['expected'])}")
        lines.append("")
    return "\n".join(lines).rstrip()


def to_chat_messages(case: EvalCase, few_shot: bool = False) -> list[dict[str, str]]:
    """Turn one test case into chat messages: system, then the conversation."""
    messages = [
        {"role": "system", "content": system_prompt(case.today.isoformat(), few_shot)}
    ]
    for message in case.messages:
        if message.role == "user":
            messages.append({"role": "user", "content": message.content})
        elif message.role == "assistant" and message.tool_call is not None:
            messages.append(
                {"role": "assistant", "content": json.dumps(message.tool_call)}
            )
        elif message.role == "assistant":
            messages.append({"role": "assistant", "content": message.content})
        else:
            messages.append(
                {"role": "user", "content": _tool_result(message.name, message.content)}
            )
    return messages


def _tool_result(name: str, content: dict[str, Any]) -> str:
    return f"[Tool result: {name}]\n{json.dumps(content)}"
