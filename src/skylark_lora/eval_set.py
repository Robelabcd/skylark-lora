"""The hand-written test set: its format, its rules, and a loader.

Every case is checked when loaded, so a mistake in the answer key
fails loudly instead of silently corrupting results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from skylark_lora.tools import IsoDate, validate_call

EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "test_set.json"

CATEGORY_TARGETS: dict[str, int] = {
    "simple_search": 15,
    "search_options": 12,
    "relative_date": 10,
    "city_airport": 8,
    "follow_up": 15,
    "bookings": 8,
    "clarify": 14,
    "decline": 8,
    "respond": 6,
    "messy": 4,
}
Category = Literal[
    "simple_search",
    "search_options",
    "relative_date",
    "city_airport",
    "follow_up",
    "bookings",
    "clarify",
    "decline",
    "respond",
    "messy",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserMessage(_Strict):
    role: Literal["user"]
    content: str = Field(min_length=1)


class AssistantMessage(_Strict):
    """An earlier assistant turn: either text, or a tool call."""

    role: Literal["assistant"]
    content: str | None = None
    tool_call: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> AssistantMessage:
        if (self.content is None) == (self.tool_call is None):
            raise ValueError(
                "assistant message needs exactly one of content or tool_call"
            )
        if self.tool_call is not None:
            result = validate_call(self.tool_call)
            if not result.ok:
                raise ValueError(f"invalid tool_call in context: {result.error}")
        return self


class ToolMessage(_Strict):
    """The result a tool returned earlier in the conversation."""

    role: Literal["tool"]
    name: str
    content: dict[str, Any]


Message = Annotated[
    UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]


class EvalCase(_Strict):
    id: str = Field(pattern=r"^[a-z_]+-\d{3}$")
    category: Category
    today: IsoDate
    messages: list[Message] = Field(min_length=1)
    expected: dict[str, Any]
    notes: str | None = None

    @model_validator(mode="after")
    def _check_case(self) -> EvalCase:
        if self.messages[-1].role != "user":
            raise ValueError("the last message must be from the user")

        result = validate_call(self.expected)
        if not result.ok:
            raise ValueError(f"expected answer is invalid: {result.error}")

        args = self.expected["arguments"]
        for key in ("departure_date", "return_date"):
            if key in args and args[key] < self.today.isoformat():
                raise ValueError(f"{key} {args[key]} is before today {self.today}")

        # An ID in the answer must be knowable: it has to appear in an earlier tool result.
        context = json.dumps([m.content for m in self.messages if m.role == "tool"])
        for key in ("offer_id", "booking_id"):
            if key in args and f'"{args[key]}"' not in context:
                raise ValueError(
                    f"{key} {args[key]!r} does not appear in any tool result"
                )
        return self


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[EvalCase]:
    """Load and check every case. Errors name the case that is wrong."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for i, item in enumerate(raw):
        try:
            cases.append(EvalCase.model_validate(item))
        except ValidationError as exc:
            case_id = item.get("id", f"#{i}") if isinstance(item, dict) else f"#{i}"
            raise ValueError(f"test case {case_id} is invalid:\n{exc}") from None
    return cases
