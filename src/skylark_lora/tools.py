"""Tool definitions for the Skylark agent: the contract the model must follow.

The model's job, for every user turn, is to output exactly one *call*:

    {"name": "search_flights", "arguments": {"origin": "IAD", ...}}

`name` is either a real tool (it does something) or a no-tool outcome
(clarify, decline, respond). `validate_call()` checks a call against
this contract and says exactly what is wrong if it fails.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


def _parse_iso_date(value: Any) -> dt.date:
    # Pydantic only turns ValueError into a validation error, so not TypeError here.
    if not isinstance(value, str):
        raise ValueError("date must be a string in YYYY-MM-DD format")  # noqa: TRY004
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("date must be in YYYY-MM-DD format")
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ValueError("date must be a real date in YYYY-MM-DD format") from None


IataCode = Annotated[
    str,
    Field(pattern=r"^[A-Z]{3}$", description="3-letter IATA airport code, e.g. IAD"),
]
IsoDate = Annotated[
    dt.date,
    BeforeValidator(_parse_iso_date),
    Field(description="Date in YYYY-MM-DD format"),
]
OfferId = Annotated[
    str,
    Field(min_length=1, description="ID of a flight offer from search results"),
]
BookingId = Annotated[
    str,
    Field(min_length=1, description="ID of an existing booking"),
]


class Args(BaseModel):
    """Base for every tool's arguments.

    extra="forbid": unknown argument names are errors, not silently ignored.
    strict=True:    no type guessing, so "2" is not accepted where 2 is expected.
    """

    model_config = ConfigDict(extra="forbid", strict=True)


# ---------- Real tools ----------


class SearchFlights(Args):
    """Search for flights between two airports on a date."""

    origin: IataCode
    destination: IataCode
    departure_date: IsoDate
    return_date: IsoDate | None = Field(None, description="Omit for one-way trips")
    adults: int = Field(1, ge=1, le=9, description="Number of adult passengers")
    cabin: Literal["economy", "premium", "business"] = "economy"
    max_stops: int | None = Field(None, ge=0, le=2, description="0 means nonstop only")

    @model_validator(mode="after")
    def _check_trip(self) -> SearchFlights:
        if self.origin == self.destination:
            raise ValueError("origin and destination must be different")
        if self.return_date is not None and self.return_date < self.departure_date:
            raise ValueError("return_date cannot be before departure_date")
        return self


class GetOfferDetails(Args):
    """Get full details (times, baggage, fare rules) for one flight offer."""

    offer_id: OfferId


class GetPriceAdvice(Args):
    """Advise whether to buy a flight offer now or wait for a lower price."""

    offer_id: OfferId


class CreatePriceWatch(Args):
    """Watch a flight offer's price and alert the user when it drops."""

    offer_id: OfferId
    target_price: float | None = Field(
        None, gt=0, description="Alert below this price in USD; omit for any drop"
    )


class ListBookings(Args):
    """List the user's existing bookings."""


class PrepareBooking(Args):
    """Prepare a booking quote for an offer. Nothing is charged until the user confirms."""

    offer_id: OfferId
    fare: Literal["basic", "main", "flex"]


class PrepareCancellation(Args):
    """Prepare a cancellation quote for a booking. Nothing changes until the user confirms."""

    booking_id: BookingId


# ---------- No-tool outcomes ----------


class Clarify(Args):
    """Ask the user a question because required information is missing or ambiguous."""

    question: str = Field(min_length=1)


class Decline(Args):
    """Politely refuse a request that is outside Skylark's scope."""

    reason: str = Field(min_length=1)


class Respond(Args):
    """Answer directly when the conversation already contains what is needed."""

    message: str = Field(min_length=1)


TOOLS: dict[str, type[Args]] = {
    "search_flights": SearchFlights,
    "get_offer_details": GetOfferDetails,
    "get_price_advice": GetPriceAdvice,
    "create_price_watch": CreatePriceWatch,
    "list_bookings": ListBookings,
    "prepare_booking": PrepareBooking,
    "prepare_cancellation": PrepareCancellation,
}
OUTCOMES: dict[str, type[Args]] = {
    "clarify": Clarify,
    "decline": Decline,
    "respond": Respond,
}
ALL_ACTIONS: dict[str, type[Args]] = {**TOOLS, **OUTCOMES}


# ---------- Validation ----------


@dataclass
class ValidationResult:
    """The verdict on one call. `error_type` feeds error analysis later."""

    ok: bool
    name: str | None = None
    arguments: Args | None = None
    error_type: Literal["malformed", "unknown_name", "invalid_arguments"] | None = None
    error: str | None = None


def validate_call(call: Any) -> ValidationResult:
    """Check one call against the contract. Never raises; always returns a verdict."""
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return ValidationResult(
            ok=False,
            error_type="malformed",
            error='call must be an object with exactly the keys "name" and "arguments"',
        )

    name, arguments = call["name"], call["arguments"]
    if name not in ALL_ACTIONS:
        return ValidationResult(
            ok=False,
            name=str(name),
            error_type="unknown_name",
            error=f"unknown name: {name!r}",
        )
    if not isinstance(arguments, dict):
        return ValidationResult(
            ok=False,
            name=name,
            error_type="malformed",
            error='"arguments" must be an object',
        )

    try:
        parsed = ALL_ACTIONS[name].model_validate(arguments)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or name}: {err['msg']}"
            for err in exc.errors()
        )
        return ValidationResult(
            ok=False, name=name, error_type="invalid_arguments", error=problems
        )
    return ValidationResult(ok=True, name=name, arguments=parsed)


def tool_schemas() -> list[dict[str, Any]]:
    """The definitions in the JSON-schema format chat models expect in their prompt."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": model.__doc__,
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in ALL_ACTIONS.items()
    ]
