import pytest

from skylark_lora.tools import ALL_ACTIONS, tool_schemas, validate_call


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


VALID_CALLS = [
    call(
        "search_flights", origin="IAD", destination="SFO", departure_date="2026-12-18"
    ),
    call(
        "search_flights",
        origin="BWI",
        destination="ORD",
        departure_date="2026-12-18",
        return_date="2026-12-22",
        adults=2,
        cabin="business",
        max_stops=0,
    ),
    call("get_offer_details", offer_id="OF123"),
    call("get_price_advice", offer_id="OF123"),
    call("create_price_watch", offer_id="OF123"),
    call("create_price_watch", offer_id="OF123", target_price=250.0),
    call("list_bookings"),
    call("prepare_booking", offer_id="OF123", fare="main"),
    call("prepare_cancellation", booking_id="BK456"),
    call("clarify", question="Which airport are you leaving from?"),
    call("decline", reason="Skylark only handles flights, not hotels."),
    call("respond", message="The cheapest option was the 9:05 AM United flight."),
]


@pytest.mark.parametrize("c", VALID_CALLS, ids=lambda c: c["name"])
def test_valid_calls_pass(c):
    result = validate_call(c)
    assert result.ok, result.error


INVALID_ARGUMENTS = [
    (
        "lowercase airport",
        call(
            "search_flights",
            origin="iad",
            destination="SFO",
            departure_date="2026-12-18",
        ),
    ),
    (
        "city name not code",
        call(
            "search_flights",
            origin="Washington",
            destination="SFO",
            departure_date="2026-12-18",
        ),
    ),
    (
        "same origin and destination",
        call(
            "search_flights",
            origin="IAD",
            destination="IAD",
            departure_date="2026-12-18",
        ),
    ),
    (
        "wrong date format",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="12/18/2026",
        ),
    ),
    (
        "compact date",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="20261218",
        ),
    ),
    (
        "impossible date",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-02-30",
        ),
    ),
    (
        "return before departure",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-12-18",
            return_date="2026-12-10",
        ),
    ),
    (
        "adults as string",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-12-18",
            adults="2",
        ),
    ),
    (
        "zero adults",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-12-18",
            adults=0,
        ),
    ),
    (
        "unknown cabin",
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-12-18",
            cabin="first",
        ),
    ),
    (
        "missing required argument",
        call("search_flights", origin="IAD", destination="SFO"),
    ),
    ("extra argument", call("get_offer_details", offer_id="OF123", seat="12A")),
    (
        "payment field sneaked in",
        call("prepare_booking", offer_id="OF123", fare="main", card_number="4111"),
    ),
    ("unknown fare", call("prepare_booking", offer_id="OF123", fare="premium")),
    (
        "negative target price",
        call("create_price_watch", offer_id="OF123", target_price=-5.0),
    ),
    ("empty offer id", call("get_price_advice", offer_id="")),
    ("empty clarify question", call("clarify", question="")),
]


@pytest.mark.parametrize(
    "label,c", INVALID_ARGUMENTS, ids=[label for label, _ in INVALID_ARGUMENTS]
)
def test_invalid_arguments_rejected(label, c):
    result = validate_call(c)
    assert not result.ok
    assert result.error_type == "invalid_arguments"


@pytest.mark.parametrize(
    "c",
    [
        "search_flights(origin='IAD')",
        {"name": "search_flights"},
        {"name": "list_bookings", "arguments": {}, "extra": 1},
        {"name": "get_offer_details", "arguments": "OF123"},
    ],
)
def test_malformed_calls_rejected(c):
    result = validate_call(c)
    assert not result.ok
    assert result.error_type == "malformed"


def test_unknown_tool_rejected():
    result = validate_call(call("book_hotel", city="Paris"))
    assert not result.ok
    assert result.error_type == "unknown_name"


def test_error_message_names_the_problem():
    result = validate_call(
        call(
            "search_flights",
            origin="IAD",
            destination="SFO",
            departure_date="2026-12-18",
            return_date="2026-12-10",
        )
    )
    assert "return_date" in result.error


def test_schemas_cover_every_action():
    schemas = tool_schemas()
    assert [s["function"]["name"] for s in schemas] == list(ALL_ACTIONS)
    assert all(s["function"]["description"] for s in schemas)
