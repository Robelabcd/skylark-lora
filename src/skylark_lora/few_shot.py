"""Four worked examples for the few-shot baseline.

Written separately from the test set, with different cities and wording,
so the few-shot prompt never shows the model an exam answer.
"""

FEW_SHOT_EXAMPLES = [
    {
        "today": "2027-04-06",
        "context": None,
        "user": "Two of us want to fly Chicago to Austin on April 22, back the 26th",
        "expected": {
            "name": "search_flights",
            "arguments": {
                "origin": "CHI",
                "destination": "AUS",
                "departure_date": "2027-04-22",
                "return_date": "2027-04-26",
                "adults": 2,
            },
        },
    },
    {
        "today": "2027-04-06",
        "context": (
            "search results OF-501 (Delta, nonstop, $210) and OF-502 (United, 1 stop, $185)"
        ),
        "user": "Book the cheaper one, main fare",
        "expected": {
            "name": "prepare_booking",
            "arguments": {"offer_id": "OF-502", "fare": "main"},
        },
    },
    {
        "today": "2027-04-06",
        "context": None,
        "user": "I need a flight from Phoenix to Seattle",
        "expected": {
            "name": "clarify",
            "arguments": {"question": "What date would you like to fly?"},
        },
    },
    {
        "today": "2027-04-06",
        "context": None,
        "user": "Can you recommend good restaurants in Austin?",
        "expected": {
            "name": "decline",
            "arguments": {"reason": "I can only help with flights."},
        },
    },
]
