# Skylark tool contract

For every user turn, the model outputs exactly one call:

```json
{"name": "search_flights", "arguments": {"origin": "IAD", "destination": "SFO", "departure_date": "2026-12-18"}}
```

The code lives in `src/skylark_lora/tools.py`. `validate_call()` accepts or rejects any call and explains why.

## Tools (they do something)

| Name | Purpose | Arguments (required in **bold**) |
|---|---|---|
| `search_flights` | Find flights | **origin**, **destination**, **departure_date**, return_date, adults (1–9, default 1), cabin (`economy` / `premium` / `business`, default economy), max_stops (0–2) |
| `get_offer_details` | Full details of one offer | **offer_id** |
| `get_price_advice` | Buy now or wait? | **offer_id** |
| `create_price_watch` | Alert when a price drops | **offer_id**, target_price (USD, > 0) |
| `list_bookings` | Show the user's bookings | none |
| `prepare_booking` | Quote a booking | **offer_id**, **fare** (`basic` / `main` / `flex`) |
| `prepare_cancellation` | Quote a cancellation | **booking_id** |

## Outcomes (no tool is called)

| Name | When | Argument |
|---|---|---|
| `clarify` | Required information is missing or ambiguous | **question** |
| `decline` | The request is outside Skylark's scope | **reason** |
| `respond` | The answer is already in the conversation | **message** |

For outcomes, only the name is scored exactly; the free text is checked separately.

## Rules

- Airports are 3-letter uppercase IATA codes (`IAD`, not `Dulles` or `iad`).
- Dates are `YYYY-MM-DD`. Today's date is given in the system prompt, so relative dates ("next Friday") have one correct answer.
- No type guessing: `2` is valid for `adults`, `"2"` is not.
- Unknown argument names are errors, so no payment field can slip into a call.
- Origin and destination must differ; a return date cannot be before departure.
- The model proposes, the user confirms: `prepare_booking` and `prepare_cancellation` only produce quotes. Nothing is charged or changed until the user confirms in the app.

## Error types

`validate_call()` labels every failure so errors can be counted later:

| Type | Meaning | Example |
|---|---|---|
| `malformed` | Not shaped like a call | plain text, missing `arguments` |
| `unknown_name` | Name is not a tool or outcome | `book_hotel` |
| `invalid_arguments` | Right tool, wrong arguments | `departure_date: "12/18/2026"` |

Checks that need the conversation (is it the right tool? the right airport?) belong to the scorer in Step 3, not here.
