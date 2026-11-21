# Writing the test set

The test set (`data/eval/test_set.json`) is the exam every model takes. It is written by hand, never used for training, and frozen before any model is scored.

## Format

Each case is one object in the JSON list:

| Field | Meaning |
|---|---|
| `id` | `<category>-<number>`, e.g. `clarify-007` |
| `category` | One of the 10 categories below |
| `today` | The date the conversation happens (`YYYY-MM-DD`) |
| `messages` | The conversation so far; the last message is always from the user |
| `expected` | The one correct call: `{"name": ..., "arguments": {...}}` |
| `notes` | Optional: why this is the right answer |

Messages come in three kinds:

```json
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}                  // an earlier text reply
{"role": "assistant", "tool_call": {"name": ..., "arguments": {...}}}
{"role": "tool", "name": "search_flights", "content": {...}}  // what the tool returned
```

## Answer rules

Every case must have exactly one correct answer under these rules.

- **Airports:** use the airport's IATA code. For a city with one main airport, use that airport (Seattle → `SEA`). For multi-airport cities named without a specific airport, use the metro code: `WAS` (Washington DC), `NYC` (New York), `CHI` (Chicago), `LON` (London), `PAR` (Paris), `TYO` (Tokyo).
- **Dates:** relative dates are resolved from `today`. A month that has already passed means next year (in December, "Jan 3" is the next January).
- **Defaults:** 1 adult, economy, any number of stops, one-way.
- **Clarify** when: no departure date; no origin; "round trip" without a return date; booking without a fare; "that flight" when it's unclear which one.
- **Decline** when: hotels, cars, visas, payment details, anything not about flights.
- **Respond** when: the answer is already visible in the conversation.
- **Look up before acting:** if the user refers to a booking without its ID, the answer is `list_bookings`, not `clarify`.
- **IDs must be knowable:** an `offer_id` or `booking_id` in the answer must appear in an earlier tool result. The checker enforces this.
- For `clarify`, `decline` and `respond`, only the name is scored exactly. Still write a sensible text.

## Category targets

| Category | Target | Covers |
|---|---|---|
| `simple_search` | 15 | All search information given clearly |
| `search_options` | 12 | Round trip, passengers, cabin, nonstop |
| `relative_date` | 10 | "tomorrow", "this Friday", "the 20th", "Jan 5" |
| `city_airport` | 8 | City or airport names → codes, metro codes |
| `follow_up` | 15 | Details, price advice, price watch, booking on an earlier offer |
| `bookings` | 8 | Listing trips, cancelling a booking |
| `clarify` | 14 | Missing or ambiguous information |
| `decline` | 8 | Out of scope |
| `respond` | 6 | Answer already in the conversation |
| `messy` | 4 | Typos, slang, shorthand, all caps |

## Writing good cases

- **Write how real people type.** Vary length, tone and wording; don't repeat one sentence pattern with different cities.
- **Spread the variety.** Use many different airports, months and `today` dates, not just December and Dulles.
- **Include near-misses**, like "find me a flight and a hotel in Denver". Decide the answer, explain it in `notes`, and stay consistent.
- **Avoid "next Friday".** People disagree on whether it means this week or the following one, so it has no single correct answer.
- **Reuse search contexts.** Follow-up, decline and respond cases can copy the same earlier search and results.
- Run `python -m pytest` often. The checker tells you exactly which case is wrong and why, and shows your progress.

## Freezing

When all 100 cases pass, commit the file and record the commit in `docs/decisions.md`. After that, the test set does not change. If a real mistake is found later, fix it, note it in the decision log, and re-score every model.
