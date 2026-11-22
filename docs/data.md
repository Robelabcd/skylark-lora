# Training data

```
python -m skylark_lora.datagen.build
```

This writes `data/train/train.jsonl` (4,000 examples) and `data/train/dev.jsonl` (400) in under a second. The seed is fixed (6366), so the same command always produces the same files. The data is rebuilt rather than committed; `docs/train_sample.json` shows 20 examples.

## How an example is made

1. **Code picks the facts:** a route, a date, passengers, a search result, a booking.
2. **Code computes the correct call** from those facts, so every answer is right by construction.
3. **A template writes the user's message,** with optional messiness (lowercase, typos, shorthand, arrows).
4. **The example goes through the test-set checker** (`EvalCase`): valid call, dates not in the past, IDs present in the conversation.

Every example has the same format as a test case, and training uses the same prompt (`prompting.py`) as the exam.

## What it teaches

The mix targets the baseline's weaknesses: 38% of examples are clarify, decline or respond.

| Category | Share | Targets |
|---|---|---|
| `simple_search` | 10% | Airport codes for ~100 airports, from city names, airport names and nicknames |
| `search_options` | 10% | Return dates, passenger counts, cabins, stops |
| `relative_date` | 10% | "tomorrow", weekdays, "the 22nd", "in two weeks", year rollover |
| `city_airport` | 8% | Metro codes (WAS, NYC, CHI, LON, PAR, TYO) vs named airports |
| `messy` | 5% | Typos, shorthand, all caps |
| `follow_up` | 12% | Acting on the right offer ("the cheapest one", "the 7:10 one"), fares, price watches, multi-turn "it" |
| `bookings` | 7% | Look up bookings first; cancel the right one |
| `clarify` | 16% | Missing origin, destination, date, return date or fare; ambiguous cities (Portland, Houston, Columbus...); which offer or booking |
| `decline` | 12% | Hotels, cars, rides, visas, weather, restaurants, trains, seats, payment details, off-topic |
| `respond` | 10% | Prices, times, stops, booking dates already shown |

## Guards

| Guard | How |
|---|---|
| No test-set message reused | Any example with a user message matching one in the test set is rejected |
| No duplicates | Identical conversations are rejected; dev never repeats a training conversation |
| No test IDs | Training uses OF-5000+ and BK-2000+; the test set uses other IDs |
| Held-out airports | 10 airports never appear in training (below) |
| Rules enforced | `tests/test_datagen.py` checks all of the above on every run |

## Held-out airports

MSY, KEF, CUN, ADD, PIT, ANC, LIS, AMS, DUB and SAN never appear in training data, not even by city name. Nine test cases use them in their answers:

`simple_search-005, -009, -011, -012, -013, -014, -015`, `search_options-012`, `messy-003`

Accuracy on these nine cases shows whether the model learned to map cities to codes in general, or only memorized the airports it was trained on.

## Known limits

- Wording comes from templates, while the test set is human-written. Paraphrasing with an open-weight teacher model is the planned stretch, and a planned ablation.
- No children or infants, no flexible-date search, no multi-city trips: the tool contract doesn't support them.
