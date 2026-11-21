# Scoring rules

These rules are fixed before any model is scored. Every model (base, commercial, LoRA) is graded by the same code: `src/skylark_lora/scoring.py`.

## Input

A predictions file has one line per test case:

```json
{"id": "clarify-002", "output": "<whatever the model wrote>"}
```

A missing line counts as an empty answer. A line for an unknown ID is an error.

```
python -m skylark_lora.scoring results/base_predictions.jsonl --details results/base_scored.jsonl
```

## The four checks

Each answer is checked in order. The first failure is its error type.

| Check | Passes when | Error type if not |
|---|---|---|
| 1. Parse | A JSON call can be found in the output | `malformed` |
| 2. Validate | The call follows the tool contract (`validate_call`) | `unknown_name` or `invalid_arguments` |
| 3. Action | The tool or outcome name matches the answer key | `wrong_action` |
| 4. Arguments | Every argument matches the answer key | `wrong_arguments` |

**A case is correct only if it passes all four.** There is no partial credit.

## Rules

- **Lenient parsing.** The first JSON object with a `"name"` key is used, even inside prose or a code fence. Arguments written as a JSON string are decoded.
- **Format compliance is reported separately.** An output is compliant only if it is exactly one JSON call, optionally inside `<tool_call>` tags, with `arguments` as an object.
- **Meaning, not spelling.** Both calls are compared with defaults filled in, so `"adults": 1` written out equals `adults` left out. A non-default value the user didn't ask for (e.g. `max_stops: 0`) is wrong.
- **Outcomes.** For `clarify`, `decline` and `respond`, only the name is scored. The free text is not.

## Metrics

| Metric | Meaning |
|---|---|
| `accuracy` | Share of cases fully correct. **The headline number.** |
| `format_compliance` | Share of outputs that are exactly one clean call |
| `valid_rate` | Share of outputs that parse and follow the contract |
| `action_accuracy` | Share with the right tool or outcome, regardless of arguments |
| `over_calling` | Of cases that need clarify, decline or respond, the share where the model called a tool instead |
| `under_calling` | Of cases that need a tool, the share where the model asked, declined or responded instead |
| `by_category` | Accuracy for each of the 10 categories |
| `errors` | Count of each error type |
| `by_argument` | For cases with the right tool and a valid call: how often each argument was right |

Latency and cost are measured when the models run (Step 4), not here.
