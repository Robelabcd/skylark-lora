# Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-18 | Public repo `Robelabcd/skylark-lora`, MIT license | Portfolio + course deliverable; secrets kept out via .gitignore |
| 2026-11-18 | Code in `src/` package, thin notebooks | Reproducible, testable, readable |
| 2026-11-18 | Base model: Qwen instruct, 1.5–4B (confirm at Step 4) | Strong small models with tool-calling support |
| 2026-11-18 | Teacher for training data: an open model, not GPT | No terms restrictions; data and model can be published and used commercially |
| 2026-11-21 | 7 tools, including `list_bookings` | Users say "cancel my SFO trip", not a booking ID; the model needs a way to look bookings up |
| 2026-11-21 | `clarify`, `decline`, `respond` as explicit outcomes | Knowing when *not* to call a tool is scored exactly like a tool call |
| 2026-11-21 | Strict validation: no type coercion, no extra arguments, IATA codes, YYYY-MM-DD dates | One correct answer per request, so scoring is exact; no payment fields can slip in |
| 2026-11-21 | Lean inputs: adults only, no children/infants or flexible dates | Fewer arguments means cleaner results within the course timeline; can be added later |
| 2026-11-21 | Multi-airport cities use IATA metro codes (WAS, NYC, CHI) | Matches how real flight search works; no needless questions to the user |
| 2026-11-21 | Test set: 100 cases in 10 categories, finished before any training data exists | A fair exam: never trained on, not shaped by the data generator, frozen before scoring |
| 2026-11-21 | Test set stored as a pretty-printed JSON list, checked on load | Easy to write by hand; mistakes in the answer key fail loudly |
| 2026-11-21 | Test set complete: 100 cases. Answers drafted with Claude; every user message rewritten by Robel; all 100 reviewed by Robel | Fast enough for the course timeline while keeping human phrasing and human review; stated openly |
| 2026-11-21 | Scoring: all-or-nothing per case, no partial credit | A search with a wrong date is a wrong search; matches standard tool-calling benchmarks |
| 2026-11-21 | Lenient parsing, with format compliance reported separately | Grades reasoning fairly while still showing formatting gains from LoRA |
| 2026-11-21 | Calls compared with defaults filled in; outcome text not scored | Same meaning scores the same; clarify/decline/respond wording has no single right answer |
| 2026-11-21 | Base model: Qwen/Qwen3-1.7B (Apache 2.0), non-thinking mode, greedy decoding | Text-only with standard attention, so a hand-written LoRA layer drops cleanly into q/k/v/o; deterministic answers |
| 2026-11-21 | One shared prompt and JSON output format for every model; tool results shown as user messages | Same exam for all models; works with any chat API; the scorer needs no per-model code |
| 2026-11-21 | Baselines: Qwen zero-shot, Qwen few-shot (4 examples not from the test set), one OpenAI model | Measures the starting point, whether better prompting is enough, and a commercial reference |
