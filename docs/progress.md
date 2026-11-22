# Progress log

Read this first at the start of every session. Update it at the end of every session.

## Next session starts here

**Step 7: training loop and first training run.** Start with the concepts (no code yet):

- How a training example becomes tokens: the prompt from `prompting.py` plus the expected call as the target.
- Why the loss only counts the answer tokens (the prompt is masked out).
- The loop: batches, AdamW on the LoRA parameters only, learning rate schedule, checkpoints, dev loss.
- Then the first real run on Colab (L4 or A100), followed by scoring on the 100-case test set.

Before starting: activate the environment and confirm `(.venv)` in the prompt (see Things to remember).

## Where things stand (2026-11-22)

| Step | What | Status |
|---|---|---|
| 0 | Project setup: repo, package, tests, lint, secrets | ✅ Done |
| 1 | Tool contract: 7 tools + clarify/decline/respond, strict validation | ✅ Done |
| 2 | Test set: 100 cases in 10 categories, frozen | ✅ Done |
| 3 | Scorer: parse → validate → action → arguments | ✅ Done |
| 4 | Baselines: prompt builder, model runners, Colab notebook | ✅ Done (Qwen); OpenAI baseline with teammate |
| 5 | Training data generator: 4,000 train + 400 dev, 10 held-out airports | ✅ Done |
| 6 | LoRA from scratch, verified against PEFT | ✅ Done |
| 7 | Training loop and first run | ⏭️ Next |
| 8 | Evaluation and error analysis | |
| 9 | Ablations (rank, layers, data size, paraphrasing) | |
| 10 | Slides and demo (presentation **Dec 9**) | |

## Results so far

Qwen3-1.7B before fine-tuning (100-case test set, Colab T4):

| Run | Accuracy | Format compliance | Over-calling | Median latency |
|---|---|---|---|---|
| Zero-shot | 40% | 100% | 39% | 4.6 s |
| Few-shot | 44% | 100% | 50% | 5.3 s |

Main failure patterns: invented airport codes (Orlando → ORL), wrong dates (year rollover, weekdays), `clarify` instead of `decline`, never `respond` (0/6), acting when it should ask. Formatting is not the problem.

## Key numbers to remember

| | |
|---|---|
| Training data | 4,000 train + 400 dev, seed 6366, 38% clarify/decline/respond |
| Held-out airports | MSY, KEF, CUN, ADD, PIT, ANC, LIS, AMS, DUB, SAN (9 test cases use them) |
| LoRA settings | q, k, v, o in all 28 blocks; r = 16, α = 32 (scaling 2), dropout 0.05 |
| Trainable parameters | 6,422,528 (~0.4% of the model); adapter file ~26 MB |
| Tests | 414 passing |

## Session log

### 2026-11-22: Steps 5–6

**Step 5: training data generator** (`src/skylark_lora/datagen/`)

- Code picks the facts, code computes the correct call, a template writes the message, and the test-set checker validates every example. Answers are correct by construction.
- `airports.json` (109 airports, 6 metros, 5 ambiguous cities), `places.py`, `dates.py`, `world.py`, `text.py`, `scenarios.py` (one builder per category), `build.py` (mix, checks, output).
- Guards: no test-set message reused, no duplicates, 10 airports held out, same seed gives the same data. My Mac reproduced the exact files (rejected 330 / 44).
- Run with `python -m skylark_lora.datagen.build`. Output goes to `data/train/`, which is not committed.

**Step 6: LoRA from scratch** (`src/skylark_lora/lora.py`)

- `h = W₀x + (α/r)·B(Ax)`: W₀ frozen, A random, B starts at zero, so the model starts identical to the original.
- `LoRALinear`, `add_lora` (fails loudly on a typo), `merge`/`unmerge`, `save_lora`/`load_lora` (adapter only, refuses the wrong rank).
- Proven against Hugging Face PEFT on a tiny random Qwen3 model: same parameter count, same outputs (to 1e-5), same merged weights. Deliberately breaking the layer made the tests fail, so they catch real bugs.

**Environment lesson:** a new terminal started without `(.venv)`, so pip installed into Anaconda's base environment. Repaired it: removed the project and openai packages from base, restored h11 0.14.0, and `pip check` now shows only Anaconda's pre-existing conflicts.

### 2026-11-21: Steps 0–4

- Set up the repo `Robelabcd/skylark-lora` (src layout, pytest, ruff, `.env` for secrets).
- Defined the tool contract (`tools.py`) and the test-set format and checker (`eval_set.py`).
- Completed the 100-case test set: answers drafted with Claude, every user message rewritten by me, all reviewed by me. Frozen.
- Built the scorer (`scoring.py`): all-or-nothing, lenient parsing, format compliance reported separately.
- Built the prompt builder, the runners and the baseline notebook; ran the Qwen3-1.7B baselines (results above).
- Set up the Hugging Face token (Colab Secrets) and the OpenAI key (`.env`, ignored by git).
- Answered the professor's follow-up questions on dataset and GPU; a new teammate joined, who owns the OpenAI baseline (guide written).

## Loose ends

- [ ] Teammate added as a collaborator on GitHub; guide link sent; target date agreed
- [ ] Professor email sent (dataset, GPU, new teammate, resubmission question), and his reply noted here
- [ ] OpenAI: auto-reload off or monthly budget set; old `llm_project` key deleted if unused
- [ ] VS Code: Python: Select Interpreter → `.venv`, so new terminals start with `(.venv)`
- [ ] Optional: delete `~/.git-home-backup`; switch the VS Code GitHub account to Robelabcd

## Things to remember

- **Check for `(.venv)` in the prompt before any pip or python command.** If it's missing: `source .venv/bin/activate`, then `which python` should end in `skylark-lora/.venv/bin/python`.
- Run tests with `python -m pytest`, not `pytest`.
- Install with `python -m pip install -e ".[dev,openai,train]"`.
- Rebuild training data with `python -m skylark_lora.datagen.build`.
- Secrets live only in `.env` (⚙️ tab) and Colab Secrets, never in `.env.example` or code.
- Baseline notebook: https://colab.research.google.com/github/Robelabcd/skylark-lora/blob/main/notebooks/01_baselines.ipynb
- Every decision and its reason goes in `docs/decisions.md`.
