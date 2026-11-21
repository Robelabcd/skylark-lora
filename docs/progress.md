# Progress log

Read this first at the start of every session. Update it at the end of every session.

## Next session starts here

**Step 5: training data.** The design is agreed in outline; three decisions are still open:

1. Wording: templates only first, teacher-model paraphrasing as a stretch? (recommended: yes)
2. Size: about 4,000 training + 400 dev examples, ~40% clarify/decline/respond? (recommended: yes)
3. Regenerate the data from a fixed seed instead of committing it? (recommended: yes)
4. Hold out 10 airports from training to test generalization? (recommended: yes)

Before starting, check the loose ends below.

## Where things stand (2026-11-21)

| Step | What | Status |
|---|---|---|
| 0 | Project setup: repo, package, tests, lint, secrets | ✅ Done |
| 1 | Tool contract: 7 tools + clarify/decline/respond, strict validation | ✅ Done |
| 2 | Test set: 100 cases in 10 categories, frozen | ✅ Done |
| 3 | Scorer: parse → validate → action → arguments | ✅ Done |
| 4 | Baselines: prompt builder, model runners, Colab notebook | ✅ Done (Qwen); OpenAI baseline with teammate |
| 5 | Training data generator | ⏭️ Next |
| 6 | LoRA layer from scratch, verified against PEFT | |
| 7 | Training loop and first run | |
| 8 | Evaluation and error analysis | |
| 9 | Ablations (rank, data size, paraphrasing) | |
| 10 | Slides and demo (presentation **Dec 9**) | |

## Results so far

Qwen3-1.7B before fine-tuning (100-case test set, Colab T4):

| Run | Accuracy | Format compliance | Over-calling | Median latency |
|---|---|---|---|---|
| Zero-shot | 40% | 100% | 39% | 4.6 s |
| Few-shot | 44% | 100% | 50% | 5.3 s |

Main failure patterns: invented airport codes (Orlando → ORL), wrong dates (year rollover, weekdays), `clarify` instead of `decline`, never `respond` (0/6), acting when it should ask. Formatting is not the problem.

## Session log

### 2026-11-21: Steps 0–4

- Set up the repo `Robelabcd/skylark-lora` (src layout, pytest, ruff, `.env` for secrets).
- Defined the tool contract (`tools.py`) and the test-set format and checker (`eval_set.py`).
- Completed the 100-case test set: answers drafted with Claude, every user message rewritten by me, all reviewed by me. Frozen.
- Built the scorer (`scoring.py`): all-or-nothing, lenient parsing, format compliance reported separately.
- Built the prompt builder, the runners and the baseline notebook; ran the Qwen3-1.7B baselines (results above).
- Set up the Hugging Face token (Colab Secrets) and the OpenAI key (`.env`, ignored by git).
- Answered the professor's follow-up questions on dataset and GPU; a new teammate joined, who owns the OpenAI baseline.

## Loose ends

- [ ] Baseline result files committed to `results/`
- [ ] Teammate added as a collaborator on GitHub; guide link sent; target date agreed
- [ ] Professor email sent (dataset, GPU, new teammate, resubmission question), and his reply noted here
- [ ] OpenAI: auto-reload off or monthly budget set; old `llm_project` key deleted if unused
- [ ] Optional: delete `~/.git-home-backup`; switch the VS Code GitHub account to Robelabcd

## Things to remember

- Run tests with `python -m pytest`, not `pytest`.
- Activate the environment first: `source .venv/bin/activate`.
- Secrets live only in `.env` (⚙️ tab) and Colab Secrets, never in `.env.example` or code.
- Baseline notebook: https://colab.research.google.com/github/Robelabcd/skylark-lora/blob/main/notebooks/01_baselines.ipynb
- Every decision and its reason goes in `docs/decisions.md`.
