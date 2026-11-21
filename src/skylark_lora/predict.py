"""Run any model over the test set and save its answers.

A model is just a function: chat messages in, (text, extra info) out.
That keeps this loop identical for Qwen, the LoRA model and the OpenAI model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from skylark_lora.eval_set import EvalCase
from skylark_lora.prompting import to_chat_messages

Generate = Callable[[list[dict[str, str]]], tuple[str, dict[str, Any]]]


def run_predictions(
    cases: list[EvalCase],
    generate: Generate,
    out_path: Path,
    few_shot: bool = False,
) -> int:
    """Answer every case not already in out_path, appending one JSON line each.

    Already-answered cases are skipped, so an interrupted run (Colab timeout,
    network error) continues where it stopped instead of paying twice.
    Returns the number of new answers written.
    """
    done = _answered_ids(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for case in cases:
            if case.id in done:
                continue
            start = time.perf_counter()
            text, extra = generate(to_chat_messages(case, few_shot))
            latency_ms = round((time.perf_counter() - start) * 1000)
            row = {"id": case.id, "output": text, "latency_ms": latency_ms, **extra}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    return written


def _answered_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["id"] for line in lines if line.strip()}
