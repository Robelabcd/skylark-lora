"""Run an OpenAI model on the test set, with the same prompt as every other model.

Needs OPENAI_API_KEY in .env (on your Mac) or in the environment (Colab).

    python -m skylark_lora.run_openai --model <model name> \\
        --out results/openai_<model name>_zero_shot.jsonl

Pick a current small model from the OpenAI pricing page and record its exact
name in docs/decisions.md. Token counts are saved, so cost can be reported.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from skylark_lora.eval_set import load_eval_set
from skylark_lora.predict import Generate, run_predictions

MAX_ATTEMPTS = 3


def make_generator(client: Any, model: str) -> Generate:
    """Wrap an OpenAI client as a generate function. `client` is injectable for tests."""

    def generate(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages
                )
                break
            except Exception:
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(2**attempt)
        usage = response.usage
        return response.choices[0].message.content or "", {
            "prompt_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

    return generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an OpenAI model on the test set.")
    parser.add_argument("--model", required=True, help="OpenAI model name")
    parser.add_argument(
        "--out", type=Path, required=True, help="Predictions file (JSONL)"
    )
    parser.add_argument("--few-shot", action="store_true", help="Add worked examples")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    generate = make_generator(OpenAI(), args.model)
    written = run_predictions(
        load_eval_set(), generate, args.out, few_shot=args.few_shot
    )
    print(f"Wrote {written} new answers to {args.out}")


if __name__ == "__main__":
    main()
