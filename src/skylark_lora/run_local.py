"""Run an open model (e.g. Qwen3-1.7B) on the test set with Hugging Face transformers.

Run this on a GPU (Colab):

    python -m skylark_lora.run_local --model Qwen/Qwen3-1.7B \\
        --out results/qwen3-1.7b_zero_shot.jsonl

Decoding is greedy (no randomness), so the same model gives the same answers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from skylark_lora.eval_set import load_eval_set
from skylark_lora.predict import Generate, run_predictions


def load_generator(model_id: str, max_new_tokens: int = 256) -> Generate:
    """Load a model and return a generate function. Imports torch only when used."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    def generate(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: answer directly, no reasoning trace
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(new_tokens.shape[0]),
        }

    return generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an open model on the test set.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument(
        "--out", type=Path, required=True, help="Predictions file (JSONL)"
    )
    parser.add_argument("--few-shot", action="store_true", help="Add worked examples")
    args = parser.parse_args(argv)

    written = run_predictions(
        load_eval_set(), load_generator(args.model), args.out, few_shot=args.few_shot
    )
    print(f"Wrote {written} new answers to {args.out}")


if __name__ == "__main__":
    main()
