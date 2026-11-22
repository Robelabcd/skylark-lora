"""Build the training and dev sets.

    python -m skylark_lora.datagen.build

Writes data/train/train.jsonl and data/train/dev.jsonl. The same seed always
produces the same files, so the data is rebuilt instead of committed.

Every example is checked with the same rules as the test set, and examples
whose user messages match a test-set message are rejected.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pydantic import Field

from skylark_lora.datagen.scenarios import BUILDERS, MIX
from skylark_lora.eval_set import EvalCase, load_eval_set

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "train"


class TrainCase(EvalCase):
    """A training example: the same format and checks as a test case, with longer IDs."""

    id: str = Field(pattern=r"^[a-z]+-\d{5}$")


def normalize(message: str) -> str:
    return " ".join(message.lower().split()).rstrip(".?!")


def exam_user_messages() -> set[str]:
    return {
        normalize(m.content)
        for c in load_eval_set()
        for m in c.messages
        if m.role == "user"
    }


def generate(
    n: int, seed: int, split: str, forbidden: set[str], seen: set[str] | None = None
) -> tuple[list[dict], Counter]:
    """n checked examples in the MIX proportions.

    forbidden: normalized user messages that must not appear (the test set's).
    seen: conversations already used (e.g. by the training set), so dev never repeats one.
    """
    rng = random.Random(seed)
    targets = {cat: round(n * share) for cat, share in MIX.items()}
    targets[max(targets, key=targets.get)] += n - sum(targets.values())
    stats: Counter = Counter()
    seen = set() if seen is None else seen
    examples: list[dict] = []
    for category, target in targets.items():
        made = attempts = 0
        while made < target:
            attempts += 1
            if attempts > 50 * target + 100:
                raise RuntimeError(
                    f"could not generate enough distinct {category} examples"
                )
            raw = BUILDERS[category](rng)
            user_messages = [
                normalize(m["content"]) for m in raw["messages"] if m["role"] == "user"
            ]
            if any(m in forbidden for m in user_messages):
                stats["rejected_test_overlap"] += 1
                continue
            key = json.dumps(raw["messages"], sort_keys=True)
            if key in seen:
                stats["rejected_duplicate"] += 1
                continue
            item = {"id": f"{split}-{len(examples):05d}", **raw}
            TrainCase.model_validate(item)  # raises if any rule is broken
            seen.add(key)
            examples.append(item)
            made += 1
    rng.shuffle(examples)
    for i, item in enumerate(examples):
        item["id"] = f"{split}-{i:05d}"
    return examples, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the training and dev sets.")
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--dev", type=int, default=400)
    parser.add_argument("--seed", type=int, default=6366)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    forbidden = exam_user_messages()
    seen: set[str] = set()
    train, train_stats = generate(args.train, args.seed, "train", forbidden, seen)
    dev, dev_stats = generate(args.dev, args.seed + 1, "dev", forbidden, seen)

    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "dev.jsonl", dev)
    print(
        f"train: {len(train)} examples  {dict(Counter(r['category'] for r in train))}"
    )
    print(f"dev:   {len(dev)} examples")
    print(f"rejected: train {dict(train_stats)}, dev {dict(dev_stats)}")


if __name__ == "__main__":
    main()
