# Skylark LoRA: Teaching a Small Open LLM to Call Tools

Fine-tuning a small open-weight language model with LoRA so it can choose and
fill in tool calls for **Skylark**, an AI airline booking agent, and comparing
it with prompting baselines and a commercial model on accuracy, latency and cost.

> **Status:** work in progress. Final project for CSCI 6366 (Neural Networks
> and Deep Learning), George Washington University, Fall 2026.

## Approach (planned)

1. Define Skylark's tools and a hand-written test set
2. Measure baselines: base model (zero- and few-shot) and a commercial model
3. Generate and verify a custom training set
4. Implement LoRA from scratch in PyTorch; verify against Hugging Face PEFT
5. Fine-tune, evaluate, and analyze errors and ablations

## Setup

```bash
git clone https://github.com/Robelabcd/skylark-lora.git
cd skylark-lora
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
