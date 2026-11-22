# LoRA, implemented from scratch

Code: `src/skylark_lora/lora.py`. Proof of correctness: `tests/test_lora.py`, which compares it with Hugging Face PEFT.

## The idea

Training all ~1.7B weights of Qwen3-1.7B needs ~27 GB of optimizer state and risks damaging what the model already knows. LoRA freezes every original weight and learns a small correction next to selected layers. For a frozen layer with weight W₀ (d_out × d_in):

```latex
h = W_0 x + \frac{\alpha}{r} \, B A x
```

| Piece | Shape | Role |
|---|---|---|
| W₀ | d_out × d_in | Original weights, frozen |
| A | r × d_in | Projects the input down to r numbers; trainable, random start |
| B | d_out × r | Projects back up; trainable, **starts at zero** |
| α / r | scalar | Keeps the update size comparable across ranks |

**Why it works:** adapting to one narrow task changes the weights in only a few directions, so the change ΔW = B·A can have low rank (Hu et al., 2021).

**Why B starts at zero:** then B·A = 0, so the model starts identical to the original. A and B can't both be zero: each one's gradient depends on the other, so neither would ever learn.

**After training:** W = W₀ + (α/r)·B·A can be computed once (merge), so inference is exactly as fast as the original model.

## Settings

| Setting | Value | Why |
|---|---|---|
| Layers | Attention q, k, v, o in all 28 blocks | Classic LoRA setup; adding feed-forward layers is a Step 9 ablation |
| Rank r | 16 | Enough capacity for one narrow task; ranks 4/8/16 compared in Step 9 |
| α | 32 (scaling 2) | Standard α = 2r |
| Dropout | 0.05 on the LoRA path | Light regularization |
| Precision | Base model in bf16, A and B in fp32 | 1.7B fits on an L4/A100 without 4-bit quantization; fp32 adapters train stably |

## Size for Qwen3-1.7B

Qwen3-1.7B: hidden size 2048, 28 blocks, 16 query heads and 8 key/value heads of size 128. Each LoRA layer adds r × (d_in + d_out) parameters:

| Layer | Shape (d_in → d_out) | LoRA parameters (r = 16) |
|---|---|---|
| q_proj | 2048 → 2048 | 65,536 |
| k_proj | 2048 → 1024 | 49,152 |
| v_proj | 2048 → 1024 | 49,152 |
| o_proj | 2048 → 2048 | 65,536 |
| **Per block** | | **229,376** |
| **28 blocks** | | **6,422,528 (~0.4% of the model)** |

The adapter file is about 26 MB (fp32), versus about 3.4 GB for the full model in bf16. Adding the feed-forward layers would raise the count to 17.4M. These numbers are computed from the published configuration and checked against the loaded model in Step 7.

## Proof of correctness

The tests build a tiny, randomly initialized Qwen3 model (same architecture, 64-dimensional, 2 blocks), so they run on a laptop CPU in seconds:

| Test | Shows |
|---|---|
| Layer starts identical / model starts identical | B = 0 means no change before training |
| Layer matches the formula | The forward pass computes W₀x + (α/r)·BAx |
| Only A and B are trainable; training changes only A and B | Gradients never touch W₀ |
| Same trainable-parameter count as PEFT | The right layers get the right shapes |
| **Same output as PEFT** | With the same A and B, logits match PEFT to 1e-5 |
| **Same merged weights as PEFT** | merge() produces PEFT's merged W |
| Merge keeps output; unmerge restores weights | Merging is exact and reversible |
| Save/load round trip; wrong rank rejected | Adapters are saved alone and reloaded safely |

The tests were also checked by deliberately breaking the layer (scaling α instead of α/r; corrupting B). The PEFT comparison failed both times, so the tests catch real mistakes.
