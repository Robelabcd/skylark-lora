"""LoRA (Low-Rank Adaptation), written from scratch in PyTorch.

For a frozen linear layer with weight W0, LoRA computes

    h = W0 x + (alpha / r) * B (A x)

A (r x d_in) starts random and B (d_out x r) starts at zero, so before training
the layer behaves exactly like the original. Only A and B are trained.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
tests/test_lora.py checks this implementation against Hugging Face PEFT.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank correction B @ A."""

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank r must be positive, got {r}")
        self.base = base
        self.base.requires_grad_(False)  # W0 (and bias) never change
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        device = base.weight.device
        # A: (r, d_in), random like nn.Linear's default init. B: (d_out, r), zeros.
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:  # the correction is already folded into W0
            return out
        h = self.dropout(x.to(self.lora_A.dtype))
        update = (h @ self.lora_A.T) @ self.lora_B.T  # x -> r numbers -> d_out numbers
        return out + (update * self.scaling).to(out.dtype)

    def delta_weight(self) -> torch.Tensor:
        """The full-size change this adapter represents: (alpha / r) * B @ A."""
        return self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the correction into W0, so inference costs nothing extra."""
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo merge(), for example to continue training."""
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, merged={self.merged}"


def add_lora(
    model: nn.Module,
    targets: Iterable[str] = ATTENTION,
    r: int = 16,
    alpha: float = 32,
    dropout: float = 0.05,
) -> list[str]:
    """Freeze the whole model, then wrap every nn.Linear whose name ends in a target.

    Returns the names of the wrapped layers. Raises if nothing matched, so a typo
    in a target name can't silently train nothing.
    """
    targets = tuple(targets)
    model.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        if name.split(".")[-1] in targets and isinstance(module, nn.Linear):
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            )
            wrapped.append(name)
    if not wrapped:
        raise ValueError(f"no nn.Linear layers named {targets} found")
    return wrapped


def lora_layers(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_lora(model: nn.Module) -> None:
    for layer in lora_layers(model):
        layer.merge()


def unmerge_lora(model: nn.Module) -> None:
    for layer in lora_layers(model):
        layer.unmerge()


# ---------- Saving and loading just the adapter ----------


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the A and B matrices: a few MB instead of the whole model."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith((".lora_A", ".lora_B"))
    }


def save_lora(model: nn.Module, path: str | Path, config: dict) -> None:
    """Save the adapter weights plus the settings needed to rebuild it (r, alpha, targets)."""
    from safetensors.torch import save_file

    save_file(
        lora_state_dict(model), str(path), metadata={"lora_config": json.dumps(config)}
    )


def load_lora(model: nn.Module, path: str | Path) -> dict:
    """Load adapter weights into a model that already has add_lora() applied.

    Returns the saved config. Raises if any adapter weight is missing or extra,
    or if the saved r and alpha differ from the model's.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        config = json.loads(f.metadata()["lora_config"])
    for layer in lora_layers(model):
        if (layer.r, layer.alpha) != (config["r"], config["alpha"]):
            raise ValueError(
                f"adapter has r={config['r']}, alpha={config['alpha']}; model has r={layer.r}, alpha={layer.alpha}"
            )
    weights = load_file(str(path))
    expected = set(lora_state_dict(model))
    if set(weights) != expected:
        raise ValueError(
            f"adapter keys don't match: missing {expected - set(weights)}, extra {set(weights) - expected}"
        )
    model.load_state_dict(weights, strict=False)
    return config
