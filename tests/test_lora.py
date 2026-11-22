"""Proof that the hand-written LoRA is correct, checked against Hugging Face PEFT.

Uses a tiny, randomly initialized Qwen3-style model: same architecture as
Qwen3-1.7B, just small, so it runs on a laptop CPU in seconds with no download.
Skipped automatically if torch, transformers or peft are not installed.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from skylark_lora.lora import (
    ATTENTION,
    LoRALinear,
    add_lora,
    count_parameters,
    load_lora,
    lora_layers,
    merge_lora,
    save_lora,
    unmerge_lora,
)

R, ALPHA = 4, 8


@pytest.fixture
def base_model():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.fixture
def tokens():
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 12))


def logits(model, tokens):
    with torch.no_grad():
        return model(input_ids=tokens).logits


def ours(base_model):
    model = copy.deepcopy(base_model)
    add_lora(model, ATTENTION, r=R, alpha=ALPHA, dropout=0.0)
    return model.eval()


def randomize_B(model):
    """B starts at zero; give it values so the adapter actually changes the output."""
    torch.manual_seed(2)
    with torch.no_grad():
        for layer in lora_layers(model):
            layer.lora_B.normal_(std=0.02)


# ---------- One layer ----------


def test_layer_starts_identical_to_base():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    layer = LoRALinear(copy.deepcopy(base), r=R, alpha=ALPHA)
    x = torch.randn(3, 16)
    assert torch.equal(layer(x), base(x))
    assert torch.count_nonzero(layer.lora_B) == 0
    assert torch.count_nonzero(layer.lora_A) > 0
    assert layer.scaling == ALPHA / R


def test_layer_matches_the_formula():
    torch.manual_seed(0)
    layer = LoRALinear(torch.nn.Linear(16, 8), r=R, alpha=ALPHA)
    torch.nn.init.normal_(layer.lora_B)
    x = torch.randn(3, 16)
    expected = layer.base(x) + (ALPHA / R) * (x @ layer.lora_A.T @ layer.lora_B.T)
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(torch.nn.Linear(4, 4), r=0, alpha=1)


# ---------- Whole model ----------


def test_model_starts_identical_to_base(base_model, tokens):
    assert torch.equal(logits(ours(base_model), tokens), logits(base_model, tokens))


def test_wraps_exactly_the_attention_layers(base_model):
    model = copy.deepcopy(base_model)
    wrapped = add_lora(model, ATTENTION, r=R, alpha=ALPHA)
    assert len(wrapped) == 2 * 4  # 2 blocks x q, k, v, o
    assert all(name.endswith(ATTENTION) for name in wrapped)


def test_typo_in_targets_fails_loudly(base_model):
    with pytest.raises(ValueError, match="no nn.Linear"):
        add_lora(copy.deepcopy(base_model), ["qproj"], r=R, alpha=ALPHA)


def test_only_A_and_B_are_trainable(base_model):
    model = ours(base_model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(
        name.endswith((".lora_A", ".lora_B")) for name in trainable
    )
    # r x (d_in + d_out) per layer: q and o are 64->64, k and v are 64->32
    expected = 2 * R * ((64 + 64) + (64 + 32) + (64 + 32) + (64 + 64))
    assert count_parameters(model)[0] == expected


def test_training_changes_only_A_and_B(base_model, tokens):
    model = ours(base_model).train()
    frozen_before = {
        n: p.clone() for n, p in model.named_parameters() if not p.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    for _ in range(3):
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None
            assert torch.equal(p, frozen_before[name]), name
    assert any(torch.count_nonzero(layer.lora_B) > 0 for layer in lora_layers(model))


# ---------- The proof: same numbers as PEFT ----------


def peft_model(base_model):
    config = peft.LoraConfig(
        r=R, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=list(ATTENTION)
    )
    return peft.get_peft_model(copy.deepcopy(base_model), config).eval()


def copy_weights_into_peft(model, reference):
    """Load our A and B into PEFT's layers, matched by layer name."""
    ours_by_name = {
        name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)
    }
    copied = 0
    for name, module in reference.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer) and hasattr(module, "lora_A"):
            mine = ours_by_name[name.removeprefix("base_model.model.")]
            with torch.no_grad():
                module.lora_A["default"].weight.copy_(mine.lora_A)
                module.lora_B["default"].weight.copy_(mine.lora_B)
            copied += 1
    assert copied == len(ours_by_name)


def test_same_trainable_parameter_count_as_peft(base_model):
    trainable_peft, _ = peft_model(base_model).get_nb_trainable_parameters()
    assert count_parameters(ours(base_model))[0] == trainable_peft


def test_same_output_as_peft(base_model, tokens):
    model = ours(base_model)
    randomize_B(model)
    reference = peft_model(base_model)
    copy_weights_into_peft(model, reference)
    ours_out, peft_out = logits(model, tokens), logits(reference, tokens)
    assert not torch.allclose(
        ours_out, logits(base_model, tokens)
    )  # the adapter does something
    assert torch.allclose(ours_out, peft_out, atol=1e-5)


def test_same_merged_weights_as_peft(base_model):
    model = ours(base_model)
    randomize_B(model)
    reference = peft_model(base_model)
    copy_weights_into_peft(model, reference)
    merge_lora(model)
    merged_reference = reference.merge_and_unload()
    ours_weights = {
        n.replace(".base.", "."): p
        for n, p in model.named_parameters()
        if ".base." in n
    }
    compared = 0
    for name, weight in merged_reference.named_parameters():
        if name in ours_weights:
            assert torch.allclose(ours_weights[name], weight, atol=1e-6), name
            compared += 1
    assert compared == len(lora_layers(model))


# ---------- Merge, save, load ----------


def test_merge_keeps_output_and_unmerge_restores_weights(base_model, tokens):
    model = ours(base_model)
    randomize_B(model)
    before = logits(model, tokens)
    weights_before = {n: p.clone() for n, p in model.named_parameters()}
    merge_lora(model)
    assert torch.allclose(logits(model, tokens), before, atol=1e-5)
    unmerge_lora(model)
    for name, p in model.named_parameters():
        assert torch.allclose(p, weights_before[name], atol=1e-6), name


def test_save_and_load_round_trip(base_model, tokens, tmp_path):
    model = ours(base_model)
    randomize_B(model)
    path = tmp_path / "adapter.safetensors"
    config = {"r": R, "alpha": ALPHA, "targets": list(ATTENTION)}
    save_lora(model, path, config)

    fresh = ours(base_model)
    assert load_lora(fresh, path) == config
    assert torch.equal(logits(fresh, tokens), logits(model, tokens))


def test_load_rejects_a_different_rank(base_model, tmp_path):
    model = ours(base_model)
    path = tmp_path / "adapter.safetensors"
    save_lora(model, path, {"r": R, "alpha": ALPHA, "targets": list(ATTENTION)})
    other = copy.deepcopy(base_model)
    add_lora(other, ATTENTION, r=R * 2, alpha=ALPHA)
    with pytest.raises(ValueError, match="adapter has r="):
        load_lora(other, path)
