"""S4 (rigor item 5): `_project_kv` seam, `adapter.kv_lora_path`, `fc` adapter."""

import pytest
import torch

from speculators.generator.drafters import drafter_family
from speculators.generator.drafters.dflash_attention import (
    KVPathDFlashAttention,
    install_kv_path,
)
from speculators.generator.model import install_adapters
from speculators.models.dflash.model_definitions import Qwen3DFlashAttention
from tests.unit.generator import _legacy_dflash_attention as legacy
from tests.unit.generator.test_dflash_generator import (
    HIDDEN,
    LAYERS,
    _bank,
    _batch,
    _cfg,
    _drafter,
    _run,
)

RANK = 4
PATHS = ("both", "context", "block")
# Which of (k_ctx, k_noise, v_ctx, v_noise) carry the LoRA delta.
ADAPTED = {
    "both": (True, True, True, True),
    "context": (True, False, True, False),
    "block": (False, True, False, True),
}


def _adapted(kv="both", targets=("k_proj", "v_proj"), fc=False):
    drafter = _drafter()
    family = drafter_family(_cfg(kv=kv, fc=fc), _bank())
    paths = install_adapters(
        drafter,
        RANK,
        2 * RANK,
        0.0,
        targets=family.adapter_targets(drafter, list(targets)),
    )
    family.prepare_drafter(drafter)
    return family, drafter.eval(), paths


def _factors(drafter, paths, *, scale, seed=3):
    g = torch.Generator().manual_seed(seed)
    factors = {}
    for key, path in paths.items():
        base = drafter.get_submodule(path).base
        a = torch.randn(RANK, base.in_features, generator=g) * 0.3
        b = torch.randn(base.out_features, RANK, generator=g) * scale
        factors[key] = (a, b)
    return factors


def _equal_outputs(left, right):
    _, left_loss, left_metrics = left
    _, right_loss, right_metrics = right
    assert torch.equal(left_loss, right_loss)
    assert left_metrics.keys() == right_metrics.keys()
    for key in left_metrics:
        assert torch.equal(
            torch.as_tensor(left_metrics[key]), torch.as_tensor(right_metrics[key])
        )


def test_seamed_forward_is_bit_identical_to_pre_seam_forward(monkeypatch):
    family, drafter, paths = _adapted(targets=("q_proj", "k_proj", "v_proj"))
    batch = _batch()
    factors = _factors(drafter, paths, scale=0.2)
    base = _drafter()
    seamed = (
        _run(family, base, batch),
        _run(family, drafter, batch, factors, paths),
    )
    monkeypatch.setattr(Qwen3DFlashAttention, "forward", legacy.forward)
    pre_seam = (
        _run(family, base, batch),
        _run(family, drafter, batch, factors, paths),
    )
    for new, old in zip(seamed, pre_seam, strict=True):
        _equal_outputs(new, old)


@pytest.mark.parametrize("kv", PATHS)
def test_kv_path_puts_lora_delta_on_selected_path_only(kv):
    _, drafter, _ = _adapted(kv=kv)
    torch.manual_seed(4)
    for layer in drafter.layers:
        for name in ("k_proj", "v_proj"):
            torch.nn.init.normal_(getattr(layer.self_attn, name).b, std=0.2)
    target_hidden = torch.randn(1, 5, HIDDEN)
    hidden_states = torch.randn(1, 3, HIDDEN)
    for layer in drafter.layers:
        attention = layer.self_attn
        if kv == "both":
            assert type(attention) is Qwen3DFlashAttention
        else:
            assert type(attention) is KVPathDFlashAttention
        got = attention._project_kv(target_hidden, hidden_states)
        inputs = (target_hidden, hidden_states) * 2
        modules = (attention.k_proj,) * 2 + (attention.v_proj,) * 2
        for value, module, x, adapted in zip(
            got, modules, inputs, ADAPTED[kv], strict=True
        ):
            want = module(x) if adapted else module.base(x)
            assert torch.equal(value, want)
            assert not torch.allclose(module(x), module.base(x))


@pytest.mark.parametrize("kv", PATHS)
def test_zero_lora_equals_base_for_every_kv_path(kv):
    family, drafter, paths = _adapted(kv=kv, fc=True)
    batch = _batch()
    factors = _factors(drafter, paths, scale=0.0)
    _equal_outputs(
        _run(family, drafter, batch, factors, paths), _run(family, _drafter(), batch)
    )


def test_kv_paths_give_distinct_losses_and_gradients():
    batch = _batch()
    losses = {}
    for kv in PATHS:
        family, drafter, paths = _adapted(kv=kv)
        factors = {
            key: tuple(t.requires_grad_(True) for t in pair)
            for key, pair in _factors(drafter, paths, scale=0.2).items()
        }
        drafter.train()
        _, loss, _ = _run(family, drafter, batch, factors, paths, training=True)
        loss.backward()
        assert all(t.grad is not None for pair in factors.values() for t in pair)
        assert all(p.grad is None for p in drafter.parameters())
        losses[kv] = loss.detach()
    assert len({value.item() for value in losses.values()}) == len(PATHS)


def test_fc_adapter_changes_forward_and_receives_gradient():
    family, drafter, paths = _adapted(targets=("q_proj",), fc=True)
    assert set(paths) == {*(f"l{i}_q_proj" for i in range(LAYERS)), "fc"}
    assert paths["fc"] == "fc"
    batch = _batch()
    zero = _factors(drafter, paths, scale=0.0)
    fc_only = {**zero, "fc": _factors(drafter, {"fc": "fc"}, scale=0.2)["fc"]}
    fc_only["fc"] = tuple(t.requires_grad_(True) for t in fc_only["fc"])
    _, base_loss, _ = _run(family, drafter, batch, zero, paths)
    _, loss, _ = _run(family, drafter, batch, fc_only, paths)
    assert not torch.equal(loss, base_loss)
    loss.backward()
    assert all(t.grad is not None and t.grad.abs().sum() > 0 for t in fc_only["fc"])


def test_install_kv_path_errors():
    _, drafter, _ = _adapted(targets=("q_proj", "o_proj"))
    with pytest.raises(ValueError, match="requires adapted k_proj or v_proj"):
        install_kv_path(drafter, "context")
    with pytest.raises(ValueError, match="Unknown kv_lora_path"):
        install_kv_path(_adapted()[1], "neither")
    # Already switched (or any other attention class) is refused.
    _, switched, _ = _adapted(kv="block")
    with pytest.raises(TypeError, match="needs Qwen3DFlashAttention"):
        install_kv_path(switched, "context")
    # Through the family: a non-default path without adapted k/v fails loudly.
    with pytest.raises(ValueError, match="requires adapted k_proj or v_proj"):
        _adapted(kv="context", targets=("q_proj", "o_proj"))


def test_v_only_adapter_switches_layers():
    _, drafter, _ = _adapted(kv="context", targets=("v_proj",))
    layers = drafter.layers
    assert all(type(layer.self_attn) is KVPathDFlashAttention for layer in layers)
