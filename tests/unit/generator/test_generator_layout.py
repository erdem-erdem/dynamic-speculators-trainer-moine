"""S2: multi-layer adapters and generator layout; legacy equivalence for EAGLE-3."""

import copy

import pytest
import torch
from torch import nn

from speculators.generator.config import Architecture
from speculators.generator.model import LoRAGenerator, install_adapters
from tests.unit.generator import _legacy_model as legacy


class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(h, h), nn.Linear(h, h // 2)
        self.v_proj, self.o_proj = nn.Linear(h, h // 2), nn.Linear(h, h)


class _Layer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.self_attn = _Attn(h)
        self.mlp = nn.Sequential()
        self.mlp.gate_proj, self.mlp.down_proj = (
            nn.Linear(h, 2 * h),
            nn.Linear(2 * h, h),
        )


class _Drafter(nn.Module):
    def __init__(self, h=8, layers=1):
        super().__init__()
        self.fc = nn.Linear(3 * h, h)
        self.layers = nn.ModuleList(_Layer(h) for _ in range(layers))


def _arch():
    return Architecture(
        encoder_hidden=16,
        example_dim=8,
        condition_dim=8,
        descriptor_dim=4,
        decoder_width=16,
        residual_blocks=1,
    )


@pytest.mark.parametrize(
    "target_modules", [None, ["q_proj", "k_proj", "v_proj", "o_proj"], ["down_proj"]]
)
def test_install_adapters_matches_legacy(target_modules):
    torch.manual_seed(0)
    base = _Drafter()
    old, new = copy.deepcopy(base), copy.deepcopy(base)
    torch.manual_seed(1)
    old_paths = legacy.install_adapters(old, 4, 8, 0.0, target_modules=target_modules)
    torch.manual_seed(1)
    new_paths = install_adapters(new, 4, 8, 0.0, target_modules=target_modules)
    assert new_paths == old_paths
    assert list(new_paths) == list(old_paths)
    old_state, new_state = old.state_dict(), new.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        torch.testing.assert_close(new_state[key], old_state[key], atol=0, rtol=0)
    assert [n for n, p in new.named_parameters() if p.requires_grad] == [
        n for n, p in old.named_parameters() if p.requires_grad
    ]


@pytest.mark.parametrize("feedback_dim", [0, 3])
def test_generator_matches_legacy(feedback_dim):
    shapes = {"k_proj": (8, 4), "o_proj": (8, 8), "q_proj": (8, 8), "v_proj": (8, 4)}
    torch.manual_seed(3)
    old = legacy.LoRAGenerator(12, shapes, 4, _arch(), 4, feedback_dim=feedback_dim)
    torch.manual_seed(3)
    new = LoRAGenerator(12, shapes, 4, _arch(), 4, feedback_dim=feedback_dim)
    old_state, new_state = old.state_dict(), new.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        torch.testing.assert_close(new_state[key], old_state[key], atol=0, rtol=0)
    # Non-zero heads so the comparison covers the decoded path, not only biases.
    torch.manual_seed(4)
    for module in (old, new):
        torch.manual_seed(4)
        for head in module.heads.values():
            for linear in head.values():
                nn.init.normal_(linear.weight, std=0.1)
    summaries, lengths = torch.randn(3, 12), torch.tensor([5, 9, 2])
    feedback = torch.randn(feedback_dim) if feedback_dim else None
    want, got = old(summaries, lengths, feedback), new(summaries, lengths, feedback)
    assert list(got) == list(want)
    for key in want:
        for g, w in zip(got[key], want[key], strict=True):
            torch.testing.assert_close(g, w, atol=0, rtol=0)
    contexts = [(summaries[:2], lengths[:2]), (summaries[2:], lengths[2:])]
    fb = torch.randn(2, feedback_dim) if feedback_dim else None
    want, got = old.forward_cohorts(contexts, fb), new.forward_cohorts(contexts, fb)
    for key in want:
        for g, w in zip(got[key], want[key], strict=True):
            torch.testing.assert_close(g, w, atol=0, rtol=0)


def _layered(head_sharing):
    shapes, layout = {}, {}
    for layer in range(3):
        for name, shape in (("q_proj", (8, 8)), ("v_proj", (8, 4))):
            shapes[f"l{layer}_{name}"] = shape
            layout[f"l{layer}_{name}"] = (name, layer)
    shapes["fc"], layout["fc"] = (24, 8), ("fc", 3)
    torch.manual_seed(0)
    return LoRAGenerator(
        12, shapes, 4, _arch(), 4, layout=layout, head_sharing=head_sharing
    )


def test_per_type_heads_are_shared_and_layers_distinguished():
    shared, separate = _layered("per_type"), _layered("per_module")
    assert set(shared.heads) == {"q_proj", "v_proj", "fc"}
    assert len(separate.heads) == 7
    assert shared.layer_embedding.num_embeddings == 4
    assert shared.module_embedding.num_embeddings == 3
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert count(shared) < count(separate)
    torch.manual_seed(1)
    for head in shared.heads.values():
        for linear in head.values():
            nn.init.normal_(linear.weight, std=0.1)
    factors = shared(torch.randn(2, 12), torch.tensor([3, 4]))
    assert set(factors) == set(shared.shapes)
    assert factors["l0_q_proj"][0].shape == (4, 8)
    assert factors["fc"][0].shape == (4, 24)
    assert not torch.equal(factors["l0_q_proj"][0], factors["l1_q_proj"][0])


def test_zero_b_heads_give_zero_delta_at_init():
    factors = _layered("per_type")(torch.randn(2, 12), torch.tensor([3, 4]))
    assert all(torch.count_nonzero(b) == 0 for _, b in factors.values())


def test_layout_validation():
    shapes = {"l0_q_proj": (8, 8), "l1_q_proj": (8, 4)}
    layout = {"l0_q_proj": ("q_proj", 0), "l1_q_proj": ("q_proj", 1)}
    with pytest.raises(ValueError, match="differing shapes"):
        LoRAGenerator(12, shapes, 4, _arch(), layout=layout, head_sharing="per_type")
    with pytest.raises(ValueError, match="layout keys"):
        LoRAGenerator(12, shapes, 4, _arch(), layout={"l0_q_proj": ("q_proj", 0)})
    with pytest.raises(ValueError, match="head_sharing"):
        LoRAGenerator(12, shapes, 4, _arch(), head_sharing="bogus")


def test_install_adapters_explicit_targets_multi_layer():
    model = _Drafter(layers=2)
    targets = {
        f"l{i}_{n}": f"layers.{i}.self_attn.{n}" for i in range(2) for n in ("q_proj",)
    }
    targets["fc"] = "fc"
    paths = install_adapters(model, 2, 4, 0.0, targets=targets)
    assert paths == dict(sorted(targets.items()))
    assert hasattr(model.get_submodule("layers.1.self_attn.q_proj"), "base")
    assert hasattr(model.fc, "base")
    with pytest.raises(ValueError, match="not nn.Linear"):
        install_adapters(_Drafter(), 2, 4, 0.0, targets={"x": "layers.0.self_attn"})
    with pytest.raises(ValueError, match="dots"):
        install_adapters(_Drafter(), 2, 4, 0.0, targets={"a.b": "fc"})
    with pytest.raises(ValueError, match="either targets"):
        install_adapters(
            _Drafter(), 2, 4, 0.0, targets={"fc": "fc"}, target_modules=["q_proj"]
        )


def test_per_type_bias_hybrid_static_part_is_per_layer():
    """Hybrid (ABL24K 2026-11-14): shared head weights, one static A/B per key."""
    hybrid, shared = _layered("per_type_bias"), _layered("per_type")
    assert set(hybrid.heads) == set(shared.heads)
    assert all(lin.bias is None for h in hybrid.heads.values() for lin in h.values())
    assert set(hybrid.biases) == {f"{k}__{f}" for k in hybrid.shapes for f in "AB"}
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    head_bias = sum(lin.bias.numel() for h in shared.heads.values() for lin in h.values())
    assert count(hybrid) == count(shared) - head_bias + sum(b.numel() for b in hybrid.biases.values())
    # At init (zero head weights, zero B) the update is zero and A is the per-key static A.
    x, n = torch.randn(2, 12), torch.tensor([3, 4])
    factors = hybrid(x, n)
    assert all(torch.count_nonzero(b) == 0 for _, b in factors.values())
    for key, (a, _) in factors.items():
        torch.testing.assert_close(a.flatten(), hybrid.biases[f"{key}__A"])
    # The static part differs per layer; the prompt part flows through shared weights.
    with torch.no_grad():
        hybrid.biases["l0_q_proj__B"].normal_()
        hybrid.biases["l1_q_proj__B"].normal_()
    f = hybrid(x, n)
    assert not torch.equal(f["l0_q_proj"][1], f["l1_q_proj"][1])
    for head in hybrid.heads.values():
        for lin in head.values():
            nn.init.normal_(lin.weight, std=0.1)
    f1, f2 = hybrid(x, n), hybrid(torch.randn(2, 12), n)
    assert not torch.equal(f1["l0_q_proj"][0], f2["l0_q_proj"][0])  # prompt-dependent
    # Gradients reach both the per-key static factors and the shared heads.
    sum(a.sum() + b.sum() for a, b in f1.values()).backward()
    assert hybrid.biases["l0_q_proj__A"].grad is not None
    assert hybrid.heads["q_proj"]["A"].weight.grad is not None
