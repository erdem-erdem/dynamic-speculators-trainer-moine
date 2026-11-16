"""GENDIAG (2026-11-16): training-time diagnostics observe without changing
training; the context-free twin ignores its input."""

import json

import pytest
import torch

from speculators.generator import campaign
from speculators.generator.config import Architecture
from speculators.generator.diagnostics import group_of, spread_stats
from speculators.generator.model import LoRAGenerator
from tests.unit.generator.test_campaign import build_tiny_campaign

SHAPES = {"l0_q": (8, 8), "l1_q": (8, 8), "l0_v": (8, 4), "l1_v": (8, 4)}
LAYOUT = {"l0_q": ("q", 0), "l1_q": ("q", 1), "l0_v": ("v", 0), "l1_v": ("v", 1)}
ARCH = dict(encoder_hidden=8, example_dim=4, condition_dim=4, descriptor_dim=2, decoder_width=8)


def _generator(sharing, mode="prompt"):
    torch.manual_seed(0)
    return LoRAGenerator(
        6, SHAPES, 2, Architecture(**ARCH, input_mode=mode), 8,
        layout=LAYOUT, head_sharing=sharing,
    )


@pytest.mark.parametrize("sharing", ["per_module", "per_type", "per_type_bias"])
def test_every_parameter_has_a_group(sharing):
    for name, _ in _generator(sharing).named_parameters():
        group_of(name)


def test_input_mode_default_is_omitted_from_the_identity():
    assert "input_mode" not in Architecture().model_dump(mode="json")
    assert Architecture(input_mode="constant").model_dump(mode="json")["input_mode"] == "constant"


def test_constant_input_twin_ignores_the_summary():
    gen = _generator("per_module", "constant")
    for p in gen.parameters():  # make the heads non-zero so ΔW depends on z
        torch.nn.init.normal_(p, std=0.3)
    one = gen(torch.randn(1, 6), torch.tensor([5]))
    two = gen(torch.randn(1, 6) * 9, torch.tensor([300]))
    for key in one:
        assert torch.equal(one[key][0], two[key][0])
        assert torch.equal(one[key][1], two[key][1])
    real = _generator("per_module")
    for p in real.parameters():
        torch.nn.init.normal_(p, std=0.3)
    a = real(torch.randn(1, 6), torch.tensor([5]))
    b = real(torch.randn(1, 6) * 9, torch.tensor([300]))
    assert not torch.equal(a["l0_q"][0], b["l0_q"][0])


def test_spread_stats_constant_rows_have_zero_spread():
    x = torch.ones(6, 5)
    stats = spread_stats(x + 1e-12 * torch.randn(6, 5), list("aabbcc"))
    assert stats["spread"] < 1e-20
    assert stats["cos_mean"] > 0.999999


def _checkpoint(result):
    return torch.load(result["checkpoint_path"], weights_only=False)["model"]


def test_diagnostics_do_not_change_training(tmp_path, monkeypatch):
    """Same checkpoint tensors and history with and without GENDIAG."""
    results = {}
    for mode in ("plain", "diag"):
        root = tmp_path / mode
        root.mkdir()
        if mode == "diag":
            monkeypatch.setenv("GENDIAG_DIR", str(root / "gendiag"))
            monkeypatch.setenv("GENDIAG_INTERVAL", "1")
            monkeypatch.setenv("GENDIAG_SNAPSHOTS", "1")
            monkeypatch.setenv("GENDIAG_PROBE_PER_CATEGORY", "2")
        else:
            monkeypatch.delenv("GENDIAG_DIR", raising=False)
        cfg, _ = build_tiny_campaign(root, monkeypatch, "dflash", [])
        job = campaign.jobs(cfg)[0]
        assert job["arm"] != "fresh_lora"
        results[mode] = campaign.run_one(cfg, job)
    plain, diag = (_checkpoint(results[m]) for m in ("plain", "diag"))
    assert plain.keys() == diag.keys()
    for key in plain:
        assert torch.equal(plain[key], diag[key]), key
    assert [h["score"] for h in results["plain"]["history"]] == [
        h["score"] for h in results["diag"]["history"]
    ]
    lines = [
        json.loads(x)
        for x in (tmp_path / "diag/gendiag/diag.jsonl").read_text().splitlines()
    ]
    assert lines[0]["kind"] == "meta"
    steps = [x for x in lines if x["kind"] == "step"]
    assert steps and steps[0]["update"] == 1
    first = steps[0]
    assert {"grad", "step", "init", "probe", "zgrad"} <= set(first)
    assert "headB_w" in first["grad"] and first["grad"]["headB_w"]["sumsq"] > 0
    # Zero-initialised head weights: no gradient reaches the trunk at step 1.
    assert first["grad"]["enc1"]["sumsq"] == 0
    assert {"summary", "z", "dW", "headA"} <= set(first["probe"]["val"])
    assert (tmp_path / "diag/gendiag/snapshots/u00001.pt").is_file()
