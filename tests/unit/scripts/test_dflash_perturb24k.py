"""Unit tests for the pure helpers of scripts/dflash_port/perturbation (PERTURB24K)."""

import sys
from pathlib import Path

import pytest
import torch

PERTURB = Path(__file__).resolve().parents[3] / "scripts/dflash_port/perturbation"
sys.path.insert(0, str(PERTURB))

import harness as H  # noqa: E402
import metrics as MT  # noqa: E402
from speculators.models.metrics import compute_accepted_length_counts  # noqa: E402


def _row(blocks=6, vocab=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    l1 = torch.randn(1, blocks * 16, vocab, generator=g) * 3
    lt = l1 + torch.randn(1, blocks * 16, vocab, generator=g)
    mask = torch.ones(1, blocks * 16)
    mask[:, ::16] = 0  # anchors
    mask[:, -5:] = 0  # truncated tail
    return l1, lt, mask


def test_identical_logits_do_not_move():
    l1, lt, mask = _row()
    slot, block = MT.row_metrics(l1, l1.clone(), lt, mask)
    k = MT.SK
    n = slot[:, k["n"]].sum()
    assert n == mask.sum()
    for key in ("kl12", "kl21", "js", "js100", "tv12", "flip", "fix", "brk"):
        assert slot[:, k[key]].abs().max() < 1e-5, key
    for key in ("ov5", "ov10", "in5", "in10"):
        assert torch.allclose(slot[:, k[key]], slot[:, k["n"]]), key
    assert block[MT.BK["eal1"]] == block[MT.BK["eal2"]]


def test_metrics_match_naive_definitions():
    l1, lt, mask = _row(seed=1)
    l2 = l1 + 0.7 * torch.randn(l1.shape, generator=torch.Generator().manual_seed(2))
    slot, block = MT.row_metrics(l1, l2, lt, mask, chunk=7)
    blocks = l1.shape[1] // 16
    p1 = torch.softmax(l1, -1).reshape(blocks, 16, -1)[:, 1:]
    p2 = torch.softmax(l2, -1).reshape(blocks, 16, -1)[:, 1:]
    pt = torch.softmax(lt, -1).reshape(blocks, 16, -1)[:, 1:]
    valid = mask.reshape(blocks, 16)[:, 1:].bool()
    kl12 = (p1 * (p1.log() - p2.log())).sum(-1)
    tv = 0.5 * (p1 - p2).abs().sum(-1)
    m = 0.5 * (p1 + p2)
    js = 0.5 * (p1 * (p1.log() - m.log())).sum(-1) + 0.5 * (p2 * (p2.log() - m.log())).sum(-1)
    klt2 = (pt * (pt.log() - p2.log())).sum(-1)
    a1, a2, at = p1.argmax(-1), p2.argmax(-1), pt.argmax(-1)
    k = MT.SK
    for key, ref in (("kl12", kl12), ("tv12", tv), ("js", js), ("klT2", klt2),
                     ("flip", (a1 != a2).float()),
                     ("fix", ((a1 != at) & (a2 == at)).float()),
                     ("brk", ((a1 == at) & (a2 != at)).float())):
        expected = (ref * valid).sum(0).double()
        assert torch.allclose(slot[:, k[key]], expected, atol=1e-4), key
    # Accepted length equals the upstream metric.
    total1, count = compute_accepted_length_counts(a1 == at, valid)
    total2, _ = compute_accepted_length_counts(a2 == at, valid)
    assert float(block[MT.BK["blocks"]]) == float(count)
    assert float(block[MT.BK["eal1"]]) == pytest.approx(float(total1))
    assert float(block[MT.BK["eal2"]]) == pytest.approx(float(total2))
    # Net churn equals the change in correct counts.
    net = slot[:, k["fix"]] - slot[:, k["brk"]]
    assert torch.allclose(net, slot[:, k["c2"]] - slot[:, k["c1"]])


def test_cached_matches_direct():
    l1, lt, mask = _row(seed=3)
    l2 = l1 + torch.randn(l1.shape, generator=torch.Generator().manual_seed(4))
    direct = MT.row_metrics(l1, l2, lt, mask, chunk=9)
    pieces = MT.prepare_theta1(l1, lt, mask, chunk=9)
    cached = MT.row_metrics_cached(pieces, l2, chunk=9)
    assert torch.allclose(direct[0], cached[0])
    assert torch.allclose(direct[1], cached[1])


def test_locations_cover_plan():
    locs = H.t1_locations()
    assert len(locs) == len(set(locs)) == 55
    assert len(H.location_members("whole")) == 36
    assert len(H.location_members("all")) == 35
    assert len(H.location_members("all_attn")) == 20
    assert H.location_members("layer1") == [(f"l0_{m}", "both") for m in H.ATTN + H.FFN]
    assert H.location_members("l3_kv@ctx") == [("l3_k", "ctx"), ("l3_v", "ctx")]
    assert H.hook_key("l3_k", "ctx") == "l3_k@ctx"
    with pytest.raises(ValueError):
        H.location_members("l9_x")


def test_delta_dense_and_lowrank():
    x = torch.randn(3, 8)
    a, b = torch.randn(2, 8), torch.randn(5, 2)
    dense = H.Delta(w=b @ a, coef=0.5)
    low = H.Delta(a=a, b=b, coef=0.5)
    assert torch.allclose(dense(x), low(x), atol=1e-5)
    with pytest.raises(ValueError):
        H.Delta()
