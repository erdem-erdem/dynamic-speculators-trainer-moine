"""Unit tests for the pure helpers of scripts/dflash_port/analysis (ANALYSIS24K)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ANALYSIS = Path(__file__).resolve().parents[3] / "scripts/dflash_port/analysis"
sys.path.insert(0, str(ANALYSIS))

import analyze_repr  # noqa: E402
import analyze_slots  # noqa: E402
import collect_slots  # noqa: E402
from speculators.models.metrics import compute_accepted_length_counts  # noqa: E402


def _flat(correct, valid, dm=None):
    n = correct.shape[0]
    dm = np.ones_like(correct) if dm is None else dm
    zeros = np.zeros(n * 15, np.float32)
    return {
        "correct": correct.reshape(-1), "valid": valid.reshape(-1),
        "dm": dm.reshape(-1), "slot": np.tile(np.arange(1, 16, dtype=np.int8), n),
        "row": np.zeros(n * 15, np.int32), "block": np.repeat(np.arange(n), 15),
        "anchor_rel": np.zeros(n * 15, np.int32), "p_top": zeros + 0.95,
        "ent": zeros, "rank": np.full(n * 15, 2), "p_d_tgt": zeros,
        "p_d_top": zeros, "t_p_pred": zeros, "t_rank_pred": np.full(n * 15, 3),
    }


def test_blocks_tau_matches_metric_and_fix_gain():
    rng = np.random.default_rng(0)
    correct = rng.random((200, 15)) < 0.8
    valid = np.ones((200, 15), bool)
    valid[:20, 10:] = False  # truncated tails
    b = analyze_slots.blocks(_flat(correct, valid))
    total, count = compute_accepted_length_counts(
        torch.as_tensor(correct), torch.as_tensor(valid))
    assert b["tau"].mean() == pytest.approx(float(total / count))
    # Fix gain: the block after correcting only its first miss.
    for i in range(200):
        if b["kind"][i] != "miss":
            assert b["fix_gain"][i] == 0
            continue
        fixed = correct[i] & valid[i]
        fixed[b["first"][i] - 1] = True
        run = int(np.cumprod(fixed).sum()) + 1
        assert b["fix_gain"][i] == run - b["tau"][i]


def test_blocks_kinds_and_onpath():
    correct = np.ones((3, 15), bool)
    valid = np.ones((3, 15), bool)
    correct[1, 4] = False  # genuine miss at slot 5
    valid[2, 7:] = False  # all valid correct, then the response ends
    dm = np.ones((3, 15), bool)
    dm[1, 2] = False  # data path left the greedy path at slot 3
    b = analyze_slots.blocks(_flat(correct, valid, dm))
    assert list(b["kind"]) == ["full", "miss", "truncated"]
    assert list(b["tau"]) == [16, 5, 8]
    assert not b["onpath"][1]


def test_compact_drops_slot0_and_empty_blocks():
    n = 3
    stats = {
        "valid": np.ones(n * 16, bool), "correct": np.ones(n * 16, bool),
        "dm": np.ones(n * 16, bool),
        **{k: np.zeros(n * 16, np.float32) for k in
           ("p_top", "ent", "p_d_tgt", "p_d_top", "t_p_pred")},
        "rank": np.ones(n * 16, np.int64), "t_rank_pred": np.ones(n * 16, np.int64),
    }
    stats["valid"][::16] = False
    stats["valid"][16:32] = False  # block 1 has no valid slot
    out = collect_slots.compact(stats, 7, np.array([5, 6, 7]))
    assert len(out["slot"]) == 30
    assert list(np.unique(out["block"])) == [0, 2]
    assert (out["row"] == 7).all()
    assert list(np.unique(out["anchor_rel"])) == [5, 7]


def test_donors_same_and_cross_category():
    rows = [("math-1", 1), ("math-1", 2), ("code-1", 3), ("code-1", 4)]
    same, cross = collect_slots.donors(rows)
    assert same == {1: 2, 2: 1, 3: 4, 4: 3}
    assert cross == {1: 3, 2: 4, 3: 1, 4: 2}


def test_linear_cka_properties():
    x = torch.randn(500, 20)
    q, _ = torch.linalg.qr(torch.randn(20, 20))
    assert analyze_repr.linear_cka(x, x @ q * 3.0) == pytest.approx(1.0, abs=1e-5)
    assert analyze_repr.linear_cka(x, torch.randn(500, 20)) < 0.2


def test_auc_matches_definition():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.4])
    labels = np.array([0, 0, 1, 1, 1], bool)
    pairs = [(p, q) for p in scores[labels] for q in scores[~labels]]
    expected = np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for p, q in pairs])
    assert analyze_repr.auc(scores, labels) == pytest.approx(expected)


def test_slot_stats_ranks_probabilities_and_data_match():
    targets = torch.tensor([[[0.0, 5.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 4.0]]])
    logits = torch.tensor([[[0.0, 2.0, 3.0], [3.0, 1.0, 0.0], [1.0, 2.0, 0.0]]])
    mask = torch.tensor([[0, 1, 1]])
    data = torch.tensor([1, 0, 1])  # last position: data token is not the argmax
    s = collect_slots.slot_stats(logits, targets, mask, data)
    assert list(s["correct"]) == [False, True, False]
    assert list(s["rank"]) == [2, 1, 3]
    assert list(s["dm"]) == [True, True, False]
    assert list(s["t_rank_pred"]) == [2, 1, 2]  # ties rank by strictly larger logits
    assert s["p_top"][2] == pytest.approx(torch.softmax(targets[0, 2], -1)[2].item())
    assert list(s["valid"]) == [False, True, True]


def test_grad_summary_shared_vs_specific_directions():
    import grad_maps

    torch.manual_seed(0)
    shape = (40, 48)
    common = torch.randn(shape)
    specific = {c: torch.randn(shape) for c in ("math", "code", "stem", "chat")}

    def accumulate(spread):
        acc, counts = {}, {}
        for c in specific:
            for h in (0, 1):
                g = common + spread * specific[c] + 0.01 * torch.randn(shape)
                acc[(c, h)] = {"w": g * 10}
                counts[(c, h)] = 10
        return acc, counts

    weights = {"w": torch.ones(shape)}
    same = grad_maps.summarize(*accumulate(0.0), weights)["w"]
    apart = grad_maps.summarize(*accumulate(3.0), weights)["w"]
    for c in specific:
        assert same["categories"][c]["heldout_capture"] == pytest.approx(1.0, abs=0.02)
        assert apart["categories"][c]["heldout_capture"] < 0.6
        assert same["categories"][c]["split_half_cos"] > 0.99
    assert same["cos"]["math|code"] > 0.99
    assert apart["cos_halves"]["math|code"] < 0.3
