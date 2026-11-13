"""D-PACE gate on the drafter's label probability (ABL24K, 2026-11-13)."""

from functools import partial

import torch

from speculators.losses import (
    dpace_label_decay,
    dpace_loss_decay,
    dpace_weights,
    loss_function,
    target_label_prob,
)
from speculators.losses.eager import ce_loss as ce_loss_eager
from speculators.losses.eager import kl_div_loss as kl_div_loss_eager
from speculators.models.dflash.metrics import compute_metrics


def _toy(blocks=3, block=4, vocab=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, blocks * block, vocab, generator=g)
    targets = torch.randn(1, blocks * block, vocab, generator=g)
    mask = torch.ones(1, blocks * block)
    mask[0, block] = 0  # one masked position
    return logits, targets, mask


def _reference_weights(q, mask, block, alpha):
    q = q.reshape(-1, block).double()
    m = mask.reshape(-1, block).double()
    s = torch.where(m > 0, (1 - alpha) * q + alpha, torch.ones_like(q))
    out = torch.zeros_like(q)
    for a in range(q.shape[0]):
        for j in range(block):
            out[a, j] = sum(
                m[a, k] * torch.prod(s[a, : k + 1]) for k in range(j, block)
            ) * m[a, j]
    return out.reshape(1, -1)


def test_dpace_loss_decay_unchanged_by_refactor():
    logits, targets, mask = _toy()
    ce = ce_loss_eager(logits, targets) * mask
    got = dpace_loss_decay(None, mask, 4, 0.5, elementwise_loss=ce)
    ref = _reference_weights(torch.exp(-ce), mask, 4, 0.5)
    torch.testing.assert_close(got.double(), ref, rtol=1e-5, atol=1e-6)


def test_label_prob_matches_softmax_and_chunking():
    logits, targets, _ = _toy()
    ids = targets.argmax(-1)
    ref = torch.softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(target_label_prob(logits, targets), ref)
    torch.testing.assert_close(target_label_prob(logits, targets, chunk=5), ref)


def test_label_gate_equals_ce_gate_and_ignores_loss_term():
    logits, targets, mask = _toy()
    q = target_label_prob(logits, targets)
    ce = ce_loss_eager(logits, targets) * mask
    via_ce = dpace_loss_decay(None, mask, 4, 0.1, elementwise_loss=ce)
    via_label = dpace_label_decay(
        None, mask, 4, 0.1, label_prob=q, elementwise_loss=kl_div_loss_eager(logits, targets)
    )
    # The label gate equals the CE gate wherever the position is unmasked,
    # whatever per-position loss it weights.
    torch.testing.assert_close(via_label, via_ce, rtol=1e-5, atol=1e-6)


def test_label_gated_ce_is_expected_accepted_length_gradient():
    """At alpha -> 0, sum_j w_j * CE_j has the gradient of -E[accepted]."""
    logits, targets, _ = _toy(blocks=2, block=5)
    mask = torch.ones(1, 10)
    leaf = logits.clone().double().requires_grad_(True)
    ids = targets.argmax(-1)
    q = torch.softmax(leaf, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    expected = torch.cumprod(q.reshape(2, 5), -1).sum()
    (grad_tau,) = torch.autograd.grad(expected, leaf)

    leaf2 = logits.clone().double().requires_grad_(True)
    ce = torch.nn.functional.cross_entropy(
        leaf2.reshape(-1, leaf2.shape[-1]), ids.reshape(-1), reduction="none"
    ).reshape(1, -1)
    w = dpace_weights(torch.exp(-ce.detach()), mask, 5, 1e-12).double()
    (grad_loss,) = torch.autograd.grad((w * ce).sum(), leaf2)
    torch.testing.assert_close(grad_loss, -grad_tau, rtol=1e-6, atol=1e-9)


def test_compute_metrics_dpace_label_path():
    logits, targets, mask = _toy()
    cfg = {"kl": (kl_div_loss_eager, 1.0)}
    loss, metrics = compute_metrics(
        logits, targets, mask, block_size=4, loss_config=cfg,
        per_position_loss_weight="dpace-label", dpace_alpha=0.1,
    )
    q = target_label_prob(logits, targets)
    decay = partial(dpace_label_decay, loss_mask=mask, block_size=4, dpace_alpha=0.1, label_prob=q)
    ref = loss_function(logits, targets, mask, torch.arange(12).unsqueeze(0) % 4,
                        loss_fn=kl_div_loss_eager, decay_fn=decay)
    torch.testing.assert_close(loss, ref)
    own, _ = compute_metrics(
        logits, targets, mask, block_size=4, loss_config=cfg,
        per_position_loss_weight="dpace", dpace_alpha=0.1,
    )
    assert not torch.allclose(loss, own)  # the gate source changes the loss
    assert torch.isfinite(metrics["loss_sum"])
