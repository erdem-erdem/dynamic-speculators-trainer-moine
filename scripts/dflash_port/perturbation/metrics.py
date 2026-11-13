"""Movement / gain / churn metrics between two drafter distributions (pure torch).

For one row: logits of the starting point theta1 (`l1`) and of the perturbed
model theta2 (`l2`) at the same anchored blocks, the verifier logits (`lt`)
and the aligned loss mask, all `[1, A * 16, V]` / `[1, A * 16]`. Slot 0 of a
block is the anchor (masked); slots 1..15 are the draft slots.

`row_metrics` returns
- `slot`: `[15, len(SLOT_KEYS)]` sums over the row's valid positions per slot;
- `block`: `[len(BLOCK_KEYS)]` per-block sums (accepted length as upstream
  `compute_accepted_length_counts`: leading run of correct valid slots + 1).

Definitions (p1, p2 drafter distributions, pT verifier distribution):
- kl12 = KL(p1||p2), kl21 = KL(p2||p1), js = JS(p1, p2) (nats),
  js100 = JS of p1, p2 restricted to the top-100 tokens of p1 and renormalised,
  tv12 = 0.5 * sum |p1 - p2|;
- flip = [argmax p1 != argmax p2]; ov5 / ov10 = |topk(p1) & topk(p2)| / k;
  in5 / in10 = [argmax p2 in topk(p1)];
- klT1 / klT2 = KL(pT||p1) / KL(pT||p2); tvT1 / tvT2 = TV(pT, p);
- c1 / c2 = [argmax p == argmax pT]; fix = !c1 & c2, brk = c1 & !c2;
- reach1 / reach2 = [the block's leading correct run reaches this slot].
"""

from __future__ import annotations

import torch

BLOCK = 16
SLOTS = BLOCK - 1
SLOT_KEYS = (
    "n",
    "kl12",
    "kl21",
    "js",
    "js100",
    "tv12",
    "flip",
    "ov5",
    "ov10",
    "in5",
    "in10",
    "klT1",
    "klT2",
    "tvT1",
    "tvT2",
    "c1",
    "c2",
    "fix",
    "brk",
    "reach1",
    "reach2",
)
BLOCK_KEYS = ("blocks", "eal1", "eal2")
SK = {k: i for i, k in enumerate(SLOT_KEYS)}
BK = {k: i for i, k in enumerate(BLOCK_KEYS)}


def _kl(lp, lq):
    """sum p (log p - log q) over the last dim, p = exp(lp)."""
    return (lp.exp() * (lp - lq)).sum(-1)


def _tv(lp, lq):
    return 0.5 * (lp.exp() - lq.exp()).abs().sum(-1)


def _js(lp, lq):
    p, q = lp.exp(), lq.exp()
    lm = torch.log((p + q) * 0.5).clamp_min(-1e30)
    return 0.5 * ((p * (lp - lm)).sum(-1) + (q * (lq - lm)).sum(-1))


def _overlap(i1, i2):
    return (i1.unsqueeze(-1) == i2.unsqueeze(-2)).any(-1).float().mean(-1)


def position_metrics(l1, l2, lt, *, top1=None):
    """Per-position metrics for rows of logits `[N, V]` -> dict of `[N]`.

    `top1` optionally carries precomputed (lp1, top10_1, top100_1, klT1, tvT1)
    pieces of the starting point to save work when one theta1 is compared
    with many theta2."""
    lp2 = torch.log_softmax(l2.float(), -1)
    lpt = torch.log_softmax(lt.float(), -1)
    if top1 is None:
        top1 = theta1_pieces(l1, lt)
    lp1 = top1["lp1"]
    out = {
        "kl12": _kl(lp1, lp2),
        "kl21": _kl(lp2, lp1),
        "js": _js(lp1, lp2),
        "tv12": _tv(lp1, lp2),
        "klT1": top1["klT1"],
        "klT2": _kl(lpt, lp2),
        "tvT1": top1["tvT1"],
        "tvT2": _tv(lpt, lp2),
    }
    i100 = top1["i100"]
    r1 = torch.log_softmax(lp1.gather(-1, i100), -1)
    r2 = torch.log_softmax(lp2.gather(-1, i100), -1)
    out["js100"] = _js(r1, r2)
    t2 = lp2.topk(10, -1).indices
    # argmax on the raw (bfloat16) logits, exactly as upstream compute_metrics:
    # bf16 logits tie often at the top and topk does not break ties like argmax.
    a1, a2 = top1["a1"], l2.argmax(-1)
    at = top1["at"]
    out["flip"] = (a1 != a2).float()
    out["ov5"] = _overlap(i100[:, :5], t2[:, :5])
    out["ov10"] = _overlap(i100[:, :10], t2[:, :10])
    out["in5"] = (i100[:, :5] == a2.unsqueeze(-1)).any(-1).float()
    out["in10"] = (i100[:, :10] == a2.unsqueeze(-1)).any(-1).float()
    c1 = a1 == at
    c2 = a2 == at
    out["c1"], out["c2"] = c1.float(), c2.float()
    out["fix"] = (~c1 & c2).float()
    out["brk"] = (c1 & ~c2).float()
    return out


def theta1_pieces(l1, lt):
    lp1 = torch.log_softmax(l1.float(), -1)
    lpt = torch.log_softmax(lt.float(), -1)
    # Top-k sets (overlaps, js100) from topk; the accuracy-relevant top-1
    # from argmax on the raw logits, which may differ from topk[0] on ties.
    i100 = lp1.topk(min(100, lp1.shape[-1]), -1).indices
    return {
        "lp1": lp1,
        "i100": i100,
        "a1": l1.argmax(-1),
        "at": lt.argmax(-1),
        "klT1": _kl(lpt, lp1),
        "tvT1": _tv(lpt, lp1),
    }


def accepted(correct, valid):
    """[A, 15] bools -> per-slot reach [A, 15] float and per-block length [A]."""
    acc = torch.logical_and(correct, valid).float()
    reach = acc.cumprod(-1)
    return reach, reach.sum(-1) + 1.0


@torch.no_grad()
def row_metrics(l1, l2, lt, mask, *, chunk=1024):
    """Per-slot and per-block sums for one row (see module docstring)."""
    if l1.shape != l2.shape or l1.shape != lt.shape:
        raise ValueError(f"Shape mismatch {l1.shape} {l2.shape} {lt.shape}")
    vocab = l1.shape[-1]
    if l1.shape[1] % BLOCK:
        raise ValueError("Positions not a multiple of the block size")
    blocks = l1.shape[1] // BLOCK
    l1 = l1.reshape(blocks, BLOCK, vocab)[:, 1:]
    l2 = l2.reshape(blocks, BLOCK, vocab)[:, 1:]
    lt = lt.reshape(blocks, BLOCK, vocab)[:, 1:]
    valid = mask.reshape(blocks, BLOCK)[:, 1:].bool()
    slot_ids = torch.arange(SLOTS, device=l1.device).expand(blocks, SLOTS)
    flat = valid.reshape(-1).nonzero().squeeze(-1)
    sums = torch.zeros(SLOTS, len(SLOT_KEYS), dtype=torch.float64, device=l1.device)
    c1_full = torch.zeros(blocks * SLOTS, dtype=torch.bool, device=l1.device)
    c2_full = torch.zeros_like(c1_full)
    s_flat = slot_ids.reshape(-1)
    l1f, l2f, ltf = (x.reshape(-1, vocab) for x in (l1, l2, lt))
    for start in range(0, flat.numel(), chunk):
        idx = flat[start : start + chunk]
        m = position_metrics(l1f[idx], l2f[idx], ltf[idx])
        s = s_flat[idx]
        sums[:, SK["n"]].index_add_(0, s, torch.ones_like(s, dtype=torch.float64))
        for key, value in m.items():
            sums[:, SK[key]].index_add_(0, s, value.double())
        c1_full[idx] = m["c1"].bool()
        c2_full[idx] = m["c2"].bool()
    c1 = c1_full.reshape(blocks, SLOTS)
    c2 = c2_full.reshape(blocks, SLOTS)
    reach1, len1 = accepted(c1, valid)
    reach2, len2 = accepted(c2, valid)
    block_valid = valid.any(-1).double()
    sums[:, SK["reach1"]] = (reach1.double() * block_valid[:, None]).sum(0)
    sums[:, SK["reach2"]] = (reach2.double() * block_valid[:, None]).sum(0)
    block = torch.stack(
        [
            block_valid.sum(),
            (len1.double() * block_valid).sum(),
            (len2.double() * block_valid).sum(),
        ]
    )
    return sums.cpu(), block.cpu()


@torch.no_grad()
def row_metrics_cached(pieces, l2, chunk=1024):
    """As `row_metrics` with theta1 precomputed by `prepare_theta1`."""
    vocab = l2.shape[-1]
    blocks = pieces["blocks"]
    l2f = l2.reshape(blocks, BLOCK, vocab)[:, 1:].reshape(-1, vocab)
    flat, s_flat, valid = pieces["flat"], pieces["s_flat"], pieces["valid"]
    sums = torch.zeros(SLOTS, len(SLOT_KEYS), dtype=torch.float64, device=l2.device)
    c1_full = torch.zeros(blocks * SLOTS, dtype=torch.bool, device=l2.device)
    c2_full = torch.zeros_like(c1_full)
    for n, start in enumerate(range(0, flat.numel(), chunk)):
        idx = flat[start : start + chunk]
        top1 = pieces["chunks"][n]
        m = position_metrics(None, l2f[idx], pieces["ltf"][idx], top1=top1)
        s = s_flat[idx]
        sums[:, SK["n"]].index_add_(0, s, torch.ones_like(s, dtype=torch.float64))
        for key, value in m.items():
            sums[:, SK[key]].index_add_(0, s, value.double())
        c1_full[idx] = m["c1"].bool()
        c2_full[idx] = m["c2"].bool()
    c1 = c1_full.reshape(blocks, SLOTS)
    c2 = c2_full.reshape(blocks, SLOTS)
    reach1, len1 = accepted(c1, valid)
    reach2, len2 = accepted(c2, valid)
    block_valid = valid.any(-1).double()
    sums[:, SK["reach1"]] = (reach1.double() * block_valid[:, None]).sum(0)
    sums[:, SK["reach2"]] = (reach2.double() * block_valid[:, None]).sum(0)
    block = torch.stack(
        [
            block_valid.sum(),
            (len1.double() * block_valid).sum(),
            (len2.double() * block_valid).sum(),
        ]
    )
    return sums.cpu(), block.cpu()


@torch.no_grad()
def prepare_theta1(l1, lt, mask, chunk=1024):
    vocab = l1.shape[-1]
    blocks = l1.shape[1] // BLOCK
    valid = mask.reshape(blocks, BLOCK)[:, 1:].bool()
    l1f = l1.reshape(blocks, BLOCK, vocab)[:, 1:].reshape(-1, vocab)
    ltf = lt.reshape(blocks, BLOCK, vocab)[:, 1:].reshape(-1, vocab)
    flat = valid.reshape(-1).nonzero().squeeze(-1)
    s_flat = torch.arange(SLOTS, device=l1.device).expand(blocks, SLOTS).reshape(-1)
    chunks = [
        theta1_pieces(l1f[flat[s : s + chunk]], ltf[flat[s : s + chunk]])
        for s in range(0, flat.numel(), chunk)
    ]
    return {
        "blocks": blocks,
        "valid": valid,
        "flat": flat,
        "s_flat": s_flat,
        "ltf": ltf,
        "chunks": chunks,
    }
