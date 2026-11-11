"""Aggregation helpers: sweep files -> per-perturbation summaries (CPU only).

Conventions (as every DFlash report): within a category values are pooled
over rows, blocks and slots; "all" is the mean over the 4 categories.
Per-position metrics are means over valid draft positions; τ deltas are per
block. Paired bootstrap CIs resample rows within each category.
"""

from __future__ import annotations

import torch

CATEGORIES = ("math", "code", "stem", "chat")
MOVE = ("kl12", "kl21", "js", "js100", "tv12", "flip", "ov5", "ov10", "in5", "in10")


def load(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["done"] != len(data["rows"]):
        raise ValueError(f"{path}: incomplete sweep ({data['done']}/{len(data['rows'])} rows)")
    data["k"] = {key: i for i, key in enumerate(data["slot_keys"])}
    return data


def _cat_index(data):
    cats = data["categories"]
    return {c: torch.tensor([i for i, x in enumerate(cats) if x == c]) for c in CATEGORIES}


def summarize(data, *, per_slot=True):
    """{name: {"meta", "all": {...}, "<cat>": {...}, "slot": {metric: [15]}}}."""
    k = data["k"]
    slot, block = data["slot"], data["block"]
    idx = _cat_index(data)
    out = {}
    for p, name in enumerate(data["names"]):
        entry = {"meta": data["meta"][p]}
        per_cat = {}
        for c, rows in idx.items():
            s = slot[p, rows].sum((0, 1))  # [K]
            b = block[p, rows].sum(0)
            per_cat[c] = _metrics(s, b, k)
        entry.update(per_cat)
        entry["all"] = {m: sum(per_cat[c][m] for c in CATEGORIES) / len(CATEGORIES)
                        for m in per_cat[CATEGORIES[0]]}
        if per_slot:
            prof = {}
            for c, rows in idx.items():
                s = slot[p, rows].sum(0)  # [15, K]
                n = s[:, k["n"]].clamp_min(1)
                prof[c] = {
                    **{m: (s[:, k[m]] / n) for m in MOVE},
                    "d_acc": (s[:, k["c2"]] - s[:, k["c1"]]) / n,
                    "d_klT": (s[:, k["klT2"]] - s[:, k["klT1"]]) / n,
                    "fix": s[:, k["fix"]] / n,
                    "brk": s[:, k["brk"]] / n,
                    "acc1": s[:, k["c1"]] / n,
                    "reach1": s[:, k["reach1"]] / block[p, rows, 0].sum(),
                    "d_reach": (s[:, k["reach2"]] - s[:, k["reach1"]]) / block[p, rows, 0].sum(),
                }
            entry["slot"] = {
                m: [float(sum(prof[c][m][i] for c in CATEGORIES) / 4) for i in range(15)]
                for m in prof[CATEGORIES[0]]
            }
        out[name] = entry
    return out


def _metrics(s, b, k):
    n = float(s[k["n"]])
    blocks = float(b[0])
    m = {key: float(s[k[key]]) / n for key in MOVE}
    m.update(
        tau1=float(b[1]) / blocks,
        d_tau=float(b[2] - b[1]) / blocks,
        acc1=float(s[k["c1"]]) / n,
        d_acc=float(s[k["c2"] ] - s[k["c1"]]) / n,
        klT1=float(s[k["klT1"]]) / n,
        d_klT=float(s[k["klT2"]] - s[k["klT1"]]) / n,
        tvT1=float(s[k["tvT1"]]) / n,
        d_tvT=float(s[k["tvT2"]] - s[k["tvT1"]]) / n,
        fix=float(s[k["fix"]]) / n,
        brk=float(s[k["brk"]]) / n,
        net=float(s[k["fix"]] - s[k["brk"]]) / n,
        positions=n,
        blocks=blocks,
    )
    return m


def bootstrap_dtau(data, names=None, resamples=2000, seed=0):
    """Paired, category-stratified bootstrap of the category-mean Δτ.
    Returns {name: (lo, hi)} 95% intervals."""
    names = names or data["names"]
    pos = [data["names"].index(n) for n in names]
    block = data["block"][pos]  # [P, R, 3]
    d = block[..., 2] - block[..., 1]  # [P, R]
    nb = block[..., 0]
    idx = _cat_index(data)
    g = torch.Generator().manual_seed(seed)
    total = torch.zeros(resamples, len(pos), dtype=torch.float64)
    for rows in idx.values():
        r = len(rows)
        pick = rows[torch.randint(0, r, (resamples, r), generator=g)]  # [B, r]
        w = torch.zeros(resamples, d.shape[1], dtype=torch.float64)
        w.scatter_add_(1, pick, torch.ones_like(pick, dtype=torch.float64))
        total += (w @ d.T) / (w @ nb.T)
    total /= len(idx)
    lo = total.quantile(0.025, 0)
    hi = total.quantile(0.975, 0)
    return {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)}
