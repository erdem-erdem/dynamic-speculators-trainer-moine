"""GENDIAG phases B4 / E2: is there a reproducible per-row signal, and does the
generator write along it?

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/gendiag/direction.py RECIPE \
        --tier T --out OUT.json [--snapshot u01000.pt] [--split val|train] [--point mean|zero]

At the point ΔW = ΔW̄ (the mean generated ΔW over the evaluated rows: where
the generator's shared part sits) or ΔW = 0 (base), for each row i the loss
gradient w.r.t. every adapted ΔW is taken twice, from the first and from the
second half of the row's supervised tokens (g_i¹, g_i²; validation anchors,
eval mode, the training loss). With ḡ the mean over rows and ḡ_c the mean of
the row's category:

- signal budget (noise-corrected through the half split):
    shared      ⟨ḡ¹, ḡ²⟩
    row         mean_i ⟨g_i¹ − ḡ¹,  g_i² − ḡ²⟩      (reproducible row-specific part)
    within_cat  mean_i ⟨g_i¹ − ḡ_c¹, g_i² − ḡ_c²⟩   (beyond category)
  each over sqrt(mean‖g¹‖² · mean‖g²‖²); `noise` is the rest.
- direction test: with D_j = ΔW_j − ΔW̄ the generated deviation of row j, the
  first-order loss change −⟨g_i, D_j⟩ (g_i = g_i¹ + g_i²) for j = i (own), 8
  rows of the same category and 8 of other categories; also with the second
  half only (held-out half: the gradient of tokens the generator's input
  never saw differently from the first half).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from controls import dense_dw, generate  # noqa: E402

DONORS = 8


def half_batches(runtime, row):
    batches = list(runtime.corpus.microbatches([row], runtime.device, runtime.dtype))
    if len(batches) != 1:
        raise RuntimeError(f"row {row}: {len(batches)} microbatches")
    batch = batches[0][0]
    mask = batch["loss_mask"]
    pos = torch.nonzero(mask[0].bool(), as_tuple=False)[:, 0]
    mid = pos[len(pos) // 2]
    first, second = mask.clone(), mask.clone()
    first[0, mid:] = 0
    second[0, :mid] = 0
    return {**batch, "loss_mask": first}, {**batch, "loss_mask": second}


def gradient(runtime, batch, point, scale, eye):
    """∂L/∂ΔW per module at ΔW = point (dense path: A = I, B = ΔW / s)."""
    leaves = {k: (point[k] / scale[k]).detach().requires_grad_(True) for k in point}
    factors = {k: (eye[v.shape[1]], leaves[k]) for k, v in point.items()}
    with torch.enable_grad():
        _, loss, _ = runtime.forward(batch, factors)
        grads = torch.autograd.grad(loss, list(leaves.values()))
    return {k: g / scale[k] for k, g in zip(leaves, grads, strict=True)}, float(loss)


def inner(a, b):
    return sum(float((a[k] * b[k]).sum()) for k in a)


def analyse(runtime, job, output, generator, args):
    started = time.time()
    rows = C.eval_rows(runtime, job, args.split, per_category=args.per_category)
    if args.per_subset:
        seen: dict[str, int] = {}
        kept = []
        for subset, row in rows:
            if seen.get(subset, 0) < args.per_subset:
                kept.append((subset, row))
                seen[subset] = seen.get(subset, 0) + 1
        rows = kept
    labels = [C.category(s) for s, _ in rows]
    dev, scale = runtime.device, C.scales(runtime)
    factors, _ = generate(generator, runtime, rows)
    keys = list(factors[0])
    n = len(rows)
    mean = None
    for f in factors:
        dw = dense_dw(f, scale, dev)
        mean = {k: dw[k] / n for k in keys} if mean is None else {k: mean[k] + dw[k] / n for k in keys}
    point = mean if args.point == "mean" else {k: torch.zeros_like(v) for k, v in mean.items()}
    eye = {d: torch.eye(d, device=dev) for d in {v.shape[1] for v in mean.values()}}
    rng = random.Random(1)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(labels):
        by_cat.setdefault(c, []).append(i)
    zeros = {k: torch.zeros_like(v) for k, v in mean.items()}
    sum1 = {k: v.clone() for k, v in zeros.items()}
    sum2 = {k: v.clone() for k, v in zeros.items()}
    cat1 = {c: {k: v.clone() for k, v in zeros.items()} for c in by_cat}
    cat2 = {c: {k: v.clone() for k, v in zeros.items()} for c in by_cat}
    per_row = []
    for i, (_, row) in enumerate(rows):
        b1, b2 = half_batches(runtime, row)
        g1, l1 = gradient(runtime, b1, point, scale, eye)
        g2, l2 = gradient(runtime, b2, point, scale, eye)
        for k in keys:
            sum1[k] += g1[k]
            sum2[k] += g2[k]
            cat1[labels[i]][k] += g1[k]
            cat2[labels[i]][k] += g2[k]
        g = {k: g1[k] + g2[k] for k in keys}
        same = rng.sample([j for j in by_cat[labels[i]] if j != i], DONORS)
        cross = rng.sample([j for j in range(n) if labels[j] != labels[i]], DONORS)

        def gain(j, grad):
            d = dense_dw(factors[j], scale, dev)
            return -sum(float((grad[k] * (d[k] - mean[k])).sum()) for k in keys)

        own_dw = dense_dw(factors[i], scale, dev)
        dev_norm = sum(float((own_dw[k] - mean[k]).pow(2).sum()) for k in keys) ** 0.5
        per_row.append({
            "category": labels[i],
            "loss1": l1, "loss2": l2,
            "g1g1": inner(g1, g1), "g2g2": inner(g2, g2), "g1g2": inner(g1, g2),
            "dev_norm": dev_norm,
            "g_norm": inner(g, g) ** 0.5,
            "own": gain(i, g), "same": [gain(j, g) for j in same], "cross": [gain(j, g) for j in cross],
            "own_h2": gain(i, g2), "same_h2": [gain(j, g2) for j in same[:4]],
        })
        if (i + 1) % 50 == 0:
            print(f"[direction] {i + 1}/{n} ({time.time() - started:.0f}s)", flush=True)
    m1 = {k: v / n for k, v in sum1.items()}
    m2 = {k: v / n for k, v in sum2.items()}
    e11 = sum(r["g1g1"] for r in per_row) / n
    e22 = sum(r["g2g2"] for r in per_row) / n
    e12 = sum(r["g1g2"] for r in per_row) / n
    shared = inner(m1, m2)
    within = 0.0
    for c, members in by_cat.items():
        mc1 = {k: v / len(members) for k, v in cat1[c].items()}
        mc2 = {k: v / len(members) for k, v in cat2[c].items()}
        within += len(members) * inner(mc1, mc2)
    within = e12 - within / n  # mean_i ⟨g_i¹ − ḡ_c¹, g_i² − ḡ_c²⟩
    norm = (e11 * e22) ** 0.5
    budget = {
        "shared": shared / norm,
        "row": (e12 - shared) / norm,
        "within_cat": within / norm,
        "between_cat": (e12 - shared - within) / norm,
        "noise": 1 - e12 / norm,
        "split_half_cos_mean": sum(r["g1g2"] / max((r["g1g1"] * r["g2g2"]) ** 0.5, 1e-300) for r in per_row) / n,
    }

    def avg(xs):
        return sum(xs) / len(xs)

    direction = {
        "own": avg([r["own"] for r in per_row]),
        "same": avg([avg(r["same"]) for r in per_row]),
        "cross": avg([avg(r["cross"]) for r in per_row]),
        "own_minus_same_share_positive": avg([float(r["own"] > avg(r["same"])) for r in per_row]),
        "own_h2": avg([r["own_h2"] for r in per_row]),
        "same_h2": avg([avg(r["same_h2"]) for r in per_row]),
        "own_cos": avg([r["own"] / max(r["g_norm"] * r["dev_norm"], 1e-300) for r in per_row]),
        "by_category": {
            c: {
                "own": avg([r["own"] for r in per_row if r["category"] == c]),
                "same": avg([avg(r["same"]) for r in per_row if r["category"] == c]),
                "cross": avg([avg(r["cross"]) for r in per_row if r["category"] == c]),
            }
            for c in by_cat
        },
    }
    return {
        "rows": [[s, r] for s, r in rows], "split": args.split, "point": args.point,
        "snapshot": str(args.snapshot) if args.snapshot else None, "run": str(output),
        "budget": budget, "direction": direction, "per_row": per_row,
        "seconds": time.time() - started,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe")
    p.add_argument("--tier", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--split", default="val", choices=("val", "train"))
    p.add_argument("--per-category", type=int, default=125)
    p.add_argument("--point", default="mean", choices=("mean", "zero"))
    p.add_argument("--per-subset", type=int, help="keep the first N rows of each subset")
    args = p.parse_args(argv)
    result = C.with_trained(args.recipe, args.tier, lambda *a: analyse(*a, args), snapshot=args.snapshot)
    C.write_json(args.out, result)
    print("[direction] budget", {k: round(v, 4) for k, v in result["budget"].items()})
    print("[direction] direction", {k: v for k, v in result["direction"].items() if k != "by_category"})


if __name__ == "__main__":
    main()
