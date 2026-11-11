"""Step 5: linear probes and CKA on drafter states (validation).

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/analysis/analyze_repr.py

Inputs (collect_slots.py --hidden): per arm, the drafter's states after each
of the 5 layers at the 15 draft slots of 6 seeded fully valid blocks per row
(the same blocks for every arm; the layer-0 input is the constant mask-token
embedding at draft slots and is skipped), and the target's 6 stored layers at
the position each slot is predicted from.

- Probes (logistic regression, standardized features, L2; rows split 80/20 by
  row id): category (4 classes, accuracy) and "this slot is accepted" (in the
  leading correct run of its block, for that arm; ROC AUC), compared with
  baselines: slot index only, the drafter's own top-1 probability, and both.
- Linear CKA per layer between arms (base vs static, base vs generator, static
  vs generator, generator vs the generator conditioned on a same-category /
  cross-category donor prompt), the relative change ‖X_a − X_b‖ / ‖X_b‖, and
  the cosine between the static and generator changes (X − X_base).
- CKA of each drafter layer (base) vs each target layer fed in (and the
  target's last layer).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARMS = ("base", "static", "generator", "generator_same", "generator_cross")
TARGET_LAYERS = ("t2", "t10", "t18", "t26", "t34", "t36")


def linear_cka(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xy = (y.T @ x).norm() ** 2
    xx = (x.T @ x).norm()
    yy = (y.T @ y).norm()
    return float(xy / (xx * yy))


def auc(scores, labels):
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    pos = labels.sum()
    neg = len(labels) - pos
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit_logistic(x_train, y_train, x_test, classes, l2=1e-3, steps=300):
    """Multinomial logistic regression (full batch, LBFGS) on the GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    xs = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    mean, std = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt, xs = (xt - mean) / std, (xs - mean) / std
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    w = torch.zeros(xt.shape[1], classes, device=device, requires_grad=True)
    b = torch.zeros(classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ w + b, yt) + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(xs @ w + b, -1).cpu().numpy()


def main():
    hdir = C.OUT / "hidden" / "validation"
    rows = json.loads((C.OUT / "slots" / "validation" / "rows.json").read_text())
    labels = dict(np.load(hdir / "labels.npz"))
    row_cat = np.array([C.CATEGORIES.index(r["category"]) for r in rows])
    cat = row_cat[labels["row"]]
    test = labels["row"] % 5 == 0
    train = ~test
    slot = labels["slot"].astype(np.int64)
    n = len(slot)
    # Per-arm accepted labels from the slot records (join on row, block, slot).
    accepted, p_top = {}, {}
    for arm in ARMS:
        data = np.load(C.OUT / "slots" / "validation" / f"slots_{arm}.npz")
        keys = {}
        nb = len(data["slot"]) // 15
        correct = (data["correct"] & data["valid"]).reshape(nb, 15)
        acc_run = np.cumprod(correct, 1).astype(bool)
        brow, bblock = data["row"].reshape(nb, 15)[:, 0], data["block"].reshape(nb, 15)[:, 0]
        for i, (r, bl) in enumerate(zip(brow, bblock, strict=True)):
            keys[(int(r), int(bl))] = i
        idx = np.array([keys[(int(r), int(bl))] for r, bl in
                        zip(labels["row"], labels["block"], strict=True)])
        accepted[arm] = acc_run[idx, slot - 1]
        p_top[arm] = data["p_d_top"].reshape(nb, 15)[idx, slot - 1].astype(np.float32)
    if not np.array_equal(accepted["base"], labels["base_accepted"]):
        raise ValueError("Accepted labels disagree with the hidden-state labels")
    states = {arm: np.load(hdir / f"drafter_{arm}.npy", mmap_mode="r") for arm in ARMS}
    target = np.load(hdir / "target.npy", mmap_mode="r")
    if states["base"].shape[1] != n:
        raise ValueError("States and labels differ in length")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {"positions": int(n), "test_positions": int(test.sum()),
              "probes": {}, "cka": {}, "rel_change": {}, "delta_cos": {},
              "drafter_vs_target_cka": {}}
    slot_onehot = np.eye(16, dtype=np.float32)[slot]
    # Baselines for the accept probe (per arm).
    for arm in ("base", "static", "generator"):
        y = accepted[arm]
        base_slot = fit_logistic(slot_onehot[train], y[train], slot_onehot[test], 2)[:, 1]
        feat = np.concatenate([slot_onehot, np.log(p_top[arm])[:, None]], 1)
        base_both = fit_logistic(feat[train], y[train], feat[test], 2)[:, 1]
        result["probes"][f"{arm}|accept_baselines"] = {
            "slot_only_auc": auc(base_slot, y[test]),
            "drafter_p_top_auc": auc(p_top[arm][test], y[test]),
            "slot_plus_p_top_auc": auc(base_both, y[test]),
            "accept_rate": float(y.mean()),
        }
    for arm in ("base", "static", "generator"):
        for layer in range(1, 6):
            x = np.asarray(states[arm][layer], dtype=np.float32)
            probs = fit_logistic(x[train], cat[train], x[test], 4)
            entry = {"category_acc": float((probs.argmax(1) == cat[test]).mean())}
            y = accepted[arm]
            feat = np.concatenate([x, slot_onehot], 1)
            probs = fit_logistic(feat[train], y[train], feat[test], 2)[:, 1]
            entry["accept_auc"] = auc(probs, y[test])
            result["probes"][f"{arm}|L{layer - 1}"] = entry
            print(arm, layer - 1, entry, flush=True)
    for t, name in enumerate(TARGET_LAYERS):
        x = np.asarray(target[t], dtype=np.float32)
        probs = fit_logistic(x[train], cat[train], x[test], 4)
        result["probes"][f"target|{name}"] = {
            "category_acc": float((probs.argmax(1) == cat[test]).mean())}
    result["probes"]["chance_category"] = float(np.bincount(cat[test]).max() / test.sum())
    pairs = (("base", "static"), ("base", "generator"), ("static", "generator"),
             ("generator", "generator_same"), ("generator", "generator_cross"))
    with torch.no_grad():
        for layer in range(1, 6):
            xs = {arm: torch.as_tensor(np.asarray(states[arm][layer]), device=device)
                  .float() for arm in ARMS}
            for a, b in pairs:
                key = f"{a}|{b}|L{layer - 1}"
                result["cka"][key] = linear_cka(xs[a], xs[b])
                result["rel_change"][key] = float((xs[a] - xs[b]).norm() / xs[a].norm())
            ds = xs["static"] - xs["base"]
            dg = xs["generator"] - xs["base"]
            dgc = xs["generator_cross"] - xs["base"]
            dgs = xs["generator_same"] - xs["base"]

            def mean_cos(u, v):
                return float(torch.nn.functional.cosine_similarity(u, v, dim=1).mean())

            result["delta_cos"][f"L{layer - 1}"] = {
                "static_vs_generator": mean_cos(ds, dg),
                "generator_vs_cross_donor": mean_cos(dg, dgc),
                "generator_vs_same_donor": mean_cos(dg, dgs),
                "generator_minus_cross_over_generator_change": float(
                    (xs["generator"] - xs["generator_cross"]).norm() / dg.norm()),
                "generator_minus_same_over_generator_change": float(
                    (xs["generator"] - xs["generator_same"]).norm() / dg.norm()),
                "static_change_rel": float(ds.norm() / xs["base"].norm()),
                "generator_change_rel": float(dg.norm() / xs["base"].norm()),
            }
            for t, name in enumerate(TARGET_LAYERS):
                y = torch.as_tensor(np.asarray(target[t]), device=device).float()
                result["drafter_vs_target_cka"][f"L{layer - 1}|{name}"] = linear_cka(
                    xs["base"], y)
            del xs
            print("cka layer", layer - 1, flush=True)
    C.write_json(C.OUT / "results" / "repr_validation.json", result)
    print(json.dumps({k: result[k] for k in ("cka", "delta_cos")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
