"""Steps 3b and 4 analysis: Δτ of every variant vs its arm's unmodified pass.

    python scripts/dflash_port/analysis/analyze_maps.py

τ is pooled per category (sum of accepted lengths / blocks) and averaged over
categories (as the evaluation reports). Δ uses the same rows and anchors;
95% CIs are paired bootstrap over rows, stratified by category (2,000
resamples). Writes analysis24k/results/{knockouts,sensitivity,masking}.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESAMPLES = 2000


def category_tau(eal_sum, eal_total, cats, index=None):
    values = []
    for c in C.CATEGORIES:
        sel = cats == c
        if index is not None:
            sel_idx = index[c]
            values.append(eal_sum[sel_idx].sum() / eal_total[sel_idx].sum())
        else:
            values.append(eal_sum[sel].sum() / eal_total[sel].sum())
    return float(np.mean(values)), dict(zip(C.CATEGORIES, values, strict=True))


def bootstrap_delta(ref, var, totals, cats, seed=0):
    """Paired, category-stratified bootstrap of the category-mean τ difference.
    ref / var: per-row accepted-length sums; totals: per-row block counts
    (identical anchors, so identical totals)."""
    rng = np.random.default_rng(seed)
    members = {c: np.nonzero(cats == c)[0] for c in C.CATEGORIES}
    deltas = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        idx = {c: rng.choice(m, len(m)) for c, m in members.items()}
        a, _ = category_tau(var, totals, cats, idx)
        b, _ = category_tau(ref, totals, cats, idx)
        deltas[i] = a - b
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def arrays(per_row):
    return (np.array([r["eal_sum"] for r in per_row]),
            np.array([r["eal_total"] for r in per_row]),
            np.array([C.category(r["subset"]) for r in per_row]))


def compare(results, reference_label, labels, ci=True):
    ref_sum, totals, cats = arrays(results[reference_label])
    ref_tau, ref_cats = category_tau(ref_sum, totals, cats)
    out = {"reference": reference_label, "tau": ref_tau, "tau_categories": ref_cats,
           "variants": {}}
    for label in labels:
        var_sum, var_tot, var_cats = arrays(results[label])
        if not np.array_equal(var_tot, totals):
            raise ValueError(f"{label}: block counts differ from the reference")
        tau, by_cat = category_tau(var_sum, totals, cats)
        entry = {"tau": tau, "delta": tau - ref_tau,
                 "delta_categories": {c: by_cat[c] - ref_cats[c] for c in C.CATEGORIES}}
        if ci:
            entry["ci"] = bootstrap_delta(ref_sum, var_sum, totals, cats)
        out["variants"][label] = entry
    return out


def knockouts():
    path = C.OUT / "knockouts" / "knockouts.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())["results"]
    out = {}
    for arm in ("base", "static", "generator"):
        labels = [k for k in data if k.startswith(f"{arm}|") and k != f"{arm}|none"]
        if not labels:
            continue
        heads = [k for k in labels if "|head:" in k]
        others = [k for k in labels if k not in heads]
        res = compare(data, f"{arm}|none", others)
        res["variants"].update(compare(data, f"{arm}|none", heads, ci=False)["variants"])
        out[arm] = res
    return out


def sensitivity():
    path = C.OUT / "knockouts" / "sensitivity.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    data = raw["results"]
    labels = [k for k in data if k != "base|none"]
    res = compare(data, "base|none", [k for k in labels if "transplant" in k])
    res["variants"].update(compare(data, "base|none",
                                   [k for k in labels if "random" in k], ci=False)["variants"])
    # Mean over seeds per module and scale.
    summary = {}
    for label, entry in res["variants"].items():
        if "random" not in label:
            continue
        _, module, scale, _seed = label.split("|", 1)[1].split(":")
        summary.setdefault(f"{module}:{scale}", []).append(entry["delta"])
    res["random_mean"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                              "max": float(np.max(v)), "n": len(v)}
                          for k, v in summary.items()}
    res["norms"] = raw["norms"]
    res["mean_rel"] = raw["mean_rel"]
    return res


def masking():
    path = C.OUT / "prompt" / "masking.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())["rows"]
    out = {"rows": len(rows),
           "prompt_len_median": float(np.median([r["prompt_len"] for r in rows])),
           "response_len_median": float(np.median([r["response_len"] for r in rows])),
           "n_mask_equals_prompt_share": float(np.mean(
               [r["n_mask"] == r["prompt_len"] for r in rows])),
           "arms": {}}
    cats = np.array([C.category(r["subset"]) for r in rows])
    for arm in ("base", "static", "generator"):
        def per(cond, i):
            return np.array([r[f"{arm}|{cond}"][i] for r in rows])
        ref, totals = per("none", 0), per("none", 1)
        tau0, cat0 = category_tau(ref, totals, cats)
        entry = {"none": {"tau": tau0, "tau_categories": cat0}}
        for cond in ("prompt", "resp_random", "resp_early"):
            var = per(cond, 0)
            if not np.array_equal(per(cond, 1), totals):
                raise ValueError("block counts differ")
            tau, cat = category_tau(var, totals, cats)
            entry[cond] = {"tau": tau, "delta": tau - tau0,
                           "delta_categories": {c: cat[c] - cat0[c] for c in C.CATEGORIES},
                           "ci": bootstrap_delta(ref, var, totals, cats)}
        out["arms"][arm] = entry
    # Generator - static gap with and without the prompt in the context.
    gaps = {}
    for cond in ("none", "prompt", "resp_random", "resp_early"):
        g = np.array([r[f"generator|{cond}"][0] for r in rows])
        s = np.array([r[f"static|{cond}"][0] for r in rows])
        t = np.array([r[f"static|{cond}"][1] for r in rows])
        gt, _ = category_tau(g, t, cats)
        st, _ = category_tau(s, t, cats)
        gaps[cond] = {"delta": gt - st, "ci": bootstrap_delta(s, g, t, cats)}
    out["generator_minus_static"] = gaps
    sink = C.OUT / "prompt" / "masking_sink.json"
    if sink.is_file():
        srows = {r["row"]: r for r in json.loads(sink.read_text())["rows"]}
        for arm in ("base", "static", "generator"):
            ref = np.array([r[f"{arm}|none"][0] for r in rows])
            totals = np.array([r[f"{arm}|none"][1] for r in rows])
            check = np.array([srows[r["row"]][f"{arm}|none"][0] for r in rows])
            if not np.array_equal(check, ref):
                raise ValueError("sink run does not reproduce the unmasked pass")
            var = np.array([srows[r["row"]][f"{arm}|prompt_keep4"][0] for r in rows])
            tau0, cat0 = category_tau(ref, totals, cats)
            tau, cat = category_tau(var, totals, cats)
            out["arms"][arm]["prompt_keep4"] = {
                "tau": tau, "delta": tau - tau0,
                "delta_categories": {c: cat[c] - cat0[c] for c in C.CATEGORIES},
                "ci": bootstrap_delta(ref, var, totals, cats)}
    return out


def main():
    for name, function in (("knockouts", knockouts), ("sensitivity", sensitivity),
                           ("masking", masking)):
        result = function()
        if result is None:
            print(name, "missing")
            continue
        C.write_json(C.OUT / "results" / f"{name}.json", result)
        print(name, "written")


if __name__ == "__main__":
    main()
