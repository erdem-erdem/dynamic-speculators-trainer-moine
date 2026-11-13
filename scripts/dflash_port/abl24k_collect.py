"""Collect ABL24K run results into one JSON (read-only over run directories).

    python scripts/dflash_port/abl24k_collect.py --out results.json \
        --base /workspace/yusuf/runs/dflash/math10k/math10k-base \
        --reference static_kl_r32 LABEL=RUN_DIR [LABEL=RUN_DIR ...]

Per run: trainable parameters, training time, the validation history
(category-mean τ and full_acc per evaluation), validation / test / OOD
category-mean τ, full_acc and per-category τ, paired bootstrap differences
(τ, full_acc; 95% CI, stratified by category, as `compare_arms.py`) against
the reference arm and against base, the LoRA-coefficient ablation (M4), and
for generators the D13 similarity and D14 cost. Missing splits are skipped,
so the script can run while lanes are still going.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_arms import CATEGORIES, bootstrap, load_rows, macro, pooled, stats  # noqa: E402

SPLITS = ("validation", "test", "ood")


def _groups(rows):
    cats = [c for c in CATEGORIES if any(v[0] == c for v in rows.values())]
    cats += sorted({v[0] for v in rows.values()} - set(cats))
    keys = sorted(rows)
    return cats, {c: [k for k in keys if rows[k][0] == c] for c in cats}


def _split_summary(rows):
    cats, by_cat = _groups(rows)
    out = {"rows": len(rows), "all": macro(rows, list(by_cat.values()))}
    out["categories"] = {c: stats(pooled(rows, by_cat[c])) for c in cats}
    out["pooled"] = stats(pooled(rows, sorted(rows)))
    return out


def _try_rows(run_dir, split):
    try:
        return load_rows(run_dir, split, "coef-2_donor-correct")
    except (IndexError, FileNotFoundError, KeyError):
        return None


def _history(run_dir):
    files = list(Path(run_dir).glob("pretraining/*/metrics.json"))
    if not files:
        return []
    hist = []
    for entry in json.loads(files[0].read_text()):
        per_cat = {}
        for subset, data in entry.get("subsets", {}).items():
            per_cat.setdefault(subset.rsplit("-", 1)[0], []).append(
                data["contexts"]["1"]["eal"]
            )
        tau = {c: sum(v) / len(v) for c, v in per_cat.items()}
        hist.append(
            {
                "update": entry["update"],
                "score": entry["score"],
                "tau": sum(tau.values()) / len(tau) if tau else None,
                "tau_categories": tau,
            }
        )
    return hist


def _offline(run_dir):
    files = [
        f
        for f in Path(run_dir).glob("offline_eval/*/*/*.json")
        if f.name.count(".") == 1
    ]
    if not files:
        return {}
    rec = json.loads(sorted(files)[-1].read_text())
    out = {"similarity": rec.get("similarity"), "cost": rec.get("cost")}
    abl = rec.get("ablations") or {}
    out["coefficients"] = {
        k.split("_")[0].removeprefix("coef-"): v.get("score")
        for k, v in abl.items()
        if k.endswith("donor-correct")
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("runs", nargs="+")
    args = p.parse_args(argv)
    runs = dict(spec.split("=", 1) for spec in args.runs)
    base_rows = {s: _try_rows(args.base, s) for s in SPLITS}
    ref_rows = {s: _try_rows(runs[args.reference], s) for s in SPLITS}
    result = {"base": {}, "runs": {}}
    for s in SPLITS:
        if base_rows[s]:
            result["base"][s] = _split_summary(base_rows[s])
    for label, run_dir in runs.items():
        entry = {"run_dir": run_dir}
        res = list(Path(run_dir).glob("pretraining/*/result.json"))
        if res:
            r = json.loads(res[0].read_text())
            entry["trainable_parameters"] = r.get("trainable_parameters")
            entry["training_seconds"] = r.get("training_seconds")
            entry["completed_updates"] = r.get("completed_updates")
        entry["history"] = _history(run_dir)
        entry.update(_offline(run_dir))
        for s in SPLITS:
            rows = _try_rows(run_dir, s)
            if not rows:
                continue
            summ = _split_summary(rows)
            _, by_cat = _groups(rows)
            groups = list(by_cat.values())
            for name, ref in (("vs_reference", ref_rows[s]), ("vs_base", base_rows[s])):
                if ref and sorted(ref) == sorted(rows) and ref is not rows:
                    a, b = macro(rows, groups), macro(ref, groups)
                    ci = bootstrap(rows, ref, groups, args.resamples, 0)
                    summ[name] = {
                        m: {"delta": a[m] - b[m], "ci": ci[m]} for m in ("tau", "full_acc")
                    }
            entry[s] = summ
        result["runs"][label] = entry
        print(label, {s: round(entry[s]["all"]["tau"], 4) for s in SPLITS if s in entry})
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
