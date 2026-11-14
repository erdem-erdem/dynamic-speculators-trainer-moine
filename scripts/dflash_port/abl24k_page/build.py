"""Build the ABL24K results page: PERTURB24K template + chart kit + this page's body, renderer and data.

    python scripts/dflash_port/abl24k_page/build.py RESULTS.json OUT.html

RESULTS.json is the output of `scripts/dflash_port/abl24k_collect.py` (labels as in ARMS
below; missing runs are skipped). The base drafter's per-slot validation accuracy (F6) is
read from the base run's offline record.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / "perturbation"
BASE_RECORD = Path("/workspace/yusuf/runs/dflash/math10k/math10k-base/offline_eval")

# label, display name, short label, kind, loss, rank, heads, panels (F4 / F7 groups)
ARMS = [
    ("static_kl_r32", "static r32, KL", "r32", "static", "KL", 32, None, ["rank", "loss", "static", "generator"]),
    ("static_kl_r64", "static r64, KL", "r64", "static", "KL", 64, None, ["rank", "static"]),
    ("static_kl_r128", "static r128, KL", "r128", "static", "KL", 128, None, ["rank", "static"]),
    ("static_kl_r256", "static r256, KL", "r256", "static", "KL", 256, None, ["rank", "static"]),
    ("static_kl_r512", "static r512, KL", "r512", "static", "KL", 512, None, ["rank", "static"]),
    ("static_kl_r1024", "static r1024, KL", "r1024", "static", "KL", 1024, None, ["rank", "static"]),
    ("static_ce_r32", "static r32, CE", "CE", "static", "CE", 32, None, ["loss", "static"]),
    ("static_kllabel_r32", "static r32, KL + label gate α 0.1", "KL+gate", "static", "KL, label gate α 0.1", 32, None, ["loss", "static"]),
    ("gen_pm256_r32", "generator per-module heads, r32 (1.1B)", "gen 1.1B", "generator", "KL", 32, "per module", ["generator", "loss"]),
    ("gen_pt256_r32", "generator shared heads, r32", "gen shared r32", "generator", "KL", 32, "shared across layers", ["generator"]),
    ("gen_pt256_r64", "generator shared heads, r64", "gen shared r64", "generator", "KL", 64, "shared across layers", ["generator"]),
    ("gen_pt128_r128", "generator shared heads, r128, decoder 128", "gen shared r128", "generator", "KL", 128, "shared across layers", ["generator"]),
    ("gen_pt32_r512", "generator shared heads, r512, decoder 32", "gen shared r512", "generator", "KL", 512, "shared across layers", ["generator"]),
    ("gen_hybrid32_r512", "hybrid: per-layer static r512 + shared heads (decoder 32)", "hybrid r512", "generator", "KL", 512, "shared weights, per-layer static", ["generator"]),
    ("ctx_static_3ep", "static r32, KL, 3 epochs (MIX24K)", "static 3ep", "context", "KL", 32, None, ["rank"]),
    ("ctx_gen_3ep", "generator 1.1B, KL, 3 epochs (MIX24K)", "gen 3ep", "context", "KL", 32, "per module", ["generator"]),
]


def r(v, d=5):
    if isinstance(v, float):
        return float(f"{v:.{d}g}")
    if isinstance(v, list):
        return [r(x, d) for x in v]
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    return v


def base_slot_acc():
    files = sorted(BASE_RECORD.glob("*/*/*.json"))
    files = [f for f in files if f.name.count(".") == 1]
    rec = json.loads(files[-1].read_text())
    res = rec["validation"] if "validation" in rec else rec
    accs = [[] for _ in range(15)]
    for data in res["subsets"].values():
        ctx = data.get("contexts", {}).get("1", {})
        for k in range(1, 16):
            if f"position_{k}_acc" in ctx:
                accs[k - 1].append(ctx[f"position_{k}_acc"])
    return [sum(a) / len(a) for a in accs] if all(accs) else None


def main():
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    R = json.loads(results.read_text())
    runs = {}
    for label, run in R["runs"].items():
        run = dict(run)
        run.pop("run_dir", None)
        for sp in ("validation", "test", "ood"):
            if sp in run:
                run[sp].pop("pooled", None)
        runs[label] = run
    try:
        slot = base_slot_acc()
    except (IndexError, KeyError, FileNotFoundError):
        slot = None
    data = {
        "arms": [dict(label=a[0], name=a[1], short=a[2], kind=a[3], loss=a[4], rank=a[5], heads=a[6], panels=a[7],
                      group="rank" if a[0].startswith("static_kl_r") else None) for a in ARMS],
        "runs": runs, "base": R["base"], "reference": "static_kl_r32", "base_slot_acc": slot,
    }
    html = (KIT / "artifact_template.html").read_text()
    html = html.replace("<title>Move vs Improve: Results</title>", "<title>ABL24K Ablations</title>")
    html = html.replace("/*__CHARTS__*/", (KIT / "artifact_charts.js").read_text())
    html = html.replace("/*__RENDER__*/", (HERE / "render.js").read_text())
    html = html.replace("<!--__BODY__-->", (HERE / "body.html").read_text())
    html = html.replace("/*__DATA__*/null", json.dumps(r(data), separators=(",", ":")))
    out.write_text(html)
    print(out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
