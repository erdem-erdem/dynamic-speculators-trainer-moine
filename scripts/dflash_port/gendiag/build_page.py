"""Build the GENDIAG results page (one self-contained HTML file).

    python scripts/dflash_port/gendiag/build_page.py --out /path/gendiag.html \
        [--results results.json]

Reads (read-only) the diagnostics, controls, direction, internals and
snapshot outputs under /workspace/yusuf/runs/dflash/gendiag/ and the
`abl24k_collect.py` results JSON; embeds a compact data object and the chart
helpers into `page/template.html`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from summarize_diag import DIAG, series  # noqa: E402

OUT = Path("/workspace/yusuf/runs/dflash/gendiag")
DIAG_RUNS = ("pt256", "pm256", "pt256-twin", "pm256-twin", "pt256-fp32", "pm256-fp32",
             "pt256-fp32-centred", "pm64", "pt64", "pt1024")


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def controls():
    out = {}
    for p in sorted((OUT / "controls").glob("*.json")):
        r = _load(p)
        out[p.stem] = {"summary": r["summary"], "wins": r["own_win_share"], "pca": r["pca"],
                       "rows": len(r["rows"])}
    return out


def directions():
    out = {}
    for p in sorted((OUT / "direction").glob("*.json")):
        r = _load(p)
        pr = r["per_row"]
        own = [x["own"] for x in pr]
        same = [sum(x["same"]) / len(x["same"]) for x in pr]
        cross = [sum(x["cross"]) / len(x["cross"]) for x in pr]
        diff = [a - b for a, b in zip(own, same, strict=True)]
        n = len(diff)
        mean = sum(diff) / n
        sd = (sum((d - mean) ** 2 for d in diff) / (n - 1)) ** 0.5
        out[p.stem] = {"budget": r["budget"], "direction": r["direction"], "rows": n,
                       "own_minus_same": {"mean": mean, "se": sd / n**0.5},
                       "own_minus_cross": {"mean": sum(a - b for a, b in zip(own, cross, strict=True)) / n}}
    return out


def internals():
    out = {}
    for p in sorted((OUT / "internals").glob("*.json")):
        r = _load(p)
        out[p.stem] = {"grad": r["grad"], "zgrad": r["zgrad"], "params": r["params"],
                       "probe_val": _slim_probe(r["probe"]["val"]),
                       "probe_train": _slim_probe(r["probe"]["train"])}
    return out


def _slim_probe(pv):
    keep = ("summary", "enc_hidden", "encoder", "cond_hidden", "z", "dec_first", "dec_out", "A", "B")
    res = {k: pv[k] for k in keep if k in pv}
    res["dW"] = pv["dW"]
    res["headA"], res["headB"] = pv["headA"], pv["headB"]
    return res


def snapshots():
    out = {}
    for d in sorted((OUT / "snapshots").glob("*")):
        runs = {}
        for p in sorted(d.glob("controls_u*.json")):
            r = _load(p)
            runs.setdefault(int(p.stem.split("_u")[1]), {})["controls"] = {
                k: {"tau": v["tau"], "vs": v.get("vs_reference")} for k, v in r["summary"].items()}
        for p in sorted(d.glob("internals_u*.json")):
            r = _load(p)
            runs.setdefault(int(p.stem.split("_u")[1]), {})["internals"] = {
                "grad": r["grad"], "dW": r["probe"]["val"]["dW"],
                "spread": {k: r["probe"]["val"][k]["spread"] for k in ("summary", "encoder", "z", "dec_out", "A", "B")}}
        out[d.name] = runs
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--results", type=Path)
    args = p.parse_args(argv)
    data = {
        "series": {n: _thin(series(n)) for n in DIAG_RUNS if (DIAG / n / "diag.jsonl").is_file()},
        "controls": controls(),
        "direction": directions(),
        "internals": internals(),
        "snapshots": snapshots(),
        "results": _load(args.results) if args.results else None,
    }
    template = (HERE / "page" / "template.html").read_text()
    charts = (HERE / "page" / "charts.js").read_text()
    body = (HERE / "page" / "body.html").read_text()
    render = (HERE / "page" / "render.js").read_text()
    html = template.replace("/*BODY*/", body).replace("/*RENDER*/", render).replace("/*CHARTS*/", charts).replace(
        "/*DATA*/", "const D = " + json.dumps(data, separators=(",", ":"), default=float) + ";")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({len(html) / 1e6:.2f} MB); series {list(data['series'])}")


def _thin(s):
    """Keep the series the page draws."""
    keys = ("update", "grad_share", "grad_group_share", "swallowed", "rel_moved", "zgrad_shared",
            "spread", "headA_dyn", "headB_dyn", "headB_static", "headB_bias", "headA_static",
            "headA_bias", "dW", "train_dW", "validation", "mean_dW_change", "rel_step")
    out = {k: s[k] for k in keys}
    out["meta"] = {k: s["meta"][k] for k in ("groups", "head_sharing", "constant_input", "dtype")}
    return out


if __name__ == "__main__":
    main()
