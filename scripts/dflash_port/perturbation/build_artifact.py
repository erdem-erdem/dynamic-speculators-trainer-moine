"""Build the PERTURB24K results page: template + chart kit + data (inline only).

    python scripts/dflash_port/perturbation/build_artifact.py OUT.html

Reads perturb24k/results/results.json and inlines a compact copy of it, so
every chart on the page is drawn from the saved outputs.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path("/workspace/yusuf/runs/dflash/perturb24k/results/results.json")


def r(v, d=5):
    if isinstance(v, float):
        return float(f"{v:.{d}g}")
    if isinstance(v, list):
        return [r(x, d) for x in v]
    if isinstance(v, dict):
        return {k: r(x, d) for k, x in v.items()}
    return v


def compact(R):
    keep_s = ("point", "kind", "location", "eps", "seed", "rank", "update", "kl", "d_tau",
              "flip", "fix", "brk", "d_klT", "ci")
    D = {"scatter": [{k: s[k] for k in keep_s if k in s} for s in R["scatter"]],
         "pareto": R["pareto"], "pareto_by_kind": R["pareto_by_kind"], "points": {},
         "free": R.get("free", {}), "t2": R.get("t2", {}), "verify": R["verify"],
         "loss_check": R.get("loss_check", {}), "free_rel_norm": R.get("free_rel_norm", {})}
    for pt, P in R["points"].items():
        D["points"][pt] = {k: P[k] for k in ("floor", "dose", "rank", "fisher", "geometry", "steps",
                                             "cross", "slot_dense_0.3", "slot_descent_0.01",
                                             "t0_summary") if k in P}
        for loc in D["points"][pt].get("dose", {}).values():
            for k in [k for k in loc if k.startswith(("js100", "ov10", "in5"))]:
                del loc[k]
        for steps in D["points"][pt].get("steps", {}).values():
            for kind in ("descent", "ascent"):
                for e in steps[kind].values():
                    for k in ("kl21", "js", "js100", "ov5", "ov10", "in5", "in10", "tau1", "acc1",
                              "klT1"):
                        e.pop(k, None)
    return r(D)


def main():
    out = Path(sys.argv[1])
    R = json.loads(RESULTS.read_text())
    data = json.dumps(compact(R), separators=(",", ":"))
    html = (HERE / "artifact_template.html").read_text()
    html = html.replace("/*__CHARTS__*/", (HERE / "artifact_charts.js").read_text())
    html = html.replace("/*__RENDER__*/", (HERE / "artifact_render.js").read_text())
    html = html.replace("<!--__BODY__-->", (HERE / "artifact_body.html").read_text())
    html = html.replace("/*__DATA__*/null", data)
    out.write_text(html)
    print(out, f"{len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
