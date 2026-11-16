"""Compact time series from GENDIAG diag.jsonl files (for the report and page).

    python scripts/dflash_port/gendiag/summarize_diag.py NAME [NAME ...] --out series.json

NAME = a directory under /workspace/yusuf/runs/dflash/gendiag/diag/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIAG = Path("/workspace/yusuf/runs/dflash/gendiag/diag")
TRUNK = ("enc1", "enc2", "enc_ln", "cond1", "cond2", "emb", "dec_in", "dec_res")
HEADS_W = ("headA_w", "headB_w")
STATIC = ("headA_b", "headB_b", "bias_A", "bias_B")
STAGES = ("summary", "enc_hidden", "encoder", "cond_hidden", "z", "dec_first", "dec_out", "A", "B")


def load(name):
    lines = [json.loads(x) for x in (DIAG / name / "diag.jsonl").read_text().splitlines() if x.strip()]
    meta = next(x for x in lines if x["kind"] == "meta")
    steps = {}
    for x in lines:
        if x["kind"] == "step":
            steps[x["update"]] = x  # a resumed run may repeat updates; keep the last
    return meta, [steps[u] for u in sorted(steps)]


def series(name):
    meta, steps = load(name)
    out = {"meta": meta, "update": [], "grad_share": {}, "grad_rms": {}, "swallowed": {}, "rel_step": {},
           "rel_moved": {}, "moved": {}, "zgrad_shared": [], "zgrad_norm": [], "spread": {}, "eta2": {},
           "participation": {}, "headA_dyn": [], "headB_dyn": [], "headB_static": [], "headB_bias": [],
           "headA_static": [], "headA_bias": [], "dW": {}, "mean_dW_change": [], "validation": [],
           "grad_group_share": {"trunk": [], "head_weights": [], "biases": []},
           "train_dW": {}}
    for s in steps:
        out["update"].append(s["update"])
        for g, v in s["grad"].items():
            out["grad_share"].setdefault(g, []).append(v["share"])
            out["grad_rms"].setdefault(g, []).append(v["rms"])
        out["grad_group_share"]["trunk"].append(sum(s["grad"][g]["share"] for g in TRUNK if g in s["grad"]))
        out["grad_group_share"]["head_weights"].append(sum(s["grad"][g]["share"] for g in HEADS_W if g in s["grad"]))
        out["grad_group_share"]["biases"].append(sum(s["grad"][g]["share"] for g in STATIC if g in s["grad"]))
        for g, v in s["step"].items():
            out["swallowed"].setdefault(g, []).append(v["swallowed_share"])
            out["rel_step"].setdefault(g, []).append(v["rel_step"])
        for g, v in s["init"].items():
            out["rel_moved"].setdefault(g, []).append(v["rel_moved"])
            out["moved"].setdefault(g, []).append(v["moved"])
        z = s.get("zgrad") or {}
        out["zgrad_shared"].append(z.get("shared_share"))
        out["zgrad_norm"].append(z.get("norm_mean_cohort"))
        pv = s["probe"]["val"]
        for st in STAGES:
            out["spread"].setdefault(st, []).append(pv[st]["spread"])
            out["eta2"].setdefault(st, []).append(pv[st]["eta2"])
            out["participation"].setdefault(st, []).append(pv[st]["participation"])
        for part in ("A", "B"):
            out[f"head{part}_dyn"].append(pv[f"head{part}"]["dynamic_over_static"])
            out[f"head{part}_static"].append(pv[f"head{part}"]["static_norm"])
            out[f"head{part}_bias"].append(pv[f"head{part}"]["bias_norm"])
        for k, v in pv["dW"].items():
            out["dW"].setdefault(k, []).append(v)
        for k, v in s["probe"]["train"]["dW"].items():
            out["train_dW"].setdefault(k, []).append(v)
        out["mean_dW_change"].append(s["probe"].get("mean_dW_change"))
        out["validation"].append(s.get("validation_score"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    result = {name: series(name) for name in args.names if (DIAG / name / "diag.jsonl").is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    for name, s in result.items():
        last = len(s["update"]) - 1
        print(name, "update", s["update"][last],
              "trunk share %.2e" % s["grad_group_share"]["trunk"][last],
              "dyn_ratio %.4f" % s["dW"]["dyn_ratio"][last],
              "z spread %.3e" % s["spread"]["z"][last],
              "B spread %.3e" % s["spread"]["B"][last])


if __name__ == "__main__":
    main()
