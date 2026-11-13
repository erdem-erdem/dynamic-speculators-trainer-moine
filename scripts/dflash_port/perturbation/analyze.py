"""Analysis of the PERTURB24K sweeps: tables, hypothesis evidence, figure data.

    .venv/bin/python scripts/dflash_port/perturbation/analyze.py [--png-dir DIR]

Reads perturb24k/{t1,t0steps,free,t2,t0}/..., writes
perturb24k/results/results.json (every number of the report and the
artifact) and PNG figures (matplotlib) to --png-dir. CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as A  # noqa: E402

OUT = Path("/workspace/yusuf/runs/dflash/perturb24k")
POINTS = ("base", "static")
LAYERS = 5
MODULES = ("q", "k", "v", "o", "gate", "up", "down")
T2_RUNS = ("full", "lora_attn_r512", "lora_ffn_r512")
T2_UPDATES = (0, 150, 300, 600)
KL_THRESHOLD = 0.01
FLIP_PP = 0.01
CAT_EPS = 0.0003  # sweep.CAT_STEP_EPS: per-category steps and slot profiles
SLOT_METRICS = ("kl12", "flip", "d_acc", "d_klT", "fix", "brk", "acc1", "reach1", "d_reach")


def members(location):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness as H  # noqa: PLC0415

    return H.location_members(location)


def t1_locations():
    import harness as H  # noqa: PLC0415

    return H.t1_locations()


def fisher_key(site, path):
    return site if path == "both" else f"{site}@{path}"


def compact(entry, slots=False):
    keep = ("kl12", "kl21", "js", "js100", "tv12", "flip", "ov5", "ov10", "in5", "in10",
            "d_tau", "d_acc", "d_klT", "d_tvT", "fix", "brk", "net", "tau1", "acc1", "klT1")
    out = {"meta": entry["meta"], "all": {k: entry["all"][k] for k in keep}}
    out["cat"] = {c: {k: entry[c][k] for k in ("kl12", "flip", "d_tau", "d_acc", "d_klT", "fix",
                                                "brk", "tau1")} for c in A.CATEGORIES}
    if slots and "slot" in entry:
        out["slot"] = {m: entry["slot"][m] for m in SLOT_METRICS}
    return out


def interp_eps(eps, values, threshold):
    """Smallest eps where `values` (increasing in eps) crosses `threshold`,
    log-log interpolation; None if never reached, "<min" if already above."""
    pairs = sorted(zip(eps, values))
    if pairs[0][1] >= threshold:
        return None if pairs[0][1] is None else pairs[0][0]
    for (e0, v0), (e1, v1) in zip(pairs, pairs[1:]):
        if v1 >= threshold:
            if v0 <= 0:
                return e1
            t = (math.log(threshold) - math.log(v0)) / (math.log(v1) - math.log(v0))
            return math.exp(math.log(e0) + t * (math.log(e1) - math.log(e0)))
    return None


def pareto(points):
    """Upper-left front: max d_tau among points with KL <= x."""
    front, best = [], -1e9
    for p in sorted(points, key=lambda p: p["kl"]):
        if p["d_tau"] > best:
            best = p["d_tau"]
            front.append(p)
    return front


def complete(path):
    """True when the sweep file exists and holds every row (partial files from
    a running job are skipped with a note; final runs must have none)."""
    if not path.exists():
        return False
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["done"] != len(data["rows"]):
        print(f"SKIP incomplete {path} ({data['done']}/{len(data['rows'])})")
        return False
    return True


def load_sweep(path, slots=False):
    data = A.load(path)
    return data, A.summarize(data, per_slot=slots)


def main():  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=None)
    args = parser.parse_args()
    R = {"points": {}}
    scatter = []

    # ---------------- T1 dense + low rank ----------------
    for point in POINTS:
        P = R["points"].setdefault(point, {})
        dense_path = OUT / "t1" / f"{point}_dense.pt"
        if not complete(dense_path):
            continue
        data, summ = load_sweep(dense_path, slots=True)
        ci = A.bootstrap_dtau(data)
        floor = [summ[n]["all"] for n in summ if summ[n]["meta"]["eps"] == 1e-4]
        P["floor"] = {k: sum(f[k] for f in floor) / len(floor)
                      for k in ("kl12", "tv12", "flip", "d_tau", "fix", "brk", "js100")}
        P["floor_ci"] = [ci[n] for n in summ if summ[n]["meta"]["eps"] == 1e-4]
        dose = {}
        for name, e in summ.items():
            m = e["meta"]
            if m["eps"] == 1e-4:
                continue
            d = dose.setdefault(m["location"], {})
            d.setdefault(m["eps"], []).append(e)
            scatter.append({"point": point, "kind": "dense", "location": m["location"],
                            "eps": m["eps"], "seed": m["seed"], "kl": e["all"]["kl12"],
                            "d_tau": e["all"]["d_tau"], "flip": e["all"]["flip"],
                            "fix": e["all"]["fix"], "brk": e["all"]["brk"],
                            "d_klT": e["all"]["d_klT"], "ci": ci[name]})
        curves = {}
        for loc, by_eps in dose.items():
            eps = sorted(by_eps)
            c = {"eps": eps}
            for metric in ("kl12", "tv12", "flip", "js100", "d_tau", "d_klT", "fix", "brk",
                           "ov10", "in5"):
                vals = [[e["all"][metric] for e in by_eps[x]] for x in eps]
                c[metric] = [sum(v) / len(v) for v in vals]
                c[metric + "_lo"] = [min(v) for v in vals]
                c[metric + "_hi"] = [max(v) for v in vals]
            c["cat_kl12"] = {cat: [sum(e[cat]["kl12"] for e in by_eps[x]) / len(by_eps[x])
                                   for x in eps] for cat in A.CATEGORIES}
            c["eps_kl001"] = interp_eps(eps, c["kl12"], KL_THRESHOLD)
            c["eps_flip1pp"] = interp_eps(eps, c["flip"], P["floor"]["flip"] + FLIP_PP)
            curves[loc] = c
        P["dose"] = curves
        # Per-slot profiles at eps = 0.3 for the group locations (seed mean).
        prof = {}
        for loc in ("fc", "all_attn", "all_ffn", "whole", "layer1", "layer5"):
            es = dose[loc][0.3]
            prof[loc] = {m: [sum(e["slot"][m][i] for e in es) / len(es) for i in range(15)]
                         for m in SLOT_METRICS}
        P["slot_dense_0.3"] = prof
        lr_path = OUT / "t1" / f"{point}_lowrank.pt"
        if complete(lr_path):
            ldata, lsum = load_sweep(lr_path)
            ranks = {}
            for name, e in lsum.items():
                m = e["meta"]
                key = f"{m['location']}|{m['eps']}"
                ranks.setdefault(key, {}).setdefault(m["rank"], []).append(e["all"])
                scatter.append({"point": point, "kind": "lowrank", "location": m["location"],
                                "eps": m["eps"], "rank": m["rank"], "seed": m["seed"],
                                "kl": e["all"]["kl12"], "d_tau": e["all"]["d_tau"],
                                "flip": e["all"]["flip"], "fix": e["all"]["fix"],
                                "brk": e["all"]["brk"], "d_klT": e["all"]["d_klT"]})
            table = {}
            for key, by_rank in ranks.items():
                loc, eps = key.split("|")
                row = {}
                for rank, es in sorted(by_rank.items()):
                    row[str(rank)] = {k: sum(x[k] for x in es) / len(es)
                                      for k in ("kl12", "flip", "tv12", "d_tau", "d_klT")}
                dense = dose[loc][float(eps)]
                row["full"] = {k: sum(x["all"][k] for x in dense) / len(dense)
                               for k in ("kl12", "flip", "tv12", "d_tau", "d_klT")}
                table[key] = row
            P["rank"] = table

    # ---------------- T0: Fisher, gradient geometry, steps ----------------
    for point in POINTS:
        P = R["points"].setdefault(point, {})
        fpath = OUT / "t0" / point / "fisher.pt"
        if not fpath.exists():
            continue
        f = torch.load(fpath, map_location="cpu", weights_only=False)
        keys = {k: i for i, k in enumerate(f["keys"])}
        cats = f["categories"]
        sq = f["sq_norms"]
        summary = json.loads((OUT / "t0" / point / "summary.json").read_text())
        P["t0_summary"] = {"counts": summary["counts"], "mean_loss": summary["mean_loss"]}
        wn = json.loads((OUT / "t1" / f"{point}_weight_norms.json").read_text()) \
            if (OUT / "t1" / f"{point}_weight_norms.json").exists() else None
        sf = torch.load(OUT / "t0" / point / "slot_fisher.pt", map_location="cpu",
                        weights_only=False)
        geom_path = OUT / "t0" / point / "category_geometry.json"
        if geom_path.exists():
            geom = json.loads(geom_path.read_text())
        else:
            geom = category_geometry(point, summary, f)
            geom_path.write_text(json.dumps(geom, indent=1))
        fisher = {}
        for loc in t1_locations():
            ks = [keys[fisher_key(s, p)] for s, p in members(loc)]
            per_row = sq[:, ks].sum(1)
            entry = {"trace": float(per_row.mean())}
            entry["cat"] = {c: float(per_row[[i for i, x in enumerate(cats) if x == c]].mean())
                            for c in A.CATEGORIES}
            sl = sf["sq_norms"][:, :, ks].sum(-1)  # [rows, 15]
            valid = (sf["valid"] > 0).double()
            entry["slot"] = [float((sl[:, s] * valid[:, s]).sum() / valid[:, s].sum().clamp_min(1))
                             for s in range(15)]
            numel = sum(geom["numel"][fisher_key(s, p)] for s, p in members(loc))
            entry["numel"] = numel
            if wn is not None:
                w2 = sum(wn[s] ** 2 for s, _ in members(loc))
                # Expected second-order loss change of a random direction of
                # relative norm 1: 0.5 * ||W||^2 * tr(F) / numel.
                entry["rand_curv"] = 0.5 * w2 * entry["trace"] / numel
            fisher[loc] = entry
        P["fisher"] = fisher
        P["geometry"] = geom["locations"]
        spath = OUT / "t0steps" / f"{point}.pt"
        if complete(spath):
            data, summ = load_sweep(spath, slots=True)
            ci = A.bootstrap_dtau(data)
            steps = {}
            cross = {}
            for name, e in summ.items():
                m = e["meta"]
                if m["source"] == "all":
                    d = steps.setdefault(m["location"], {"eps": [], "descent": {}, "ascent": {},
                                                          "grad_norm": m["grad_norm"],
                                                          "weight_norm": m["weight_norm"]})
                    d[m["kind"]][m["eps"]] = {**compact(e)["all"], "ci": ci[name],
                                              "cat": compact(e)["cat"]}
                    scatter.append({"point": point, "kind": m["kind"], "location": m["location"],
                                    "eps": m["eps"], "kl": e["all"]["kl12"],
                                    "d_tau": e["all"]["d_tau"], "flip": e["all"]["flip"],
                                    "fix": e["all"]["fix"], "brk": e["all"]["brk"],
                                    "d_klT": e["all"]["d_klT"], "ci": ci[name]})
                else:
                    cross.setdefault(m["location"], {})[m["source"]] = {
                        c: {"d_tau": e[c]["d_tau"], "d_klT": e[c]["d_klT"], "d_acc": e[c]["d_acc"]}
                        for c in A.CATEGORIES}
                    cross[m["location"]][m["source"]]["all"] = {
                        "d_tau": e["all"]["d_tau"], "d_klT": e["all"]["d_klT"]}
            for loc, d in steps.items():
                d["eps"] = sorted(d["descent"])
                if f"descent|{loc}|{CAT_EPS}|all" in summ:
                    e = summ[f"descent|{loc}|{CAT_EPS}|all"]
                    for cat in A.CATEGORIES:
                        cross.setdefault(loc, {}).setdefault("all", {})
                    if loc in cross:
                        cross[loc]["all"] = {c: {"d_tau": e[c]["d_tau"], "d_klT": e[c]["d_klT"],
                                                 "d_acc": e[c]["d_acc"]} for c in A.CATEGORIES}
                        cross[loc]["all"]["all"] = {"d_tau": e["all"]["d_tau"],
                                                    "d_klT": e["all"]["d_klT"]}
            P["steps"] = steps
            P["cross"] = {k: v for k, v in cross.items() if len(v) > 1}
            prof = {}
            for loc in ("all_attn", "all_ffn", "whole", "fc"):
                n = f"descent|{loc}|{CAT_EPS}|all"
                if n in summ:
                    prof[loc] = summ[n]["slot"]
            P["slot_descent"] = prof

    # ---------------- Free points and T2 (from base) ----------------
    free_path = OUT / "free" / "base.pt"
    if free_path.exists():
        data, summ = load_sweep(free_path, slots=True)
        ci = A.bootstrap_dtau(data)
        R["free"] = {n: {**compact(e, slots=True), "ci": ci[n]} for n, e in summ.items()}
        for n, e in summ.items():
            scatter.append({"point": "base", "kind": "free", "location": n, "kl": e["all"]["kl12"],
                            "d_tau": e["all"]["d_tau"], "flip": e["all"]["flip"],
                            "fix": e["all"]["fix"], "brk": e["all"]["brk"],
                            "d_klT": e["all"]["d_klT"], "ci": ci[n]})
    R["free_rel_norm"] = free_rel_norms()
    R["t2"] = {}
    for run in T2_RUNS:
        hist = []
        for u in T2_UPDATES:
            p = OUT / "t2" / run / f"eval_u{u}.pt"
            if not complete(p):
                continue
            data, summ = load_sweep(p, slots=True)
            ci = A.bootstrap_dtau(data)
            (name, e), = summ.items()
            hist.append({"update": u, **compact(e, slots=True), "ci": ci[name]})
            if u > 0:
                scatter.append({"point": "base", "kind": "train", "location": run, "update": u,
                                "kl": e["all"]["kl12"], "d_tau": e["all"]["d_tau"],
                                "flip": e["all"]["flip"], "fix": e["all"]["fix"],
                                "brk": e["all"]["brk"], "d_klT": e["all"]["d_klT"],
                                "ci": ci[name]})
        if hist:
            R["t2"][run] = {"history": hist}
            s = OUT / "t2" / run / "summary.json"
            if s.exists():
                js = json.loads(s.read_text())
                R["t2"][run].update(validation500=js["validation500"], wandb=js["wandb"],
                                    trainable=js["trainable_parameters"],
                                    rel_norm=[h.get("eval/rel_delta_norm") for h in js["history"]],
                                    seconds=js["seconds"])
    verify = json.loads((OUT / "verify" / "verify.json").read_text())
    R["verify"] = {k: verify[k] for k in ("scores", "recorded", "max_row_abs_diff_vs_recorded",
                                          "max_own_vs_upstream_count_diff", "rows")}
    R["correlations"] = correlations(R)
    lc = OUT / "results" / "loss_check.json"
    R["loss_check"] = json.loads(lc.read_text())["points"] if lc.exists() else {}
    R["scatter"] = scatter
    R["pareto"] = {pt: pareto([s for s in scatter if s["point"] == pt]) for pt in POINTS}
    R["pareto_by_kind"] = {
        pt: {k: pareto([s for s in scatter if s["point"] == pt and s["kind"] == k])
             for k in sorted({s["kind"] for s in scatter if s["point"] == pt})}
        for pt in POINTS}
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print("wrote", OUT / "results" / "results.json", len(scatter), "scatter points")
    if args.png_dir:
        import figures  # noqa: PLC0415

        figures.make_all(R, args.png_dir)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def correlations(R):
    """Spearman rank correlations across locations (per θ₁): Fisher-predicted
    random-direction curvature vs measured KL at ε=0.3; first-order gain
    ‖g‖‖W‖ vs the best one-step Δτ; random sensitivity vs best step Δτ."""
    out = {}
    for point, P in R["points"].items():
        if "dose" not in P or "fisher" not in P:
            continue
        locs = [loc for loc in P["dose"] if loc in P["fisher"]]
        kl = [P["dose"][loc]["kl12"][P["dose"][loc]["eps"].index(0.3)] for loc in locs]
        entry = {"locations": len(locs),
                 "fisher_curv_vs_kl_eps0.3": spearman([P["fisher"][loc]["rand_curv"] for loc in locs], kl)}
        steps = P.get("steps", {})
        sl = [loc for loc in locs if loc in steps]
        if len(sl) > 3:
            best = [max(v["d_tau"] for v in steps[loc]["descent"].values()) for loc in sl]
            entry["step_locations"] = len(sl)
            entry["first_order_vs_best_step_dtau"] = spearman(
                [steps[loc]["grad_norm"] * steps[loc]["weight_norm"] for loc in sl], best)
            entry["kl_eps0.3_vs_best_step_dtau"] = spearman(
                [P["dose"][loc]["kl12"][P["dose"][loc]["eps"].index(0.3)] for loc in sl], best)
        out[point] = entry
    return out


def free_rel_norms():
    """Relative size ‖ΔW‖/‖W‖ over q/k/v/o of all layers of each static adapter
    (ΔW = 2 B A; base weight norms from the T1 base sweep)."""
    import harness as H  # noqa: PLC0415

    wn_path = OUT / "t1" / "base_weight_norms.json"
    if not wn_path.exists():
        return {}
    wn = json.loads(wn_path.read_text())
    out = {}
    for name, run_dir in H.FREE_STATICS.items():
        ck = run_dir / "pretraining" / run_dir.name / "checkpoint.pt"
        state = torch.load(ck, map_location="cpu", weights_only=False)["model"]
        d2 = w2 = 0.0
        for i in range(LAYERS):
            for m in ("q", "k", "v", "o"):
                path = f"layers.{i}.self_attn.{m}_proj"
                a, b = state[f"{path}.a"].float(), state[f"{path}.b"].float()
                d2 += float((2.0 * b @ a).pow(2).sum())
                w2 += wn[f"l{i}_{m}"] ** 2
        out[name] = (d2 / w2) ** 0.5
    return out


def category_geometry(point, summary, fisher):
    """Cosines between per-category mean gradients per location, with the
    noise-only ceiling from the per-row Fisher (E|g_hat - g|^2 = (trF - |g|^2)/n)."""
    grads = torch.load(OUT / "t0" / point / "mean_grads_bf16.pt", map_location="cpu")
    cats = list(A.CATEGORIES)
    keys = {k: i for i, k in enumerate(fisher["keys"])}
    rows_cat = fisher["categories"]
    numel = {}
    for k, v in grads[cats[0]].items():
        numel[k] = v.numel()
    for i in range(LAYERS):
        for m in ("k", "v"):
            numel[f"l{i}_{m}"] = numel[f"l{i}_{m}@ctx"]
    out = {"numel": numel, "locations": {}}
    for loc in t1_locations():
        mem = members(loc)

        def vec(cat):
            parts = []
            for site, path in mem:
                if path == "both" and site.endswith(("_k", "_v")):
                    parts.append((grads[cat][f"{site}@ctx"].float()
                                  + grads[cat][f"{site}@blk"].float()).reshape(-1))
                else:
                    parts.append(grads[cat][fisher_key(site, path)].float().reshape(-1))
            return torch.cat(parts)

        vs = {c: vec(c) for c in cats}
        ks = [keys[fisher_key(s, p)] for s, p in mem]
        entry = {"norm": {}, "noise": {}, "cos": {}, "cos_ceiling": {}}
        for c in cats:
            idx = [i for i, x in enumerate(rows_cat) if x == c]
            tr = float(fisher["sq_norms"][idx][:, ks].sum(1).mean())
            g2 = float(vs[c].pow(2).sum())
            entry["norm"][c] = g2 ** 0.5
            entry["noise"][c] = max(tr - g2, 0.0) / len(idx)
        mean = sum(vs.values()) / len(cats)
        for i, c1 in enumerate(cats):
            for c2 in cats[i + 1:]:
                cos = float((vs[c1] @ vs[c2]) / (vs[c1].norm() * vs[c2].norm()))
                entry["cos"][f"{c1}|{c2}"] = cos
                # cos of two noisy estimates of the same direction at these norms
                n1, n2 = entry["norm"][c1] ** 2, entry["norm"][c2] ** 2
                e1, e2 = entry["noise"][c1], entry["noise"][c2]
                s1 = max(n1 - e1, 1e-30)
                s2 = max(n2 - e2, 1e-30)
                entry["cos_ceiling"][f"{c1}|{c2}"] = math.sqrt(s1 * s2 / (n1 * n2))
        entry["shared_share"] = {
            c: float((vs[c] @ mean) ** 2 / (vs[c].pow(2).sum() * mean.pow(2).sum()))
            for c in cats}
        out["locations"][loc] = entry
    return out


if __name__ == "__main__":
    main()
