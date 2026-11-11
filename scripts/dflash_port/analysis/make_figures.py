"""Figures (PNG, for reporting/dflash/analysis24k/) and the compact data file
used by the HTML artifact, from analysis24k/results/*.json and the raw
grad / attention outputs.

    python scripts/dflash_port/analysis/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FIG = C.REPO / "reporting/dflash/analysis24k"
RES = C.OUT / "results"
MODULES = ["q", "k_ctx", "k_blk", "v_ctx", "v_blk", "o", "gate", "up", "down"]
PAIRS = ["math|code", "math|stem", "math|chat", "code|stem", "code|chat", "stem|chat"]
COLORS = {"base": "#8a8a85", "static": "#2a78d6", "generator": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})


def load(name):
    path = RES / name
    return json.loads(path.read_text()) if path.is_file() else None


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


def errors_figs(data, out):
    err = load("errors_test.json")
    if err is None:
        return
    base = err["arms"]["base"]
    classes = ["miss: target >=0.9, rank 1-5", "miss: target 0.5-0.9, rank 1-5",
               "miss: target <0.5, rank 1-5", "miss: rank >5", "full block (16)",
               "truncated"]
    rows = {}
    for scope in ["all"] + list(C.CATEGORIES):
        s = base["all_blocks"] if scope == "all" else base["categories"][scope]
        g = s["grid"]
        miss = s["share_miss"]
        rows[scope] = [
            miss * g[">=0.9|1-5"]["share"], miss * g["0.5-0.9|1-5"]["share"],
            miss * g["<0.5|1-5"]["share"],
            miss * (g[">=0.9|>5"]["share"] + g["0.5-0.9|>5"]["share"] + g["<0.5|>5"]["share"]),
            s["share_full"], s["share_truncated"],
        ]
    out["block_outcomes"] = {"classes": classes, "rows": rows}
    fig, ax = plt.subplots(figsize=(7, 2.8))
    palette = ["#2a78d6", "#7fb0ea", "#c9dcf3", "#eb6834", "#1baf7a", "#cccccc"]
    names = list(rows)
    left = np.zeros(len(names))
    for i, cls in enumerate(classes):
        vals = np.array([rows[n][i] for n in names])
        ax.barh(names, vals, left=left, color=palette[i], label=cls, edgecolor="white")
        left += vals
    ax.invert_yaxis()
    ax.set_xlabel("share of blocks (base drafter, test)")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    save(fig, "block_outcomes.png")
    # Static / generator vs base: Δτ contribution by the base block class.
    contrib = {}
    for arm in ("static", "generator"):
        t = err["transitions"][f"base->{arm}"]["all"]
        contrib[arm] = {k: t[k]["delta_tau_contribution"] for k in t
                        if isinstance(t[k], dict)}
        contrib[arm]["total"] = t["delta_tau"]
        contrib[arm]["blocks_improved"] = t["blocks_improved"]
        contrib[arm]["blocks_worse"] = t["blocks_worse"]
    out["transition_contrib"] = contrib
    keys = ["full", "miss|>=0.9|1-5", "miss|0.5-0.9|1-5", "miss|<0.5|1-5",
            "miss|>=0.9|>5", "miss|0.5-0.9|>5", "miss|<0.5|>5"]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    x = np.arange(len(keys))
    for j, arm in enumerate(("static", "generator")):
        ax.bar(x + (j - 0.5) * 0.38, [contrib[arm].get(k, 0) for k in keys], 0.36,
               color=COLORS[arm], label=arm)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x, [k.replace("miss|", "miss ") for k in keys], rotation=20, fontsize=7)
    ax.set_ylabel("Δτ vs base (test)")
    ax.legend(fontsize=7)
    save(fig, "transition_contrib.png")
    # First-miss slot distribution and on-path slot accuracy.
    out["first_miss_slot"] = {c: base["categories"][c]["first_miss_slot"]
                              for c in C.CATEGORIES}
    out["slot_acc_onpath"] = {arm: err["arms"][arm]["slot_level"]["by_slot_onpath"]
                              for arm in ("base", "static", "generator")}
    out["error_summary"] = {
        arm: {k: err["arms"][arm]["all_blocks"][k] for k in (
            "tau", "share_full", "share_truncated", "share_miss",
            "fix_all_first_misses_tau_gain", "onpath_share", "fix_gain_onpath",
            "fix_gain_offpath", "rank", "confidence", "addressable", "strict",
            "target_uncertain", "offpath", "median", "drafter_pick_is_target_second",
            "tie")}
        for arm in ("base", "static", "generator")}
    out["error_categories"] = {
        c: {k: base["categories"][c][k] for k in (
            "tau", "share_miss", "addressable", "strict", "target_uncertain",
            "offpath", "fix_all_first_misses_tau_gain")}
        for c in C.CATEGORIES}
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for c in C.CATEGORIES:
        d = base["categories"][c]["first_miss_slot"]
        ax.plot(range(1, 16), [d[str(s)] for s in range(1, 16)], marker="o", ms=3, label=c)
    ax.set_xlabel("slot of the first miss")
    ax.set_ylabel("share of misses")
    ax.legend(fontsize=7)
    save(fig, "first_miss_slot.png")


def module_name(layer, m):
    return f"l{layer}_{m}"


def grad_figs(out):
    for point in ("base", "static"):
        path = C.grads_dir(point) / "summary.json"
        if not path.is_file():
            continue
        s = json.loads(path.read_text())["modules"]
        def grid(fn):
            return np.array([[fn(s[module_name(l, m)]) for m in MODULES] for l in range(5)])
        cats = C.CATEGORIES
        gain = grid(lambda e: np.mean([e["categories"][c]["top32_gain"] for c in cats]))
        share = grid(lambda e: np.mean([e["categories"][c]["top32_share"] for c in cats]))
        capture = grid(lambda e: np.mean([e["categories"][c]["shared_capture"] for c in cats]))
        reliab = grid(lambda e: np.mean([e["categories"][c]["split_half_cos"] for c in cats]))
        cross = grid(lambda e: np.mean([e["cos_halves"][p] for p in PAIRS]))
        corrected = grid(lambda e: np.mean([
            e["cos_halves"][p] / np.sqrt(max(e["categories"][p.split("|")[0]]["split_half_cos"], 1e-6)
                                         * max(e["categories"][p.split("|")[1]]["split_half_cos"], 1e-6))
            for p in PAIRS]))
        has_heldout = "heldout_capture" in s["l0_q"]["categories"]["math"]
        heldout = grid(lambda e: np.nanmean([np.nan if e["categories"][c]["heldout_capture"] is None else e["categories"][c]["heldout_capture"] for c in cats])) \
            if has_heldout else None
        rel_gain = grid(lambda e: np.mean([e["categories"][c]["top32_gain"] for c in cats])
                        / e["weight_fro"])
        fc = s["fc"]
        entry = {
            "modules": MODULES, "gain": gain.tolist(), "share": share.tolist(),
            "capture": capture.tolist(), "split_half": reliab.tolist(),
            "cross_cos": cross.tolist(), "corrected_cos": corrected.tolist(),
            "rel_gain": rel_gain.tolist(),
            "heldout_capture": heldout.tolist() if heldout is not None else None,
            "heldout_capture_per_category": {
                c: grid(lambda e, c=c: np.nan if e["categories"][c]["heldout_capture"] is None else e["categories"][c]["heldout_capture"]).tolist()
                for c in cats} if has_heldout else None,
            "fc": {"gain": float(np.mean([fc["categories"][c]["top32_gain"] for c in cats])),
                   "share": float(np.mean([fc["categories"][c]["top32_share"] for c in cats])),
                   "capture": float(np.mean([fc["categories"][c]["shared_capture"] for c in cats])),
                   "split_half": float(np.mean([fc["categories"][c]["split_half_cos"] for c in cats])),
                   "cross_cos": float(np.mean([fc["cos_halves"][p] for p in PAIRS])),
                   "heldout_capture": float(np.nanmean([np.nan if fc["categories"][c]["heldout_capture"] is None else fc["categories"][c]["heldout_capture"] for c in cats]))
                   if has_heldout else None},
            "per_category_gain": {c: grid(lambda e, c=c: e["categories"][c]["top32_gain"]).tolist()
                                  for c in cats},
            "per_category_capture": {c: grid(lambda e, c=c: e["categories"][c]["shared_capture"]).tolist()
                                     for c in cats},
            "pair_cos": {p: grid(lambda e, p=p: e["cos_halves"][p]).tolist() for p in PAIRS},
            "mean_loss": json.loads(path.read_text())["mean_loss"],
        }
        if point == "base":
            entry["static_delta_cos"] = [[s[module_name(l, m)].get("static_delta_cos_neg_grad")
                                          for m in MODULES] for l in range(5)]
        out[f"grads_{point}"] = entry
        fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
        for ax, mat, title, cmap in (
            (axes[0], gain, "rank-32 first-order gain ‖P32 G‖ (mean over categories)", "Blues"),
            (axes[1], heldout if heldout is not None else capture,
             "held-out share a shared rank-32 update captures" if heldout is not None
             else "share a shared rank-32 update captures (in-sample)", "Greens"),
            (axes[2], corrected, "cross-category cosine / split-half (noise-corrected)", "RdBu"),
        ):
            kw = {"vmin": -1, "vmax": 1} if cmap == "RdBu" else {}
            im = ax.imshow(mat, cmap=cmap, aspect="auto", **kw)
            ax.set_xticks(range(len(MODULES)), MODULES, rotation=45, fontsize=7)
            ax.set_yticks(range(5), [f"L{i}" for i in range(5)])
            ax.set_title(title, fontsize=8)
            for i in range(5):
                for j in range(len(MODULES)):
                    ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.04)
        save(fig, f"grad_maps_{point}.png")


def knockout_figs(out):
    ko = load("knockouts.json")
    if ko is not None:
        heads = {}
        for arm in ("base", "static"):
            if arm not in ko:
                continue
            v = ko[arm]["variants"]
            n_heads = max(int(k.split(":")[2]) for k in v if "|head:" in k) + 1
            heads[arm] = [[v[f"{arm}|head:{l}:{h}"]["delta"] for h in range(n_heads)]
                          for l in range(5)]
        out["head_knockouts"] = heads
        mods = {}
        for arm in ko:
            v = ko[arm]["variants"]
            mods[arm] = {k.split("|")[1]: {"delta": e["delta"], "ci": e.get("ci")}
                         for k, e in v.items() if "|head:" not in k}
            mods[arm]["_tau"] = ko[arm]["tau"]
        out["module_knockouts"] = mods
        fig, axes = plt.subplots(2, 1, figsize=(9, 3.6))
        for ax, arm in zip(axes, ("base", "static"), strict=True):
            if arm not in heads:
                continue
            im = ax.imshow(np.array(heads[arm]), cmap="Reds_r", aspect="auto", vmax=0)
            ax.set_yticks(range(5), [f"L{i}" for i in range(5)])
            ax.set_title(f"Δτ when one head is zeroed ({arm})", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.02)
        axes[1].set_xlabel("head")
        save(fig, "head_knockouts.png")
        fig, ax = plt.subplots(figsize=(8, 2.8))
        kinds = ["attn", "mlp", "ctx", "blk"]
        x = np.arange(5)
        for j, kind in enumerate(kinds):
            ax.bar(x + (j - 1.5) * 0.2, [mods["base"][f"{kind}:{l}"]["delta"] for l in range(5)],
                   0.19, label=kind)
        ax.set_xticks(x, [f"L{i}" for i in range(5)])
        ax.set_ylabel("Δτ (base)")
        ax.legend(fontsize=7, ncol=4)
        save(fig, "module_knockouts.png")
    sens = load("sensitivity.json")
    if sens is not None:
        names = ["fc"] + [f"l{l}_{m}" for l in range(5)
                          for m in ("q", "k", "v", "o", "gate", "up", "down")]
        rand1 = [sens["random_mean"][f"{n}:x1"]["mean"] for n in names]
        rand4 = [sens["random_mean"][f"{n}:x4"]["mean"] for n in names]
        trans = [sens["variants"].get(f"base|transplant:{n}", {}).get("delta") for n in names]
        out["sensitivity"] = {"names": names, "random_x1": rand1, "random_x4": rand4,
                              "transplant": trans, "norms": sens["norms"],
                              "mean_rel": sens["mean_rel"],
                              "transplant_ci": [sens["variants"].get(f"base|transplant:{n}", {}).get("ci")
                                                for n in names]}
        fig, ax = plt.subplots(figsize=(10, 2.8))
        x = np.arange(len(names))
        ax.bar(x - 0.27, rand1, 0.26, label="random rank-32, static norm (mean of 3)", color="#8a8a85")
        ax.bar(x, rand4, 0.26, label="random, 4x norm", color="#555")
        ax.bar(x + 0.27, [t if t is not None else 0 for t in trans], 0.26,
               label="trained static ΔW of this module alone", color="#2a78d6")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x, names, rotation=90, fontsize=6)
        ax.set_ylabel("Δτ (base, 200 rows)")
        ax.legend(fontsize=7)
        save(fig, "sensitivity.png")


def prompt_figs(out):
    mk = load("masking.json")
    if mk is not None:
        out["masking"] = mk
        fig, ax = plt.subplots(figsize=(6, 2.6))
        conds = ["prompt", "resp_random", "resp_early"]
        x = np.arange(len(conds))
        for j, arm in enumerate(("base", "static", "generator")):
            vals = [mk["arms"][arm][c]["delta"] for c in conds]
            err = np.array([[mk["arms"][arm][c]["delta"] - mk["arms"][arm][c]["ci"][0],
                             mk["arms"][arm][c]["ci"][1] - mk["arms"][arm][c]["delta"]]
                            for c in conds]).T
            ax.bar(x + (j - 1) * 0.27, vals, 0.25, yerr=err, color=COLORS[arm], label=arm)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(x, ["prompt removed", "as many random\nresponse tokens", "as many earliest\nresponse tokens"])
        ax.set_ylabel("Δτ vs full context")
        ax.legend(fontsize=7)
        save(fig, "masking.png")
    path = C.OUT / "prompt" / "attention.json"
    if path.is_file():
        att = json.loads(path.read_text())
        res = {}
        for arm, by_cat in att["results"].items():
            sums = sum(np.array(v["sums"]) for v in by_cat.values())  # L, H, F
            q = sum(np.array(v["queries"]) for v in by_cat.values())
            uni = sum(np.array(v["uniform_prompt_sum"]) for v in by_cat.values())
            frac = sums / q[:, None, None]
            res[arm] = {
                "fields": next(iter(by_cat.values()))["fields"],
                "per_head": frac.tolist(),
                "per_layer": frac.mean(1).tolist(),
                "uniform_prompt": (uni / q).tolist(),
                "per_category_layer": {c: (np.array(v["sums"]) / np.array(v["queries"])[:, None, None]).mean(1).tolist()
                                       for c, v in by_cat.items()},
            }
        out["attention"] = res
        out["attention_equivalence"] = att["equivalence"]
        base = np.array(res["base"]["per_head"])  # L, H, F
        fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
        im = axes[0].imshow(base[..., 0] - base[..., 5], aspect="auto", cmap="Purples")
        axes[0].set_title("base: attention mass on prompt tokens (excl. first 4), per head", fontsize=8)
        axes[0].set_yticks(range(5), [f"L{i}" for i in range(5)])
        fig.colorbar(im, ax=axes[0], fraction=0.03)
        layers = np.arange(5)
        for k, (idx, name) in enumerate(((0, "prompt"), (5, "first 4 tokens"), (1, "response ctx"),
                                        (2, "anchor"), (3, "block masks"))):
            axes[1].plot(layers, base.mean(1)[:, idx], marker="o", label=name)
        axes[1].plot(layers, res["base"]["uniform_prompt"], ls="--", color="#999",
                     label="prompt share if uniform")
        axes[1].set_xticks(layers, [f"L{i}" for i in layers])
        axes[1].set_ylabel("mean attention mass")
        axes[1].legend(fontsize=6)
        save(fig, "attention.png")


def repr_figs(out):
    rp = load("repr_validation.json")
    if rp is None:
        return
    out["repr"] = rp
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
    layers = range(5)
    for key, label in (("base|static", "static vs base"), ("base|generator", "generator vs base"),
                       ("generator|generator_cross", "generator: own vs other-category prompt"),
                       ("generator|generator_same", "generator: own vs same-category prompt")):
        axes[0].plot(layers, [rp["rel_change"][f"{key}|L{l}"] for l in layers], marker="o", label=label)
        axes[1].plot(layers, [1 - rp["cka"][f"{key}|L{l}"] for l in layers], marker="o", label=label)
    axes[0].set_ylabel("‖X_a − X_b‖ / ‖X_a‖")
    axes[1].set_ylabel("1 − linear CKA")
    axes[0].legend(fontsize=6)
    for arm in ("base", "static", "generator"):
        axes[2].plot(layers, [rp["probes"][f"{arm}|L{l}"]["accept_auc"] for l in layers],
                     marker="o", color=COLORS[arm], label=f"{arm}: probe on state")
        axes[2].axhline(rp["probes"][f"{arm}|accept_baselines"]["drafter_p_top_auc"],
                        color=COLORS[arm], ls="--", lw=0.8)
    axes[2].set_ylabel("accept-probe AUC (dashed: drafter top-1 prob)")
    axes[2].legend(fontsize=6)
    for ax in axes:
        ax.set_xticks(list(layers), [f"L{i}" for i in layers])
    save(fig, "representations.png")


def main():
    out = {}
    errors_figs(None, out)
    grad_figs(out)
    knockout_figs(out)
    prompt_figs(out)
    repr_figs(out)
    C.write_json(C.OUT / "results" / "artifact_data.json", out)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
