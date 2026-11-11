"""Static PNG figures (F1-F9) of the PERTURB24K report from results.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODULES = ("q", "k", "v", "o", "gate", "up", "down")
CATS = ("math", "code", "stem", "chat")
GROUPS = ("fc", "all_attn", "all_ffn", "all", "whole")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6})


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=110)
    plt.close(fig)


def f1_dose(R, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for r, point in enumerate(("base", "static")):
        dose = R["points"][point].get("dose")
        if not dose:
            continue
        floor = R["points"][point]["floor"]
        for c, metric in enumerate(("kl12", "tv12", "flip")):
            ax = axes[r, c]
            for i, loc in enumerate(GROUPS + ("layer1", "layer5")):
                d = dose[loc]
                ax.plot(d["eps"], d[metric], color=C[i % 8], lw=1.6, marker="o", ms=3, label=loc)
                ax.fill_between(d["eps"], d[metric + "_lo"], d[metric + "_hi"], color=C[i % 8],
                                alpha=0.2, lw=0)
            for m in ("l2_q", "l2_down"):
                d = dose[m]
                ax.plot(d["eps"], d[metric], color="#898781", lw=1, ls="--", label=m)
            ax.axhline(floor[metric], color="#52514e", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{point}: {metric}")
            if r == 1:
                ax.set_xlabel("relative norm ε = ‖ΔW‖/‖W‖")
    axes[0, 0].legend(fontsize=7, ncol=2)
    _save(fig, out, "f1_dose_response.png")


def f2_heat(R, out, eps=0.3, metric="kl12"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, point in zip(axes, ("base", "static")):
        dose = R["points"][point].get("dose")
        if not dose:
            continue
        cols = list(MODULES) + ["kv@ctx", "kv@blk"]
        M = np.zeros((5, len(cols)))
        for i in range(5):
            for j, m in enumerate(cols):
                loc = f"l{i}_{m}" if "@" not in m else f"l{i}_{m}"
                d = dose[loc]
                M[i, j] = d[metric][d["eps"].index(eps)]
        im = ax.imshow(np.log10(M), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(5), [f"layer {i + 1}" for i in range(5)])
        for i in range(5):
            for j in range(len(cols)):
                ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", fontsize=7,
                        color="white" if np.log10(M[i, j]) > np.log10(M).mean() else "black")
        ax.set_title(f"{point}: KL(p1‖p2) at ε={eps}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="log10 KL")
    _save(fig, out, "f2_sensitivity_heatmap.png")


KIND_STYLE = {"dense": (C[0], "o", 10), "lowrank": (C[6], "s", 10), "descent": (C[2], "^", 18),
              "ascent": (C[7], "v", 14), "train": (C[1], "D", 30), "free": (C[3], "*", 90)}


def f3_scatter(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, point in zip(axes, ("base", "static")):
        pts = [s for s in R["scatter"] if s["point"] == point]
        for kind, (col, mk, size) in KIND_STYLE.items():
            ks = [s for s in pts if s["kind"] == kind]
            if ks:
                ax.scatter([s["kl"] for s in ks], [s["d_tau"] for s in ks], c=col, marker=mk,
                           s=size, alpha=0.75, label=f"{kind} ({len(ks)})", lw=0)
        front = R["pareto"][point]
        ax.plot([p["kl"] for p in front], [p["d_tau"] for p in front], color="#0b0b0b", lw=1,
                drawstyle="steps-post", label="Pareto front")
        ax.set_xscale("log")
        ax.set_ylim(-0.6, 0.35)
        ax.axhline(0, color="#898781", lw=0.8)
        ax.set_xlabel("movement KL(p₁‖p₂) per position (nats)")
        ax.set_ylabel("Δτ vs θ₁")
        ax.set_title(f"θ₁ = {point}")
        ax.legend(fontsize=7, loc="lower left")
    _save(fig, out, "f3_movement_vs_gain.png")


def f4_churn(R, out, items):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    labels = [i[0] for i in items]
    fix = [i[1]["fix"] * 100 for i in items]
    brk = [i[1]["brk"] * 100 for i in items]
    x = np.arange(len(items))
    ax.bar(x - 0.2, fix, 0.38, color=C[2], label="fixes (wrong→right)")
    ax.bar(x + 0.2, brk, 0.38, color=C[7], label="breaks (right→wrong)")
    for xi, f, b in zip(x, fix, brk):
        ax.text(xi, max(f, b) + 0.1, f"net {f - b:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("% of draft positions")
    ax.legend(fontsize=8)
    _save(fig, out, "f4_churn.png")


def f5_slots(R, out, profiles):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    s = np.arange(1, 16)
    for i, (label, prof) in enumerate(profiles):
        axes[0].plot(s, prof["kl12"], color=C[i % 8], label=label)
        axes[1].plot(s, prof["flip"], color=C[i % 8])
        axes[2].plot(s, np.array(prof["d_acc"]) * 100, color=C[i % 8])
    axes[0].set_yscale("log")
    axes[0].set_title("movement KL per slot")
    axes[1].set_title("top-1 flip rate per slot")
    axes[2].set_title("Δ accuracy per slot (pp)")
    axes[2].axhline(0, color="#898781", lw=0.8)
    for a in axes:
        a.set_xlabel("draft slot")
    axes[0].legend(fontsize=7)
    _save(fig, out, "f5_slot_profiles.png")


def f6_rank(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, point in zip(axes, ("base", "static")):
        table = R["points"][point].get("rank")
        if not table:
            continue
        for i, (key, row) in enumerate(sorted(table.items())):
            ranks = [r for r in row if r != "full"]
            xs = [int(r) for r in ranks] + [4096]
            ys = [row[r]["kl12"] for r in ranks] + [row["full"]["kl12"]]
            ax.plot(xs, ys, marker="o", ms=3, color=C[i % 8], label=key)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("rank of ΔW (4096 = dense)")
        ax.set_ylabel("KL(p1‖p2)")
        ax.set_title(f"θ₁ = {point}: same ‖ΔW‖, varying rank")
        ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "f6_rank.png")


def f7_paths(R, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (run, d) in enumerate(R["t2"].items()):
        h = d["history"]
        ax.plot([max(x["all"]["kl12"], 1e-5) for x in h], [x["all"]["d_tau"] for x in h],
                marker="o", color=C[i], label=run)
        for x in h:
            ax.annotate(str(x["update"]), (max(x["all"]["kl12"], 1e-5), x["all"]["d_tau"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    for name, e in R.get("free", {}).items():
        ax.scatter(e["all"]["kl12"], e["all"]["d_tau"], marker="*", s=80, color=C[3])
        ax.annotate(name, (e["all"]["kl12"], e["all"]["d_tau"]), fontsize=6, xytext=(3, -8),
                    textcoords="offset points")
    front = R["pareto_by_kind"]["base"].get("descent", [])
    ax.plot([p["kl"] for p in front], [p["d_tau"] for p in front], color="#898781", ls="--",
            lw=1, drawstyle="steps-post", label="one gradient step (front)")
    ax.set_xscale("log")
    ax.set_xlabel("KL(p_base‖p) per position")
    ax.set_ylabel("Δτ vs base")
    ax.legend(fontsize=7)
    _save(fig, out, "f7_training_paths.png")


def f8_category(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, point in zip(axes, ("base", "static")):
        cross = R["points"][point].get("cross", {}).get("all_attn")
        if not cross:
            continue
        M = np.array([[cross[src][tgt]["d_klT"] for tgt in CATS] for src in CATS + ("all",)])
        im = ax.imshow(-M * 1000, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4), CATS)
        ax.set_yticks(range(5), [f"step on {c}" for c in CATS + ("all",)])
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{-M[i, j] * 1000:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"{point}: KL-to-target reduction ×1e3 (all_attn, ε=3e-4)")
        ax.grid(False)
        fig.colorbar(im, ax=ax)
    _save(fig, out, "f8_category_cross_gain.png")


def make_all(R, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f1_dose(R, out)
    f2_heat(R, out)
    f3_scatter(R, out)
    f6_rank(R, out)
    f7_paths(R, out)
    f8_category(R, out)
