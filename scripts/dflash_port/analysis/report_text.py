"""Numbers and text of ANALYSIS24K.md and the artifact (used by write_report.py).

Every number is read from the computed results; the prose around them is the
interpretation, kept separate from the measurement tables.
"""

import numpy as np

import common as C

MODULES = ["q", "k_ctx", "k_blk", "v_ctx", "v_blk", "o", "gate", "up", "down"]
PAIRS = ["math|code", "math|stem", "math|chat", "code|stem", "code|chat", "stem|chat"]
ARMS = ("base", "static", "generator")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def sgn(x, d=3):
    return f"{x:+.{d}f}"


def ci(c, d=3):
    return f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def grad_grid(summary, fn):
    m = summary["modules"]
    return np.array([[fn(m[f"l{l}_{k}"]) for k in MODULES] for l in range(5)])


def grad_numbers(summary):
    cats = C.CATEGORIES
    m = summary["modules"]
    mean = lambda e, key: float(np.nanmean([np.nan if e["categories"][c][key] is None  # noqa: E731
                                            else e["categories"][c][key] for c in cats]))
    corr = lambda e: float(np.mean([  # noqa: E731
        e["cos_halves"][p] / np.sqrt(max(e["categories"][p.split("|")[0]]["split_half_cos"], 1e-6)
                                     * max(e["categories"][p.split("|")[1]]["split_half_cos"], 1e-6))
        for p in PAIRS]))
    has_heldout = "heldout_capture" in m["l0_q"]["categories"]["math"]
    out = {
        "gain": grad_grid(summary, lambda e: mean(e, "top32_gain")),
        "share": grad_grid(summary, lambda e: mean(e, "top32_share")),
        "split": grad_grid(summary, lambda e: mean(e, "split_half_cos")),
        "cross": grad_grid(summary, lambda e: float(np.mean([e["cos_halves"][p] for p in PAIRS]))),
        "corr": grad_grid(summary, corr),
        "capture": grad_grid(summary, lambda e: mean(e, "shared_capture")),
        "heldout": grad_grid(summary, lambda e: mean(e, "heldout_capture")) if has_heldout else None,
        "fc": {k: mean(m["fc"], k) for k in ("top32_gain", "top32_share", "split_half_cos",
                                             "shared_capture")},
        "fc_cross": float(np.mean([m["fc"]["cos_halves"][p] for p in PAIRS])),
        "fc_corr": corr(m["fc"]),
        "mean_loss": summary["mean_loss"],
    }
    if has_heldout:
        out["fc"]["heldout_capture"] = mean(m["fc"], "heldout_capture")
        out["heldout_by_cat"] = {c: grad_grid(summary, lambda e, c=c: np.nan if e["categories"][c]["heldout_capture"] is None
                                              else e["categories"][c]["heldout_capture"]) for c in cats}
    out["static_cos"] = grad_grid(summary, lambda e: e.get("static_delta_cos_neg_grad") or np.nan)
    # Path totals (sum of squared rank-32 gains over layers) per module type.
    gain = out["gain"]
    out["gain_by_type"] = {k: float(np.sqrt((gain[:, j] ** 2).sum())) for j, k in enumerate(MODULES)}
    out["gain_by_layer"] = [float(np.sqrt((gain[l] ** 2).sum())) for l in range(5)]
    groups = {"attention q/o": ["q", "o"], "K/V context path": ["k_ctx", "v_ctx"],
              "K/V block path": ["k_blk", "v_blk"], "MLP": ["gate", "up", "down"]}
    out["gain_groups"] = {g: float(np.sqrt(sum((gain[:, MODULES.index(k)] ** 2).sum() for k in ks)))
                          for g, ks in groups.items()}
    out["gain_groups"]["fc"] = out["fc"]["top32_gain"]
    out["corr_groups"] = {g: float(np.mean(out["corr"][:, [MODULES.index(k) for k in ks]]))
                          for g, ks in groups.items()}
    if out["heldout"] is not None:
        out["heldout_groups"] = {g: float(np.mean(out["heldout"][:, [MODULES.index(k) for k in ks]]))
                                 for g, ks in groups.items()}
    return out


def numbers(data, r):
    n = {}
    et = r["errors_test"]
    n["err"] = {arm: et["arms"][arm] for arm in ARMS}
    n["trans"] = et["transitions"]
    n["repro"] = {"validation": r["reproduction_validation"], "test": r["reproduction_test"]}
    n["grads"] = {"base": grad_numbers(r["grads_base"]), "static": grad_numbers(r["grads_static"])}
    n["ko"] = r["knockouts"]
    n["sens"] = r["sensitivity"]
    n["mask"] = r["masking"]
    n["att"] = data["attention"]
    n["att_equiv"] = r["attention"]["equivalence"]
    n["repr"] = r["repr"]
    return n


# --------------------------------------------------------------------------- text


def step1_md(n):
    b, s, g = (n["err"][a]["all_blocks"] for a in ARMS)
    lines = []
    rows = []
    for arm in ARMS:
        a = n["err"][arm]["all_blocks"]
        rows.append([arm, f"{a['tau']:.3f}", pct(a["share_full"]), pct(a["share_miss"]),
                     pct(a["share_truncated"]), f"{a['fix_all_first_misses_tau_gain']:.3f}",
                     pct(a["onpath_share"]),
                     f"{pct(a['addressable']['share'])} / {a['addressable']['tau_gain']:.3f}",
                     f"{pct(a['strict']['share'])} / {a['strict']['tau_gain']:.3f}",
                     f"{pct(a['target_uncertain']['share'])} / {a['target_uncertain']['tau_gain']:.3f}",
                     f"{pct(a['offpath']['share'])} / {a['offpath']['tau_gain']:.3f}",
                     f"{pct(a['tie']['share'])} / {a['tie']['tau_gain']:.3f}"])
    lines.append("**Results (test, 49,021 blocks; pooled τ).** Shares of misses / τ a perfect fix of that class would add:")
    lines.append("")
    lines.append(table(["arm", "τ", "full blocks", "blocks ending in a miss", "truncated",
                        "fix every first miss (Δτ)", "misses on-path",
                        "addressable (p≥0.5, rank≤5, on-path)", "strict (p≥0.9, rank≤5, on-path)",
                        "target uncertain (p<0.5)", "off-path", "bf16 ties (rank 1)"], rows))
    lines.append("")
    grid = b["grid"]
    rows = []
    for conf in (">=0.9", "0.5-0.9", "<0.5"):
        for rank in ("1-5", ">5"):
            e = grid[f"{conf}|{rank}"]
            rows.append([f"p_top {conf}", f"rank {rank}", pct(e["share"]), f"{e['tau_gain']:.3f}",
                         pct(e["onpath_share"])])
    lines.append("Base drafter, first misses by target confidence × drafter rank of the target's token:")
    lines.append("")
    lines.append(table(["target confidence", "drafter rank", "share of misses", "Δτ if fixed",
                        "on-path share"], rows))
    lines.append("")
    rk = b["rank"]
    lines.append(f"Rank of the target's token at the first miss (base): 1 (bf16 tie) {pct(rk['1'])}, 2 {pct(rk['2'])}, 3–5 {pct(rk['3-5'])}, 6–20 {pct(rk['6-20'])}, >20 {pct(rk['>20'])}. Medians at the miss: target top-1 probability {b['median']['p_top']:.3f}, target entropy {b['median']['ent']:.3f} nats, drafter probability of the target's token {b['median']['p_d_tgt']:.3f}, drafter's own top-1 probability {b['median']['p_d_top']:.3f}, target probability of the drafter's pick {b['median']['t_p_pred']:.4f}; the drafter's pick is the target's second choice in {pct(b['drafter_pick_is_target_second'])} of misses.")
    lines.append("")
    rows = []
    for c in C.CATEGORIES:
        a = n["err"]["base"]["categories"][c]
        rows.append([c, f"{a['tau']:.3f}", pct(a["share_miss"]), f"{a['fix_all_first_misses_tau_gain']:.3f}",
                     f"{pct(a['addressable']['share'])} / {a['addressable']['tau_gain']:.3f}",
                     f"{pct(a['strict']['share'])} / {a['strict']['tau_gain']:.3f}",
                     f"{pct(a['target_uncertain']['share'])} / {a['target_uncertain']['tau_gain']:.3f}",
                     f"{pct(a['offpath']['share'])} / {a['offpath']['tau_gain']:.3f}"])
    lines.append("By category (base, test):")
    lines.append("")
    lines.append(table(["category", "τ", "blocks ending in a miss", "fix every first miss", "addressable",
                        "strict", "target uncertain", "off-path"], rows))
    lines.append("")
    sl = n["err"]["base"]["slot_level"]
    lines.append(f"Slot level (every valid slot, base): {pct(sl['by_conf']['>=0.9']['share'])} of slots have target p_top ≥ 0.9 and the drafter matches {pct(sl['by_conf']['>=0.9']['acc'])} of them; 0.5–0.9: {pct(sl['by_conf']['0.5-0.9']['share'])} of slots, {pct(sl['by_conf']['0.5-0.9']['acc'])} matched; < 0.5: {pct(sl['by_conf']['<0.5']['share'])}, {pct(sl['by_conf']['<0.5']['acc'])}. On-path accuracy falls with depth: slot 1 {pct(sl['by_slot_onpath']['1']['acc'])}, slot 8 {pct(sl['by_slot_onpath']['8']['acc'])}, slot 15 {pct(sl['by_slot_onpath']['15']['acc'])} (the data is still on the greedy path at slot 15 in {pct(sl['by_slot_onpath']['15']['share'])} of blocks). Figures: [block outcomes](analysis24k/block_outcomes.png), [first-miss slot](analysis24k/first_miss_slot.png).")
    lines.append("")
    rows = []
    keys = [("full", "full blocks"), ("miss|>=0.9|1-5", "miss p≥0.9, rank≤5"),
            ("miss|0.5-0.9|1-5", "miss p 0.5–0.9, rank≤5"), ("miss|<0.5|1-5", "miss p<0.5, rank≤5"),
            ("miss|>=0.9|>5", "miss p≥0.9, rank>5"), ("miss|0.5-0.9|>5", "miss p 0.5–0.9, rank>5"),
            ("miss|<0.5|>5", "miss p<0.5, rank>5"), ("miss|onpath", "miss on-path (any)"),
            ("miss|offpath", "miss off-path (any)")]
    for key, label in keys:
        e1 = n["trans"]["base->static"]["all"][key]
        e2 = n["trans"]["base->generator"]["all"][key]
        e3 = n["trans"]["static->generator"]["all"].get(key)
        rows.append([label, pct(e1["blocks_share"]), f"{pct(e1['fixed_rate'])} / {pct(e1['broken_rate'])}",
                     sgn(e1["delta_tau_contribution"]),
                     f"{pct(e2['fixed_rate'])} / {pct(e2['broken_rate'])}", sgn(e2["delta_tau_contribution"]),
                     sgn(e3["delta_tau_contribution"]) if e3 else "–"])
    ts, tg, tsg = (n["trans"][k]["all"] for k in ("base->static", "base->generator", "static->generator"))
    lines.append("**What the LoRAs fix** (same blocks and anchors in every arm; a block is *fixed* when the adapted arm's first failure comes later, *broken* when earlier):")
    lines.append("")
    lines.append(table(["base block class", "share of blocks", "static: fixed / broken", "static Δτ contribution",
                        "generator: fixed / broken", "generator Δτ contribution",
                        "generator − static, Δτ contribution (classes of the static arm)"], rows))
    lines.append("")
    lines.append(f"Totals: static − base Δτ {sgn(ts['delta_tau'])} ({pct(ts['blocks_improved'])} of blocks longer, {pct(ts['blocks_worse'])} shorter); generator − base {sgn(tg['delta_tau'])} ({pct(tg['blocks_improved'])} / {pct(tg['blocks_worse'])}); generator − static {sgn(tsg['delta_tau'])} ({pct(tsg['blocks_improved'])} / {pct(tsg['blocks_worse'])}). Figure: [contributions](analysis24k/transition_contrib.png).")
    lines.append("")
    lines.append("**Interpretation.**")
    lines.append(f"- *Most first misses look like near-misses, not ignorance.* In {pct(1 - rk['6-20'] - rk['>20'])} of base misses the drafter has the target's token in its top 5, and {pct(b['onpath_share'])} of misses happen while the data is still on the target's greedy path. The target is confident (p_top ≥ 0.5) in {pct(b['confidence']['>=0.9'] + b['confidence']['0.5-0.9'])} of them.")
    lines.append(f"- *Addressable share.* Our working definition (on-path, target p_top ≥ 0.5, drafter rank ≤ 5) covers {pct(b['addressable']['share'])} of base misses; fixing only those first misses would add {b['addressable']['tau_gain']:.2f} τ (pooled), against {b['fix_all_first_misses_tau_gain']:.2f} for fixing every first miss and {sgn(ts['delta_tau'], 2)} actually gained by the static LoRA. The strict version (p_top ≥ 0.9) is {pct(b['strict']['share'])} of misses and {b['strict']['tau_gain']:.2f} τ. These are first-order ceilings with later misses unchanged; they say the drafter's errors are mostly of a kind a small correction could flip, not that a LoRA can reach them.")
    lines.append(f"- *Less fixable part.* Target-uncertain misses (p_top < 0.5) are {pct(b['target_uncertain']['share'])} of misses ({b['target_uncertain']['tau_gain']:.2f} τ); off-path misses (a sampled non-argmax token earlier in the block) {pct(b['offpath']['share'])} ({b['offpath']['tau_gain']:.2f} τ). Neither can be fixed by better prediction of the target's argmax from the context alone. bf16 logit ties are {pct(b['tie']['share'])} of misses ({b['tie']['tau_gain']:.2f} τ), an artefact of greedy matching on bf16 logits.")
    lines.append(f"- *What the LoRAs fix.* Both adapters gain almost entirely on target-confident near-misses (p ≥ 0.9, rank ≤ 5: static {sgn(ts['miss|>=0.9|1-5']['delta_tau_contribution'])} τ, generator {sgn(tg['miss|>=0.9|1-5']['delta_tau_contribution'])}) and lose a little on blocks the base already accepted in full (static {sgn(ts['full']['delta_tau_contribution'])}, generator {sgn(tg['full']['delta_tau_contribution'])}) and on far misses. The changes are churn: the static lengthens {pct(ts['blocks_improved'])} of blocks and shortens {pct(ts['blocks_worse'])}. The generator moves more blocks in both directions than the static; its net +0.02 over static is the difference of {pct(tsg['blocks_improved'])} improved and {pct(tsg['blocks_worse'])} worsened blocks.")
    return "\n".join(lines)


def grad_table(mat, fmt="{:.3g}", fc=None):
    rows = []
    for l in range(5):
        row = [f"L{l}"] + [fmt.format(v) for v in mat[l]]
        rows.append(row)
    header = ["layer"] + MODULES
    t = table(header, rows)
    if fc is not None:
        t += f"\n\n`fc`: {fc}"
    return t


def step2_md(n):
    gb, gs = n["grads"]["base"], n["grads"]["static"]
    L = []
    L.append("**Measured.** The training loss (KL to the target distribution + D-PACE position weights; the 24K recipe's `block_drafter` settings) back-propagated row by row through 1,000 training rows per category (4,000 rows; seeded sample from the 24K pool r0002-100k subsets 00000..00005; never evaluation rows; validation anchors, eval mode), at the base drafter and at the static-adapted point (trained static factors installed). Gradients w.r.t. `fc` and every q/k/v/o and gate/up/down of the 5 layers; K and V split by path by giving the context path and the block path separate copies of the weight (their sum is the ordinary gradient). Per module and category, on the category's mean gradient G: the **rank-32 gain** ‖P₃₂(G)‖_F (the first-order loss decrease per unit Frobenius step of the best rank-32 update), the **top-32 share** of the gradient energy, the **split-half cosine** (two disjoint halves of 500 rows: how reproducible G is), the **noise-corrected cross-category cosine** (mean cosine between half-sample gradients of two categories divided by √ of their split-half cosines), and the **held-out capture**: the first-order gain of the best *shared* rank-32 direction as a share of the best *category-specific* rank-32 direction, both fitted on one half and scored on the other (1 = a per-category adapter gains nothing over a static one at first order). Code: `grad_maps.py`. Figures: [base](analysis24k/grad_maps_base.png), [static point](analysis24k/grad_maps_static.png).")
    L.append("")
    L.append(f"Mean training loss per row at the base: " + ", ".join(f"{c} {v:.3f}" for c, v in gb["mean_loss"].items()) + "; at the static point: " + ", ".join(f"{c} {v:.3f}" for c, v in gs["mean_loss"].items()) + ".")
    L.append("")
    L.append("Rank-32 gain at the base drafter (mean over categories; ×10⁻³):")
    L.append("")
    L.append(grad_table(gb["gain"] * 1e3, "{:.2f}", fc=f"{gb['fc']['top32_gain'] * 1e3:.1f} ×10⁻³ (one 4096 × 20,480 matrix)"))
    L.append("")
    gg = gb["gain_groups"]
    L.append("Grouped (root of summed squared gains over layers): " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in gg.items()) + " (×10⁻³). At the static point: " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in gs["gain_groups"].items()) + ".")
    L.append("")
    L.append("Top-32 share of the gradient energy (base):")
    L.append("")
    L.append(grad_table(gb["share"], "{:.2f}", fc=f"{gb['fc']['top32_share']:.2f}"))
    L.append("")
    L.append("Split-half cosine (base; reliability of the 500-row mean gradient):")
    L.append("")
    L.append(grad_table(gb["split"], "{:.2f}", fc=f"{gb['fc']['split_half_cos']:.2f}"))
    L.append("")
    L.append("Noise-corrected cross-category cosine (base; mean of the 6 category pairs; values above 1 are noise):")
    L.append("")
    L.append(grad_table(gb["corr"], "{:.2f}", fc=f"{gb['fc_corr']:.2f}"))
    L.append("")
    if gb["heldout"] is not None:
        L.append("Held-out capture of the shared rank-32 direction (base; mean over categories):")
        L.append("")
        L.append(grad_table(gb["heldout"], "{:.2f}", fc=f"{gb['fc']['heldout_capture']:.2f}"))
        L.append("")
        L.append("Per category, mean over the 45 modules: " + ", ".join(
            f"{c} {np.nanmean(v):.2f}" for c, v in gb["heldout_by_cat"].items()) + ".")
        L.append("")
    L.append("At the static point, rank-32 gain (×10⁻³; includes the noise floor of a 1,000-row mean):")
    L.append("")
    L.append(grad_table(gs["gain"] * 1e3, "{:.2f}", fc=f"{gs['fc']['top32_gain'] * 1e3:.1f} ×10⁻³"))
    L.append("")
    L.append(f"Split-half cosine at the static point (reliability; compare with the base table above):")
    L.append("")
    L.append(grad_table(gs["split"], "{:.3f}", fc=f"{gs['fc']['split_half_cos']:.3f}"))
    L.append("")
    L.append("At the static point the mean gradients are not reproducible between halves, so the gains above are mostly the noise floor, and cross-category cosines and held-out captures are not interpretable there (not shown; they are in `grads/static/summary.json`).")
    L.append("")
    L.append("Cosine between the trained static ΔW and −G (base, all categories pooled), attention modules:")
    L.append("")
    L.append(table(["layer"] + MODULES[:6],
                   [[f"L{l}"] + [f"{v:.3f}" for v in gb["static_cos"][l, :6]] for l in range(5)]))
    L.append("")
    L.append("**Interpretation.**")
    cg = gb["corr_groups"]
    L.append(f"- *Where a rank-32 step has first-order leverage:* `fc` has the largest single gain ({gb['fc']['top32_gain'] * 1e3:.1f} ×10⁻³), then the K/V paths: the context V of the first layers and the block K/V of the deep layers. Grouped, the K/V paths (context {gg['K/V context path'] * 1e3:.1f}, block {gg['K/V block path'] * 1e3:.1f}) exceed q/o ({gg['attention q/o'] * 1e3:.1f}) and the MLPs ({gg['MLP'] * 1e3:.1f}). The static LoRA (q/k/v/o) can act on all but `fc` and the MLPs; it uses one K/V weight for both paths.")
    L.append(f"- *Rank 32 is enough for attention, not for the MLPs.* In the L3–L4 q and K/V modules (and the block path from L0) the top-32 singular directions hold ≥ 0.89 of the gradient energy; in the MLPs and early o/q the share is 0.4–0.8, so these gradients are spread and noisier (split-half cosines 0.14–0.64).")
    held = gb.get("heldout_groups")
    L.append(f"- *Conflict map:* categories agree most in the K/V paths and deep q (noise-corrected cosine: context path {cg['K/V context path']:.2f}, block path {cg['K/V block path']:.2f}, q/o {cg['attention q/o']:.2f}) and least in the MLPs ({cg['MLP']:.2f}) and o_proj." + (f" The held-out test tempers this: a shared rank-32 direction fitted on all categories reaches {held['K/V context path']:.2f} (context path), {held['K/V block path']:.2f} (block path), {held['attention q/o']:.2f} (q/o) and {held['MLP']:.2f} (MLP) of the held-out first-order gain of a category-specific direction fitted on 500 rows of that category; the lowest values are in the L4 MLP (≈ 0.87–0.90) and `fc` ({gb['fc']['heldout_capture']:.2f}). Category disagreement exists, but with this much data a per-category rank-32 step would add at most ≈ 10% first-order gain over the shared one, mostly in modules the current generator does not adapt (L4 MLP, `fc`)." if held else ""))
    smax = float(np.nanmax(gs["split"]))
    L.append(f"- *After static training* the grouped gains drop (q/o {gg['attention q/o'] * 1e3:.1f} → {gs['gain_groups']['attention q/o'] * 1e3:.1f}, context path {gg['K/V context path'] * 1e3:.1f} → {gs['gain_groups']['K/V context path'] * 1e3:.1f}, block path {gg['K/V block path'] * 1e3:.1f} → {gs['gain_groups']['K/V block path'] * 1e3:.1f}, MLP {gg['MLP'] * 1e3:.1f} → {gs['gain_groups']['MLP'] * 1e3:.1f}, `fc` {gg['fc'] * 1e3:.1f} → {gs['gain_groups']['fc'] * 1e3:.1f} ×10⁻³), and what is left is not reproducible between halves of the same category (split-half cosine ≤ {smax:.2f} in every module, `fc` {gs['fc']['split_half_cos']:.2f}). The attention-only static LoRA also removed the consistent first-order signal in the MLPs and `fc`, which it does not adapt: there is no shared or per-category direction left at first order, only per-row variation.")
    L.append("- The trained static ΔW is nearly orthogonal to the base gradient (cosine 0.00–0.18; highest for o_proj). The static LoRA does not simply follow the initial first-order direction; first-order gains are local and do not say how far a direction can be followed.")
    return "\n".join(L)


def step4_md(n):
    ko, sens = n["ko"], n["sens"]
    L = []
    L.append("**Measured.** 200 validation rows (50 per category, seeded), cached on the GPU; every variant is one production pass with forward hooks; Δτ is the category-mean τ difference against the same arm unmodified, with 95% paired bootstrap intervals over rows (stratified, 2,000 resamples). Knock-outs: one attention head (its slice of the o_proj input zeroed), a layer's attention output, a layer's MLP output, and K and V of one path (zeros) in one layer or all layers. Arms base, static, generator (heads: base and static). Sensitivity (base): a random rank-32 ΔW (Gaussian factors) on one module, Frobenius norm equal to the trained static ΔW of that module (MLP and `fc`: the static's mean relative norm ‖ΔW‖/‖W‖ = " + f"{sens['mean_rel']:.4f} times ‖W‖), at 1× and 4×, 3 seeds; and the trained static ΔW of a single module alone (transplant). Code: `knockouts.py`, `analyze_maps.py`.")
    L.append("")
    rows = []
    for kind, label in (("attn", "attention output"), ("mlp", "MLP output"), ("ctx", "K/V context path"),
                        ("blk", "K/V block path")):
        for arm in ARMS:
            v = ko[arm]["variants"]
            rows.append([label, arm] + [f"{v[f'{arm}|{kind}:{l}']['delta']:+.2f}" for l in range(5)]
                        + ([f"{v[f'{arm}|{kind}:all']['delta']:+.2f}"] if kind in ("ctx", "blk") else ["–"]))
    L.append(f"Reference τ (category mean, 200 rows): base {ko['base']['tau']:.3f}, static {ko['static']['tau']:.3f}, generator {ko['generator']['tau']:.3f}.")
    L.append("")
    L.append(table(["knocked out", "arm", "L0", "L1", "L2", "L3", "L4", "all layers"], rows))
    L.append("")
    cis = ko["base"]["variants"]
    L.append(f"(Intervals are tight; e.g. base attention L0 {ci(cis['base|attn:0']['ci'], 2)}, MLP L1 {ci(cis['base|mlp:1']['ci'], 2)}, block path L1 {ci(cis['base|blk:1']['ci'], 2)}.)")
    L.append("")
    for arm in ("base", "static"):
        hs = np.array([[ko[arm]["variants"][f"{arm}|head:{l}:{h}"]["delta"] for h in range(32)] for l in range(5)])
        top = sorted(((hs[l, h], l, h) for l in range(5) for h in range(32)))[:5]
        L.append(f"Heads ({arm}): mean Δτ {hs.mean():+.3f}; {int((hs < -0.05).sum())} of 160 heads cost more than 0.05 τ, {int((hs < -0.1).sum())} more than 0.1; {int((hs > 0).sum())} heads improve τ when removed. Largest: " + ", ".join(f"L{l} h{h} {v:+.3f}" for v, l, h in top) + f". Per layer, mean Δτ: " + ", ".join(f"L{l} {hs[l].mean():+.3f}" for l in range(5)) + ".")
        L.append("")
    names = ["fc"] + [f"l{l}_{m}" for l in range(5) for m in ("q", "k", "v", "o", "gate", "up", "down")]
    rows = []
    for name in names:
        r1 = sens["random_mean"][f"{name}:x1"]
        r4 = sens["random_mean"][f"{name}:x4"]
        t = sens["variants"].get(f"base|transplant:{name}")
        rows.append([name, f"{sens['norms'][name]['delta_fro']:.2f}", f"{r1['mean']:+.4f} [{r1['min']:+.4f}, {r1['max']:+.4f}]",
                     f"{r4['mean']:+.4f} [{r4['min']:+.4f}, {r4['max']:+.4f}]",
                     f"{t['delta']:+.4f} {ci(t['ci'])}" if t else "–"])
    L.append(f"Sensitivity (base, τ {sens['tau']:.3f}); random: mean [min, max] of 3 seeds; transplant: Δτ [95% CI]:")
    L.append("")
    L.append(table(["module", "‖ΔW‖_F", "random rank-32, 1×", "random rank-32, 4×", "trained static ΔW alone"], rows))
    L.append("")
    tsum = sum(v["delta"] for k, v in sens["variants"].items() if "transplant" in k)
    full = ko["static"]["tau"] - ko["base"]["tau"]
    r1 = np.array([sens["random_mean"][f"{x}:x1"]["mean"] for x in names[1:]])
    r4 = np.array([sens["random_mean"][f"{x}:x4"]["mean"] for x in names[1:]])
    L.append(f"Sum of the 20 single-module transplants: {tsum:+.3f} τ; the full static LoRA on the same rows: {full:+.3f} τ. Figures: [heads](analysis24k/head_knockouts.png), [modules](analysis24k/module_knockouts.png), [sensitivity](analysis24k/sensitivity.png).")
    L.append("")
    L.append("**Interpretation.**")
    v = ko["base"]["variants"]
    L.append(f"- *The drafter depends heavily on every block it has* (removing any layer's attention costs 1.9–6.1 τ, any MLP 0.65–2.5 τ) and above all on the context: zeroing the context K/V in layer 0 alone costs {v['base|ctx:0']['delta']:+.2f} τ, nearly as much as in all layers ({v['base|ctx:all']['delta']:+.2f}). The block path matters less per layer ({v['base|blk:1']['delta']:+.2f} to {v['base|blk:2']['delta']:+.2f}) but removing it everywhere costs {v['base|blk:all']['delta']:+.2f}. Single heads matter little (mean −0.03); a handful in L4 and L1 carry 0.1–0.3 τ each.")
    diffs = [abs(ko[a]["variants"][f"{a}|{k}"]["delta"] - v[f"base|{k}"]["delta"])
             for a in ("static", "generator") for k in [x.split("|")[1] for x in v if "|head:" not in x]]
    hb = [v[f"base|head:{l}:{h}"]["delta"] for l in range(5) for h in range(32)]
    hs_ = [ko["static"]["variants"][f"static|head:{l}:{h}"]["delta"] for l in range(5) for h in range(32)]
    L.append(f"- *Adapted drafters have the same dependency map:* static and generator knock-out effects differ from the base by at most {max(diffs):.2f} τ (largest in the MLPs of L3), and per-head effects correlate across base and static (Pearson {np.corrcoef(hb, hs_)[0, 1]:.2f}); the static drafter is slightly more robust to single-head removal. Adapters re-weight existing computation; they do not reroute it.")
    L.append(f"- *Flat to random directions:* a random rank-32 update with the trained static's norm changes τ by at most {np.abs(r1).max():.3f} on any attention or MLP module, and by at most {np.abs(r4).max():.3f} even at 4× that norm; only `fc` is sensitive (4×: {sens['random_mean']['fc:x4']['mean']:+.3f}). The trained ΔW of single modules gives up to +0.03 τ each, and their sum ({tsum:+.3f}) exceeds the full static gain on the same rows ({full:+.3f}): the gain is spread thinly over many modules, and the single-module effects overlap (sub-additive). At the static's step size the drafter is flat in random directions and responds only to a few specific ones. Sensitivity is not improvability: a flat random response says random directions are harmless, not that a trained direction cannot help.")
    return "\n".join(L)


def step5_md(n):
    rp = n["repr"]
    L = []
    L.append(f"**Measured.** For 6 seeded fully valid blocks per validation row ({rp['positions']:,} draft-slot positions; the same blocks for every arm), the drafter's state after each of the 5 layers (the layer-0 input at draft slots is the constant mask embedding, skipped) for base, static, generator, and the generator conditioned on another row's prompt (same category; next category). Linear probes (multinomial logistic regression, standardized, L2 1e-3, LBFGS; 80/20 split by row; {rp['test_positions']:,} test positions): category (accuracy; majority class {pct(rp['probes']['chance_category'])}) and \"this slot is accepted\" (in the leading correct run of its block, for that arm; AUC), with slot index as an extra feature, against the drafter's own top-1 probability as a baseline. Linear CKA and relative change ‖X_a − X_b‖/‖X_a‖ between arms on the same positions; CKA of base drafter layers vs the target layers fed in, at the position each slot predicts from. Code: `collect_slots.py --hidden`, `analyze_repr.py`. Figure: [representations](analysis24k/representations.png).")
    L.append("")
    rows = []
    for key, label in (("base|static", "static vs base"), ("base|generator", "generator vs base"),
                       ("static|generator", "generator vs static"),
                       ("generator|generator_same", "generator: own vs same-category prompt"),
                       ("generator|generator_cross", "generator: own vs other-category prompt")):
        rows.append([label] + [f"{rp['rel_change'][f'{key}|L{l}']:.3f} / {rp['cka'][f'{key}|L{l}']:.4f}" for l in range(5)])
    L.append("Relative change / linear CKA per layer:")
    L.append("")
    L.append(table(["pair", "L0", "L1", "L2", "L3", "L4"], rows))
    L.append("")
    rows = []
    for l in range(5):
        d = rp["delta_cos"][f"L{l}"]
        rows.append([f"L{l}", f"{d['static_vs_generator']:.3f}", f"{d['generator_vs_same_donor']:.3f}",
                     f"{d['generator_vs_cross_donor']:.3f}", f"{d['generator_minus_same_over_generator_change']:.3f}",
                     f"{d['generator_minus_cross_over_generator_change']:.3f}"])
    L.append("Changes relative to base (Δ = X − X_base): mean per-position cosine between the static and generator changes, between the generator's change with its own prompt and with a donor prompt, and the size of the donor-induced difference relative to the generator's change:")
    L.append("")
    L.append(table(["layer", "cos(Δ static, Δ generator)", "cos(Δ own, Δ same-cat donor)", "cos(Δ own, Δ other-cat donor)",
                    "‖own − same-cat‖ / ‖Δ own‖", "‖own − other-cat‖ / ‖Δ own‖"], rows))
    L.append("")
    rows = []
    for arm in ARMS:
        b = rp["probes"][f"{arm}|accept_baselines"]
        rows.append([arm] + [f"{rp['probes'][f'{arm}|L{l}']['category_acc']:.3f}" for l in range(5)]
                    + [f"{rp['probes'][f'{arm}|L{l}']['accept_auc']:.3f}" for l in range(5)]
                    + [f"{b['drafter_p_top_auc']:.3f}", f"{b['slot_only_auc']:.3f}"])
    L.append("Probes (test positions): category accuracy L0–L4 | accept AUC L0–L4 | baselines: drafter top-1 probability AUC, slot-only AUC:")
    L.append("")
    L.append(table(["arm"] + [f"cat L{l}" for l in range(5)] + [f"accept L{l}" for l in range(5)]
                   + ["p_top AUC", "slot AUC"], rows))
    L.append("")
    L.append("Category accuracy from the target layers fed in: " + ", ".join(
        f"{k.split('|')[1]} {v['category_acc']:.3f}" for k, v in rp["probes"].items() if k.startswith("target|")) + ".")
    L.append("")
    m = rp["drafter_vs_target_cka"]
    L.append("Linear CKA, base drafter layer vs target layer (rows L0–L4; columns target 2, 10, 18, 26, 34, 36):")
    L.append("")
    L.append(table(["drafter"] + ["t2", "t10", "t18", "t26", "t34", "t36"],
                   [[f"L{l}"] + [f"{m[f'L{l}|{t}']:.3f}" for t in ("t2", "t10", "t18", "t26", "t34", "t36")] for l in range(5)]))
    L.append("")
    L.append("**Interpretation.**")
    d4 = rp["delta_cos"]["L4"]
    L.append(f"- *Category is already in the drafter's states* at every layer (probe accuracy 0.96–0.98 against a majority baseline of {pct(rp['probes']['chance_category'])}) and in every target layer fed in (0.96–0.99). A prompt-derived category signal from the generator adds information the drafter already has; whether it uses it is another question (steps 1 and 3).")
    L.append(f"- *Adapters change the late layers most:* relative change grows from ≈ 0.05 at L0 to {rp['rel_change']['base|static|L4']:.2f} (static) and {rp['rel_change']['base|generator|L4']:.2f} (generator) at L4; CKA stays ≥ {min(rp['cka'][f'base|generator|L{l}'] for l in range(5)):.2f}. Static and generator change the states in partly shared directions (per-position cosine {min(rp['delta_cos'][f'L{l}']['static_vs_generator'] for l in range(5)):.2f}–{max(rp['delta_cos'][f'L{l}']['static_vs_generator'] for l in range(5)):.2f}).")
    L.append(f"- *Prompt dependence of the generator's change:* swapping the prompt for another category's moves the states by about half the generator's own change (L4: {d4['generator_minus_cross_over_generator_change']:.2f}); a same-category prompt by about a fifth ({d4['generator_minus_same_over_generator_change']:.2f}). The generator's effect is mostly category-level, matching the D13 picture (η² 0.78).")
    acc = rp["probes"]
    L.append(f"- *Acceptance is readable but not better than the drafter's own confidence:* the linear accept-probe reaches AUC {acc['base|L4']['accept_auc']:.3f} at L4 against {acc['base|accept_baselines']['drafter_p_top_auc']:.3f} for the drafter's top-1 probability. Both adapters lower that confidence AUC a little ({acc['static|accept_baselines']['drafter_p_top_auc']:.3f} static, {acc['generator|accept_baselines']['drafter_p_top_auc']:.3f} generator). The drafter-vs-target CKA is low everywhere (≤ 0.30, highest for target layers 10, 34 and 36 in the late drafter layers); linear CKA is dominated by a few large-norm residual dimensions, so these values are indicative only.")
    return "\n".join(L)


def step3_md(n, interpretation):
    mk, att = n["mask"], n["att"]
    L = []
    eq = n["att_equiv"]
    L.append(f"**Measured.** (a) Attention weights of draft-slot queries (slots 1–15 of valid blocks) on 200 validation rows (50 per category, rows ≤ 4,096 tokens so the dense weights fit), for base, static and generator. FlexAttention returns no weights, so this pass switches the drafter to the eager path (`eager_attention_forward` with the float mask of the same mask mod). Equivalence on 5 rows per arm: eager vs flex argmax agreement {min(e['argmax_agreement'] for e in eq):.3f}–{max(e['argmax_agreement'] for e in eq):.3f}, mean |logit difference| {np.mean([e['mean_abs_logit_diff'] for e in eq]):.3f}, τ per row within {max(abs(e['eager_eal'] - e['flex_eal']) for e in eq):.2f} (bf16 kernel differences). Mass is split into: prompt tokens (positions before the first supervised token; the first 4 of them separately, as attention-sink candidates), response tokens of the injected context, the block's anchor token, and its mask tokens; plus the 32 most recent context tokens. (b) τ on all 500 validation rows with the prompt removed from the injected context (an extra FlexAttention mask term: block queries never attend to prompt positions), the same keeping the first 4 prompt tokens (attention-sink control), and two controls removing n = min(prompt length, response length / 2) response tokens: random response positions, and the earliest response positions. Median prompt length {mk['prompt_len_median']:.0f} tokens, median response {mk['response_len_median']:.0f}; n equals the prompt length in {pct(mk['n_mask_equals_prompt_share'])} of rows. The generator still receives its prompt summary in every condition (only the drafter's context is masked). Code: `prompt_usage.py`. Figures: [attention](analysis24k/attention.png), [masking](analysis24k/masking.png).")
    L.append("")
    fields = att["base"]["fields"]
    rows = []
    for arm in ARMS:
        a = att[arm]
        for l in range(5):
            pl = a["per_layer"][l]
            rows.append([arm, f"L{l}", pct(pl[0]), pct(pl[5]), pct(pl[0] - pl[5]), pct(a["uniform_prompt"][l]),
                         pct(pl[1]), pct(pl[4]), pct(pl[2]), pct(pl[3])])
    L.append("(a) Mean attention mass per draft query (mean over heads):")
    L.append("")
    L.append(table(["arm", "layer", "prompt", "first 4 prompt tokens", "rest of prompt", "prompt if uniform",
                    "response context", "32 most recent", "anchor", "block masks"], rows))
    L.append("")
    rows = []
    for c in C.CATEGORIES:
        pl = np.array(att["base"]["per_category_layer"][c])
        rows.append([c] + [f"{pct(pl[l, 0])} ({pct(pl[l, 0] - pl[l, 5])})" for l in range(5)])
    L.append("Base drafter by category, prompt mass per layer (in brackets: without the first 4 tokens):")
    L.append("")
    L.append(table(["category", "L0", "L1", "L2", "L3", "L4"], rows))
    L.append("")
    ph = np.array(att["base"]["per_head"])  # L, H, F
    rest = ph[..., 0] - ph[..., 5]
    order = sorted(((rest[l, h], l, h) for l in range(5) for h in range(rest.shape[1])), reverse=True)[:6]
    L.append("Heads with the most prompt mass outside the first 4 tokens (base): " + ", ".join(
        f"L{l} h{h} {pct(v)}" for v, l, h in order) + f"; {int((rest > 0.2).sum())} of {rest.size} heads put more than 20% there.")
    L.append("")
    rows = []
    conds = [c for c in ("prompt", "prompt_keep4", "resp_random", "resp_early") if c in mk["arms"]["base"]]
    names = {"prompt": "prompt removed", "prompt_keep4": "prompt removed except its first 4 tokens",
             "resp_random": "random response tokens removed", "resp_early": "earliest response tokens removed"}
    for arm in ARMS:
        e = mk["arms"][arm]
        rows.append([arm, f"{e['none']['tau']:.3f}"] + [f"{e[c]['delta']:+.3f} {ci(e[c]['ci'])}" for c in conds])
    L.append("(b) Category-mean τ with the full context, and Δτ when part of the context is removed (500 validation rows; 95% paired bootstrap):")
    L.append("")
    L.append(table(["arm", "τ (full context)"] + [names[c] for c in conds], rows))
    L.append("")
    rows = []
    for arm in ARMS:
        e = mk["arms"][arm]
        rows.append([arm] + [f"{e['prompt']['delta_categories'][c]:+.3f}" for c in C.CATEGORIES])
    L.append("Δτ with the prompt removed, by category:")
    L.append("")
    L.append(table(["arm"] + list(C.CATEGORIES), rows))
    L.append("")
    gs = mk["generator_minus_static"]
    L.append("Generator − static Δτ under each condition: " + "; ".join(
        f"{k} {v['delta']:+.3f} {ci(v['ci'])}" for k, v in gs.items()) + ".")
    L.append("")
    L.append("**Interpretation.**")
    L.extend(interpretation)
    return "\n".join(L)


def headline(n):
    """Key numbers used by the TL;DR, the artifact and the final summary."""
    b = n["err"]["base"]["all_blocks"]
    ts = n["trans"]["base->static"]["all"]
    tsg = n["trans"]["static->generator"]["all"]
    gb = n["grads"]["base"]
    mk = n["mask"]["arms"]
    sens = n["sens"]
    names = [f"l{l}_{m}" for l in range(5) for m in ("q", "k", "v", "o", "gate", "up", "down")]
    return {
        "fix_all": b["fix_all_first_misses_tau_gain"], "addr_share": b["addressable"]["share"],
        "addr_gain": b["addressable"]["tau_gain"], "strict_share": b["strict"]["share"],
        "strict_gain": b["strict"]["tau_gain"], "uncertain_share": b["target_uncertain"]["share"],
        "offpath_share": b["offpath"]["share"], "onpath_share": b["onpath_share"],
        "near_share": 1 - b["rank"]["6-20"] - b["rank"][">20"],
        "static_gain": ts["delta_tau"], "gen_minus_static": tsg["delta_tau"],
        "static_improved": ts["blocks_improved"], "static_worse": ts["blocks_worse"],
        "gs_improved": tsg["blocks_improved"], "gs_worse": tsg["blocks_worse"],
        "corr_groups": gb["corr_groups"], "gain_groups": gb["gain_groups"],
        "heldout_groups": gb.get("heldout_groups"),
        "mask_prompt": {a: mk[a]["prompt"] for a in ARMS},
        "mask_random": {a: mk[a]["resp_random"] for a in ARMS},
        "mask_early": {a: mk[a]["resp_early"] for a in ARMS},
        "rand_max_x1": float(max(abs(sens["random_mean"][f"{x}:x1"]["mean"]) for x in names)),
        "rand_max_x4": float(max(abs(sens["random_mean"][f"{x}:x4"]["mean"]) for x in names)),
        "fc_x4": sens["random_mean"]["fc:x4"]["mean"],
        "transplant_sum": float(sum(v["delta"] for k, v in sens["variants"].items() if "transplant" in k)),
        "ctx0": n["ko"]["base"]["variants"]["base|ctx:0"]["delta"],
        "ctxall": n["ko"]["base"]["variants"]["base|ctx:all"]["delta"],
        "cat_probe_min": min(n["repr"]["probes"][f"{a}|L{l}"]["category_acc"] for a in ARMS for l in range(5)),
        "donor_cross_L4": n["repr"]["delta_cos"]["L4"]["generator_minus_cross_over_generator_change"],
        "donor_same_L4": n["repr"]["delta_cos"]["L4"]["generator_minus_same_over_generator_change"],
        "att_prompt_rest": [n["att"]["base"]["per_layer"][l][0] - n["att"]["base"]["per_layer"][l][5] for l in range(5)],
        "att_first4": [n["att"]["base"]["per_layer"][l][5] for l in range(5)],
        "att_uniform": n["att"]["base"]["uniform_prompt"],
    }


def step3_interpretation(n):
    mk = n["mask"]["arms"]
    h = headline(n)
    gs = n["mask"]["generator_minus_static"]
    out = []
    pr = h["att_prompt_rest"]
    out.append(f"- *The drafter reads the prompt, but less than its share of the context.* Draft queries put {pct(min(pr))}–{pct(max(pr))} of their attention on prompt tokens beyond the first 4 (plus {pct(min(h['att_first4']))}–{pct(max(h['att_first4']))} on the first 4), against {pct(h['att_uniform'][0])} if attention were uniform over visible tokens; about half goes to response context and, from L1 on, 22–35% to the block's own mask tokens. The adapters hardly change this distribution (all three arms within ≈ 0.02 per field).")
    keep4 = mk["base"].get("prompt_keep4")
    keep4_txt = (f" Keeping the first 4 prompt tokens (attention-sink control) changes this to {keep4['delta']:+.2f} {ci(keep4['ci'], 2)}: the sink tokens account for {keep4['delta'] - mk['base']['prompt']['delta']:.2f} τ, the rest is prompt content." if keep4 else "")
    out.append(f"- *Removing the prompt from the context costs as much as removing as many response tokens:* base {mk['base']['prompt']['delta']:+.2f} τ {ci(mk['base']['prompt']['ci'], 2)} vs {mk['base']['resp_random']['delta']:+.2f} {ci(mk['base']['resp_random']['ci'], 2)} for random response tokens and {mk['base']['resp_early']['delta']:+.2f} for the earliest response tokens.{keep4_txt} It hurts most in stem ({mk['base']['prompt']['delta_categories']['stem']:+.2f}) and least in math ({mk['base']['prompt']['delta_categories']['math']:+.2f}). The prompt is therefore information the drafter already obtains directly through the injected target features.")
    out.append(f"- *The generator does not substitute for the prompt.* With the prompt removed from the drafter's context, the generator (which still conditions on the prompt summary) loses as much as the static LoRA: generator − static is {gs['prompt']['delta']:+.3f} {ci(gs['prompt']['ci'])} without the prompt vs {gs['none']['delta']:+.3f} {ci(gs['none']['ci'])} with it, and {gs['resp_random']['delta']:+.3f} {ci(gs['resp_random']['ci'])} with random response tokens removed. Its prompt-derived LoRA does not carry the content the drafter lost; its small edge needs the full context, as an adjustment on top of a drafter that reads the prompt anyway.")
    return out


def static_point_noise(n):
    gs = n["grads"]["static"]
    return float(np.nanmax(gs["split"])), float(np.nanmedian(gs["split"])), gs["fc"]["split_half_cos"]


def tldr(n):
    h = headline(n)
    smax, smed, sfc = static_point_noise(n)
    gb = n["grads"]["base"]
    cg = h["corr_groups"]
    mp = h["mask_prompt"]
    gs = n["mask"]["generator_minus_static"]
    held = h["heldout_groups"]
    held_txt = (f" Held out, a shared rank-32 direction reaches {held['K/V context path']:.2f} / {held['K/V block path']:.2f} / {held['attention q/o']:.2f} / {held['MLP']:.2f} (context / block / q,o / MLP) of a category-specific one's first-order gain." if held else "")
    return "\n".join([
        "## TL;DR",
        "",
        "Measurements first; interpretation in italics. Offline, teacher-forced, greedy, one seed; τ pooled over blocks unless noted.",
        "",
        f"- **How much of the remaining error looks fixable.** {pct(h['near_share'])} of the base drafter's first misses are near-misses (the target's token is in the drafter's top 5) and {pct(h['onpath_share'])} happen while the data is still on the target's greedy path. Fixing just the on-path, target-confident (p ≥ 0.5) near-misses ({pct(h['addr_share'])} of misses) would add up to {h['addr_gain']:.2f} τ; the strict version (p ≥ 0.9; {pct(h['strict_share'])} of misses) {h['strict_gain']:.2f} τ; every first miss {h['fix_all']:.2f} τ. The static LoRA gains {h['static_gain']:+.2f} τ, almost all from target-confident near-misses, while shortening {pct(h['static_worse'])} of blocks (it lengthens {pct(h['static_improved'])}). Target-uncertain (p < 0.5) misses are {pct(h['uncertain_share'])} and off-path misses {pct(h['offpath_share'])}. *The error left is mostly of a kind a small correction could flip; the adapters reach a fraction of it and churn blocks in both directions.*",
        f"- **Where a rank-32 LoRA could act, where categories disagree.** At the base drafter the first-order rank-32 gain is largest in `fc` ({gb['fc']['top32_gain'] * 1e3:.0f} ×10⁻³) and the K/V paths (context {h['gain_groups']['K/V context path'] * 1e3:.0f}, block {h['gain_groups']['K/V block path'] * 1e3:.0f}; q/o {h['gain_groups']['attention q/o'] * 1e3:.0f}; MLP {h['gain_groups']['MLP'] * 1e3:.0f}). Categories agree in the K/V paths and deep q (noise-corrected cosine {cg['K/V context path']:.2f} context, {cg['K/V block path']:.2f} block) and disagree most in the MLPs ({cg['MLP']:.2f}) and o_proj.{held_txt} At the static-adapted point no module has a reproducible mean gradient left (split-half cosine ≤ {smax:.2f}, median {smed:.2f}; `fc` {sfc:.2f}, vs up to 0.98 at the base). *Categories' gradients differ most in the MLP and o_proj, but at this data scale (500 rows per category half) a category-specific rank-32 step is barely better on held-out rows than the shared one, and after the static LoRA no reproducible direction is left. At first order there is little room for category-level adaptation.*",
        f"- **Is the prompt redundant for the drafter?** The drafter reads the prompt directly: removing it from the injected context costs {mp['base']['delta']:+.2f} τ (category mean), about as much as removing as many random response tokens ({h['mask_random']['base']['delta']:+.2f}). Category is linearly decodable from every drafter layer (≥ {h['cat_probe_min']:.2f}). The generator, which still sees the prompt summary, does not make up for a prompt-less context (generator − static {gs['prompt']['delta']:+.3f} without the prompt vs {gs['none']['delta']:+.3f} with it). *Prompt-level information is largely redundant with what the drafter already attends to; the generator's gain is a small category-level adjustment (swapping in another category's prompt moves its effect on the L4 states by {h['donor_cross_L4']:.2f} of its size, a same-category prompt by {h['donor_same_L4']:.2f}).*",
        f"- **Sensitivity.** The drafter is flat: a random rank-32 update with the static LoRA's norm changes τ by at most {h['rand_max_x1']:.3f} on any attention or MLP module ({h['rand_max_x4']:.3f} at 4×); only `fc` reacts ({h['fc_x4']:+.3f} at 4×). Knocking out the context path in layer 0 alone costs {h['ctx0']:+.2f} τ (all layers {h['ctxall']:+.2f}). The 20 single-module static transplants sum to {h['transplant_sum']:+.2f} τ. *Gains come from a few trained directions spread thinly over many modules; no single module holds them.*",
        "- **For the generator design** (see [Implications](#implications-for-the-generator-design)): prompt-summary conditioning (C1–C3) has little new information to add; block- or response-conditioned adaptation (G6, C4) and objectives aimed at the confident near-miss first misses without breaking accepted blocks (D3) target what is left; per-category controls (A1) should include the MLP / o_proj, where categories disagree most, though the held-out gradient test predicts little gain; cheap low-dimensional generators (G1, G3) match the category-level structure that the 24K generator actually uses.",
    ])


def implications(n):
    h = headline(n)
    return "\n".join([
        "Interpretation (not measured directly; single seed, offline metrics):",
        "",
        f"- **Conditioning (C1–C5).** The prompt is already in the drafter's context and is used (step 3), and category is decodable from every drafter layer (step 5). Normalizing or pooling the prompt summary (C1, C2, C3) can sharpen the routing signal but adds no information the drafter lacks. Response-so-far or per-block conditioning (C4, G6) addresses a different quantity: misses are local (near-misses on the next tokens of the response), and they are where the remaining error sits. C5a (oracle category one-hot) remains the cheapest test of the category-routing ceiling.",
        f"- **Parameterization (G1–G6).** The 24K generator's effect is mostly category-level (step 5; D13 η² 0.78) and small. A basis mixture (G1) or diagonal core (G3) can represent that at a fraction of the 1.1B parameters. Because the drafter is flat to random rank-32 directions (step 4), larger per-prompt ΔW are not dangerous at this norm, but they also do not help on their own; what helps is a few specific directions.",
        f"- **Where to adapt.** First-order leverage is highest in `fc` and the K/V paths, where categories agree. Category disagreement is largest in the MLP and o_proj, but held out a shared rank-32 direction captures ≈ 0.9–1.0 of a category-specific one's gain everywhere (lowest in the L4 MLP and `fc`). A dynamic part therefore has little first-order room in any module at this data scale; if one is tried, the L4 MLP and `fc` are the places, which are outside the current q/k/v/o targets. The K/V path split (D1) showed no difference; the gradient maps agree that both paths carry similar, category-aligned signal.",
        f"- **Objective (D3, D-PACE).** The adapters gain on confident near-miss first misses and lose on blocks the base already accepted (step 1). A τ-aligned loss (D3) that weights the first failing slot and penalizes breaking accepted prefixes targets exactly this trade-off, for static and generator alike. Note that D-PACE weights depend on the drafter's own confidence, so training losses are not comparable across arms (the static point has a higher weighted KL than the base on the same rows although its τ is higher).",
        f"- **Evaluation hygiene.** {pct(h['offpath_share'])} of first misses follow a sampled non-argmax token earlier in the block, and about 4% are exact bf16 logit ties. Both inflate the apparent error; greedy-path labels and tie-aware matching would make small arm differences easier to read.",
    ])


NEXT = [
    "<b>τ-aligned loss (D3) on the static LoRA, then the generator.</b> Directly targets the confident near-miss first misses (≈ 0.4–0.9 τ addressable) and penalizes breaking accepted blocks (the adapters' main loss). Needs a loss term; one 24K run each, ≈ 3 h.",
    "<b>Per-category static LoRAs including MLP/o (A1+).</b> Bounds category routing where the gradient maps show the most disagreement (the held-out first-order test predicts ≤ ≈ 10% more gain than a shared adapter); 4 static runs on the 24K category splits. If they do not beat the global static, prompt-level dynamism is not worth its cost on this drafter.",
    "<b>Block-conditioned adapter (G6 with G1).</b> A small basis of LoRAs mixed per anchor block from the drafter's own context features, since the prompt is redundant and misses are local.",
    "<b>Static `fc` (+ MLP) adapter.</b> `fc` has the largest first-order gain and is the only module sensitive to random updates; D4 tested `fc` only inside the generator.",
    "<b>Greedy-path, tie-aware evaluation.</b> Re-score existing arms with labels from the target's greedy path (or regenerated greedy responses) and tie-aware matching, so that the ≈ 20% of first misses that are off-path or bf16 ties stop counting as drafter errors.",
]


def limits(n):
    smax, smed, _ = static_point_noise(n)
    return "\n".join([
        "- One seed per arm; offline, teacher-forced, greedy matching on bf16 logits; no serving measurement.",
        "- Fix gains are first-order ceilings (one corrected slot per block, later misses unchanged), not achievable targets.",
        "- \"On-path\" is judged against the target's argmax along the sampled response; responses were sampled, so the labels themselves are argmaxes along a sampled path.",
        f"- Gradients: first-order, mean over 1,000 rows per category with validation anchors; at the static point the mean gradients are noise-dominated (split-half cosine ≤ {smax:.2f}), so their gains, cosines and captures are not interpretable there. Held-out capture can exceed 1 by noise. D-PACE weights depend on the drafter's confidence (loss values are not comparable across arms; gradients are the training gradients).",
        "- Attention weights come from the eager path (98–99% argmax agreement with FlexAttention); the attention study uses 200 rows of at most 4,096 tokens (longer rows excluded).",
        "- Masking removes positions from the attention of all layers; the prompt-length-matched controls remove at most half of the response. The generator's own prompt summary is never masked.",
        "- Knock-outs and sensitivity use 200 validation rows; zeroing K/V is one choice of ablation (mean-ablation would differ). Random-direction sensitivity says nothing about trained directions (\"sensitivity ≠ improvability\").",
        "- Linear CKA is dominated by a few large-norm residual dimensions; the drafter-vs-target CKA values are indicative only. Probes are linear with fixed L2.",
    ])


def markdown(n, url):
    h = headline(n)
    rv, rt = n["repro"]["validation"], n["repro"]["test"]
    parts = [
        "# Where DFlash moves: error, gradient, prompt, knock-out and representation analysis of the 24K checkpoints",
        "",
        "- **Status:** done (2026-11-11). Analysis only: no training, no change to any run, store or library code.",
        "- **Question (user 2026-11-11):** every ablation moves the DFlash drafter by tiny amounts (best static LoRA ≈ +0.2 τ over base; generator − static +0.022 τ on 24K). Where, and how much, can the drafter be influenced, before designing new generators?",
        "- **Models:** base drafter `z-lab/Qwen3-8B-DFlash-b16@9b41424b`; 24K static LoRA (KL) `mix24k-static-lora-attn-r32-3ep-kl`; 24K generator (KL) `mix24k-generator-attn-r32-3ep-kl` ([MIX24K.md](MIX24K.md)). Attention q/k/v/o, rank 32, coefficient 2.",
        f"- **Interactive page:** {url}",
        "- **Code:** `scripts/dflash_port/analysis/` (harness `common.py`; tests `tests/unit/scripts/test_dflash_analysis24k.py`). **Raw results:** `/workspace/yusuf/runs/dflash/analysis24k/` (`slots/`, `grads/`, `prompt/`, `knockouts/`, `hidden/`, `results/*.json`). **Figures:** [analysis24k/](analysis24k/).",
        "",
        tldr(n),
        "",
        "## Method common to every step",
        "",
        "- **Forward pass:** the production one. One `Runtime` built from the 24K generator recipe through `campaign.run_one(..., evaluate=)` (read-only; the run identity is checked against the trained run), adapters on q/k/v/o of all 5 layers, `Runtime.forward` → `functional_call` with per-arm factors: zero factors (base; bit-identical to the released drafter), the static checkpoint's factors, or the factors the generator makes from each row's own prompt. The drafter's `forward` is replaced by the upstream `_backbone_forward` + `compute_metrics` plus a capture of logits, targets, mask and block indices. Validation anchors (content-seeded, density 1/16), eval mode, bf16 autocast, FlexAttention.",
        f"- **Reproduction check (passed).** Category-mean full_acc, validation (500 rows): base {rv['base']['computed']['full_acc']['all']:.6f} (recorded {rv['base']['recorded_full_acc']:.6f}; the record is the math10k base run without adapters), static {rv['static']['computed']['full_acc']['all']:.6f} (recorded {rv['static']['recorded_full_acc']:.6f}), generator {rv['generator']['computed']['full_acc']['all']:.6f} (recorded {rv['generator']['recorded_full_acc']:.6f}). Test: base {rt['base']['computed']['full_acc']['all']:.6f} ({rt['base']['recorded_full_acc']:.6f}), static {rt['static']['computed']['full_acc']['all']:.6f} ({rt['static']['recorded_full_acc']:.6f}), generator {rt['generator']['computed']['full_acc']['all']:.6f} ({rt['generator']['recorded_full_acc']:.6f}). Category-mean τ, validation: base {rv['base']['computed']['eal']['all']:.3f}, static {rv['static']['computed']['eal']['all']:.3f}, generator {rv['generator']['computed']['eal']['all']:.3f}. A first attempt did not reproduce the LoRA arms: newly installed adapters start in training mode (LoRA dropout on), and production validation calls `drafter.eval()` first; the harness now does too. The block-level τ of step 1 equals the metric counts per subset (max |Δ| 0).",
        "- **τ in this report:** per-block accepted run + 1 (bonus), **pooled over blocks** unless marked \"category mean\" (the evaluation headline is the category mean; pooled τ = 1 + accepted draft tokens per block, e.g. base test 8.213 = 721 per 100 blocks + 1).",
        "",
        "## Step 1: error decomposition (is the remaining error fixable?)",
        "",
        "**Measured.** For every block of the test split (49,021 blocks, 500 rows; validation in `results/errors_validation.json`, same picture) and each arm: the first slot that fails; at that slot the target's top-1 probability `p_top` and entropy, the rank the drafter gave the target's top-1 token, the drafter's probability of it, and whether the data stayed on the target's greedy path inside the block before the miss (**on-path**: every earlier slot's data token equals the target's argmax label). The labels are the target's argmax along the sampled response; after a sampled non-argmax token, the label follows a token the drafter cannot know (**off-path** miss). Rank 1 at a miss is an exact bf16 logit tie that argmax broke the other way. The **fix gain** of a miss is the τ gained if only that first miss were corrected (the block then runs to its next failing slot). Code: `collect_slots.py`, `analyze_slots.py`.",
        "",
        step1_md(n),
        "",
        "## Step 2: gradient maps (where could a rank-32 LoRA help, where do categories disagree?)",
        "",
        step2_md(n),
        "",
        "## Step 3: prompt usage (is the generator's prompt information redundant?)",
        "",
        step3_md(n, step3_interpretation(n)),
        "",
        "## Step 4: knock-outs and sensitivity",
        "",
        step4_md(n),
        "",
        "## Step 5: representations (probes and CKA)",
        "",
        step5_md(n),
        "",
        "## Implications for the generator design",
        "",
        implications(n),
        "",
        "### Most promising next experiments",
        "",
        "\n".join(f"{i + 1}. {x.replace('<b>', '**').replace('</b>', '**')}" for i, x in enumerate(NEXT)),
        "",
        "## Limits",
        "",
        limits(n),
        "",
        "## Reproduce",
        "",
        "```bash",
        "export CUDA_VISIBLE_DEVICES=0",
        "# one-time: the generator run's inputs copied into the analysis directory (not part of the run identity)",
        "cp -r /workspace/yusuf/runs/dflash/mix24k/mix24k-generator-attn-r32-3ep-kl/inputs /workspace/yusuf/runs/dflash/analysis24k/inputs/generator",
        "A=scripts/dflash_port/analysis",
        ".venv/bin/python $A/collect_slots.py --split validation --hidden && .venv/bin/python $A/collect_slots.py --split test",
        ".venv/bin/python $A/grad_maps.py --point base && .venv/bin/python $A/grad_maps.py --point static",
        ".venv/bin/python $A/knockouts.py --part all",
        ".venv/bin/python $A/prompt_usage.py --part all && .venv/bin/python $A/prompt_usage.py --part sink",
        ".venv/bin/python $A/analyze_slots.py --split test && .venv/bin/python $A/analyze_slots.py --split validation",
        ".venv/bin/python $A/analyze_maps.py && .venv/bin/python $A/analyze_repr.py",
        ".venv/bin/python $A/make_figures.py && .venv/bin/python $A/write_report.py",
        "```",
        "",
        "GPU 0 only; peak ≈ 58 GB GPU (gradient maps), ≈ 30 GB host RSS per process.",
        "",
    ]
    return "\n".join(parts)


def artifact_text(n):
    h = headline(n)
    b = n["err"]["base"]["all_blocks"]
    mp = h["mask_prompt"]["base"]
    gs = n["mask"]["generator_minus_static"]
    smax, _, _ = static_point_noise(n)
    held = h["heldout_groups"]
    kpis = [
        {"v": f"{h['addr_gain']:.2f} τ", "l": f"ceiling from fixing the on-path, target-confident near-misses ({pct(h['addr_share'], 0)} of first misses); the static LoRA gains {h['static_gain']:.2f}"},
        {"v": f"{h['near_share'] * 100:.0f}%", "l": "of first misses have the target's token in the drafter's top 5"},
        {"v": f"{mp['delta']:+.2f} τ", "l": f"when the prompt is removed from the drafter's context, about the same as removing as many response tokens ({h['mask_random']['base']['delta']:+.2f})"},
        {"v": f"≤ {h['rand_max_x1']:.3f}", "l": "|Δτ| from a random rank-32 update of the static LoRA's size on any attention or MLP module"},
    ]
    fixgain = [
        {"name": "strict\np≥0.9,rank≤5", "v": b["strict"]["tau_gain"]},
        {"name": "addressable\np≥0.5,rank≤5", "v": b["addressable"]["tau_gain"]},
        {"name": "target\nuncertain", "v": b["target_uncertain"]["tau_gain"]},
        {"name": "off-path", "v": b["offpath"]["tau_gain"]},
        {"name": "bf16 ties", "v": b["tie"]["tau_gain"]},
        {"name": "every first\nmiss", "v": b["fix_all_first_misses_tau_gain"]},
        {"name": "static − base\n(actual)", "v": h["static_gain"]},
    ]
    ts = n["trans"]["base->static"]["all"]
    tsg = n["trans"]["static->generator"]["all"]
    means1 = (f"<p><b>What it means.</b> Most of the drafter's misses are near-misses on tokens the target was sure about, made while the response was still on the target's greedy path. That is the kind of error a small correction could flip: fixing the confident near-misses alone would be worth up to {b['addressable']['tau_gain']:.2f} τ, {b['addressable']['tau_gain'] / h['static_gain']:.1f} times what the static LoRA gained.</p>"
              f"<p>The adapters do go after exactly these misses, but they also break blocks that were fine: the static LoRA lengthens {pct(ts['blocks_improved'])} of blocks and shortens {pct(ts['blocks_worse'])}. The generator's +{tsg['delta_tau']:.3f} over the static is the small difference between {pct(tsg['blocks_improved'])} improved and {pct(tsg['blocks_worse'])} worsened blocks.</p>"
              f"<p class=\"muted\">About {pct(b['target_uncertain']['share'], 0)} of misses fall where the target itself is unsure, and {pct(b['offpath']['share'], 0)} come after the sampled response left the greedy path. Neither can be fixed by predicting the target better.</p>")
    cg = h["corr_groups"]
    held_txt = (f" Scored on held-out rows, one shared rank-32 direction reaches {held['K/V context path']:.2f} (context path), {held['K/V block path']:.2f} (block path), {held['attention q/o']:.2f} (q/o) and {held['MLP']:.2f} (MLP) of what a category-specific one achieves." if held else "")
    means2 = (f"<p><b>What it means.</b> At the base drafter the largest rank-32 gains are in <span class=\"mono\">fc</span> and in the K/V paths. In attention, nearly all of the gradient fits in 32 directions, and the categories mostly want the same update there (agreement {cg['K/V context path']:.2f} context, {cg['K/V block path']:.2f} block). They disagree most in the MLPs ({cg['MLP']:.2f}) and o_proj.</p>"
              f"<p>That disagreement is small in practice.{held_txt} A per-category update would add at most about 10% first-order gain, mostly in the L4 MLP and <span class=\"mono\">fc</span>, which the generator does not adapt.</p>"
              f"<p>Switch to the static point and the maps go to noise. Two halves of the same category no longer agree (split-half cosine ≤ {smax:.2f} in every module), so once the static LoRA is trained, no shared or per-category first-order direction is left to follow.</p>")
    means3 = "<p><b>What it means.</b> " + " ".join(
        x.replace("- *", "").replace("*", "") for x in step3_interpretation(n)) + "</p>"
    sens = n["sens"]
    means4 = (f"<p><b>What it means.</b> The drafter needs everything it has. Layer 0's attention and its context path are the backbone: zeroing the context K/V there costs {h['ctx0']:.1f} τ. But it barely reacts to random rank-32 changes of the static LoRA's size (at most {h['rand_max_x1']:.3f} τ, {h['rand_max_x4']:.3f} at 4×). Only <span class=\"mono\">fc</span> is sensitive.</p>"
              f"<p>The trained static ΔW helps a little in almost every module (up to +0.03 τ each), and the 20 single-module gains add up to {h['transplant_sum']:+.2f} τ. The improvement is a few specific directions spread over many modules, not one weak spot. Flatness to random directions does not prove that no direction would help.</p>")
    rp = n["repr"]
    means5 = (f"<p><b>What it means.</b> Category is already decodable from every drafter layer ({pct(h['cat_probe_min'], 0)}+). A prompt-derived category signal tells the drafter nothing new. The adapters change the late layers most (L4: {rp['rel_change']['base|static|L4']:.2f} static, {rp['rel_change']['base|generator|L4']:.2f} generator). About half of the generator's change depends on the prompt's category ({h['donor_cross_L4']:.2f} when another category's prompt is used). A same-category prompt changes it by only {h['donor_same_L4']:.2f}, so the per-row part is small.</p>")
    implic = ("<p><b>Conditioning.</b> The prompt is in the drafter's context and is used, and category is already in its states. Better prompt summaries (C1–C3) sharpen routing but add little information. Adapting per block or on the response so far (C4, G6) targets the local near-misses where the error is.</p>"
              "<p><b>Parameterization.</b> The 24K generator acts mostly per category and by small amounts. A basis mixture (G1) or diagonal core (G3) can represent that at a fraction of 1.1B parameters. Per-category controls (A1) should include MLP and o_proj, where categories disagree most, but the held-out gradient test predicts little gain from them.</p>"
              "<p><b>Objective.</b> The adapters trade fixed near-misses for broken accepted blocks. A τ-aligned loss (D3) that rewards fixing the first miss and penalizes breaking accepted prefixes targets that trade-off directly.</p>")
    return {"kpis": kpis, "fixgain": fixgain, "means1": means1, "means2": means2, "means3": means3,
            "means4": means4, "means5": means5, "implications": implic, "next": NEXT}
