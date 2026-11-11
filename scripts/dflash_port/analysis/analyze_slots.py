"""Step 1 analysis: error decomposition from the per-slot records.

    python scripts/dflash_port/analysis/analyze_slots.py --split validation

Reads analysis24k/slots/<split>/slots_<arm>.npz (collect_slots.py) and writes
analysis24k/results/errors_<split>.json.

Near-miss: the drafter ranks the target's token 1-5 (rank 1 at a miss is an
exact bf16 logit tie broken the other way by argmax). Addressable: on-path,
target-confident (p_top >= 0.5) near-misses; strict: p_top >= 0.9.

Per block (a 16-token anchor block, slots 1..15): the first slot that is not
(correct and valid). If that slot is invalid, the block ran into the end of the
response ("truncated"); if no slot fails, all 15 were accepted ("full").
τ_block = first failing slot (accepted run + bonus token), 16 when full; the
pooled mean equals the evaluation's τ (checked against the metric counts).

At a genuine first miss: the target's top-1 probability (confidence) and
entropy, the rank the drafter gave the target's token, the drafter's
probability of it, and whether the in-block data path was the target's greedy
path before the miss ("on-path": every earlier slot's data token equals the
target's argmax label; off-path misses follow a sampled non-argmax token the
drafter could not know).

"Fix gain" of a miss: the τ gained if only that first miss were corrected
(the block then runs to its next failing slot); summed over a miss class and
divided by the number of blocks, it is the τ a perfect fix of that class
would add (one fix per block, later misses unchanged).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARMS = ("base", "static", "generator")
RANK_BINS = ((1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 20, "6-20"),
             (21, 10**9, ">20"))
CONF_BINS = ((0.9, 1.01, ">=0.9"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "<0.5"))
ANCHOR_BINS = ((0, 64, "0-63"), (64, 256, "64-255"), (256, 1024, "256-1023"),
               (1024, 10**9, ">=1024"))


def load(split, arm):
    data = np.load(C.OUT / "slots" / split / f"slots_{arm}.npz")
    return {k: data[k] for k in data.files}


def blocks(slots):
    """Reshape flat slot records (15 per block) into per-block arrays."""
    n = len(slots["slot"]) // 15
    if not (slots["slot"].reshape(n, 15) == np.arange(1, 16)).all():
        raise ValueError("Slot records are not 15 per block in order")
    out = {k: v.reshape(n, 15) for k, v in slots.items()}
    ok = out["correct"] & out["valid"]
    fail = ~ok
    has_fail = fail.any(1)
    first = np.where(has_fail, fail.argmax(1), 15)  # 0-based slot index
    tau = first + 1.0
    idx = np.minimum(first, 14)
    take = lambda key: out[key][np.arange(n), idx]  # noqa: E731
    kind = np.where(~has_fail, "full",
                    np.where(take("valid"), "miss", "truncated"))
    # Next failing slot after the first miss (for the fix gain).
    after = fail.copy()
    after[np.arange(n), idx] = False
    after[np.arange(15)[None, :] <= idx[:, None]] = False
    has_next = after.any(1)
    nxt = np.where(has_next, after.argmax(1), 15)
    fix_gain = np.where(kind == "miss", nxt - first, 0.0)
    # On-path: data token equals the argmax label at every slot before the miss.
    dm = out["dm"].astype(bool)
    before = np.arange(15)[None, :] < idx[:, None]
    onpath = np.where(before, dm, True).all(1)
    return {
        "row": out["row"][:, 0], "block": out["block"][:, 0],
        "anchor_rel": out["anchor_rel"][:, 0], "kind": kind, "first": first + 1,
        "tau": tau, "fix_gain": fix_gain, "onpath": onpath,
        "p_top": take("p_top").astype(np.float32), "ent": take("ent").astype(np.float32),
        "rank": take("rank"), "p_d_tgt": take("p_d_tgt").astype(np.float32),
        "p_d_top": take("p_d_top").astype(np.float32),
        "t_p_pred": take("t_p_pred").astype(np.float32),
        "t_rank_pred": take("t_rank_pred"),
        "slot_correct": out["correct"], "slot_valid": out["valid"],
        "slot_p_top": out["p_top"].astype(np.float32), "slot_dm": dm,
        "slot_rank": out["rank"],
    }


def bin_label(value, bins):
    for lo, hi, label in bins:
        if lo <= value < hi or (lo == hi == value):
            return label
    raise ValueError(value)


def labels(values, bins):
    out = np.empty(len(values), dtype=object)
    for lo, hi, label in bins:
        if lo == hi:
            out[values == lo] = label
        else:
            out[(values >= lo) & (values < hi)] = label
    return out


def classify(b):
    """Miss classes at the first genuine miss."""
    miss = b["kind"] == "miss"
    conf = labels(b["p_top"], CONF_BINS)
    rank = labels(b["rank"], ((1, 1, "1"), (2, 5, "2-5"), (6, 10**9, ">5")))
    # rank 1 at a miss = an exact bf16 logit tie the argmax broke the other way.
    near = b["rank"] <= 5
    addressable = miss & (b["p_top"] >= 0.5) & near & b["onpath"]
    strict = miss & (b["p_top"] >= 0.9) & near & b["onpath"]
    return {"miss": miss, "conf": conf, "rank": rank, "near": near,
            "addressable": addressable, "strict": strict}


def summary(b, rows, mask=None):
    """Error decomposition of a set of blocks."""
    if mask is None:
        mask = np.ones(len(b["tau"]), bool)
    n = int(mask.sum())
    cls = classify(b)
    miss = cls["miss"] & mask
    m = int(miss.sum())
    out = {
        "blocks": n,
        "tau": float(b["tau"][mask].mean()),
        "share_full": float((b["kind"][mask] == "full").mean()),
        "share_truncated": float((b["kind"][mask] == "truncated").mean()),
        "share_miss": m / n,
        "misses": m,
    }
    if m == 0:
        return out
    gain = b["fix_gain"]
    out["fix_all_first_misses_tau_gain"] = float(gain[miss].sum() / n)
    out["onpath_share"] = float(b["onpath"][miss].mean())
    out["fix_gain_onpath"] = float(gain[miss & b["onpath"]].sum() / n)
    out["fix_gain_offpath"] = float(gain[miss & ~b["onpath"]].sum() / n)
    out["rank"] = {
        label: float(((b["rank"] >= lo) & (b["rank"] <= hi))[miss].mean())
        for lo, hi, label in RANK_BINS
    }
    tie = miss & (b["rank"] == 1)
    out["tie"] = {"share": float(tie.sum() / m), "tau_gain": float(gain[tie].sum() / n)}
    out["confidence"] = {
        label: float(((b["p_top"] >= lo) & (b["p_top"] < hi))[miss].mean())
        for lo, hi, label in CONF_BINS
    }
    grid = {}
    for clo, chi, clabel in CONF_BINS:
        for rlabel, rsel in (("1-5", cls["near"]), (">5", b["rank"] > 5)):
            sel = miss & (b["p_top"] >= clo) & (b["p_top"] < chi) & rsel
            grid[f"{clabel}|{rlabel}"] = {
                "share": float(sel.sum() / m),
                "tau_gain": float(gain[sel].sum() / n),
                "onpath_share": float(b["onpath"][sel].mean()) if sel.any() else None,
            }
    out["grid"] = grid
    for name in ("addressable", "strict"):
        sel = cls[name] & mask
        out[name] = {"share": float(sel.sum() / m), "tau_gain": float(gain[sel].sum() / n)}
    uncertain = miss & (b["p_top"] < 0.5)
    out["target_uncertain"] = {"share": float(uncertain.sum() / m),
                               "tau_gain": float(gain[uncertain].sum() / n)}
    offpath = miss & ~b["onpath"]
    out["offpath"] = {"share": float(offpath.sum() / m),
                      "tau_gain": float(gain[offpath].sum() / n)}
    out["median"] = {
        key: float(np.median(b[key][miss]))
        for key in ("p_top", "ent", "p_d_tgt", "p_d_top", "t_p_pred")
    }
    out["drafter_pick_is_target_second"] = float((b["t_rank_pred"][miss] == 2).mean())
    out["drafter_pick_target_prob_ge_0.1"] = float((b["t_p_pred"][miss] >= 0.1).mean())
    out["first_miss_slot"] = {
        str(s): float((b["first"][miss] == s).mean()) for s in range(1, 16)
    }
    return out


def slot_level(b, mask):
    """Accuracy of all valid slots by target-confidence bin and slot."""
    valid = b["slot_valid"] & mask[:, None]
    correct = b["slot_correct"]
    out = {"by_conf": {}, "by_slot": {}, "by_slot_onpath": {}}
    for lo, hi, label in CONF_BINS:
        sel = valid & (b["slot_p_top"] >= lo) & (b["slot_p_top"] < hi)
        out["by_conf"][label] = {"share": float(sel.sum() / valid.sum()),
                                 "acc": float(correct[sel].mean())}
    # Cumulative on-path indicator: data path greedy before each slot.
    dm = b["slot_dm"]
    prior = np.concatenate([np.ones((len(dm), 1), bool),
                            np.cumprod(dm, 1)[:, :-1].astype(bool)], 1)
    for s in range(15):
        sel = valid[:, s]
        out["by_slot"][str(s + 1)] = float(correct[sel, s].mean())
        sel_on = sel & prior[:, s]
        out["by_slot_onpath"][str(s + 1)] = {
            "acc": float(correct[sel_on, s].mean()),
            "share": float(sel_on.sum() / sel.sum()),
        }
    return out


def transitions(ref, other, rows_cat):
    """How the other arm changes the reference arm's blocks (same anchors)."""
    if not (np.array_equal(ref["row"], other["row"])
            and np.array_equal(ref["block"], other["block"])):
        raise ValueError("Arms have different blocks")
    cls = classify(ref)
    delta = other["tau"] - ref["tau"]
    n = len(delta)
    out = {"delta_tau": float(delta.mean()),
           "blocks_improved": float((delta > 0).mean()),
           "blocks_worse": float((delta < 0).mean())}
    groups = {
        "full": ref["kind"] == "full",
        "truncated": ref["kind"] == "truncated",
    }
    for clo, chi, clabel in CONF_BINS:
        for rlabel, rsel in (("1-5", cls["near"]), (">5", ref["rank"] > 5)):
            groups[f"miss|{clabel}|{rlabel}"] = (
                cls["miss"] & (ref["p_top"] >= clo) & (ref["p_top"] < chi) & rsel)
    groups["miss|offpath"] = cls["miss"] & ~ref["onpath"]
    groups["miss|onpath"] = cls["miss"] & ref["onpath"]
    for name, sel in groups.items():
        if not sel.any():
            continue
        fixed = sel & (other["first"] > ref["first"])
        broke = sel & (other["first"] < ref["first"])
        out[name] = {
            "blocks_share": float(sel.mean()),
            "fixed_rate": float(fixed.sum() / sel.sum()),
            "broken_rate": float(broke.sum() / sel.sum()),
            "delta_tau_contribution": float(delta[sel].sum() / n),
        }
    # Where the other arm loses: blocks the reference accepted further.
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    split_dir = C.OUT / "slots" / args.split
    rows = json.loads((split_dir / "rows.json").read_text())
    repro = json.loads((split_dir / "reproduction.json").read_text())
    row_cat = np.array([r["category"] for r in rows])
    row_subset = np.array([r["subset"] for r in rows])
    result = {"split": args.split, "arms": {}}
    per_arm = {}
    for arm in ARMS:
        b = blocks(load(args.split, arm))
        per_arm[arm] = b
        cat = row_cat[b["row"]]
        # Check: pooled block τ per subset equals the metric counts' τ.
        counts = repro[arm]["counts"]
        worst = 0.0
        for subset, c in counts.items():
            sel = row_subset[b["row"]] == subset
            worst = max(worst, abs(b["tau"][sel].mean() - c["eal_sum"] / c["eal_total"]))
        if worst > 1e-6:
            raise ValueError(f"{arm}: block τ disagrees with the metric counts ({worst})")
        entry = {"tau_check_max_abs_diff": worst,
                 "all_blocks": summary(b, rows),
                 "slot_level": slot_level(b, np.ones(len(cat), bool)),
                 "categories": {}, "anchor_bins": {}}
        for c in C.CATEGORIES:
            entry["categories"][c] = summary(b, rows, cat == c)
            entry["categories"][c]["slot_level"] = slot_level(b, cat == c)
        for lo, hi, label in ANCHOR_BINS:
            sel = (b["anchor_rel"] >= lo) & (b["anchor_rel"] < hi)
            entry["anchor_bins"][label] = summary(b, rows, sel)
        # Category-mean τ (as reported: mean over categories of pooled τ).
        entry["tau_category_mean_blocks"] = float(np.mean([
            entry["categories"][c]["tau"] for c in C.CATEGORIES]))
        result["arms"][arm] = entry
    result["transitions"] = {}
    for ref, other in (("base", "static"), ("base", "generator"), ("static", "generator")):
        key = f"{ref}->{other}"
        result["transitions"][key] = {"all": transitions(per_arm[ref], per_arm[other], rows)}
        for c in C.CATEGORIES:
            sel = row_cat[per_arm[ref]["row"]] == c
            sub = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v)
                   for k, v in per_arm[ref].items()}
            sub_o = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v)
                     for k, v in per_arm[other].items()}
            result["transitions"][key][c] = transitions(sub, sub_o, rows)
    out = C.OUT / "results" / f"errors_{args.split}.json"
    C.write_json(out, result)
    for arm in ARMS:
        a = result["arms"][arm]["all_blocks"]
        print(arm, round(a["tau"], 4), "miss", round(a["share_miss"], 3),
              "addressable", a["addressable"], "uncertain", a["target_uncertain"],
              "offpath", a["offpath"], "fixall", a["fix_all_first_misses_tau_gain"])


if __name__ == "__main__":
    main()
