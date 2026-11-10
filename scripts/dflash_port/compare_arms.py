# ruff: noqa: E501  (Markdown table templates)
"""Paired comparison of evaluated arms on the same rows, with bootstrap CIs.

    python scripts/dflash_port/compare_arms.py --split test \\
        --arm base=RUN_DIR[:file-suffix] --arm static=RUN_DIR --arm generator=RUN_DIR \\
        [--variant coef-1_donor-correct] [--reference base] [--resamples 10000]

Reads the per-row counts every offline evaluation stores (`cohorts[].counts`:
eal_sum / eal_total / full_acc_sum / full_acc_total, one row per cohort) and
prints Markdown tables: per arm and category, full_acc, τ (accepted length per
block, including the verifier's bonus token), accepted draft tokens per block
(τ - 1) and acceptance rate ((τ - 1) / draft slots); the total accepted draft
tokens over the split; and, for every arm against the reference, the paired
difference in τ and full_acc with a 95% bootstrap interval (rows resampled
with replacement, stratified by category, the same resample for both arms).
Within a category, numbers are pooled over draft blocks. The "all" column is
the macro average over categories (each category weighs equally), matching the
headline `score` of the offline-eval records; the token totals are pooled
counts over every block of the split.

`--split validation` reads the main offline-eval record: the base arm's
validation, or a trained arm's `--variant` ablation (default
coef-2_donor-correct: the training scale). test / ood read `<label>.<split>.json`.
"""

import argparse
import json
import random
from pathlib import Path

DRAFT_SLOTS = 15  # block 16, anchor excluded
CATEGORIES = ("math", "code", "stem", "chat")


def _category(subset):
    return subset.rsplit("-", 1)[0]


def load_rows(run_dir, split, variant):
    """{primary_id: (category, counts)} for one arm."""
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("offline_eval/*/*/*.json"))
    if split == "validation":
        base = [f for f in files if f.name.count(".") == 1]
        record = json.loads(base[-1].read_text())
        result = (
            record["validation"]
            if "validation" in record
            else record["ablations"][variant]
        )
    else:
        chosen = [f for f in files if f.name.endswith(f".{split}.json")]
        result = json.loads(chosen[-1].read_text())[split]
    rows = {}
    for subset, data in result["subsets"].items():
        for cohort in data["cohorts"]:
            (pid,) = cohort["primary_ids"]
            rows[str(pid)] = (_category(subset), cohort["counts"])
    return rows


def pooled(rows, keys):
    s = dict.fromkeys(("eal_sum", "eal_total", "full_acc_sum", "full_acc_total"), 0.0)
    for key in keys:
        for k in s:
            s[k] += rows[key][1][k]
    return s


def stats(s):
    tau = s["eal_sum"] / s["eal_total"]
    return {
        "full_acc": s["full_acc_sum"] / s["full_acc_total"],
        "tau": tau,
        "accepted": tau - 1,
        "rate": (tau - 1) / DRAFT_SLOTS,
        "blocks": s["eal_total"],
        "accepted_total": s["eal_sum"] - s["eal_total"],
    }


def macro(rows, groups):
    """Mean over category groups of each group's pooled metrics."""
    per = [stats(pooled(rows, g)) for g in groups]
    return {
        m: sum(p[m] for p in per) / len(per)
        for m in ("full_acc", "tau", "accepted", "rate")
    }


def bootstrap(arm, ref, groups, resamples, seed):
    rng = random.Random(seed)
    deltas = {"tau": [], "full_acc": []}
    for _ in range(resamples):
        sample = [rng.choices(group, k=len(group)) for group in groups]
        a, b = macro(arm, sample), macro(ref, sample)
        for m, values in deltas.items():
            values.append(a[m] - b[m])
    out = {}
    for m, values in deltas.items():
        values.sort()
        out[m] = (
            values[int(0.025 * len(values))],
            values[int(0.975 * len(values)) - 1],
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test", "ood"))
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--variant", default="coef-2_donor-correct")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    arms = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        variant = args.variant
        if "@" in path:
            path, variant = path.split("@", 1)
        arms[label] = load_rows(path, args.split, variant)
    names = list(arms)
    keys = sorted(arms[names[0]])
    for label, rows in arms.items():
        if sorted(rows) != keys:
            raise ValueError(f"{label} was evaluated on different rows")
    categories = [c for c in CATEGORIES if any(arms[names[0]][k][0] == c for k in keys)]
    categories += sorted({arms[names[0]][k][0] for k in keys} - set(categories))
    by_cat = {c: [k for k in keys if arms[names[0]][k][0] == c] for c in categories}
    print(f"**{args.split}** ({len(keys)} rows)\n")
    print("| Arm | | " + " | ".join(categories) + " | all (category mean) |")
    print("|---|---|" + "---|" * (len(categories) + 1))
    for label, rows in arms.items():
        cells = {c: stats(pooled(rows, by_cat[c])) for c in categories}
        cells["all"] = macro(rows, list(by_cat.values()))
        cols = [*categories, "all"]
        print(
            f"| {label} | full_acc | "
            + " | ".join(f"{cells[c]['full_acc']:.4f}" for c in cols)
            + " |"
        )
        print("| | τ | " + " | ".join(f"{cells[c]['tau']:.3f}" for c in cols) + " |")
        print(
            "| | accepted draft tokens / block | "
            + " | ".join(f"{cells[c]['accepted']:.2f} of {DRAFT_SLOTS}" for c in cols)
            + " |"
        )
        print(
            "| | acceptance rate | "
            + " | ".join(f"{cells[c]['rate']:.1%}" for c in cols)
            + " |"
        )
    print()
    print("| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |")
    print("|---|---|---|---|")
    for label, rows in arms.items():
        s = stats(pooled(rows, keys))
        print(
            f"| {label} | {s['blocks']:,.0f} | {s['accepted_total']:,.0f} | {100 * s['accepted']:.0f} |"
        )
    reference = args.reference or names[0]
    print()
    print(
        f"Paired differences vs **{reference}** (all = category mean; 95% bootstrap interval, {args.resamples:,} resamples of rows, stratified by category):\n"
    )
    print(
        "| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |"
    )
    print("|---|---|---|---|---|---|---|")
    for label in names:
        if label == reference:
            continue
        for scope, groups in [
            ("all", list(by_cat.values())),
            *((c, [by_cat[c]]) for c in categories),
        ]:
            a, b = macro(arms[label], groups), macro(arms[reference], groups)
            ci = bootstrap(
                arms[label], arms[reference], groups, args.resamples, args.seed
            )
            print(
                f"| {label} | {scope} | {a['tau'] - b['tau']:+.3f} | [{ci['tau'][0]:+.3f}, {ci['tau'][1]:+.3f}] "
                f"| {a['full_acc'] - b['full_acc']:+.4f} | [{ci['full_acc'][0]:+.4f}, {ci['full_acc'][1]:+.4f}] "
                f"| {100 * (a['accepted'] - b['accepted']):+.1f} |"
            )
    print()


if __name__ == "__main__":
    main()
