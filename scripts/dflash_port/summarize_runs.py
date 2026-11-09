# ruff: noqa: E501  (Markdown table templates)
"""Markdown tables of DFlash run results, for `reporting/dflash/` (read-only).

    python scripts/dflash_port/summarize_runs.py RUN_DIR [RUN_DIR ...] [--base BASE_DIR]

Per run: validation history (score = full_acc, τ = eal, per category), the
offline evaluation (reproduction, M4 coefficients, M7 donors, K=4 view, the
M4 per-row oracle, D13 similarity, D14 cost) and test / OOD results when
present. With `--base`: the base drafter's validation, test and OOD rows.
Every number is read from the run's JSON outputs; nothing is recomputed
except τ means per category, which are read per subset.
"""

import argparse
import json
from pathlib import Path

CATEGORIES = ("math", "code", "stem", "chat")


def _subset_category(name):
    return name.rsplit("-", 1)[0]


def per_category(result):
    """{category: (full_acc, eal)} from an evaluation result (one context size)."""
    out = {}
    for name, subset in result["subsets"].items():
        (context,) = subset["contexts"].values()
        out[_subset_category(name)] = (context["full_acc"], context["eal"])
    return out


def overall(result):
    (context,) = result["contexts"].values()
    return result["score"], context["eal"]


def row(label, result):
    score, eal = overall(result)
    cats = per_category(result)
    names = [c for c in CATEGORIES if c in cats] + sorted(set(cats) - set(CATEGORIES))
    cells = " | ".join(f"{cats[c][0]:.4f} / {cats[c][1]:.2f}" for c in names)
    return f"| {label} | {score:.4f} | {eal:.3f} | {cells} |", names


def table(rows, first="Arm"):
    lines, header = [], None
    for label, result in rows:
        line, names = row(label, result)
        if header is None:
            header = names
            lines.append(
                f"| {first} | score | τ | "
                + " | ".join(f"{n} (acc / τ)" for n in names)
                + " |"
            )
            lines.append("|---|---|---|" + "---|" * len(names))
        lines.append(line)
    return "\n".join(lines)


def _find(run_dir, suffix):
    found = sorted(run_dir.glob(f"offline_eval/*/*/*{suffix}"))
    return json.loads(found[-1].read_text()) if found else None


def summarize_run(run_dir):
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "run_record_train.json").read_text())
    job = record["job"]
    result = json.loads(
        (run_dir / job["stage"] / job["label"] / "result.json").read_text()
    )
    out = [f"### {job['label']}", ""]
    out.append(
        f"Arm `{job['arm']}`, {result['completed_updates']} updates, git "
        f"`{record['git_sha'][:10]}`, trainable parameters "
        f"{result.get('trainable_parameters', 'n/a'):,}."
        if isinstance(result.get("trainable_parameters"), int)
        else f"Arm `{job['arm']}`, {result['completed_updates']} updates, git "
        f"`{record['git_sha'][:10]}`."
    )
    out += ["", "**Validation during training** (200 fixed rows):", ""]
    out.append(
        table([(f"update {h['update']}", h) for h in result["history"]], "Update")
    )
    evaluation = _find(run_dir, f"{job['label']}.json")
    if evaluation is not None:
        rep = evaluation["reproduction"]
        out += [
            "",
            f"**Offline reproduction:** {rep['score']:.6f} vs training "
            f"{rep['training_validation_score']:.6f} "
            f"({'bit-exact' if rep['reproduces_training_validation'] else 'not bitwise'}).",
            "",
            "**Offline variants** (M4 coefficient × M7 donor; `_k-4`: 4 rows per LoRA):",
            "",
        ]
        abl = evaluation["ablations"]
        out.append(table([(name, abl[name]) for name in abl], "Variant"))
        if "m4_per_row" in evaluation:
            out += [
                "",
                "**M4 per-row τ** (best fixed coefficient vs per-row oracle):",
                "",
            ]
            out.append(
                "| Category | best fixed s | τ at best fixed | per-row oracle τ | oracle gain | rows best at each s |"
            )
            out.append("|---|---|---|---|---|---|")
            for cat, info in evaluation["m4_per_row"]["categories"].items():
                best = info["best_fixed"]
                counts = ", ".join(f"{s}: {n}" for s, n in info["best_counts"].items())
                out.append(
                    f"| {cat} | {best} | {info['row_mean'][best]:.3f} | "
                    f"{info['oracle']:.3f} | +{info['oracle_gain_over_best_fixed']:.3f} | {counts} |"
                )
        if "similarity" in evaluation:
            s = evaluation["similarity"]["all"]
            out += [
                "",
                "**D13 similarity of the generated ΔW** (all modules pooled): "
                f"{s['samples']} LoRAs; cos within {s['cos_in']:.4f}, across "
                f"{s['cos_cross']:.4f} (gap {s['cos_gap']:.4f}); dyn_ratio "
                f"{s['dyn_ratio']:.4f}; η² {s['eta2']:.3f}; effective rank "
                f"{s['effective_rank']:.2f}; centroid accuracy {s['centroid_accuracy']:.3f}.",
            ]
        if "cost" in evaluation:
            c = evaluation["cost"]
            out += [
                "",
                f"**D14 cost** ({c['device']}): {c['generator_parameters']:,} generator "
                f"parameters; {c['adapted_modules']} modules, rank {c['rank']}; "
                f"factors {c['factor_bytes'] / 1e6:.1f} MB per LoRA; "
                f"{c['ms_per_call']:.2f} ms per call; {c['ms_per_batched_call']:.2f} ms "
                f"for {c['batched_cohorts']} batched; merge {c['ms_merge']:.2f} ms.",
            ]
    for split in ("test", "ood"):
        data = _find(run_dir, f"{job['label']}.{split}.json")
        if data is not None:
            out += ["", f"**{split}:**", "", table([(job["label"], data[split])])]
    return "\n".join(out) + "\n"


def summarize_base(base_dir):
    base_dir = Path(base_dir)
    parts = ["### Base drafter", ""]
    for split in ("validation", "test", "ood"):
        name = "m0-base.json" if split == "validation" else f"m0-base.{split}.json"
        found = sorted(base_dir.glob(f"offline_eval/*/base/*{name}"))
        if found:  # one table per split: the OOD set has its own columns
            data = json.loads(found[-1].read_text())
            parts += [f"**{split}:**", "", table([("base", data[split])]), ""]
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args(argv)
    parts = [summarize_base(args.base)] if args.base else []
    parts += [summarize_run(run) for run in args.runs]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
