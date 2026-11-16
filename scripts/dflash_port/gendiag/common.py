"""Shared helpers for the GENDIAG offline analyses (2026-11-16).

`with_trained(recipe, tier, fn)` runs `fn(runtime, job, output, generator)`
inside the read-only evaluation lifecycle of a trained generator run
(`campaign.run_one(..., evaluate=)`, as `run_arm --mode eval`); the run's
identity is checked against its configuration. Nothing is written to the run.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
RUNS = Path("/workspace/yusuf/runs/dflash")
OUT = RUNS / "gendiag"
sys.path.insert(0, str(REPO / "scripts/dflash_port"))


def category(subset: str) -> str:
    return subset.rsplit("-", 1)[0]


def with_trained(recipe, tier, fn, *, snapshot=None):
    from run_arm import resolve  # noqa: PLC0415

    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.offline_eval import _load_trained  # noqa: PLC0415
    from speculators.generator.structure_study import training_job  # noqa: PLC0415

    os.chdir(REPO)
    resolved = resolve(
        Path(recipe).resolve(), tier=tier, output_root=RUNS, wandb=False,
        store_mode="read", hub=False, materialize=True,
    )
    cfg = resolved.cfg
    job = training_job(cfg)

    def run(runtime, job, output):
        generator, checkpoint = _load_trained(runtime, job, output)
        if generator is None:
            raise ValueError("GENDIAG analyses need a generator run")
        if snapshot is not None:
            state = torch.load(snapshot, map_location="cpu", weights_only=False)["model"]
            generator.load_state_dict(state, strict=True)
        generator.eval()
        runtime.drafter.eval()
        return fn(runtime, job, output, generator)

    return campaign.run_one(cfg, job, evaluate=run)


def eval_rows(runtime, job, split, per_category=None, seed=0):
    """[(subset, row index)]: the validation rows, the test rows, or a seeded
    sample of `per_category` training rows per category."""
    corpus = runtime.corpus
    if split == "val":
        return [(s, i) for s in job["validation_subsets"] for i in corpus.indices(s, validation=True)]
    if split == "test":
        raise ValueError("test rows are kept for final numbers only")
    if split != "train":
        raise ValueError(split)
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for s in job["subsets"]:
        by_cat.setdefault(category(s), []).extend((s, i) for i in corpus.indices(s))
    rows = []
    for cat in sorted(by_cat):
        rows += rng.sample(by_cat[cat], per_category)
    return rows


def scales(runtime):
    return {k: runtime.drafter.get_submodule(p).scale for k, p in runtime.paths.items()}


@torch.no_grad()
def row_counts(runtime, row, factors):
    """Summed metric counts of one row (one LoRA) through the production forward."""
    out: dict[str, float] = {}
    for batch, _ in runtime.corpus.microbatches([row], runtime.device, runtime.dtype):
        _, _, metrics = runtime.forward(batch, factors)
        for k, v in metrics.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def headline(records, rows, key):
    """Category mean of subset-pooled `key` (as campaign_train.evaluate)."""
    pooled: dict[str, list[float]] = {}
    for (subset, _), counts in zip(rows, records, strict=True):
        p = pooled.setdefault(subset, [0.0, 0.0])
        p[0] += counts[f"{key}_sum"]
        p[1] += counts[f"{key}_total"]
    by_cat: dict[str, list[float]] = {}
    for subset, (s, t) in pooled.items():
        by_cat.setdefault(category(subset), []).append(s / t)
    cats = {c: sum(v) / len(v) for c, v in by_cat.items()}
    return sum(cats.values()) / len(cats), cats


def paired_delta(rec_a, rec_b, rows, key="eal", draws=2000, seed=17):
    """Point Δ = headline(b) − headline(a) and a 95% bootstrap CI over rows,
    resampled within subset."""
    point = headline(rec_b, rows, key)[0] - headline(rec_a, rows, key)[0]
    rng = random.Random(seed)
    by_subset: dict[str, list[int]] = {}
    for idx, (subset, _) in enumerate(rows):
        by_subset.setdefault(subset, []).append(idx)
    deltas = []
    for _ in range(draws):
        pick = [i for members in by_subset.values() for i in (rng.choice(members) for _ in members)]
        sub_rows = [rows[i] for i in pick]
        deltas.append(
            headline([rec_b[i] for i in pick], sub_rows, key)[0]
            - headline([rec_a[i] for i in pick], sub_rows, key)[0]
        )
    deltas.sort()
    return {"delta": point, "lo": deltas[int(0.025 * draws)], "hi": deltas[int(0.975 * draws) - 1]}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else float(v)))
