"""Offline evaluation tools for trained generator / static-LoRA arms (S6c).

- `Variant`: inference-time changes applied inside the ordinary validation
  loop (`campaign_train.evaluate(..., variant=)`):
  - `coefficient` (M4): the LoRA scale used at inference, replacing the
    training scale alpha/rank. Default grid `DEFAULT_COEFFICIENTS`; 0 (not in
    the grid) reproduces the frozen base drafter and is used as a test check.
  - `donor` (M7): which rows condition the generator: `correct` (the
    evaluated cohort, as in training validation), `fixed` (one fixed set of
    rows for every cohort and subset: a single generated LoRA),
    `cross_category` (rows of the next category, cyclic in sorted order).
    Only with `allow_same_category` (user 2026-11-07: flagged for later):
    `same_category`, other rows of the same category.
    Donor rows come from the job's unseen donor subsets (`donor_subsets`,
    user 2026-11-09: never trained on, never evaluated). Jobs without them
    (EAGLE-3 protocol) use validation rows (never the cohort's own) and, for
    `fixed`, training rows. Donor draws are deterministic (seeded by cohort
    position); donor sets have the cohort's size.
  - `group_size`: rows sharing one generated LoRA, overriding the job's
    validation group size (M4 reports K=4 means next to K=1).
- `similarity` (D13): is the generator dynamic? Exact Frobenius geometry of
  ΔW = s·B·A over many contexts, via Gram matrices (no ΔW is materialised).
- `generator_cost` (D14): generator latency, parameters, factor bytes and the
  time to merge ΔW into the base weights.

Metric definitions (ours; the EAGLE-3 GEN-005 script is not in the repo):
- `G[i, j] = <ΔW_i, ΔW_j>_F = s² · tr((B_iᵀ B_j)(A_j A_iᵀ))`, per module;
  the pooled ("all") Gram is the sum over modules (concatenated ΔW).
- `cos_in` / `cos_cross`: mean cosine over distinct pairs of the same /
  different categories; `cos_gap = cos_in - cos_cross`.
- `dyn_ratio`: Σ‖ΔW_i - mean‖² / Σ‖ΔW_i‖², in [0, 1]; 0 for a static LoRA.
- `eta2`: between-category share of that variance (Σ_c n_c‖μ_c - μ‖² / Σ‖ΔW_i - μ‖²).
- `effective_rank`: participation ratio (Σλ)² / Σλ² of the centred Gram.
- `centroid_accuracy`: leave-one-out nearest-centroid category accuracy.
"""

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from speculators.generator.campaign_train import (
    conditioning_subset,
    groups,
    validation_group_size,
)
from speculators.generator.model import AdapterLinear

DONORS = ("correct", "fixed", "same_category", "cross_category")
# M7 controls run by default (user 2026-11-07, option C).
DEFAULT_DONORS = ("correct", "fixed", "cross_category")
# M4 grid (user 2026-11-07); the reference recipe trains at alpha/rank = 2.
DEFAULT_COEFFICIENTS = (0.5, 1.0, 2.0, 4.0, 8.0)
# M4 continuity: K=4 category means next to the K=1 (one LoRA per row)
# sweep, as the EAGLE-3 validation convention (user 2026-11-09).
CONTINUITY_GROUP_SIZE = 4
# Per-row M4 table metric: accept length (eal, the headline metric).
PER_ROW_METRIC = "eal"


@contextmanager
def _conditioning_bound(generator, size):
    """Let a generator trained on at most `max_examples` conditioning rows read
    `size` rows for one evaluation (M4 K=4 continuity of a K=1 generator).
    The example encoder is a set encoder (mean, std and log-count pooling, no
    per-slot parameters), so `max_examples` is only an input bound; the rows
    are still outside the trained K, which the result records."""
    trained = generator.max_examples
    generator.max_examples = max(trained, size)
    try:
        yield
    finally:
        generator.max_examples = trained


def _category(subset: str) -> str:
    return subset.rsplit("-", 1)[0]


@dataclass(frozen=True)
class Variant:
    coefficient: float | None = None
    donor: Literal["correct", "fixed", "same_category", "cross_category"] = "correct"
    donor_seed: int = 0
    subsets: tuple[str, ...] = ()  # the evaluated subsets (donor pool)
    allow_same_category: bool = False
    rows: frozenset[int] | None = None  # evaluate only these rows
    donor_subsets: tuple[str, ...] = ()  # unseen donor pool (validation splits)
    group_size: int | None = None  # rows per generated LoRA (None: the job's)

    def __post_init__(self):
        if self.donor not in DONORS:
            raise ValueError(f"Unknown donor {self.donor!r}")
        if self.coefficient is not None and self.coefficient < 0:
            raise ValueError("coefficient must be nonnegative")
        if self.donor == "same_category" and not self.allow_same_category:
            raise ValueError(
                "same_category donors are off by default; set allow_same_category"
            )
        if self.donor != "correct" and not self.subsets:
            raise ValueError("Donor variants need the evaluated subsets")
        if self.group_size is not None and self.group_size < 1:
            raise ValueError("group_size must be positive")

    @property
    def label(self) -> str:
        coefficient = "train" if self.coefficient is None else f"{self.coefficient:g}"
        label = f"coef-{coefficient}_donor-{self.donor}"
        if self.group_size is not None:
            label += f"_k-{self.group_size}"
        return label

    def identity(self) -> dict:
        return {
            "coefficient": self.coefficient,
            "donor": self.donor,
            "donor_seed": self.donor_seed,
            "subsets": list(self.subsets),
            "rows": None if self.rows is None else sorted(self.rows),
            "donor_subsets": list(self.donor_subsets),
            "group_size": self.group_size,
        }

    def select(self, indices):
        """The evaluated rows of one subset (all, or the ablation set's)."""
        if self.rows is None:
            return indices
        return [index for index in indices if index in self.rows]

    def donor_pool(self, runtime, subset, rows):
        """Candidate donor rows of the same category (`same_category`) or the
        next category (`cross_category`, sorted order, cyclic): the unseen
        donor subsets' rows when the job has them, else validation rows other
        than the cohort's own."""
        category = _category(subset)
        if self.donor == "same_category":
            donor = category
        else:
            categories = sorted({_category(s) for s in self.subsets})
            if len(categories) < 2:  # noqa: PLR2004
                raise ValueError("cross_category donors need two categories")
            donor = categories[(categories.index(category) + 1) % len(categories)]
        own = set(rows)
        names = self.donor_subsets or self.subsets
        return sorted(
            index
            for name in names
            if _category(name) == donor
            for index in runtime.corpus.indices(name, validation=True)
            if index not in own
        )

    def conditioning(self, runtime, subset, rows, position):
        if self.donor == "correct":
            return rows
        if self.donor == "fixed":
            # One LoRA for every cohort, from rows that are never evaluated:
            # the first rows (by primary id) of the first unseen donor subset
            # (user 2026-11-09), or, without donor subsets, training rows of
            # the first subset (user 2026-11-07). Group-size rows, so the
            # fixed set has the size of every cohort's own conditioning.
            size = validation_group_size(runtime, self)
            if self.donor_subsets:
                donor = min(self.donor_subsets)
                pool = runtime.corpus.indices(donor, validation=True)
            else:
                donor = min(self.subsets)
                pool = runtime.corpus.indices(donor, validation=False)
            evaluated = {
                index
                for name in self.subsets
                for index in runtime.corpus.indices(name, validation=True)
            }
            fixed = sorted(
                (index for index in pool if index not in evaluated),
                key=lambda i: runtime.corpus.rows[i]["primary_id"],
            )
            if len(fixed) < size:
                raise ValueError(f"{donor} has too few rows for a fixed donor")
            return fixed[:size]
        pool = self.donor_pool(runtime, subset, rows)
        if len(pool) < len(rows):
            raise ValueError("Donor pool has fewer rows than the conditioning set")
        seed = self.donor_seed * 1_000_003 + position
        return random.Random(seed).sample(pool, len(rows))

    def factors(self, runtime, factors):
        if self.coefficient is None:
            return factors
        if factors is None:  # static arm: the trained adapter factors
            factors = runtime.adapter_state()
        result = {}
        for name, (a, b) in factors.items():
            scale = runtime.drafter.get_submodule(runtime.paths[name]).scale
            result[name] = (a, b * (self.coefficient / scale))
        return result


@torch.no_grad()
def collect_factors(runtime, subsets, generator, *, conditioning=None):
    """Generated factors for the validation cohorts (the same cohorts and
    conditioning draws as validation). Returns [(subset, {key: (a, b)})]."""
    if generator is None:
        raise ValueError("collect_factors needs a generator")
    generator.eval()
    size = validation_group_size(runtime)
    samples = []
    for subset in subsets:
        indices = runtime.corpus.indices(subset, validation=True)
        cohorts = list(groups(indices, runtime.cfg.seed + 9000, size, size))
        for position, cohort in enumerate(cohorts):
            rows = conditioning_subset(
                cohort, conditioning, runtime.cfg.seed + 9000 + position
            )
            factors = generator(*runtime.corpus.context(rows, "last", runtime.device))
            samples.append(
                (
                    subset,
                    {k: (a.float(), b.float()) for k, (a, b) in factors.items()},
                )
            )
    return samples


def per_row_table(sweep, metric=PER_ROW_METRIC):
    """M4 per-row view (user 2026-11-09). `sweep` maps coefficient -> an
    `evaluate(..., per_cohort=True)` result with one row per cohort. Returns
    each row's metric for every coefficient and its best coefficient, and per
    category: the row-mean for every coefficient, the best fixed coefficient,
    and the per-row oracle (each row at its own best coefficient)."""
    coefficients = sorted(sweep)
    rows = {}
    for coefficient in coefficients:
        for subset, result in sweep[coefficient]["subsets"].items():
            for cohort in result["cohorts"]:
                if len(cohort["primary_ids"]) != 1:
                    raise ValueError("per_row_table needs one row per cohort")
                key = cohort["primary_ids"][0]
                entry = rows.setdefault(
                    key, {"category": _category(subset), "subset": subset, "by": {}}
                )
                entry["by"][f"{coefficient:g}"] = cohort["metrics"][metric]
    labels = [f"{c:g}" for c in coefficients]
    for key, entry in rows.items():
        if sorted(entry["by"]) != sorted(labels):
            raise ValueError(f"Row {key} is missing coefficients")
        best = max(entry["by"].values())
        entry["best"] = [c for c in labels if entry["by"][c] == best]
    categories = {}
    for category in sorted({entry["category"] for entry in rows.values()}):
        members = [e for e in rows.values() if e["category"] == category]
        means = {c: sum(e["by"][c] for e in members) / len(members) for c in labels}
        best_fixed = max(means, key=means.get)
        oracle = sum(max(e["by"].values()) for e in members) / len(members)
        categories[category] = {
            "rows": len(members),
            "row_mean": means,
            "best_fixed": best_fixed,
            "oracle": oracle,
            "oracle_gain_over_best_fixed": oracle - means[best_fixed],
            "best_counts": {c: sum(c in e["best"] for e in members) for c in labels},
        }
    return {
        "metric": metric,
        "coefficients": labels,
        "rows": rows,
        "categories": categories,
    }


def gram(factors_list, scale):
    """Exact Frobenius Gram of ΔW_i = scale·B_i·A_i (n×n, float64)."""
    a = torch.stack([pair[0] for pair in factors_list]).double()  # n, r, in
    b = torch.stack([pair[1] for pair in factors_list]).double()  # n, out, r
    bb = torch.einsum("ior,jos->ijrs", b, b)  # B_iᵀ B_j
    aa = torch.einsum("jsk,irk->ijsr", a, a)  # A_j A_iᵀ
    return scale**2 * torch.einsum("ijrs,ijsr->ij", bb, aa)


def geometry(g, labels):
    """Similarity metrics of one Gram matrix (see module docstring)."""
    n = len(labels)
    labels = list(labels)
    norms = g.diagonal().clamp_min(0)
    if n < 2 or bool((norms <= 0).any()):  # noqa: PLR2004
        raise ValueError("geometry needs >= 2 nonzero updates")
    cos = g / torch.sqrt(norms[:, None] * norms[None, :])
    same = torch.tensor([[x == y for y in labels] for x in labels])
    off = ~torch.eye(n, dtype=torch.bool)
    cats = sorted(set(labels))
    result = {
        "samples": n,
        "mean_norm": float(norms.sqrt().mean()),
        "cos_in": float(cos[same & off].mean()) if (same & off).any() else math.nan,
        "cos_cross": float(cos[~same].mean()) if (~same).any() else math.nan,
    }
    result["cos_gap"] = result["cos_in"] - result["cos_cross"]
    centring = torch.eye(n, dtype=g.dtype) - 1.0 / n
    centred = centring @ g @ centring
    total = float(centred.trace())
    result["dyn_ratio"] = total / float(norms.sum())
    member = torch.tensor([[x == c for x in labels] for c in cats], dtype=g.dtype)
    counts = member.sum(1)
    means = member / counts[:, None]  # category-mean weights (C×n)
    between = sum(
        float(counts[c]) * float((means[c] - 1.0 / n) @ g @ (means[c] - 1.0 / n))
        for c in range(len(cats))
    )
    result["eta2"] = between / total if total > 0 else math.nan
    eigen = torch.linalg.eigvalsh(centred).clamp_min(0)
    # A static set (all ΔW equal) has no variance: effective rank 0.
    result["effective_rank"] = (
        float(eigen.sum() ** 2 / (eigen**2).sum()) if total > 0 else 0.0
    )
    correct = 0
    for i in range(n):
        best, choice = math.inf, None
        for c, name in enumerate(cats):
            weights = member[c].clone()
            if labels[i] == name:
                weights[i] = 0
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            # ‖x_i - μ_c‖² = G_ii - 2 wᵀG_i + wᵀGw
            distance = float(g[i, i] - 2 * weights @ g[i] + weights @ g @ weights)
            if distance < best:
                best, choice = distance, name
        correct += choice == labels[i]
    result["centroid_accuracy"] = correct / n
    return result


def similarity(samples, scales):
    """D13 report: per-module and pooled geometry of generated ΔW."""
    labels = [_category(subset) for subset, _ in samples]
    keys = sorted(samples[0][1])
    grams = {
        key: gram([factors[key] for _, factors in samples], scales[key]) for key in keys
    }
    return {
        "categories": sorted(set(labels)),
        "all": geometry(sum(grams.values()), labels),
        "modules": {key: geometry(value, labels) for key, value in grams.items()},
    }


def _timed(runtime, function, repeats):
    def sync():
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)

    function()  # warm-up
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    sync()
    return (time.perf_counter() - started) * 1000 / repeats


@torch.no_grad()
def generator_cost(runtime, generator, contexts, *, repeats=20):
    """D14: ms per generator call (one cohort; all `contexts` batched), params,
    factor bytes and ΔW merge time (W + s·B·A for every adapted module)."""
    generator.eval()
    factors = generator(*contexts[0])
    scales = {
        key: runtime.drafter.get_submodule(path).scale
        for key, path in runtime.paths.items()
    }

    def merge():
        for key, (a, b) in factors.items():
            weight = runtime.drafter.get_submodule(runtime.paths[key]).base.weight
            torch.addmm(
                weight, b.to(weight.dtype), a.to(weight.dtype), alpha=scales[key]
            )

    return {
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "factor_bytes": sum(
            t.numel() * t.element_size() for pair in factors.values() for t in pair
        ),
        "adapted_modules": len(factors),
        "rank": next(iter(factors.values()))[0].shape[0],
        "ms_per_call": _timed(runtime, lambda: generator(*contexts[0]), repeats),
        "batched_cohorts": len(contexts),
        "ms_per_batched_call": _timed(
            runtime, lambda: generator.forward_cohorts(contexts), repeats
        ),
        "ms_merge": _timed(runtime, merge, repeats),
        "device": str(runtime.device),
        "repeats": repeats,
    }


def _load_trained(runtime, job, output):
    """The trained arm from `checkpoint.pt`: (generator or None, checkpoint)."""
    checkpoint = output / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    if job["arm"] == "fresh_lora":
        expected = {f"{path}.{x}" for path in runtime.paths.values() for x in "ab"}
        if set(state) != expected:
            raise ValueError(f"{checkpoint} does not hold this run's adapter factors")
        # The trained adapter dtype (campaign_train converts the factors to
        # job["adapter_dtype"] before training); evaluation must match it.
        dtype = getattr(torch, job.get("adapter_dtype", "float32"))
        for path in runtime.paths.values():
            adapter = runtime.drafter.get_submodule(path)
            for name in ("a", "b"):
                value = state[f"{path}.{name}"]
                if value.dtype != dtype:
                    raise ValueError(
                        f"{checkpoint}: {path}.{name} is {value.dtype}, "
                        f"the job trained {dtype}"
                    )
                setattr(
                    adapter,
                    name,
                    torch.nn.Parameter(
                        value.to(adapter.a.device).clone(), requires_grad=False
                    ),
                )
        return None, checkpoint
    generator = runtime.new_generator("last")
    generator.load_state_dict(state, strict=True)
    return generator, checkpoint


def campaign_job(cfg, label):
    """A campaign job by label (structure-study jobs come from
    `structure_study.training_job(cfg)`)."""
    from speculators.generator import campaign  # noqa: PLC0415

    matches = [job for job in campaign.jobs(cfg) if job["label"] == label]
    if len(matches) != 1:
        raise ValueError(f"No single campaign job labelled {label!r}")
    return matches[0]


SPLITS = ("validation", "test", "ood")


def _held_out(cfg, job, split, ood):
    """Subsets of a held-out split: the job's untouched test subsets, or the
    offline out-of-distribution set (`ood_set`, rows appended to the local
    inputs, never part of the training identity). Returns (subset names,
    local inputs or None, record fields)."""
    if split == "test":
        if not job.get("test_subsets"):
            raise ValueError("This job has no test subsets")
        return list(job["test_subsets"]), None, {}
    if ood is None:
        raise ValueError('split="ood" needs the OOD set directory')
    from speculators.generator.local_campaign import (  # noqa: PLC0415
        load_local_inputs,
    )
    from speculators.generator.ood_set import with_ood  # noqa: PLC0415

    inputs, manifest = with_ood(load_local_inputs(cfg), ood)
    fields = {
        "ood_set": {
            "path": str(Path(ood).resolve()),
            "manifest_sha256": hashlib.sha256(
                (Path(ood) / "manifest.json").read_bytes()
            ).hexdigest(),
            "dataset": manifest["dataset"],
            "selection": manifest["selection"],
            "sampling": manifest["sampling"],
        }
    }
    return list(manifest["subsets"]), inputs, fields


def _expose(runtime, inputs, names):
    """Make held-out OOD subsets visible to the (already identity-checked)
    evaluation corpus."""
    if inputs is None:
        return
    for name in names:
        runtime.corpus.subsets[name] = inputs.subsets[name]


def evaluate_job(  # noqa: C901 -- one record of every offline eval
    cfg,
    job,
    *,
    name,
    coefficients=DEFAULT_COEFFICIENTS,
    donors=DEFAULT_DONORS,
    allow_same_category=False,
    continuity_group_size=CONTINUITY_GROUP_SIZE,
    similarity_analysis=True,
    cost_cohorts=8,
    split="validation",
    ood=None,
):
    """Evaluate a completed training job offline (a campaign job, or the
    structure-study job used for matched arms, as EAGLE-3 GEN-009/010); writes
    `<output_root>/offline_eval/<name>/<stage>/<job label>[.test].json`.

    `split="validation"` (the job's fixed validation rows):
    1. Reproduction: the trained arm must reproduce training's final
       validation (checks the checkpoint).
    2. M4 coefficients x M7 donors (static arm: coefficients only), every
       evaluation with per-row records; the M4 per-row table; K=
       `continuity_group_size` category means for every coefficient (None:
       skipped); D13 similarity of the generated LoRAs.
    3. D14 cost of the generator (`cost_cohorts` batched cohorts).

    `split="test"`: the trained arm on the untouched test rows only
    (`job["test_subsets"]`, final numbers), with per-row records.
    `split="ood"`: the trained arm on the offline out-of-distribution set
    `ood` (`ood_set.py`), with per-row records.
    """
    # Imported here: campaign imports this module's dependencies.
    from speculators.bank.artifacts import file_digest, write_json  # noqa: PLC0415
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.campaign_train import evaluate  # noqa: PLC0415
    from speculators.provenance import (  # noqa: PLC0415
        find_repo_root,
        require_git_sha,
    )

    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    held_out, inputs, fields = (
        ([], None, {}) if split == "validation" else _held_out(cfg, job, split, ood)
    )
    sha = require_git_sha(find_repo_root(Path(__file__)))

    def run(runtime, job, output):
        generator, checkpoint = _load_trained(runtime, job, output)
        conditioning = job.get("training_conditioning_examples")
        record = {
            "git_sha": sha,
            "split": split,
            "job": job,
            "run": str(output.resolve()),
            "checkpoint_sha256": file_digest(checkpoint),
            "group_size": validation_group_size(runtime),
        }
        if split != "validation":
            _expose(runtime, inputs, held_out)
            record.update(fields)
            record[split] = evaluate(
                runtime, held_out, generator, conditioning, per_cohort=True
            )
            path = (
                cfg.output_root
                / "offline_eval"
                / name
                / job["stage"]
                / f"{job['label']}.{split}.json"
            )
            write_json(path, record)
            return record
        subsets = tuple(job["validation_subsets"])
        donor_subsets = tuple(job.get("donor_subsets", ()))
        trained = json.loads((output / "result.json").read_text())["history"][-1]
        full = evaluate(runtime, list(subsets), generator, conditioning)
        # Exact on CPU; reported (not enforced): GPU kernels may differ.
        record["reproduction"] = {
            "score": full["score"],
            "training_validation_score": trained["score"],
            "reproduces_training_validation": full["score"] == trained["score"],
            "validation_examples": full["validation_examples"],
        }
        ablations, sweep = {}, {}
        for coefficient in coefficients:
            for donor in donors:
                if generator is None and donor != "correct":
                    continue  # a static LoRA has no conditioning
                variant = Variant(
                    coefficient,
                    donor,
                    subsets=subsets,
                    allow_same_category=allow_same_category,
                    donor_subsets=donor_subsets,
                )
                result = evaluate(
                    runtime,
                    list(subsets),
                    generator,
                    conditioning,
                    variant,
                    per_cohort=True,
                )
                ablations[variant.label] = {"variant": variant.identity(), **result}
                if donor == "correct":
                    sweep[coefficient] = result
            if (
                continuity_group_size is not None
                and continuity_group_size != record["group_size"]
            ):
                variant = Variant(coefficient, group_size=continuity_group_size)
                beyond = (
                    generator is not None
                    and continuity_group_size > generator.max_examples
                )
                with (
                    _conditioning_bound(generator, continuity_group_size)
                    if beyond
                    else nullcontext()
                ):
                    result = evaluate(
                        runtime, list(subsets), generator, conditioning, variant
                    )
                ablations[variant.label] = {
                    "variant": variant.identity(),
                    # True: more conditioning rows than the generator trained on.
                    "beyond_trained_examples": beyond,
                    **result,
                }
        record["ablations"] = ablations
        if record["group_size"] == 1 and sweep:
            record["m4_per_row"] = per_row_table(sweep)
        scales = {
            key: runtime.drafter.get_submodule(path).scale
            for key, path in runtime.paths.items()
        }
        if similarity_analysis and generator is not None:
            samples = collect_factors(
                runtime, subsets, generator, conditioning=conditioning
            )
            record["similarity"] = similarity(samples, scales)
        if cost_cohorts and generator is not None:
            # Timing only: any validation cohorts serve.
            size = record["group_size"]
            indices = runtime.corpus.indices(subsets[0], validation=True)
            cohorts = list(groups(indices, runtime.cfg.seed + 9000, size, size))
            contexts = [
                runtime.corpus.context(cohort, "last", runtime.device)
                for cohort in cohorts[:cost_cohorts]
            ]
            record["cost"] = generator_cost(runtime, generator, contexts)
        path = (
            cfg.output_root
            / "offline_eval"
            / name
            / job["stage"]
            / f"{job['label']}.json"
        )
        write_json(path, record)
        return record

    return campaign.run_one(cfg, job, evaluate=run, local_inputs=inputs)


def evaluate_base(cfg, job, *, name, split="validation", ood=None):
    """Base arm: the released drafter, unmodified (no adapters installed),
    evaluated exactly as the trained arms are validated (same rows, cohorts
    and validation anchors), with per-row records. `split="test"` uses the
    job's untouched test rows, `split="ood"` the offline out-of-distribution
    set `ood`. Needs no training run. Writes
    `<output_root>/offline_eval/<name>/base/<job label>[.<split>].json`."""
    # Imported here: campaign imports this module's dependencies.
    from speculators.bank.artifacts import write_json  # noqa: PLC0415
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.campaign_train import evaluate  # noqa: PLC0415
    from speculators.provenance import (  # noqa: PLC0415
        find_repo_root,
        require_git_sha,
    )

    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    held_out, inputs, fields = (
        (list(job["validation_subsets"]), None, {})
        if split == "validation"
        else _held_out(cfg, job, split, ood)
    )
    sha = require_git_sha(find_repo_root(Path(__file__)))

    def run(runtime, job, _output):
        if runtime.paths or any(
            isinstance(module, AdapterLinear) for module in runtime.drafter.modules()
        ):
            raise RuntimeError("Base evaluation found adapter modules in the drafter")
        _expose(runtime, inputs, held_out)
        record = {
            "git_sha": sha,
            "arm": "base",
            "split": split,
            "job": job,
            "drafter": {
                "repository": runtime.bank.models.drafter,
                "revision": runtime.bank.models.drafter_revision,
            },
            "group_size": validation_group_size(runtime),
            **fields,
            split: evaluate(runtime, held_out, None, per_cohort=True),
        }
        suffix = "" if split == "validation" else f".{split}"
        path = (
            cfg.output_root
            / "offline_eval"
            / name
            / "base"
            / f"{job['label']}{suffix}.json"
        )
        write_json(path, record)
        return record

    return campaign.run_one(cfg, job, evaluate=run, base_only=True, local_inputs=inputs)
