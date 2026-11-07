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
    `same_category`, other validation rows of the same category, never the
    cohort's own. Donor draws are deterministic (seeded by cohort position).
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

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from speculators.generator.campaign_train import conditioning_subset, groups
from speculators.generator.model import AdapterLinear

DONORS = ("correct", "fixed", "same_category", "cross_category")
# M7 controls run by default (user 2026-11-07, option C).
DEFAULT_DONORS = ("correct", "fixed", "cross_category")
# M4 grid (user 2026-11-07); the reference recipe trains at alpha/rank = 2.
DEFAULT_COEFFICIENTS = (0.5, 1.0, 2.0, 4.0, 8.0)
# M4 / M7 / D13 are small ablations (user 2026-11-07): one fixed, seeded set of
# validation rows per category, shared by every variant and arm.
ABLATION_ROWS_PER_CATEGORY = 24
ABLATION_SEED = 20261107


def _category(subset: str) -> str:
    return subset.rsplit("-", 1)[0]


@dataclass(frozen=True)
class Variant:
    coefficient: float | None = None
    donor: Literal["correct", "fixed", "same_category", "cross_category"] = "correct"
    donor_seed: int = 0
    subsets: tuple[str, ...] = ()  # the evaluated subsets (donor pool)
    allow_same_category: bool = False
    rows: frozenset[int] | None = None  # evaluate only these rows (ablation set)

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

    @property
    def label(self) -> str:
        coefficient = "train" if self.coefficient is None else f"{self.coefficient:g}"
        return f"coef-{coefficient}_donor-{self.donor}"

    def identity(self) -> dict:
        return {
            "coefficient": self.coefficient,
            "donor": self.donor,
            "donor_seed": self.donor_seed,
            "subsets": list(self.subsets),
            "rows": None if self.rows is None else sorted(self.rows),
        }

    def select(self, indices):
        """The evaluated rows of one subset (all, or the ablation set's)."""
        if self.rows is None:
            return indices
        return [index for index in indices if index in self.rows]

    def donor_pool(self, runtime, subset, rows):
        """Candidate donor rows: the same category's validation rows other than
        the cohort's own (`same_category`), or the next category's
        (`cross_category`, categories in sorted order, cyclic)."""
        category = _category(subset)
        if self.donor == "same_category":
            donor = category
        else:
            categories = sorted({_category(s) for s in self.subsets})
            if len(categories) < 2:  # noqa: PLR2004
                raise ValueError("cross_category donors need two categories")
            donor = categories[(categories.index(category) + 1) % len(categories)]
        own = set(rows)
        return sorted(
            index
            for name in self.subsets
            if _category(name) == donor
            for index in runtime.corpus.indices(name, validation=True)
            if index not in own
        )

    def conditioning(self, runtime, subset, rows, position):
        if self.donor == "correct":
            return rows
        if self.donor == "fixed":
            donor = min(self.subsets)
            return sorted(runtime.corpus.indices(donor, validation=True))[: len(rows)]
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


def ablation_rows(runtime, subsets, per_category, seed=ABLATION_SEED):
    """A fixed, seeded set of `per_category` validation rows per category,
    spread as evenly as possible over the category's subsets."""
    selected = set()
    for category in sorted({_category(s) for s in subsets}):
        names = sorted(s for s in subsets if _category(s) == category)
        share, extra = divmod(per_category, len(names))
        for position, name in enumerate(names):
            pool = sorted(runtime.corpus.indices(name, validation=True))
            count = share + (position < extra)
            if len(pool) < count:
                raise ValueError(f"{name} has fewer than {count} validation rows")
            rng = random.Random(f"{seed}:{name}")
            selected.update(rng.sample(pool, count))
    return frozenset(selected)


@torch.no_grad()
def collect_factors(runtime, subsets, generator, *, rows=None, conditioning=None):
    """Generated factors for validation cohorts (the same cohorts and
    conditioning draws as validation, restricted to `rows` when given).
    Returns [(subset, {key: (a, b)})]."""
    if generator is None:
        raise ValueError("collect_factors needs a generator")
    generator.eval()
    selector = Variant(rows=rows)
    samples = []
    for subset in subsets:
        indices = selector.select(runtime.corpus.indices(subset, validation=True))
        cohorts = list(groups(indices, runtime.cfg.seed + 9000, 4, 4))
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
        with torch.no_grad():
            for path in runtime.paths.values():
                adapter = runtime.drafter.get_submodule(path)
                adapter.a.copy_(state[f"{path}.a"])
                adapter.b.copy_(state[f"{path}.b"])
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


def evaluate_job(
    cfg,
    job,
    *,
    name,
    coefficients=DEFAULT_COEFFICIENTS,
    donors=DEFAULT_DONORS,
    allow_same_category=False,
    rows_per_category=ABLATION_ROWS_PER_CATEGORY,
    similarity_analysis=True,
    cost_cohorts=8,
):
    """Evaluate a completed training job offline (a campaign job, or the
    structure-study job used for matched arms, as EAGLE-3 GEN-009/010); writes
    `<output_root>/offline_eval/<name>/<stage>/<job label>.json`.

    1. Reproduction: the trained arm on the full validation set, which must
       reproduce training's final validation (checks the checkpoint).
    2. Ablations on the small fixed set (`rows_per_category` per category):
       M4 coefficients x M7 donors (static arm: coefficients only), and D13
       similarity of the generated LoRAs for that set's cohorts.
    3. D14 cost of the generator (`cost_cohorts` batched cohorts).
    """
    # Imported here: campaign imports this module's dependencies.
    from speculators.bank.artifacts import file_digest, write_json  # noqa: PLC0415
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.campaign_train import evaluate  # noqa: PLC0415
    from speculators.provenance import (  # noqa: PLC0415
        find_repo_root,
        require_git_sha,
    )

    sha = require_git_sha(find_repo_root(Path(__file__)))

    def run(runtime, job, output):
        generator, checkpoint = _load_trained(runtime, job, output)
        subsets = tuple(job["validation_subsets"])
        conditioning = job.get("training_conditioning_examples")
        trained = json.loads((output / "result.json").read_text())["history"][-1]
        full = evaluate(runtime, list(subsets), generator, conditioning)
        record = {
            "git_sha": sha,
            "job": job,
            "run": str(output.resolve()),
            "checkpoint_sha256": file_digest(checkpoint),
            # Exact on CPU; reported (not enforced): GPU kernels may differ.
            "reproduction": {
                "score": full["score"],
                "training_validation_score": trained["score"],
                "reproduces_training_validation": full["score"] == trained["score"],
                "validation_examples": full["validation_examples"],
            },
        }
        rows = ablation_rows(runtime, subsets, rows_per_category)
        ablations = {}
        for coefficient in coefficients:
            for donor in donors:
                if generator is None and donor != "correct":
                    continue  # a static LoRA has no conditioning
                variant = Variant(
                    coefficient,
                    donor,
                    subsets=subsets,
                    allow_same_category=allow_same_category,
                    rows=rows,
                )
                ablations[variant.label] = {
                    "variant": variant.identity(),
                    **evaluate(
                        runtime, list(subsets), generator, conditioning, variant
                    ),
                }
        record["ablation_set"] = {
            "rows_per_category": rows_per_category,
            "seed": ABLATION_SEED,
            "rows": sorted(rows),
        }
        record["ablations"] = ablations
        scales = {
            key: runtime.drafter.get_submodule(path).scale
            for key, path in runtime.paths.items()
        }
        if similarity_analysis and generator is not None:
            samples = collect_factors(
                runtime, subsets, generator, rows=rows, conditioning=conditioning
            )
            record["similarity"] = similarity(samples, scales)
        if cost_cohorts and generator is not None:
            # Timing only: any validation cohorts serve (not the ablation set).
            indices = runtime.corpus.indices(subsets[0], validation=True)
            cohorts = list(groups(indices, runtime.cfg.seed + 9000, 4, 4))
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

    return campaign.run_one(cfg, job, evaluate=run)


def evaluate_base(cfg, job, *, name, rows_per_category=ABLATION_ROWS_PER_CATEGORY):
    """Base arm: the released drafter, unmodified (no adapters installed),
    evaluated on `job`'s validation split exactly as the trained arms are
    validated (same rows, cohorts and validation anchors), plus on the small
    ablation row set. Needs no training run. Writes
    `<output_root>/offline_eval/<name>/base/<job label>.json`."""
    # Imported here: campaign imports this module's dependencies.
    from speculators.bank.artifacts import write_json  # noqa: PLC0415
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.campaign_train import evaluate  # noqa: PLC0415
    from speculators.provenance import (  # noqa: PLC0415
        find_repo_root,
        require_git_sha,
    )

    sha = require_git_sha(find_repo_root(Path(__file__)))

    def run(runtime, job, _output):
        if runtime.paths or any(
            isinstance(module, AdapterLinear) for module in runtime.drafter.modules()
        ):
            raise RuntimeError("Base evaluation found adapter modules in the drafter")
        subsets = tuple(job["validation_subsets"])
        rows = ablation_rows(runtime, subsets, rows_per_category)
        record = {
            "git_sha": sha,
            "arm": "base",
            "job": job,
            "drafter": {
                "repository": runtime.bank.models.drafter,
                "revision": runtime.bank.models.drafter_revision,
            },
            "validation": evaluate(runtime, list(subsets), None),
            "ablation_set": {
                "rows_per_category": rows_per_category,
                "seed": ABLATION_SEED,
                "rows": sorted(rows),
            },
            "ablation": evaluate(
                runtime, list(subsets), None, variant=Variant(rows=rows)
            ),
        }
        path = cfg.output_root / "offline_eval" / name / "base" / f"{job['label']}.json"
        write_json(path, record)
        return record

    return campaign.run_one(cfg, job, evaluate=run, base_only=True)
