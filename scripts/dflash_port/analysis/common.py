"""Shared harness for the 24K drafter analysis (ANALYSIS24K).

One Runtime (the 24K generator recipe, adapters installed on q/k/v/o of all
5 layers) serves all three arms through the production forward
(`Runtime.forward` -> `functional_call` with per-arm factors):

- base: zero LoRA factors (b = 0, bit-identical to the released drafter);
- static: the trained static LoRA factors (checkpoint of the static run);
- generator: factors generated from each row's own prompt summary.

The drafter's `forward` is replaced (as production does with its uncompiled
forward) by `analysis_forward`, which runs the upstream `_backbone_forward`
and `compute_metrics` unchanged and additionally exposes logits, targets,
the aligned loss mask and the anchored block indices in `CAPTURE`.

Nothing is written to the run directories: the configuration points
`input_root` / `staging_dir` at the analysis directory (input_root is not part
of the run identity), and `campaign.run_one(..., evaluate=)` is read-only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
RUNS = Path("/workspace/yusuf/runs/dflash")
OUT = RUNS / "analysis24k"
TMP = OUT / "tmp"
GEN_RECIPE = REPO / "examples/dflash-lora/mix24k/generator.yaml"
STATIC_RECIPE = REPO / "examples/dflash-lora/mix24k/static_lora.yaml"
GEN_RUN = RUNS / "mix24k/mix24k-generator-attn-r32-3ep-kl"
STATIC_RUN = RUNS / "mix24k/mix24k-static-lora-attn-r32-3ep-kl"
BASE_EVAL = RUNS / "math10k/math10k-base/offline_eval/math10k/base"
BLOCK = 16
CATEGORIES = ("math", "code", "stem", "chat")

CAPTURE: dict = {}


def grads_dir(point):
    """Gradient-map output of a point (env ANALYSIS24K_GRADS_<POINT> overrides,
    for dry runs of the report on an earlier run)."""
    override = os.environ.get(f"ANALYSIS24K_GRADS_{point.upper()}")
    return Path(override) if override else OUT / "grads" / point


def category(subset: str) -> str:
    return subset.rsplit("-", 1)[0]


def build_config(ram_cache_gib: int = 16):
    """The 24K generator recipe resolved as run_arm does, with the inputs and
    staging redirected to the analysis directory."""
    sys.path.insert(0, str(REPO / "scripts/dflash_port"))
    from run_arm import resolve  # noqa: PLC0415

    from speculators.generator.structure_study import (  # noqa: PLC0415
        StructureStudy,
        training_job,
    )

    os.chdir(REPO)
    resolved = resolve(
        GEN_RECIPE,
        tier="mix24k",
        output_root=RUNS,
        wandb=False,
        store_mode="read",
        hub=False,
        materialize=False,
    )
    values = resolved.cfg.model_dump()
    inputs = OUT / "inputs" / "generator"
    if not (inputs / "training_recipe.resolved.yaml").is_file():
        raise FileNotFoundError(f"Copy the generator run's inputs to {inputs} first")
    values["input_root"] = inputs
    values["training_recipe"] = values["bank_config"] = (
        inputs / "training_recipe.resolved.yaml"
    )
    values["staging_dir"] = TMP / "staging"
    values["online"]["ram_cache_gib"] = ram_cache_gib
    cfg = StructureStudy.model_validate(values)
    return cfg, training_job(cfg)


def with_runtime(function, ram_cache_gib: int = 16):
    """Run `function(runtime, job, output)` inside the read-only evaluation
    lifecycle of the generator job (identity checked against the run)."""
    from speculators.generator import campaign  # noqa: PLC0415

    cfg, job = build_config(ram_cache_gib)
    copied = OUT / "inputs" / "generator"
    original = GEN_RUN / "inputs"
    for path in copied.glob("bank_*.json"):
        mine = path.read_text().replace(str(copied), str(original))
        if json.loads(mine) != json.loads((original / path.name).read_text()):
            raise ValueError(f"{path} differs from the run's copy")
    strict = campaign.require_trained

    def require_trained(output, identity):
        # The bank paths point at the byte-identical analysis copy of the
        # run's inputs; map them back, then apply the strict check.
        text = json.dumps(identity, default=str).replace(str(copied), str(original))
        return strict(output, json.loads(text))

    campaign.require_trained = require_trained
    try:
        return campaign.run_one(cfg, job, evaluate=function)
    finally:
        campaign.require_trained = strict


def analysis_forward(
    self,
    hidden_states,
    input_ids,
    loss_mask,
    verifier_last_hidden_states,
    document_ids,
    position_ids=None,
    loss_config=None,
    gamma=4.0,
    max_anchors=512,
    per_position_loss_weight="fixed-exp-decay",
    dpace_alpha=0.5,
    **kwargs,
):
    """Upstream DFlash forward (`DFlashDraftModel.forward`), exposing tensors."""
    from speculators.models.dflash.metrics import compute_metrics  # noqa: PLC0415

    hidden, logits, targets, mask, indices = self._backbone_forward(
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids,
        max_anchors=max_anchors,
        **kwargs,
    )
    loss, metrics = compute_metrics(
        logits,
        targets,
        mask,
        self.block_size,
        gamma=gamma,
        loss_config=loss_config,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=self.config.sample_from_anchor,
    )
    CAPTURE.update(
        logits=logits, targets=targets, mask=mask, indices=indices, loss=loss
    )
    return None, loss, metrics


def install_analysis_forward(runtime):
    drafter = runtime.drafter
    # Adapters are installed after attach_drafter's eval(): new AdapterLinear
    # modules start in training mode (LoRA dropout on). Validation calls
    # drafter.eval() first (campaign_train.evaluate_subset); so do we.
    drafter.eval()
    drafter.forward = torch.compiler.disable(
        analysis_forward.__get__(drafter), recursive=False
    )


def zero_factors(runtime):
    return {
        name: (a.detach().clone(), torch.zeros_like(b))
        for name, (a, b) in runtime.adapter_state().items()
    }


def static_factors(runtime):
    """The trained static LoRA factors, keyed like the runtime's adapters."""
    checkpoint = (
        STATIC_RUN / "pretraining" / STATIC_RUN.name / "checkpoint.pt"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    expected = {f"{p}.{x}" for p in runtime.paths.values() for x in "ab"}
    if set(state) != expected:
        raise ValueError(f"{checkpoint} does not match the runtime adapters")
    return {
        name: (
            state[f"{path}.a"].to(runtime.device),
            state[f"{path}.b"].to(runtime.device),
        )
        for name, path in runtime.paths.items()
    }


def load_generator(runtime):
    checkpoint = GEN_RUN / "pretraining" / GEN_RUN.name / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    generator = runtime.new_generator("last")
    generator.load_state_dict(state, strict=True)
    generator.eval()
    return generator


@torch.no_grad()
def generator_factors(runtime, generator, rows):
    factors = generator(*runtime.corpus.context(list(rows), "last", runtime.device))
    return {k: (a.float(), b.float()) for k, (a, b) in factors.items()}


class Arms:
    """Factor providers for the three arms (generator: per row)."""

    def __init__(self, runtime, *, generator=True):
        self.runtime = runtime
        self.zero = zero_factors(runtime)
        self.static = static_factors(runtime)
        self.generator = load_generator(runtime) if generator else None

    def factors(self, arm, row, donor=None):
        if arm == "base":
            return self.zero
        if arm == "static":
            return self.static
        if arm == "generator":
            return generator_factors(
                self.runtime, self.generator, [row if donor is None else donor]
            )
        raise ValueError(arm)


def split_rows(runtime, job, split):
    """[(subset, row index)] of the validation or test split, in subset order."""
    subsets = job["validation_subsets"] if split == "validation" else job["test_subsets"]
    return [
        (subset, index)
        for subset in subsets
        for index in runtime.corpus.indices(subset, validation=True)
    ]


def load_batch(runtime, row):
    batches = list(runtime.corpus.microbatches([row], runtime.device, runtime.dtype))
    if len(batches) != 1:
        raise RuntimeError(f"Row {row} gave {len(batches)} microbatches")
    return batches[0][0]


def to_cpu(batch):
    return {k: v.to("cpu", non_blocking=False) for k, v in batch.items()}


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def run(runtime, batch, factors, **overrides):
    """One production forward; returns the metric counts (floats)."""
    _, _, metrics = runtime.forward(batch, factors, **overrides)
    return {k: float(v) for k, v in metrics.items()}


def score(per_subset_counts):
    """Validation score as `campaign_train.evaluate`: pooled full_acc per
    subset, mean over subsets of a category, mean over categories. Also τ."""
    out = {}
    for key in ("full_acc", "eal"):
        by_cat = {}
        for subset, counts in per_subset_counts.items():
            value = counts[f"{key}_sum"] / counts[f"{key}_total"]
            by_cat.setdefault(category(subset), []).append(value)
        cats = {c: sum(v) / len(v) for c, v in by_cat.items()}
        out[key] = {"all": sum(cats.values()) / len(cats), **cats}
    return out


def accumulate(store, subset, counts):
    target = store.setdefault(subset, {})
    for key, value in counts.items():
        target[key] = target.get(key, 0.0) + value


def prompt_boundary(loss_mask_row):
    """Index of the first supervised (response) token of a row."""
    nonzero = torch.nonzero(loss_mask_row.bool(), as_tuple=False)
    if nonzero.numel() == 0:
        raise ValueError("Row has no supervised tokens")
    return int(nonzero[0, 0])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=_plain))


def _plain(value):
    return value.tolist() if hasattr(value, "tolist") else float(value)


def recorded_scores():
    """The recorded validation/test category-mean scores of the three arms."""
    out = {}
    base = json.loads((BASE_EVAL / "math10k-base.json").read_text())["validation"]
    base_test = json.loads((BASE_EVAL / "math10k-base.test.json").read_text())["test"]
    out["base"] = {"validation": base, "test": base_test}
    for arm, run_dir in (("static", STATIC_RUN), ("generator", GEN_RUN)):
        folder = run_dir / "offline_eval/mix24k/pretraining"
        val = json.loads((folder / f"{run_dir.name}.json").read_text())
        test = json.loads((folder / f"{run_dir.name}.test.json").read_text())["test"]
        out[arm] = {
            "validation": val["ablations"]["coef-2_donor-correct"],
            "test": test,
        }
    return out


def recorded_row_metrics(record):
    """{primary_id: metrics} of one recorded evaluation (per-cohort records)."""
    rows = {}
    for result in record["subsets"].values():
        for cohort in result["cohorts"]:
            rows[cohort["primary_ids"][0]] = cohort["metrics"]
    return rows
