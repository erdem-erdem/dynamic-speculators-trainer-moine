"""Shared harness of the Move-vs-Improve perturbation study (PERTURB24K).

One Runtime (the 24K generator recipe: LoRA adapters installed on q/k/v/o of
all 5 drafter layers, rank 32, alpha 64) serves every starting point through
the production forward (`Runtime.forward` -> `functional_call`):

- base: zero LoRA factors (b = 0; bit-identical to the released drafter);
- static: the trained 24K static LoRA (KL) factors. "Merged into the weights"
  is realised exactly as trained (W + scale * B @ A through the adapter path),
  which avoids rounding the merged weight to bfloat16;
- generator (a free data point): factors generated per row from its prompt.

Weight perturbations dW are added on a separate path at every site
(y = W x + scale * B A x + dW x), with forward hooks on fc / q / o / gate /
up / down and a replaced `_project_kv` for K and V, so the context path
(target features) and the block path (draft tokens) can be perturbed
separately. With no perturbation active the hooks add nothing.

The drafter's forward is replaced by `capture_forward` (upstream
`_backbone_forward` + `compute_metrics` unchanged) which exposes logits,
verifier targets and the aligned loss mask in `CAPTURE`.

The configuration resolves the recipe as `run_arm.py` does, with inputs and
staging redirected to this study's directory, so nothing is written to any
run directory (`campaign.run_one(..., evaluate=)` is read-only).
Adapted from the ANALYSIS24K harness (`scripts/dflash_port/analysis/common.py`
in the main checkout; D8: reuse, do not depend on it).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MAIN = Path("/workspace/yusuf/dyn-spec/dynamic-speculators-trainer")
RUNS = Path("/workspace/yusuf/runs/dflash")
OUT = RUNS / "perturb24k"
TMP = OUT / "tmp"
GEN_RECIPE = MAIN / "examples/dflash-lora/mix24k/generator.yaml"
GEN_RUN = RUNS / "mix24k/mix24k-generator-attn-r32-3ep-kl"
STATIC_RUN = RUNS / "mix24k/mix24k-static-lora-attn-r32-3ep-kl"
FREE_STATICS = {
    "static24k_kl": STATIC_RUN,
    "static10k_math_ce": RUNS / "math10k/math10k-static-lora-attn-r32-3ep",
    "static4k_m0_ce": RUNS / "tier1/m0-static-lora-attn-r32-4k-1500",
}
BASE_EVAL = RUNS / "math10k/math10k-base/offline_eval/math10k/base"
BLOCK = 16
SLOTS = BLOCK - 1
CATEGORIES = ("math", "code", "stem", "chat")
LAYERS = 5
ATTN = ("q", "k", "v", "o")
FFN = ("gate", "up", "down")

CAPTURE: dict = {}


def category(subset: str) -> str:
    return subset.rsplit("-", 1)[0]


# --------------------------------------------------------------------------
# Configuration and runtime lifecycle
# --------------------------------------------------------------------------


def build_config(ram_cache_gib: int = 16):
    sys.path.insert(0, str(MAIN / "scripts/dflash_port"))
    from run_arm import resolve  # noqa: PLC0415

    from speculators.generator.structure_study import (  # noqa: PLC0415
        StructureStudy,
        training_job,
    )

    os.chdir(MAIN)
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
    lifecycle of the 24K generator job (identity checked against the run)."""
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
        text = json.dumps(identity, default=str).replace(str(copied), str(original))
        return strict(output, json.loads(text))

    campaign.require_trained = require_trained
    try:
        return campaign.run_one(cfg, job, evaluate=function)
    finally:
        campaign.require_trained = strict


def capture_forward(
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

    _, logits, targets, mask, indices = self._backbone_forward(
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
    CAPTURE.update(logits=logits, targets=targets, mask=mask, indices=indices)
    return None, loss, metrics


def install_capture(runtime):
    drafter = runtime.drafter
    drafter.eval()
    if drafter.config.sample_from_anchor:
        raise ValueError("Study assumes sample_from_anchor=False (slot 0 = anchor)")
    drafter.forward = torch.compiler.disable(
        capture_forward.__get__(drafter), recursive=False
    )


# --------------------------------------------------------------------------
# Starting points (adapter factors)
# --------------------------------------------------------------------------


def zero_factors(runtime):
    return {
        name: (a.detach().clone(), torch.zeros_like(b))
        for name, (a, b) in runtime.adapter_state().items()
    }


def static_factors(runtime, run_dir=STATIC_RUN):
    checkpoint = run_dir / "pretraining" / run_dir.name / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    expected = {f"{p}.{x}" for p in runtime.paths.values() for x in "ab"}
    if set(state) != expected:
        raise ValueError(f"{checkpoint} does not match the runtime adapters")
    return {
        name: (
            state[f"{path}.a"].float().to(runtime.device),
            state[f"{path}.b"].float().to(runtime.device),
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
def generator_factors(runtime, generator, row):
    factors = generator(*runtime.corpus.context([row], "last", runtime.device))
    return {k: (a.float(), b.float()) for k, (a, b) in factors.items()}


def adapter_scale(runtime):
    scales = {
        runtime.drafter.get_submodule(path).scale for path in runtime.paths.values()
    }
    if len(scales) != 1:
        raise ValueError(f"Mixed adapter scales {scales}")
    return scales.pop()


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def split_rows(runtime, job, split="validation"):
    subsets = job["validation_subsets"] if split == "validation" else job["test_subsets"]
    return [
        (subset, index)
        for subset in subsets
        for index in runtime.corpus.indices(subset, validation=True)
    ]


def sweep_rows(runtime, job, per_category=25, seed="perturb24k"):
    """The fixed 100 validation rows of the sweeps: `per_category` per category,
    a seeded sample, in subset order."""
    rows = split_rows(runtime, job, "validation")
    chosen = []
    for cat in CATEGORIES:
        members = [(s, r) for s, r in rows if category(s) == cat]
        chosen += sorted(random.Random(f"{seed}-{cat}").sample(members, per_category))
    return chosen


def training_rows(runtime, job, per_category, seed):
    """A seeded sample of 24K training rows per category; never eval rows."""
    evaluated = {
        i
        for s in job["validation_subsets"] + job["test_subsets"]
        for i in runtime.corpus.indices(s, validation=True)
    }
    chosen = []
    for cat in CATEGORIES:
        pool = [
            i
            for s in job["subsets"]
            if category(s) == cat
            for i in runtime.corpus.indices(s)
        ]
        if evaluated & set(pool):
            raise ValueError("Training pool overlaps evaluation rows")
        chosen += [(cat, i) for i in random.Random(f"{seed}-{cat}").sample(pool, per_category)]
    return chosen


def training_pool(runtime, job):
    evaluated = {
        i
        for s in job["validation_subsets"] + job["test_subsets"]
        for i in runtime.corpus.indices(s, validation=True)
    }
    pool = [i for s in job["subsets"] for i in runtime.corpus.indices(s)]
    if evaluated & set(pool):
        raise ValueError("Training pool overlaps evaluation rows")
    return pool


def load_batch(runtime, row, training=False):
    batches = list(
        runtime.corpus.microbatches([row], runtime.device, runtime.dtype, training=training)
    )
    if len(batches) != 1:
        raise RuntimeError(f"Row {row} gave {len(batches)} microbatches")
    return batches[0][0]


# --------------------------------------------------------------------------
# Weight sites and additive perturbations
# --------------------------------------------------------------------------


def all_sites():
    """Every perturbable weight site. K/V have one weight but two paths."""
    sites = ["fc"]
    for i in range(LAYERS):
        sites += [f"l{i}_{m}" for m in ATTN + FFN]
    return sites


def site_module_path(site):
    if site == "fc":
        return "fc"
    layer, module = site.split("_", 1)
    i = int(layer[1:])
    if module in ATTN:
        return f"layers.{i}.self_attn.{module}_proj"
    return f"layers.{i}.mlp.{module}_proj"


def base_linear(module):
    from speculators.generator.model import AdapterLinear  # noqa: PLC0415

    return module.base if isinstance(module, AdapterLinear) else module


def site_weight(runtime, site):
    return base_linear(runtime.drafter.get_submodule(site_module_path(site))).weight


def effective_weights(runtime, factors):
    """{site: W (+ scale * B @ A when the site has an adapter)} in float32."""
    scale = adapter_scale(runtime) if runtime.paths else 0.0
    by_path = {path: key for key, path in runtime.paths.items()}
    out = {}
    for site in all_sites():
        path = site_module_path(site)
        w = site_weight(runtime, site).detach().float()
        key = by_path.get(path)
        if key is not None and factors is not None:
            a, b = factors[key]
            w = w + scale * b.float() @ a.float()
        out[site] = w
    return out


class Delta:
    """An additive linear map dW applied as `coef * F.linear(x, W)` (dense) or
    `coef * F.linear(F.linear(x, A), B)` (low rank)."""

    __slots__ = ("a", "b", "coef", "w")

    def __init__(self, w=None, *, a=None, b=None, coef=1.0):
        if (w is None) == (a is None or b is None):
            raise ValueError("Delta needs W, or both A and B")
        self.w, self.a, self.b, self.coef = w, a, b, coef

    def __call__(self, x):
        if self.w is not None:
            y = F.linear(x, self.w)
        else:
            y = F.linear(F.linear(x, self.a), self.b)
        return y if self.coef == 1.0 else y * self.coef


class DeltaHooks:
    """Installs once; `active` maps site keys to Delta-like callables.

    Keys: `fc`, `l{i}_{q,o,gate,up,down}`, `l{i}_{k,v}` (both K/V paths) and
    `l{i}_{k,v}@ctx` / `l{i}_{k,v}@blk` (one path)."""

    def __init__(self, drafter):
        self.active: dict = {}
        self.handles = []
        for site in all_sites():
            module = drafter.get_submodule(site_module_path(site))
            if site.endswith(("_k", "_v")):
                continue
            self.handles.append(module.register_forward_hook(self._hook(site)))
        for i, layer in enumerate(drafter.layers):
            attn = layer.self_attn
            attn._project_kv = self._project(attn, i)

    def _hook(self, site):
        def hook(_module, inputs, output):
            delta = self.active.get(site)
            if delta is None:
                return output
            return output + delta(inputs[0]).to(output.dtype)

        return hook

    def _project(self, attn, i):
        original = type(attn)._project_kv

        def project(target_hidden, hidden_states):
            k_ctx, k_blk, v_ctx, v_blk = original(attn, target_hidden, hidden_states)
            if not self.active:
                return k_ctx, k_blk, v_ctx, v_blk
            out = []
            for name, x, y in (
                ("k", target_hidden, k_ctx),
                ("k", hidden_states, k_blk),
                ("v", target_hidden, v_ctx),
                ("v", hidden_states, v_blk),
            ):
                path = "ctx" if x is target_hidden else "blk"
                for key in (f"l{i}_{name}", f"l{i}_{name}@{path}"):
                    delta = self.active.get(key)
                    if delta is not None:
                        y = y + delta(x).to(y.dtype)
                out.append(y)
            return tuple(out)

        return project

    def set(self, active):
        self.active = dict(active)

    def clear(self):
        self.active = {}


# Location vocabulary -------------------------------------------------------


def location_members(location):
    """[(site, path)] of a named location; path in {"both", "ctx", "blk"}."""
    if location == "fc":
        return [("fc", "both")]
    if location in ("all_attn", "all_ffn", "all", "whole"):
        mods = {"all_attn": ATTN, "all_ffn": FFN}.get(location, ATTN + FFN)
        members = [(f"l{i}_{m}", "both") for i in range(LAYERS) for m in mods]
        return members + ([("fc", "both")] if location == "whole" else [])
    if location.startswith("layer"):
        i = int(location[len("layer"):]) - 1
        return [(f"l{i}_{m}", "both") for m in ATTN + FFN]
    if location.endswith(("_kv@ctx", "_kv@blk")):
        layer, rest = location.split("_", 1)
        path = rest.split("@")[1]
        return [(f"{layer}_k", path), (f"{layer}_v", path)]
    if location in all_sites():
        return [(location, "both")]
    raise ValueError(f"Unknown location {location!r}")


def t1_locations():
    locs = ["fc"]
    locs += [f"l{i}_{m}" for i in range(LAYERS) for m in ATTN + FFN]
    locs += [f"l{i}_kv@{p}" for i in range(LAYERS) for p in ("ctx", "blk")]
    locs += ["all_attn", "all_ffn", "all"]
    locs += [f"layer{i + 1}" for i in range(LAYERS)]
    locs += ["whole"]
    return locs


def hook_key(site, path):
    return site if path == "both" else f"{site}@{path}"


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float))
    tmp.replace(path)


def recorded_scores():
    out = {}
    base = json.loads((BASE_EVAL / "math10k-base.json").read_text())["validation"]
    out["base"] = base
    for arm, run_dir in (("static", STATIC_RUN), ("generator", GEN_RUN)):
        folder = run_dir / "offline_eval/mix24k/pretraining"
        val = json.loads((folder / f"{run_dir.name}.json").read_text())
        out[arm] = val["ablations"]["coef-2_donor-correct"]
    return out


def recorded_row_metrics(record):
    rows = {}
    for result in record["subsets"].values():
        for cohort in result["cohorts"]:
            rows[cohort["primary_ids"][0]] = cohort["metrics"]
    return rows


def score(per_subset_counts):
    """Validation score as `campaign_train.evaluate`: pooled per subset, mean
    over a category's subsets, mean over categories (full_acc and eal=τ)."""
    out = {}
    for key in ("full_acc", "eal"):
        by_cat = {}
        for subset, counts in per_subset_counts.items():
            value = counts[f"{key}_sum"] / counts[f"{key}_total"]
            by_cat.setdefault(category(subset), []).append(value)
        cats = {c: sum(v) / len(v) for c, v in by_cat.items()}
        out[key] = {"all": sum(cats.values()) / len(cats), **cats}
    return out


def memory_report():
    rss = 0
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) / 1e6
    except OSError:
        pass
    cg = 0.0
    try:
        cg = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
    except OSError:
        pass
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return f"rss={rss:.1f}GB cgroup={cg:.0f}GiB gpu_peak={gpu:.1f}GB"
