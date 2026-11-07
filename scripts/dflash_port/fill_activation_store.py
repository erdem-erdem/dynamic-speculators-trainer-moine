"""Fill (or check) a persistent activation store for a campaign recipe (S6b).

Needs a running vLLM extraction server serving the recipe's target at its
layers, launched with `scripts/launch_vllm.py train ... --strict-provenance
--provenance-dir <store>/provenance/vllm`; the fill refuses to start without
that provenance.

    python scripts/dflash_port/fill_activation_store.py RECIPE.yaml \\
        [--subsets chat-00000 ...] [--compare N] [--machine machine.local.yaml]

RECIPE is a DFlash structure-study recipe (e.g. examples/dflash-lora/m0/
generator.yaml: resolved like a training run, with this box's machine config,
store mode forced to write; default subsets = exactly the job's training and
validation subsets) or a campaign recipe with `online.activation_store.mode:
write` (default subsets: its `experiment_subsets`).

Fills every train/validation row of the subsets, then with `--compare N`
re-extracts N rows through vLLM, bypassing the store, and reports
bit-identity / max |diff| (online ≡ offline check, P0).
Writes `<store>/fills/<UTC time>.json`.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

from speculators.bank.artifacts import write_json
from speculators.generator.activation_store import (
    check_extraction_provenance,
    compare_with_fresh,
    fill_store,
    store_indices,
)
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import OnlineActivations
from speculators.provenance import find_repo_root, require_git_sha

# `scripts/` is not a package: the sibling run layer resolves recipes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--subsets", nargs="+")
    parser.add_argument("--compare", type=int, default=0)
    parser.add_argument("--machine", type=Path, default=run_arm.DEFAULT_MACHINE)
    args = parser.parse_args()
    sha = require_git_sha(find_repo_root(Path(__file__)))
    values = yaml.safe_load(args.recipe.read_text())
    if "extends" in values or "family" in values:  # structure-study recipe
        resolved = run_arm.resolve(
            args.recipe.resolve(),
            tier="store-fill",
            output_root=Path("output/dflash-lora/store-fill"),
            wandb=False,
            store_mode="write",
            machine=args.machine,
            hub=False,
            materialize=True,
        )
        cfg = resolved.cfg
        job = run_arm.training_job(cfg)
        default_subsets = list(
            dict.fromkeys(job["subsets"] + job["validation_subsets"])
        )
    else:
        cfg = CampaignConfig.load(args.recipe)
        default_subsets = cfg.experiment_subsets
    store = cfg.online.activation_store
    if cfg.activation_source != "vllm" or store is None or store.mode != "write":
        raise ValueError(
            "Recipe needs activation_source: vllm and "
            "online.activation_store.mode: write"
        )
    check_extraction_provenance(Path(store.path) / "provenance" / "vllm")
    inputs = load_local_inputs(cfg)
    subsets = args.subsets or default_subsets
    indices = store_indices(inputs, subsets)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    started = time.time()
    try:
        cache.check_server()
        stats = fill_store(cache, indices)
        comparison = (
            compare_with_fresh(cache, indices[: args.compare]) if args.compare else None
        )
    finally:
        cache.close()
    record = {
        "git_sha": sha,
        "recipe": str(args.recipe.resolve()),
        "store": cache.store.identity,
        "subsets": subsets,
        "rows": len(indices),
        "stats": stats,
        "comparison": comparison,
        "seconds": time.time() - started,
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    write_json(cache.store.root / "fills" / f"{stamp}.json", record)
    print(record)


if __name__ == "__main__":
    main()
