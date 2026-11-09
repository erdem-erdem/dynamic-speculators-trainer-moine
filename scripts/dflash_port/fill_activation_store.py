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
import hashlib
import shutil
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
from speculators.generator.ood_set import with_ood
from speculators.provenance import find_repo_root, require_git_sha

# `scripts/` is not a package: the sibling run layer resolves recipes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--subsets", nargs="+")
    parser.add_argument("--compare", type=int, default=0)
    # Fill only the offline out-of-distribution set's rows (ood_set.py).
    parser.add_argument("--ood", type=Path)
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
        # Untouched test and unseen donor subsets: validation rows only.
        held_out = job.get("test_subsets", []) + job.get("donor_subsets", [])
    else:
        cfg = CampaignConfig.load(args.recipe)
        default_subsets = cfg.experiment_subsets
        held_out = []
    store = cfg.online.activation_store
    if cfg.activation_source != "vllm" or store is None or store.mode != "write":
        raise ValueError(
            "Recipe needs activation_source: vllm and "
            "online.activation_store.mode: write"
        )
    provenance = Path(store.path) / "provenance" / "vllm"
    check_extraction_provenance(provenance)
    # Every extractor launch rewrites <store>/provenance/vllm; keep this fill's
    # own launch record next to its fill record so a later launch cannot erase it.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fills = Path(store.path) / "fills"
    snapshot = fills / f"{stamp}.provenance"
    fills.mkdir(parents=True, exist_ok=True)
    shutil.copytree(provenance, snapshot)
    inputs = load_local_inputs(cfg)
    ood_manifest = None
    if args.ood is not None:
        if args.subsets:
            raise ValueError("--ood fills only the OOD rows; drop --subsets")
        inputs, ood_manifest = with_ood(inputs, args.ood)
        subsets, held_out = [], list(ood_manifest["subsets"])
    else:
        subsets = args.subsets or default_subsets
        held_out = [] if args.subsets else held_out
    indices = store_indices(inputs, subsets, held_out)
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
        "validation_only_subsets": held_out,
        "ood_set": None
        if args.ood is None
        else {
            "path": str(args.ood.resolve()),
            "manifest_sha256": hashlib.sha256(
                (args.ood / "manifest.json").read_bytes()
            ).hexdigest(),
        },
        "extraction_provenance": str(snapshot.relative_to(store.path)),
        "rows": len(indices),
        "stats": stats,
        "comparison": comparison,
        "seconds": time.time() - started,
    }
    write_json(cache.store.root / "fills" / f"{stamp}.json", record)
    print(record)


if __name__ == "__main__":
    main()
