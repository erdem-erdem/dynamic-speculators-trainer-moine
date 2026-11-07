"""Fill (or check) a persistent activation store for a campaign recipe (S6b).

Needs a running vLLM extraction server (`scripts/launch_vllm.py train ...` with
`--provenance-dir` pointing at the store) serving the recipe's target at its
layers. The recipe must set `online.activation_store: {path, mode: write}`.

    python scripts/dflash_port/fill_activation_store.py RECIPE.yaml \\
        [--subsets chat-00000 ...] [--compare N]

Fills every train/validation row of the subsets (default: the recipe's
`experiment_subsets`), then with `--compare N` re-extracts N rows through vLLM,
bypassing the store, and reports bit-identity / max |diff| (online ≡ offline
check, P0). Writes `<store>/fills/<UTC time>.json`.
"""

import argparse
import time
from pathlib import Path

from speculators.bank.artifacts import write_json
from speculators.generator.activation_store import (
    compare_with_fresh,
    fill_store,
    store_indices,
)
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import OnlineActivations
from speculators.provenance import find_repo_root, require_git_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--subsets", nargs="+")
    parser.add_argument("--compare", type=int, default=0)
    args = parser.parse_args()
    sha = require_git_sha(find_repo_root(Path(__file__)))
    cfg = CampaignConfig.load(args.recipe)
    store = cfg.online.activation_store
    if cfg.activation_source != "vllm" or store is None or store.mode != "write":
        raise ValueError(
            "Recipe needs activation_source: vllm and "
            "online.activation_store.mode: write"
        )
    inputs = load_local_inputs(cfg)
    subsets = args.subsets or cfg.experiment_subsets
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
