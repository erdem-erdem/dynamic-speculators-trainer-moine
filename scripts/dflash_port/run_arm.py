"""Run one DFlash experiment arm: train, base-only eval, or offline eval.

    python scripts/dflash_port/run_arm.py RECIPE --tier tier1 --mode train \\
        [--output-root /workspace/yusuf/runs/dflash] [--no-wandb] \\
        [--store-mode read|write] [--dry-run]

Modes:
- `train`: the structure-study training job (local checkpoints + HF Hub push
  every `hub_upload_interval` and at the end; W&B unless --no-wandb).
- `base`: the released drafter, no adapters, on the same validation split
  (`offline_eval.evaluate_base`).
- `eval`: offline evaluation of the completed trained arm
  (`offline_eval.evaluate_job`: reproduction, M4 x M7 on the small set, D13, D14).

The arm writes to `<output-root>/<tier>/<recipe name>/` (training job under
`pretraining/<name>/`, offline evaluation under `offline_eval/<tier>/`), and a
`run_record_<mode>.json` with the command, git SHA, host, GPUs, configuration
digest, seed, status and outcome. --dry-run resolves and prints everything
without loading models, data, vLLM, HF or W&B.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from speculators.bank.artifacts import digest, write_json
from speculators.generator.structure_study import StructureStudy, training_job
from speculators.provenance import find_repo_root, require_git_sha

DEFAULT_OUTPUT_ROOT = Path("/workspace/yusuf/runs/dflash")
MODES = ("train", "base", "eval")


def resolve(recipe, *, tier, output_root, wandb, store_mode):
    """The recipe with run-layer overrides applied (validated again)."""
    cfg = StructureStudy.load(recipe)
    root = Path(output_root).resolve() / tier / cfg.name
    values = cfg.model_dump()
    values.update(output_root=root, input_root=root / "inputs")
    values["wandb"]["enabled"] = wandb
    if store_mode is not None:
        if values["online"]["activation_store"] is None:
            raise ValueError("--store-mode needs online.activation_store in the recipe")
        values["online"]["activation_store"]["mode"] = store_mode
    return StructureStudy.model_validate(values)


def gpus():
    """`nvidia-smi` GPU list, or the reason it is unavailable (recorded as is)."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "unavailable: nvidia-smi not found"
    try:
        result = subprocess.run(  # noqa: S603 -- fixed arguments
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return result.stdout.strip().splitlines()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--store-mode", choices=("read", "write"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = resolve(
        args.recipe,
        tier=args.tier,
        output_root=args.output_root,
        wandb=not args.no_wandb,
        store_mode=args.store_mode,
    )
    job = training_job(cfg)
    configuration = cfg.model_dump(mode="json")
    record = {
        "mode": args.mode,
        "tier": args.tier,
        "recipe": str(args.recipe.resolve()),
        "command": [sys.executable, *sys.argv],
        "git_sha": require_git_sha(find_repo_root(Path(__file__))),
        "host": socket.gethostname(),
        "gpus": gpus(),
        "configuration_digest": digest(configuration),
        "seed": cfg.seed,
        "output_root": str(cfg.output_root),
        "job": job,
        "wandb": cfg.wandb.enabled,
        "activation_store": configuration["online"]["activation_store"],
        "hub_repo": cfg.repo_id if args.mode == "train" else None,
    }
    if args.dry_run:
        print(json.dumps({**record, "configuration": configuration}, indent=2))
        return record
    # Imported here: dry runs need no model code.
    from speculators.generator import offline_eval  # noqa: PLC0415
    from speculators.generator.structure_study import (  # noqa: PLC0415
        train_structure,
    )

    path = cfg.output_root / f"run_record_{args.mode}.json"
    record.update(status="running", started=time.time())
    write_json(path, record)
    try:
        if args.mode == "train":
            result = train_structure(cfg)
            outcome = {"score": result["score"], "updates": result["completed_updates"]}
        elif args.mode == "base":
            result = offline_eval.evaluate_base(cfg, job, name=args.tier)
            outcome = {"score": result["validation"]["score"]}
        else:
            result = offline_eval.evaluate_job(cfg, job, name=args.tier)
            outcome = {
                "reproduction": result["reproduction"],
                "ablations": {k: v["score"] for k, v in result["ablations"].items()},
            }
    except BaseException as error:
        record.update(status="failed", error=repr(error), finished=time.time())
        write_json(path, record)
        raise
    record.update(status="completed", outcome=outcome, finished=time.time())
    write_json(path, record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    main()
