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
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from speculators.bank.artifacts import digest, write_json
from speculators.bank.config import BankConfig
from speculators.generator.structure_study import StructureStudy, training_job
from speculators.provenance import find_repo_root, require_git_sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/workspace/yusuf/runs/dflash")
DEFAULT_MACHINE = ROOT / "examples/dflash-lora/machine.local.yaml"
MACHINE_KEYS = {"target_model_path", "drafter_revision", "drafter_model_path"}
PLACEHOLDER = "PIN_AT_P0"
MODES = ("train", "base", "eval")


@dataclass(frozen=True)
class Resolved:
    cfg: StructureStudy
    bank: BankConfig  # the training recipe with this box's drafter pin
    unresolved: list[str]


def load_machine(path):
    """This box's paths and pins (written by setup.sh --prepare-drafter);
    None when the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    machine = yaml.safe_load(path.read_text())
    if set(machine) != MACHINE_KEYS:
        raise ValueError(f"{path} must have exactly {sorted(MACHINE_KEYS)}")
    return machine


def resolve(
    recipe,
    *,
    tier,
    output_root,
    wandb,
    store_mode,
    machine=DEFAULT_MACHINE,
    hub=True,
    materialize=False,
):
    """The recipe with run-layer and machine overrides applied (validated
    again). The training recipe resolved with this box's drafter pin lives at
    `<run>/inputs/training_recipe.resolved.yaml` (written when `materialize`);
    a real run refuses any unresolved PIN_AT_P0."""
    cfg = StructureStudy.load(recipe)
    root = Path(output_root).resolve() / tier / cfg.name
    values = cfg.model_dump()
    values.update(output_root=root, input_root=root / "inputs")
    values["wandb"]["enabled"] = wandb
    values["hub_push"] = hub
    if store_mode is not None:
        if values["online"]["activation_store"] is None:
            raise ValueError("--store-mode needs online.activation_store in the recipe")
        values["online"]["activation_store"]["mode"] = store_mode
    bank = BankConfig.load(cfg.training_recipe)
    settings = load_machine(machine) if machine is not None else None
    if settings is not None:
        values["online"]["target_model_path"] = Path(settings["target_model_path"])
        values["online"]["drafter_model_path"] = Path(settings["drafter_model_path"])
        bank.models.drafter_revision = settings["drafter_revision"]
    resolved = root / "inputs" / "training_recipe.resolved.yaml"
    values["training_recipe"] = values["bank_config"] = resolved
    unresolved = [
        name
        for name, value in (
            ("models.drafter_revision", bank.models.drafter_revision),
            ("online.drafter_model_path", str(values["online"]["drafter_model_path"])),
        )
        if PLACEHOLDER in value
    ]
    if materialize:
        if unresolved:
            raise ValueError(
                f"Unresolved {unresolved}: run scripts/dflash_port/setup.sh "
                f"--prepare-drafter <SHA> (writes {machine})"
            )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(bank.model_dump(mode="json")))
    return Resolved(StructureStudy.model_validate(values), bank, unresolved)


def configuration_digest(resolved):
    """Digest of what determines results: the resolved configuration and
    training recipe, without operational toggles (W&B, HF push, store mode)."""
    values = resolved.cfg.model_dump(mode="json")
    values["resolved_training_recipe"] = resolved.bank.model_dump(mode="json")
    values.pop("wandb")
    values.pop("hub_push")
    if values["online"]["activation_store"] is not None:
        values["online"]["activation_store"].pop("mode")
    return digest(values)


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
    parser.add_argument("--machine", type=Path, default=DEFAULT_MACHINE)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--store-mode", choices=("read", "write"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    recipe = args.recipe.resolve()
    os.chdir(ROOT)  # recipes use repository-relative paths
    resolved = resolve(
        recipe,
        tier=args.tier,
        output_root=args.output_root,
        wandb=not args.no_wandb,
        store_mode=args.store_mode,
        machine=args.machine,
        hub=not args.no_hub,
        materialize=not args.dry_run,
    )
    cfg = resolved.cfg
    job = training_job(cfg)
    configuration = cfg.model_dump(mode="json")
    record = {
        "mode": args.mode,
        "tier": args.tier,
        "recipe": str(recipe),
        "machine": str(args.machine),
        "command": [sys.executable, *sys.argv],
        "git_sha": require_git_sha(find_repo_root(Path(__file__))),
        "host": socket.gethostname(),
        "gpus": gpus(),
        "configuration_digest": configuration_digest(resolved),
        "seed": cfg.seed,
        "output_root": str(cfg.output_root),
        "job": job,
        "wandb": cfg.wandb.enabled,
        "activation_store": configuration["online"]["activation_store"],
        "hub_repo": cfg.repo_id if args.mode == "train" and cfg.hub_push else None,
        "drafter": {
            "repository": resolved.bank.models.drafter,
            "revision": resolved.bank.models.drafter_revision,
            "path": configuration["online"]["drafter_model_path"],
        },
        "unresolved": resolved.unresolved,
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
