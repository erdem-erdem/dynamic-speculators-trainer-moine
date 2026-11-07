"""Run a tier matrix of DFlash arms, in order, stopping at the first failure.

    python scripts/dflash_port/run_matrix.py examples/dflash-lora/matrices/tier1.yaml \\
        [--output-root ROOT] [--no-wandb] [--dry-run]

Matrix file:
    tier: tier1
    runs:
      - {recipe: examples/dflash-lora/m0/base.yaml, mode: base}
      - {recipe: examples/dflash-lora/m0/generator.yaml, mode: train}
      ...
Each entry runs `run_arm.py` in its own process (fresh CUDA state). Recipe
paths are relative to the repository root. Entries already completed (their
`run_record_<mode>.json` says completed) are skipped; training also resumes
from its own checkpoints.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

# `scripts/` is not a package: import the sibling run layer by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_arm import (
    DEFAULT_MACHINE,
    DEFAULT_OUTPUT_ROOT,
    MODES,
    configuration_digest,
    resolve,
)

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    matrix = yaml.safe_load(Path(path).read_text())
    if set(matrix) != {"tier", "runs"} or not matrix["runs"]:
        raise ValueError("A matrix has exactly `tier` and a nonempty `runs` list")
    for entry in matrix["runs"]:
        if set(entry) != {"recipe", "mode"} or entry["mode"] not in MODES:
            raise ValueError(f"Bad matrix entry: {entry}")
        if not (ROOT / entry["recipe"]).is_file():
            raise FileNotFoundError(ROOT / entry["recipe"])
    return matrix


def plan_entry(entry, tier, args):
    """The run_arm command for one matrix entry and whether it is completed
    (a completed record of another configuration raises)."""
    recipe = ROOT / entry["recipe"]
    command = [
        sys.executable,
        str(ROOT / "scripts/dflash_port/run_arm.py"),
        str(recipe),
        "--tier",
        tier,
        "--mode",
        entry["mode"],
        "--output-root",
        str(args.output_root),
        "--machine",
        str(args.machine),
    ]
    command += ["--no-wandb"] if args.no_wandb else []
    command += ["--no-hub"] if args.no_hub else []
    resolved = resolve(
        recipe,
        tier=tier,
        output_root=args.output_root,
        wandb=not args.no_wandb,
        store_mode=None,
        machine=args.machine,
        hub=not args.no_hub,
    )
    output = resolved.cfg.output_root
    record = output / f"run_record_{entry['mode']}.json"
    done = False
    if record.is_file():
        saved = json.loads(record.read_text())
        done = saved["status"] == "completed"
        if done and saved["configuration_digest"] != configuration_digest(resolved):
            raise ValueError(
                f"{record} is completed for a different configuration than "
                f"{entry['recipe']} now resolves to; move it away to rerun"
            )
    return {
        "command": command,
        "completed": done,
        "output": output,
        "unresolved": resolved.unresolved,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--machine", type=Path, default=DEFAULT_MACHINE)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    matrix = load(args.matrix)
    plan = [plan_entry(entry, matrix["tier"], args) for entry in matrix["runs"]]
    if args.dry_run:
        for item in plan:
            state = "done" if item["completed"] else "todo"
            print(f"[{state}] {' '.join(item['command'])}\n        -> {item['output']}")
            if item["unresolved"]:
                print(f"        unresolved (setup.sh): {item['unresolved']}")
        return plan
    for item in plan:
        if item["completed"]:
            print(f"skip (completed): {' '.join(item['command'])}")
            continue
        print(f"run: {' '.join(item['command'])}", flush=True)
        # Commands are built above from validated matrix entries only.
        subprocess.run(item["command"], check=True, cwd=ROOT)  # noqa: S603
    return plan


if __name__ == "__main__":
    main()
