"""Publish every completed training run of a tier matrix to the HF Hub.

    python scripts/dflash_port/push_matrix.py MATRIX \\
        --visibility public|private [--output-root ROOT] [--dry-run]

For each `train` entry: the run directory (as `run_matrix.py` resolves it) is
published with `push_run.py` to `erdem-erdem/dyn-spec-dflash-<run name>`
(EXPERIMENT_MATRIX.md §4 naming). Runs that are not completed are listed and
skipped; runs with a `hub.json` (already published) are skipped.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import push_run
import run_matrix
from run_arm import DEFAULT_MACHINE, DEFAULT_OUTPUT_ROOT

OWNER = "erdem-erdem"
PREFIX = "dyn-spec-dflash-"


def repo_for(run_dir):
    return f"{OWNER}/{PREFIX}{Path(run_dir).name}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    # No default: publishing visibility is always an explicit choice.
    parser.add_argument("--visibility", required=True, choices=("public", "private"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--machine", type=Path, default=DEFAULT_MACHINE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    matrix = run_matrix.load(args.matrix)
    plan_args = argparse.Namespace(
        output_root=args.output_root, machine=args.machine, no_wandb=False, no_hub=True
    )
    reports = []
    for entry in matrix["runs"]:
        if entry["mode"] != "train":
            continue
        output = run_matrix.plan_entry(entry, matrix["tier"], plan_args)["output"]
        record = output / "run_record_train.json"
        if (
            not record.is_file()
            or json.loads(record.read_text())["status"] != "completed"
        ):
            print(f"skip (not completed): {output}", flush=True)
            continue
        if (output / "hub.json").is_file():
            print(f"skip (published): {output}", flush=True)
            continue
        print(f"push: {output} -> {repo_for(output)}", flush=True)
        report = push_run.push(
            output,
            repo_for(output),
            args.visibility == "private",
            dry_run=args.dry_run,
        )
        print(json.dumps(report), flush=True)
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
