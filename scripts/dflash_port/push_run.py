"""Publish one trained DFlash generator / static-LoRA run to an HF model repo.

    python scripts/dflash_port/push_run.py /workspace/yusuf/runs/dflash/tier1/<run> \\
        --repo erdem-erdem/dyn-spec-dflash-<run name> --visibility public|private \\
        [--require-eval] [--dry-run]

The serving bundle export (`speculators.generator.publish.push_run`) raises for
DFlash until P5, so this publishes the run directory itself (training
checkpoint with optimizer state, results, metrics, offline evaluations, every
provenance file) and the activation store's extraction provenance.

1. Refuses a run whose `run_record_train.json` is not completed, whose eval
   record (if present, or required with `--require-eval`) is not completed, or
   whose `checkpoint.pt` differs from the SHA256 in `result.json`.
2. Repo layout: `run/` = the run directory as is (lock files excluded);
   `provenance/store/` = the store's identity, extractor record, vLLM launch
   provenance and fill records; `MANIFEST.json` (size + SHA256 of every file);
   `README.md` (generated model card).
3. One commit, files read in place (no copy of the checkpoint). Prints the
   commit SHA and writes `<run>/hub.json` (local only, not uploaded).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 16 * 1024 * 1024
LOCAL_ONLY = {"hub.json"}
STORE_FILES = ("identity.json", "extractor.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _record(run_dir, mode):
    path = run_dir / f"run_record_{mode}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def check_run(run_dir, *, require_eval=False):
    """The completed training record and the training result directory."""
    train = _record(run_dir, "train")
    if train is None or train.get("status") != "completed":
        raise RuntimeError(f"{run_dir}: training is not completed")
    evaluation = _record(run_dir, "eval")
    if evaluation is None and require_eval:
        raise RuntimeError(f"{run_dir}: no offline evaluation (--require-eval)")
    if evaluation is not None and evaluation.get("status") != "completed":
        raise RuntimeError(f"{run_dir}: offline evaluation is not completed")
    job = train["job"]
    result_dir = run_dir / job["stage"] / job["label"]
    result = json.loads((result_dir / "result.json").read_text())
    if sha256(result_dir / "checkpoint.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"{result_dir}/checkpoint.pt differs from result.json")
    return train, result_dir, result


def collect(run_dir, store_root):
    """{path in repo: local file} for the run and its store provenance."""
    files = {
        f"run/{path.relative_to(run_dir).as_posix()}": path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and not path.name.endswith(".lock")
        and path.relative_to(run_dir).as_posix() not in LOCAL_ONLY
    }
    store_paths = [store_root / name for name in STORE_FILES]
    store_paths += sorted((store_root / "provenance").rglob("*"))
    store_paths += sorted((store_root / "fills").rglob("*"))
    for path in store_paths:
        if not path.is_file():
            if path.name in STORE_FILES:
                raise FileNotFoundError(f"{path} missing: not an activation store")
            continue
        files[f"provenance/store/{path.relative_to(store_root).as_posix()}"] = path
    if not any(name.startswith("provenance/store/provenance/vllm/") for name in files):
        raise FileNotFoundError(f"{store_root}: no vLLM extraction provenance")
    return files


def _fmt(value):
    return f"{value:.4f}"


def model_card(train, result, evaluation, store_repo):
    job, drafter = train["job"], train["drafter"]
    history = result["history"]
    first, last = history[0], history[-1]
    categories = sorted(last["categories"])
    lines = [
        "---",
        "tags:",
        "- speculators",
        "- dflash",
        "- speculative-decoding",
        "- lora-generator",
        "---",
        "",
        f"# {job['label']}",
        "",
        f"DFlash drafter adaptation run (`{job['arm']}` arm) from the "
        "dyn-spec DFlash generator port. Training checkpoint (with optimizer "
        "state), results, offline evaluations and full provenance; not a "
        "serving bundle (the DFlash serving export is not ported yet).",
        "",
        "| | |",
        "|---|---|",
        "| Target | Qwen/Qwen3-8B (store identity in `provenance/store/`) |",
        f"| Drafter | `{drafter['repository']}@{drafter['revision']}` |",
        f"| Arm / stage | `{job['arm']}` / `{job['stage']}` |",
        f"| Updates | {result['completed_updates']} |",
        f"| Seed | {train['seed']} |",
        f"| Git SHA | `{train['git_sha']}` |",
        f"| Checkpoint SHA256 | `{result['checkpoint_sha256']}` |",
    ]
    if store_repo:
        url = f"https://huggingface.co/datasets/{store_repo}"
        lines.append(f"| Activation store | [`{store_repo}`]({url}) |")
    lines += [
        "",
        "## Validation (200 fixed rows, one LoRA per row; full_acc)",
        "",
        "| Update | Score | τ (eal) | " + " | ".join(categories) + " |",
        "|---|---|---|" + "---|" * len(categories),
    ]
    for entry in (first, last) if first is not last else (last,):
        cells = [_fmt(entry["categories"][c]) for c in categories]
        eal = entry["contexts"]["1"]["eal"]
        lines.append(
            f"| {entry['update']} | {_fmt(entry['score'])} | {eal:.3f} | "
            + " | ".join(cells)
            + " |"
        )
    if evaluation is not None:
        lines += [
            "",
            "## Offline evaluation",
            "",
            "Results in `run/offline_eval/` (reproduction, M4 coefficients, "
            "M7 donors, D13 similarity, D14 cost; per-row metrics).",
        ]
    lines += [
        "",
        "## Files",
        "",
        "- `run/`: the run directory (`run_record_*.json`, `inputs/`, "
        f"`{job['stage']}/{job['label']}/`: `checkpoint.pt`, `result.json`, "
        "`metrics.json`, `timing.json`, `train_command.txt`, `speculators.patch`, "
        "`provenance/`).",
        "- `provenance/store/`: activation-store identity, extractor record, "
        "vLLM launch provenance, fill records.",
        "- `MANIFEST.json`: size and SHA256 of every file.",
        "",
    ]
    return "\n".join(lines)


def build(run_dir, *, require_eval=False):
    run_dir = Path(run_dir).resolve()
    train, _, result = check_run(run_dir, require_eval=require_eval)
    store_root = Path(train["job"]["vllm_provenance_dir"]).resolve().parents[1]
    files = collect(run_dir, store_root)
    manifest = {
        "run": train["job"]["label"],
        "git_sha": train["git_sha"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "files": {
            name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in files.items()
        },
    }
    manifest["bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    store_hub = store_root / "hub.json"  # written by push_store.py after its upload
    store_repo = (
        json.loads(store_hub.read_text())["repo"] if store_hub.is_file() else None
    )
    card = model_card(train, result, _record(run_dir, "eval"), store_repo)
    return run_dir, files, manifest, card


def push(run_dir, repo, private, *, require_eval=False, dry_run=False, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415
    from huggingface_hub.utils import validate_repo_id  # noqa: PLC0415

    validate_repo_id(repo)
    run_dir, files, manifest, card = build(run_dir, require_eval=require_eval)
    report = {
        "repo": repo,
        "private": private,
        "files": len(files) + 2,
        "bytes": manifest["bytes"],
        "dry_run": dry_run,
    }
    if dry_run:
        return report
    api = api or HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in files.items()
    ]
    operations += [
        CommitOperationAdd(
            path_in_repo="MANIFEST.json",
            path_or_fileobj=(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode(),
        ),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
    ]
    commit = api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish {manifest['run']} (run directory and provenance)",
    )
    report.update(revision=commit.oid, url=commit.commit_url)
    (run_dir / "hub.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("push",))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--repo", required=True)
    # No default: publishing visibility is always an explicit choice.
    parser.add_argument("--visibility", required=True, choices=("public", "private"))
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = push(
        args.run_dir,
        args.repo,
        args.visibility == "private",
        require_eval=args.require_eval,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
