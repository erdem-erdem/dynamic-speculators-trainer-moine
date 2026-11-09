"""Publish an activation store to the HF Hub, or verify a downloaded copy.

    STORE=output/dflash-lora/activation-store-4k
    python scripts/dflash_port/push_store.py push $STORE \\
        --repo erdem-erdem/dflash-qwen3-8b-activations-4k \\
        --visibility public|private
    python scripts/dflash_port/push_store.py verify $STORE

`push`:
1. refuses a store with unfinished (`*.pending`) entries;
2. writes `MANIFEST.json`: every file's size and SHA256, plus the store's
   identity and extractor records;
3. uploads the folder to the dataset repo (resumable `upload_large_folder`);
4. prints the commit SHA to pin (`setup.sh --fetch-store <SHA>`) and writes
   it to `<store>/hub.json`.

`verify` re-hashes every file of a store against its `MANIFEST.json` and
raises on any missing, extra or changed file. `setup.sh --fetch-store` runs it
after every download.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = "MANIFEST.json"
# Written after the upload; never part of the manifest or the repo.
LOCAL_ONLY = {"hub.json"}
CHUNK = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def store_files(root):
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != MANIFEST
        and path.relative_to(root).as_posix() not in LOCAL_ONLY
        and ".cache" not in path.relative_to(root).parts
        and not path.name.endswith(".lock")
    )
    pending = [path for path in files if path.suffix == ".pending"]
    if pending:
        raise RuntimeError(
            f"{root} has {len(pending)} unfinished entries (e.g. {pending[0]}); "
            "finish or remove the fill first"
        )
    return files


def hash_all(root, files, workers):
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(sha256, files))
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": h}
        for path, h in zip(files, digests, strict=True)
    }


def write_manifest(root, workers):
    for name in ("identity.json", "extractor.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"{root / name} missing: not an activation store")
    files = store_files(root)
    manifest = {
        "identity": json.loads((root / "identity.json").read_text()),
        "extractor": json.loads((root / "extractor.json").read_text()),
        "files": hash_all(root, files, workers),
    }
    manifest["entries"] = sum(
        name.endswith(".safetensors") for name in manifest["files"]
    )
    manifest["bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root, workers):
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing")
    manifest = json.loads(manifest_path.read_text())
    found = hash_all(root, store_files(root), workers)
    missing = sorted(set(manifest["files"]) - set(found))
    extra = sorted(set(found) - set(manifest["files"]))
    changed = sorted(
        name
        for name in set(found) & set(manifest["files"])
        if found[name] != manifest["files"][name]
    )
    if missing or extra or changed:
        raise ValueError(
            f"{root} does not match {MANIFEST}: {len(missing)} missing "
            f"{missing[:3]}, {len(extra)} extra {extra[:3]}, "
            f"{len(changed)} changed {changed[:3]}"
        )
    return manifest


def push(root, repo, private, workers):
    from huggingface_hub import HfApi  # noqa: PLC0415

    manifest = write_manifest(root, workers)
    print(
        f"Manifest: {manifest['entries']} entries, {manifest['bytes'] / 1e9:.1f} GB",
        flush=True,
    )
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo,
        folder_path=root,
        repo_type="dataset",
        ignore_patterns=["*.pending", "*.lock", *LOCAL_ONLY],
        num_workers=workers,
    )
    commit = api.list_repo_commits(repo, repo_type="dataset")[0].commit_id
    record = {
        "repo": repo,
        "repo_type": "dataset",
        "private": private,
        "revision": commit,
        "entries": manifest["entries"],
        "bytes": manifest["bytes"],
        "manifest_sha256": sha256(root / MANIFEST),
    }
    (root / "hub.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pushing = commands.add_parser("push")
    pushing.add_argument("store", type=Path)
    pushing.add_argument("--repo", required=True)
    # No default: publishing visibility is always an explicit choice.
    pushing.add_argument("--visibility", required=True, choices=("public", "private"))
    pushing.add_argument("--workers", type=int, default=16)
    checking = commands.add_parser("verify")
    checking.add_argument("store", type=Path)
    checking.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    root = args.store.resolve()
    if args.command == "push":
        push(root, args.repo, args.visibility == "private", args.workers)
    else:
        manifest = verify(root, args.workers)
        print(f"{root}: {manifest['entries']} entries match {MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())
