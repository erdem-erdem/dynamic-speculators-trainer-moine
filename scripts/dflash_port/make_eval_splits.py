"""Build (or verify) a stored evaluation split that can never leak into training.

    python scripts/dflash_port/make_eval_splits.py build --release RELEASE \\
        --seed SEED --rows N --validation SUBSET ... --test SUBSET ... \\
        --output examples/dflash-lora/<run>/eval_splits.json
    python scripts/dflash_port/make_eval_splits.py verify FILE

Candidates are the validation splits of the named subsets. A row is eligible
only if neither its primary id, nor its prompt tokens, nor its full token
sequence occurs in ANY train split of ANY local trajectory release (every
category and subset under datasets/): the trajectory data holds duplicate
prompts across rows (2026-11-10: 221 r0002-100k train rows repeat the prompt
of a validation row), so subset membership alone does not prevent leakage.
Prompts also repeat across validation splits, so a row whose prompt was
already chosen (subsets in the given order) is skipped. From each subset N
eligible rows are drawn with a seeded sample (by primary id), after keeping
one row per prompt within a split. The file stores every chosen row (primary
id, prompt sha256) and the check; pool recipes read it (`pool.rows`).
`verify` re-runs the check.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TRAJECTORIES = ROOT / "datasets/eagle3-qwen3-8b-trajectories-v1/releases"


def _hash(values):
    return hashlib.sha256(json.dumps([int(v) for v in values]).encode()).hexdigest()


def rows(path):
    table = pq.read_table(path, columns=["primary_id", "input_ids", "prompt_length"])
    for pid, ids, plen in zip(
        table["primary_id"].to_pylist(),
        table["input_ids"].to_pylist(),
        table["prompt_length"].to_pylist(),
        strict=True,
    ):
        yield str(pid), _hash(ids[:plen]), _hash(ids)


def train_keys():
    files = sorted(TRAJECTORIES.glob("*/*/*/train.parquet"))
    ids, prompts, sequences, count = set(), set(), set(), 0
    for path in files:
        for pid, prompt, sequence in rows(path):
            ids.add(pid)
            prompts.add(prompt)
            sequences.add(sequence)
            count += 1
    check = {
        "train_split_files": len(files),
        "train_rows": count,
        "releases": sorted({p.parts[-4] for p in files}),
        "by": ["primary_id", "prompt token sha256", "full token sha256"],
    }
    return (ids, prompts, sequences), check


def _validation_path(release, name):
    return TRAJECTORIES / release / name.split("-")[0] / name / "validation.parquet"


def build(args):
    (ids, prompts, sequences), check = train_keys()
    subsets, excluded, used = {}, 0, set()
    for role, names in (("validation", args.validation), ("test", args.test)):
        for name in names:
            candidates = sorted(rows(_validation_path(args.release, name)))
            first = {}
            for r in candidates:  # one row per prompt within the split
                first.setdefault(r[1], r)
            # Also skip prompts already chosen: prompts repeat across splits.
            clean = [
                r
                for r in sorted(first.values())
                if r[0] not in ids
                and r[1] not in prompts
                and r[2] not in sequences
                and r[1] not in used
            ]
            excluded += len(candidates) - len(clean)
            if len(clean) < args.rows:
                raise ValueError(f"{name}: {len(clean)} eligible rows < {args.rows}")
            chosen = sorted(
                random.Random(f"{args.seed}:{name}").sample(clean, args.rows)
            )
            used.update(r[1] for r in chosen)
            subsets[name] = {
                "role": role,
                "candidates": len(candidates),
                "eligible": len(clean),
                "rows": [{"primary_id": r[0], "prompt_sha256": r[1]} for r in chosen],
            }
    chosen_prompts = [r["prompt_sha256"] for s in subsets.values() for r in s["rows"]]
    if len(set(chosen_prompts)) != len(chosen_prompts):
        raise ValueError("A prompt repeats within the evaluation rows")
    record = {
        "release": args.release,
        "seed": args.seed,
        "rows_per_subset": args.rows,
        "counts": {
            role: {
                category: sum(
                    len(s["rows"])
                    for n, s in subsets.items()
                    if s["role"] == role and n.startswith(category + "-")
                )
                for category in ("math", "code", "stem", "chat")
            }
            for role in ("validation", "test")
        },
        "excluded_candidates": excluded,
        "subsets": subsets,
        "check": {**check, "overlap": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(
        json.dumps({k: record[k] for k in ("counts", "excluded_candidates", "check")})
    )


def verify(args):
    record = json.loads(args.file.read_text())
    (ids, prompts, sequences), check = train_keys()
    bad = []
    for name, subset in record["subsets"].items():
        stored = {r["primary_id"]: r["prompt_sha256"] for r in subset["rows"]}
        found = {
            pid: (prompt, sequence)
            for pid, prompt, sequence in rows(_validation_path(record["release"], name))
            if pid in stored
        }
        if len(found) != len(stored):
            bad.append(f"{name}: rows missing from the release")
        for pid, (prompt, sequence) in found.items():
            if prompt != stored[pid]:
                bad.append(f"{name}/{pid}: prompt changed")
            if pid in ids or prompt in prompts or sequence in sequences:
                bad.append(f"{name}/{pid}: occurs in a train split")
    if bad:
        raise ValueError(f"Evaluation split check failed: {bad[:10]}")
    print(json.dumps({"verified": str(args.file), **check, "overlap": 0}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--release", required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--rows", type=int, required=True)
    b.add_argument("--validation", nargs="+", required=True)
    b.add_argument("--test", nargs="+", required=True)
    b.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    (build if args.command == "build" else verify)(args)


if __name__ == "__main__":
    main()
