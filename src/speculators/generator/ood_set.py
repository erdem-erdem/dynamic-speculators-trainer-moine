"""Offline out-of-distribution evaluation set (user 2026-11-09).

Offline evaluation is teacher-forced: it replays a target response and needs
the target's hidden states for it. Serving needs prompts only. This module
builds the offline set:

1. Prompts: `RedHatAI/speculator_benchmarks` at the fork's pinned revision,
   `max_samples` per benchmark, chosen with the fork's seeded round-robin
   over source categories (`serving.validation.sample_domain`, as
   `serving.external_validation`). HumanEval uses the raw prompt; the others
   the Qwen3 chat template with thinking off (as the r0001 trajectories).
2. Responses: the target, served by `launch_vllm.py responses`, with the
   r0001 sampling settings (`R0001_SAMPLING`). Token ids come back from vLLM
   (`return_token_ids`), so nothing is re-tokenised.
3. Rows: trajectory-format rows (input_ids = prompt + response, loss on the
   response) in one subset per benchmark, `ood-<name>-00000`, validation
   only, written as parquet with a manifest of hashes and settings.

`with_ood(inputs, directory)` appends the rows to loaded local inputs (row
indices of the trajectories are unchanged) for store filling and offline
evaluation. They never enter a training job or its identity.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

REPO = "RedHatAI/speculator_benchmarks"
REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
DEFAULT_BENCHMARKS = ("HumanEval", "math_reasoning", "summarization")
RAW_PROMPT = {"HumanEval"}  # completion-style source prompts (fork convention)
# The r0001 trajectory responses (data.json manifest and row metadata).
R0001_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 0,
    "max_tokens": 4096,
}
CONTEXT_LENGTH = 8192
REQUIRED = ("input_ids", "loss_mask", "seq_len", "prompt_length", "primary_id")
VERSION = 1


def subset_name(benchmark):
    return f"ood-{benchmark.lower().replace('_', '-')}-00000"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_prompts(tokenizer, benchmarks, max_samples, seed, *, download=None):
    """Seeded prompt selection, formatted and tokenised. Returns
    ({benchmark: [record]}, [source file record])."""
    from speculators.generator.serving.validation import (  # noqa: PLC0415
        sample_domain,
    )

    if download is None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        def download(name):
            return hf_hub_download(
                REPO, f"{name}.jsonl", repo_type="dataset", revision=REVISION
            )

    selected, sources = {}, []
    for name in benchmarks:
        path = Path(download(name))
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        raw = name in RAW_PROMPT
        available = []
        for index, row in enumerate(rows):
            text = row["prompt"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or non-text prompt: {name}/{index}")
            if not raw:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            budget = min(R0001_SAMPLING["max_tokens"], CONTEXT_LENGTH - len(tokens))
            if not tokens or budget < 1:
                continue
            source_category = row.get("category", name)
            source_id = row.get("task_id", row.get("question_id", index))
            available.append(
                {
                    "row_id": f"{name}/{index:05d}/{source_id}",
                    "subset": f"{name}-{str(source_category).replace('/', '_')}",
                    "benchmark": name,
                    "prompt_format": "raw" if raw else "chat",
                    "prompt_token_ids": tokens,
                    "max_tokens": budget,
                }
            )
        chosen = sample_domain(available, max_samples, seed)
        if len(chosen) != max_samples:
            raise ValueError(f"{name}: {len(chosen)} eligible prompts < {max_samples}")
        selected[name] = chosen
        sources.append(
            {
                "file": f"{name}.jsonl",
                "sha256": _digest(path),
                "source_rows": len(rows),
                "eligible_rows": len(available),
                "selected_rows": len(chosen),
                "prompt_format": "raw" if raw else "chat",
            }
        )
    return selected, sources


def generate(post, model, record):
    """One response through `post(payload) -> json` (vLLM /v1/completions)."""
    payload = {
        "model": model,
        "prompt": record["prompt_token_ids"],
        "max_tokens": record["max_tokens"],
        "temperature": R0001_SAMPLING["temperature"],
        "top_p": R0001_SAMPLING["top_p"],
        "top_k": R0001_SAMPLING["top_k"],
        "seed": R0001_SAMPLING["seed"],
        "return_token_ids": True,
    }
    response = post(payload)
    (choice,) = response["choices"]
    tokens = choice.get("token_ids")
    if not tokens:
        raise ValueError(f"{record['row_id']}: vLLM returned no token ids")
    echoed = response.get("prompt_token_ids") or choice.get("prompt_token_ids")
    if echoed is not None and list(echoed) != record["prompt_token_ids"]:
        raise ValueError(f"{record['row_id']}: vLLM changed the prompt tokens")
    return {"token_ids": list(tokens), "finish_reason": choice["finish_reason"]}


def build(
    output,
    tokenizer,
    post,
    model,
    *,
    benchmarks,
    max_samples,
    seed,
    provenance,
    workers=32,
    download=None,
):
    """Select, generate and write the set (refuses an existing directory)."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new OOD set directory: {output}")
    selected, sources = select_prompts(
        tokenizer, benchmarks, max_samples, seed, download=download
    )
    pending = output.with_name(output.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    subsets = {}
    with ThreadPoolExecutor(workers) as pool:
        for name, records in selected.items():
            responses = list(pool.map(lambda r: generate(post, model, r), records))
            rows = []
            for record, response in zip(records, responses, strict=True):
                prompt = record["prompt_token_ids"]
                ids = prompt + response["token_ids"]
                rows.append(
                    {
                        "input_ids": ids,
                        "loss_mask": [0] * len(prompt)
                        + [1] * len(response["token_ids"]),
                        "seq_len": len(ids),
                        "prompt_length": len(prompt),
                        "primary_id": hashlib.sha256(
                            f"{REPO}@{REVISION}/{record['row_id']}".encode()
                        ).hexdigest(),
                        "row_id": record["row_id"],
                        "source_category": record["subset"],
                        "prompt_format": record["prompt_format"],
                        "finish_reason": response["finish_reason"],
                    }
                )
            path = pending / f"{subset_name(name)}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            subsets[subset_name(name)] = {
                "benchmark": name,
                "file": path.name,
                "sha256": _digest(path),
                "rows": len(rows),
                "truncated": sum(r["finish_reason"] == "length" for r in rows),
            }
    manifest = {
        "version": VERSION,
        "dataset": {"id": REPO, "revision": REVISION},
        "selection": {
            "benchmarks": list(benchmarks),
            "max_samples": max_samples,
            "seed": seed,
            "method": "seeded_source_category_round_robin",
            "raw_prompt": sorted(RAW_PROMPT),
            "chat_template": {"enable_thinking": False},
        },
        "sampling": R0001_SAMPLING,
        "context_length": CONTEXT_LENGTH,
        "sources": sources,
        "subsets": subsets,
        "provenance": provenance,
    }
    (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pending.rename(output)
    return manifest


def load(directory):
    """The set's rows per subset, checked against the manifest."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError(f"{directory}: OOD set version {manifest['version']}")
    rows = {}
    for name, entry in manifest["subsets"].items():
        path = directory / entry["file"]
        if _digest(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match the OOD manifest")
        rows[name] = pq.read_table(path).to_pylist()
    return manifest, rows


def with_ood(inputs, directory):
    """Local inputs with the OOD rows appended as validation-only subsets.
    Trajectory row indices are unchanged; `identity` is not changed (the OOD
    set is recorded by evaluations, never by training)."""
    from datasets import Dataset, concatenate_datasets  # noqa: PLC0415

    manifest, rows = load(directory)
    features = inputs.rows.features
    offset = len(inputs.rows)
    tables, subsets = [], dict(inputs.subsets)
    for name, members in rows.items():
        if name in subsets:
            raise ValueError(f"OOD subset {name} collides with a trajectory subset")
        records = []
        for r in members:
            values = {
                "input_ids": r["input_ids"],
                "loss_mask": r["loss_mask"],
                "seq_len": r["seq_len"],
                "prompt_length": r["prompt_length"],
                "primary_id": r["primary_id"],
                "category": name.rsplit("-", 1)[0],
                "subset_id": name,
                "role": "ood",
                "membership": "validation",
            }
            # Columns the trajectories lack are dropped; the ones every
            # consumer reads must exist.
            absent = sorted(set(REQUIRED) - set(features))
            if absent:
                raise ValueError(f"Trajectory rows lack required columns {absent}")
            records.append(
                {
                    **dict.fromkeys(features),
                    **{k: v for k, v in values.items() if k in features},
                }
            )
        tables.append(Dataset.from_list(records, features=features))
        subsets[name] = {
            "id": name,
            "train_indices": [],
            "validation_indices": list(range(offset, offset + len(records))),
        }
        offset += len(records)
    merged = replace(
        inputs, rows=concatenate_datasets([inputs.rows, *tables]), subsets=subsets
    )
    return merged, manifest
