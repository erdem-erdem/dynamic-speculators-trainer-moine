"""Persistent, content-addressed store of vLLM-extracted activations (S6b).

Online campaigns extract target hidden states through vLLM per row. With an
`online.activation_store`, every extraction is keyed by what determines it:

    target model + pinned revision, extracted layers, dtype, kind, token ids

so one store serves every run (arms, ablations, the 4K and 24K pools) that
reads the same rows from the same target, whatever drafter is trained.

Modes:
- `write`: extract through vLLM when an entry is missing and save it; present
  entries are read (a fill pass or a first online run populates the store);
- `read`: offline; never contacts vLLM. A missing entry raises.

Entries are the exact payloads the online path checks and uses, so an offline
run consumes the same tensors as the online run that wrote them.
"""

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import torch
from safetensors.torch import load, save_file

from speculators.bank.artifacts import digest, write_json
from speculators.bank.progress import report

STORE_VERSION = 1
HASH_LINE = re.compile(r"[0-9a-f]{64}  \S+")


def check_extraction_provenance(provenance):
    """The extractor must have been launched with `launch_vllm.py
    --strict-provenance --provenance-dir <store>/provenance/vllm`: hashed
    target weights and an identified vLLM (git repo or install fingerprint).
    Returns the extractor record; raises otherwise."""
    provenance = Path(provenance)
    absent = [
        name
        for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt")
        if not (provenance / name).is_file()
    ]
    if absent:
        raise FileNotFoundError(
            f"Extraction provenance missing in {provenance}: {absent}; launch vLLM "
            "with --strict-provenance --provenance-dir <store>/provenance/vllm"
        )
    weights = (provenance / "checkpoint_sha256.txt").read_text()
    lines = [line for line in weights.splitlines() if line]
    if not lines or not all(HASH_LINE.fullmatch(line) for line in lines):
        raise ValueError(f"{provenance}/checkpoint_sha256.txt has no weight hashes")
    patch = (provenance / "vllm.patch").read_text()
    install = provenance / "vllm_install.txt"
    if patch.startswith("# repo:"):
        build = patch.splitlines()[0]
    elif install.is_file():
        build = install.read_text().strip()
    else:
        raise ValueError(f"{provenance}: vLLM is neither a git repo nor fingerprinted")
    command = (provenance / "vllm_command.txt").read_text().splitlines()
    version = next((line for line in command if line.startswith("# vllm:")), None)
    if version is None:
        raise ValueError(f"{provenance}/vllm_command.txt records no vLLM version")
    return {
        "vllm": version.removeprefix("# vllm:").strip(),
        "vllm_build": build,
        "target_weights_sha256": hashlib.sha256(weights.encode()).hexdigest(),
    }


SUMMARY_KEY = "summary"


class MissingActivationError(FileNotFoundError):
    """A `read`-mode store has no entry for a requested row."""


class ActivationStore:
    def __init__(self, settings, bank):
        if bank.models.target_revision in {None, "", "main"}:
            raise ValueError("Activation store requires a pinned target revision")
        self.mode = settings.mode
        self.root = Path(settings.path).resolve()
        self.identity = {
            "version": STORE_VERSION,
            "target": bank.models.target,
            "target_revision": bank.models.target_revision,
            "layers": list(bank.hidden_states.layer_ids),
            "dtype": bank.hidden_states.dtype,
        }
        record = self.root / "identity.json"
        extractor_record = self.root / "extractor.json"
        if record.is_file():
            found = json.loads(record.read_text())
            if found != self.identity:
                raise ValueError(
                    f"Activation store {self.root} holds {found}, "
                    f"run needs {self.identity}"
                )
        elif self.mode == "read":
            raise FileNotFoundError(f"No activation store at {self.root}")
        else:
            write_json(record, self.identity)
        # Which extractor produced the entries (not part of the content key).
        if self.mode == "write":
            self.extractor = check_extraction_provenance(
                self.root / "provenance" / "vllm"
            )
            if extractor_record.is_file():
                found = json.loads(extractor_record.read_text())
                if found != self.extractor:
                    raise ValueError(
                        f"Activation store {self.root} was extracted by {found}; "
                        f"this extractor is {self.extractor}. Entries from two "
                        "extractors are never mixed: use a new store path."
                    )
            else:
                write_json(extractor_record, self.extractor)
        else:
            if not extractor_record.is_file():
                raise FileNotFoundError(f"{extractor_record} missing")
            self.extractor = json.loads(extractor_record.read_text())

    def path(self, tokens, *, summary) -> Path:
        key = digest(
            {
                **self.identity,
                "kind": SUMMARY_KEY if summary else "full",
                "tokens": [int(token) for token in tokens],
            }
        )
        return self.root / key[:2] / f"{key}.safetensors"

    def get(self, tokens, *, summary):
        """The stored payload, None when absent in `write` mode."""
        path = self.path(tokens, summary=summary)
        if not path.is_file():
            if self.mode == "read":
                raise MissingActivationError(
                    f"Offline activation store {self.root} has no "
                    f"{'summary' if summary else 'full'} entry for a "
                    f"{len(tokens)}-token row ({path.name})"
                )
            return None
        payload = load(path.read_bytes())
        return payload[SUMMARY_KEY] if summary else payload

    def put(self, tokens, payload, *, summary) -> bool:
        """Save an entry; False when it already exists."""
        if self.mode != "write":
            raise RuntimeError("Activation store is read-only")
        path = self.path(tokens, summary=summary)
        if path.is_file():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {SUMMARY_KEY: payload} if summary else payload
        pending = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.pending")
        try:
            save_file({k: v.contiguous() for k, v in tensors.items()}, pending)
            # Exclusive, atomic publish: concurrent writers of the same content
            # (threads or processes) never overwrite; exactly one wins.
            try:
                os.link(pending, path)
            except FileExistsError:
                return False
        finally:
            pending.unlink(missing_ok=True)
        return True


def store_indices(inputs, subset_names):
    """Every train and validation row of the named subsets, in a stable order."""
    unknown = sorted(set(subset_names) - set(inputs.subsets))
    if unknown:
        raise ValueError(f"Unknown subsets: {unknown}")
    return list(
        dict.fromkeys(
            index
            for name in subset_names
            for key in ("train_indices", "validation_indices")
            for index in inputs.subsets[name][key]
        )
    )


def fill_store(cache, indices, *, log_every=500):
    """Extract every (summary, full) entry the online path needs into a `write`
    store. Present entries are read back and checked, not re-extracted."""
    if cache.store is None or cache.store.mode != "write":
        raise ValueError("fill_store requires online.activation_store mode write")
    keys = [(index, summary) for index in indices for summary in (True, False)]
    window = 2 * cache.settings.concurrency
    for start in range(0, len(keys), window):
        futures = [
            cache.pool.submit(cache._generate, *key)  # noqa: SLF001
            for key in keys[start : start + window]
        ]
        for future in futures:
            future.result()
        # Filling never trains: drop full payloads instead of caching them.
        with cache.lock:
            cache.ram.full.clear()
            cache.ram.bytes = 0
        done = min(start + window, len(keys))
        if done // log_every != (done - len(futures)) // log_every or done == len(keys):
            report(f"Activation store fill: {done}/{len(keys)} {cache.snapshot()}")
    return cache.snapshot()


def compare_with_fresh(cache, indices):
    """Online ≡ offline check: re-extract rows through vLLM, bypassing the
    store, and compare with the stored entries. Returns per-kind max |diff| and
    whether every tensor is bit-identical."""
    store = cache.store
    if store is None:
        raise ValueError("compare_with_fresh requires an activation store")
    result = {"rows": len(indices), "bit_identical": True, "max_abs_diff": {}}
    cache.store = None
    try:
        for index in indices:
            for summary in (True, False):
                row = cache.rows[index]
                tokens = (
                    row["input_ids"][: row["prompt_length"]]
                    if summary
                    else row["input_ids"]
                )
                stored = store.get(tokens, summary=summary)
                if stored is None:
                    raise MissingActivationError(f"No stored entry for row {index}")
                fresh = cache._generate(index, summary)  # noqa: SLF001
                pairs = (
                    {"summary": (stored, fresh)}
                    if summary
                    else {key: (stored[key], fresh[key]) for key in stored}
                )
                if not summary and stored.keys() != fresh.keys():
                    raise ValueError(f"Stored/fresh payload keys differ for {index}")
                for key, (old, new) in pairs.items():
                    if old.shape != new.shape or old.dtype != new.dtype:
                        raise ValueError(f"Stored/fresh {key} differ in shape/dtype")
                    equal = torch.equal(old, new)
                    result["bit_identical"] &= equal
                    diff = (
                        0.0
                        if equal or not old.is_floating_point()
                        else float((old.float() - new.float()).abs().max())
                    )
                    if not old.is_floating_point() and not equal:
                        raise ValueError(f"Stored/fresh {key} (integer) differ")
                    previous = result["max_abs_diff"].get(key, 0.0)
                    result["max_abs_diff"][key] = max(previous, diff)
    finally:
        cache.store = store
        with cache.lock:
            cache.ram.full.clear()
            cache.ram.bytes = 0
    return result
