"""Pinned, drafter-agnostic preparation and load-time verification of DFlash drafters.

Any released DFlash checkpoint (D12: not tied to one publisher) is prepared the
same way: the source is downloaded at an exact commit SHA, converted with the
upstream `DFlashConverter` (verifier-owned embedding / LM head / norm filled
from the verifier), and recorded in `preparation.json`:

- source repository and commit SHA, SHA256 of the source config and weights;
- SHA256 of the converted weights and config;
- the geometry the generator depends on (aux layers, block size, mask id,
  draft layers, hidden size, vocabulary).

`verify_prepared_drafter` re-hashes the converted weights and checks the record
against the bank before a local drafter directory is used for training. Any
mismatch raises; nothing is substituted.
"""

import json
import re
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from speculators.bank.artifacts import file_digest, write_json
from speculators.provenance import atomic_write

PREPARATION_VERSION = 1
SOURCE_PATTERNS = ["config.json", "*.safetensors", "*.safetensors.index.json"]
_SHA = re.compile(r"[0-9a-f]{40}")


def _geometry(config: dict) -> dict:
    layer = config["transformer_layer_config"]
    return {
        "aux_hidden_state_layer_ids": config["aux_hidden_state_layer_ids"],
        "block_size": config["block_size"],
        "mask_token_id": config["mask_token_id"],
        "draft_layers": layer["num_hidden_layers"],
        "hidden_size": layer["hidden_size"],
        "draft_vocab_size": config["draft_vocab_size"],
    }


def weight_digests(path: Path) -> dict[str, str]:
    """SHA256 of every safetensors file (and index) in a checkpoint directory."""
    files = sorted(path.glob("*.safetensors")) + sorted(
        path.glob("*.safetensors.index.json")
    )
    if not any(name.suffix == ".safetensors" for name in files):
        raise FileNotFoundError(f"No safetensors weights in {path}")
    return {name.name: file_digest(name) for name in files}


def _load_native(source, output, verifier) -> list[str]:
    """Save a native speculators DFlash checkpoint, filling only the
    verifier-owned tensors its files omit; trained drafter weights are never
    overwritten. Returns the names that were missing from the source files."""
    from safetensors import safe_open  # noqa: PLC0415

    from speculators.models.dflash import (  # noqa: PLC0415
        DFlashDraftModel,
        DFlashSpeculatorConfig,
    )

    stored = set()
    for name in sorted(source.glob("*.safetensors")):
        with safe_open(name, framework="pt", device="cpu") as tensors:
            stored.update(tensors.keys())
    config = DFlashSpeculatorConfig.from_pretrained(source)
    config.speculators_config.verifier.name_or_path = str(verifier)
    with warnings.catch_warnings():
        # Upstream warns and keeps weight=1.0 when the verifier has no final norm.
        warnings.filterwarnings("error", message="Could not find final norm")
        model = DFlashDraftModel.from_pretrained(str(source), config=config)
        # Default flags: embed_tokens / lm_head only where NaN (missing);
        # verifier_lm_head / verifier_norm are verifier-owned and always loaded.
        model._load_verifier_weights()  # noqa: SLF001
    state = model.state_dict()
    for name in stored:
        # Loaded source tensors must survive unchanged.
        if name in state and not _same_as_stored(source, name, state[name]):
            raise ValueError(f"Prepared DFlash drafter changed source tensor {name}")
    nonfinite = [
        name
        for name, tensor in state.items()
        if tensor.is_floating_point() and not tensor.isfinite().all()
    ]
    if nonfinite:
        raise ValueError(f"Prepared DFlash drafter has non-finite tensors: {nonfinite}")
    model.save_pretrained(str(output))
    return sorted(set(state) - stored)


def _same_as_stored(source, name, tensor) -> bool:
    from safetensors import safe_open  # noqa: PLC0415

    for path in sorted(source.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if name in tensors.keys():  # noqa: SIM118 -- safe_open has no __contains__
                return torch.equal(tensors.get_tensor(name).to(tensor.dtype), tensor)
    raise KeyError(name)


def prepare_dflash(repository, revision, verifier, output_root) -> Path:
    """Prepare `repository@revision` into `output_root/<revision>` with a record.

    External (publisher-format) checkpoints are converted; native speculators
    DFlash checkpoints are loaded as they are.
    """
    if not _SHA.fullmatch(revision):
        raise ValueError(f"DFlash drafter revision must be a commit SHA: {revision!r}")
    source = Path(
        snapshot_download(repository, revision=revision, allow_patterns=SOURCE_PATTERNS)
    )
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"{repository}@{revision} has no config.json")
    source_weights = weight_digests(source)
    output = Path(output_root) / revision
    if output.exists():
        raise FileExistsError(f"Prepared drafter already exists: {output}")
    source_config = json.loads((source / "config.json").read_text())
    if "speculators_model_type" in source_config:
        if source_config["speculators_model_type"] != "dflash":
            raise ValueError(f"{repository} is not a DFlash checkpoint")
        source_format = "speculators"
        filled = _load_native(source, output, verifier)
    else:
        # Imported here: conversion pulls in the model code.
        from speculators.convert.dflash.converter import (  # noqa: PLC0415
            DFlashConverter,
        )

        source_format = "external"
        DFlashConverter().convert(
            source, output, base_model=str(verifier), validate=True
        )
        filled = "converter: embed_tokens, lm_head, verifier_lm_head, verifier_norm"
    config = json.loads((output / "config.json").read_text())
    if config.get("speculators_model_type") != "dflash":
        raise ValueError(f"Prepared checkpoint is not DFlash: {output}")
    atomic_write(output / "original_config.json", (source / "config.json").read_text())
    write_json(
        output / "preparation.json",
        {
            "preparation_version": PREPARATION_VERSION,
            "family": "dflash",
            "source_repository": repository,
            "source_revision": revision,
            "source_format": source_format,
            "source_config_sha256": file_digest(source / "config.json"),
            "source_weights_sha256": source_weights,
            "verifier": str(verifier),
            "verifier_filled": filled,
            "config_sha256": file_digest(output / "config.json"),
            "weights_sha256": weight_digests(output),
            **_geometry(config),
        },
    )
    return output


def verify_prepared_drafter(path, bank) -> dict:
    """Check a prepared local DFlash drafter against its record and the bank."""
    path = Path(path)
    record_path = path / "preparation.json"
    if not record_path.is_file():
        raise FileNotFoundError(f"Missing DFlash preparation record: {record_path}")
    record = json.loads(record_path.read_text())
    if record.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError(f"Unsupported preparation record: {record_path}")
    mismatches = []

    def check(name, found, expected):
        if found != expected:
            mismatches.append(f"{name}: found {found!r}, expected {expected!r}")

    check("config_sha256", file_digest(path / "config.json"), record["config_sha256"])
    check(
        "weights_sha256",
        weight_digests(path),
        record["weights_sha256"],
    )
    geometry = _geometry(json.loads((path / "config.json").read_text()))
    check("geometry", geometry, {key: record[key] for key in geometry})
    check("drafter", bank.models.drafter, record["source_repository"])
    check("drafter_revision", bank.models.drafter_revision, record["source_revision"])
    check(
        "aux layers",
        bank.hidden_states.layer_ids[:-1],
        record["aux_hidden_state_layer_ids"],
    )
    if mismatches:
        raise ValueError(f"Prepared DFlash drafter {path} mismatch: {mismatches}")
    return record
