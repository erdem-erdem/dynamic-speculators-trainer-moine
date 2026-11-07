"""S6b: persistent activation store; offline runs equal the online run."""

import json

import pytest
import torch

from speculators.generator import campaign, online_activations
from speculators.generator.activation_store import (
    ActivationStore,
    MissingActivationError,
    compare_with_fresh,
    fill_store,
    store_indices,
)
from speculators.generator.campaign_config import ActivationStoreSettings
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import OnlineActivations
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401

# Fields of result.json that legitimately differ between two runs: timing,
# throughput, cache/process accounting, and file hashes/paths (checkpoints embed
# the output paths; their tensors are compared separately).
VOLATILE = (
    "seconds",
    "elapsed",
    "time",
    "throughput",
    "per_second",
    "cache",
    "data",
    "process_metrics",
    "checkpoint_sha256",
)


def _store(cfg, tmp_path, mode):
    cfg.online.activation_store = ActivationStoreSettings(
        path=tmp_path / "store", mode=mode
    )


def _cache(cfg):
    inputs = load_local_inputs(cfg)
    return inputs, OnlineActivations(cfg, inputs.rows, inputs.banks["math"])


def _never_called(*_args, **_kwargs):
    raise AssertionError("offline run contacted vLLM")


def _stable(value):
    if isinstance(value, dict):
        return {
            key: _stable(item)
            for key, item in value.items()
            if not any(word in key for word in VOLATILE)
            and not key.endswith("path")
            and key != "profile"  # pilot profile: digest of the config (paths)
        }
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            item
            for key in sorted(left.keys() | right.keys())
            for item in _differences(left.get(key), right.get(key), f"{prefix}/{key}")
        ]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            item
            for i, (a, b) in enumerate(zip(left, right, strict=True))
            for item in _differences(a, b, f"{prefix}[{i}]")
        ]
    return [] if left == right else [(prefix, left, right)]


def _key(root, path):
    parts = path.relative_to(root).parts
    # Pilot directories are named by a digest of the config (incl. paths).
    return "/".join(parts[:1] + parts[2:] if parts[0] == "pilot" else parts)


def _results(root):
    return {
        _key(root, path): _stable(json.loads(path.read_text()))
        for path in sorted(root.rglob("result.json"))
    }


def _tensors(value, prefix=""):
    if torch.is_tensor(value):
        return {prefix: value}
    items = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, (list, tuple))
        else ()
    )
    return {
        name: tensor
        for key, item in items
        for name, tensor in _tensors(item, f"{prefix}/{key}").items()
    }


def _checkpoints(root):
    return {
        _key(root, path): _tensors(torch.load(path, weights_only=False))
        for path in sorted(root.rglob("checkpoint.pt"))
    }


def test_fill_writes_every_entry_once_and_reads_back(
    online_campaign,  # noqa: F811
    tmp_path,
):
    cfg, _, requests = online_campaign
    _store(cfg, tmp_path, "write")
    inputs, cache = _cache(cfg)
    indices = store_indices(inputs, cfg.experiment_subsets[:2])
    # Content-addressed: rows with identical tokens share an entry.
    distinct = {
        (
            summary,
            tuple(
                row["input_ids"][: row["prompt_length"]]
                if summary
                else row["input_ids"]
            ),
        )
        for row in (inputs.rows[i] for i in indices)
        for summary in (True, False)
    }
    assert 1 < len(distinct) < 2 * len(indices)
    try:
        stats = fill_store(cache, indices)
        assert stats["store_writes"] == len(distinct)
        extracted = len(requests)
        assert extracted >= len(distinct)
    finally:
        cache.close()
    assert not list(cfg.online.shared_dir.iterdir())
    inputs, cache = _cache(cfg)
    try:
        stats = fill_store(cache, indices)
        assert stats["store_writes"] == 0
        assert stats["store_hits"] == 2 * len(indices)
        assert len(requests) == extracted  # nothing re-extracted
    finally:
        cache.close()


def test_offline_campaign_equals_online_campaign(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, downloads, requests = online_campaign
    root = cfg.output_root
    _store(cfg, tmp_path, "write")
    assert campaign.run_campaign(cfg, "all")["runs"] == 4
    online = _results(root)
    online_checkpoints = _checkpoints(root)
    assert requests
    assert online
    assert online_checkpoints
    identity = json.loads(
        next(root.rglob("resolved_experiment.json")).read_text()
    )  # campaign provenance names the store content key
    assert identity["campaign"]["activation_store"]["layers"]

    cfg.output_root = tmp_path / "offline"
    cfg.staging_dir = tmp_path / "offline-staging"
    cfg.online.summary_cache_dir = tmp_path / "offline-summaries"
    _store(cfg, tmp_path, "read")
    monkeypatch.setattr(online_activations, "generate_hidden_states", _never_called)
    assert campaign.run_campaign(cfg, "all")["runs"] == 4
    offline = _results(cfg.output_root)
    assert offline.keys() == online.keys()
    assert _differences(offline, online) == []
    offline_checkpoints = _checkpoints(cfg.output_root)
    assert offline_checkpoints.keys() == online_checkpoints.keys()
    for name, tensors in online_checkpoints.items():
        assert tensors, name
        assert offline_checkpoints[name].keys() == tensors.keys(), name
        for key, tensor in tensors.items():
            assert torch.equal(offline_checkpoints[name][key], tensor), (name, key)
    assert not downloads


def test_read_mode_fails_loudly(online_campaign, tmp_path):  # noqa: F811
    cfg, _, _ = online_campaign
    _store(cfg, tmp_path, "read")
    with pytest.raises(FileNotFoundError, match="No activation store"):
        _cache(cfg)
    _store(cfg, tmp_path, "write")
    _, cache = _cache(cfg)
    cache.close()  # creates an empty store
    _store(cfg, tmp_path, "read")
    _, cache = _cache(cfg)
    try:
        with pytest.raises(MissingActivationError, match="no full entry"):
            cache.get_full(0)
        with pytest.raises(MissingActivationError, match="no summary entry"):
            cache.get_summary(1)
        with pytest.raises(RuntimeError, match="read-only"):
            cache.store.put([1, 2], {}, summary=False)
    finally:
        cache.close()


def test_store_identity_mismatch_and_unpinned_target(
    online_campaign,  # noqa: F811
    tmp_path,
):
    cfg, _, _ = online_campaign
    _store(cfg, tmp_path, "write")
    inputs, cache = _cache(cfg)
    cache.close()
    bank = inputs.banks["math"].model_copy(deep=True)
    bank.hidden_states.layer_ids = [*bank.hidden_states.layer_ids[:-2], 99]
    with pytest.raises(ValueError, match="holds"):
        ActivationStore(cfg.online.activation_store, bank)
    bank.models.target_revision = "main"
    with pytest.raises(ValueError, match="pinned target revision"):
        ActivationStore(cfg.online.activation_store, bank)


def test_compare_with_fresh_reports_equality_and_drift(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, _, _ = online_campaign
    _store(cfg, tmp_path, "write")
    inputs, cache = _cache(cfg)
    indices = store_indices(inputs, cfg.experiment_subsets[:1])[:3]
    try:
        fill_store(cache, indices)
        same = compare_with_fresh(cache, indices)
        assert same["bit_identical"]
        assert same["max_abs_diff"] == {
            "summary": 0.0,
            "hidden_states": 0.0,
            "token_ids": 0.0,
        }
        assert cache.store is not None
        original = online_activations.generate_hidden_states

        def drift(*args, **kwargs):
            path = original(*args, **kwargs)
            from safetensors.torch import load_file, save_file  # noqa: PLC0415

            payload = load_file(path)
            payload["hidden_states"] = payload["hidden_states"] + 1e-3
            save_file(payload, path)
            return path

        monkeypatch.setattr(online_activations, "generate_hidden_states", drift)
        drifted = compare_with_fresh(cache, indices)
        assert not drifted["bit_identical"]
        assert drifted["max_abs_diff"]["hidden_states"] == pytest.approx(1e-3, rel=0.01)
    finally:
        cache.close()
    assert torch.is_tensor(cache.summaries[indices[0]])


def test_fill_script_writes_record(online_campaign, tmp_path, monkeypatch):  # noqa: F811
    import runpy  # noqa: PLC0415
    import sys  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    cfg, _, _ = online_campaign
    _store(cfg, tmp_path, "write")
    recipe = tmp_path / "recipe.yaml"
    values = cfg.model_dump(mode="json")
    for key in ("training_recipe", "trajectory_recipe"):  # unset in this fixture
        if values[key] is None:
            values.pop(key)
    recipe.write_text(yaml.safe_dump(values))
    provenance = tmp_path / "store" / "provenance" / "vllm"
    script = "scripts/dflash_port/fill_activation_store.py"
    monkeypatch.setattr(sys, "argv", [script, str(recipe)])
    with pytest.raises(FileNotFoundError, match="Extraction provenance missing"):
        runpy.run_path(script, run_name="__main__")
    provenance.mkdir(parents=True)
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("test\n")
    script = "scripts/dflash_port/fill_activation_store.py"
    subset = cfg.experiment_subsets[0]
    monkeypatch.setattr(
        sys, "argv", [script, str(recipe), "--subsets", subset, "--compare", "2"]
    )
    runpy.run_path(script, run_name="__main__")
    (record,) = (tmp_path / "store" / "fills").glob("*.json")
    record = json.loads(record.read_text())
    assert record["subsets"] == [subset]
    assert record["comparison"]["bit_identical"]
    assert record["stats"]["store_writes"] > 0
