"""Activation-store publishing: manifest, verification and the Hub upload."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module():
    spec = importlib.util.spec_from_file_location(
        "push_store", ROOT / "scripts/dflash_port/push_store.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _store(root):
    (root / "ab").mkdir(parents=True)
    (root / "identity.json").write_text('{"version": 1}')
    (root / "extractor.json").write_text('{"vllm": "0.31.0"}')
    (root / "ab" / "ab01.safetensors").write_bytes(b"one")
    (root / "ab" / "ab02.safetensors").write_bytes(b"two")
    (root / "provenance" / "vllm").mkdir(parents=True)
    (root / "provenance" / "vllm" / "vllm_command.txt").write_text("# vllm: 0.31.0")
    return root


def test_manifest_and_verify(tmp_path):
    push_store = _module()
    root = _store(tmp_path / "store")
    manifest = push_store.write_manifest(root, workers=2)
    assert manifest["entries"] == 2
    assert manifest["identity"] == {"version": 1}
    assert "provenance/vllm/vllm_command.txt" in manifest["files"]
    # A downloaded copy carries HF's cache folder and lock files: ignored.
    (root / ".cache" / "huggingface").mkdir(parents=True)
    (root / ".cache" / "huggingface" / "x.metadata").write_text("meta")
    (root / "ab" / "ab01.safetensors.lock").write_text("")
    assert push_store.verify(root, workers=2)["entries"] == 2
    (root / "ab" / "ab02.safetensors").write_bytes(b"TWO")
    with pytest.raises(ValueError, match="1 changed"):
        push_store.verify(root, workers=2)
    (root / "ab" / "ab02.safetensors").unlink()
    with pytest.raises(ValueError, match="1 missing"):
        push_store.verify(root, workers=2)


def test_refuses_unfinished_entries_and_non_stores(tmp_path):
    push_store = _module()
    root = _store(tmp_path / "store")
    (root / "ab" / "ab03.123.456.pending").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="unfinished"):
        push_store.write_manifest(root, workers=1)
    with pytest.raises(FileNotFoundError, match="not an activation store"):
        push_store.write_manifest(tmp_path / "empty", workers=1)


def test_push_uploads_and_pins_the_commit(tmp_path, monkeypatch):
    push_store = _module()
    root = _store(tmp_path / "store")
    calls = {}

    class Api:
        def create_repo(self, repo, **kwargs):
            calls["create"] = (repo, kwargs)

        def upload_large_folder(self, **kwargs):
            calls["upload"] = kwargs

        def list_repo_commits(self, repo, repo_type):
            return [SimpleNamespace(commit_id="c" * 40)]

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))
    with pytest.raises(SystemExit):  # visibility is never defaulted
        push_store.main(["push", str(root), "--repo", "u/r"])
    push_store.main(["push", str(root), "--repo", "u/r", "--visibility", "private"])
    assert calls["create"] == (
        "u/r",
        {"repo_type": "dataset", "private": True, "exist_ok": True},
    )
    assert calls["upload"]["repo_type"] == "dataset"
    assert "hub.json" in calls["upload"]["ignore_patterns"]
    record = json.loads((root / "hub.json").read_text())
    assert record["revision"] == "c" * 40
    assert record["entries"] == 2
    # hub.json is local only: a re-verify of the pushed store still passes.
    assert push_store.verify(root, workers=1)["entries"] == 2
