"""Run-directory publishing: refusals, repo layout, manifest and model card."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module():
    spec = importlib.util.spec_from_file_location(
        "push_run", ROOT / "scripts/dflash_port/push_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validation(update, score):
    return {
        "update": update,
        "score": score,
        "categories": {"chat": score, "math": score},
        "contexts": {"1": {"eal": 7.0 + score}},
    }


def _run(tmp_path, *, eval_status="completed"):
    store = tmp_path / "store"
    (store / "provenance" / "vllm").mkdir(parents=True)
    (store / "provenance" / "vllm" / "vllm_command.txt").write_text("# vllm: 0.31.0")
    (store / "fills" / "t.provenance").mkdir(parents=True)
    (store / "fills" / "t.json").write_text("{}")
    (store / "fills" / "t.provenance" / "vllm_command.txt").write_text("cmd")
    (store / "identity.json").write_text('{"version": 1}')
    (store / "extractor.json").write_text('{"vllm": "0.31.0"}')
    (store / "ab").mkdir()
    (store / "ab" / "entry.safetensors").write_bytes(b"never uploaded")
    run = tmp_path / "run"
    result_dir = run / "pretraining" / "label"
    result_dir.mkdir(parents=True)
    (result_dir / "checkpoint.pt").write_bytes(b"weights")
    result = {
        "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest(),
        "completed_updates": 2,
        "history": [_validation(0, 0.5), _validation(2, 0.6)],
    }
    (result_dir / "result.json").write_text(json.dumps(result))
    (run / ".structure.lock").write_text("")
    record = {
        "status": "completed",
        "git_sha": "a" * 40,
        "seed": 42,
        "drafter": {"repository": "z-lab/x", "revision": "b" * 40},
        "job": {
            "label": "label",
            "stage": "pretraining",
            "arm": "direct",
            "vllm_provenance_dir": str(store / "provenance" / "vllm"),
        },
    }
    (run / "run_record_train.json").write_text(json.dumps(record))
    if eval_status is not None:
        (run / "run_record_eval.json").write_text(json.dumps({"status": eval_status}))
    return run, store


def test_layout_manifest_and_card(tmp_path):
    push_run = _module()
    run, _ = _run(tmp_path)
    _, files, manifest, card = push_run.build(run, require_eval=True)
    assert set(files) == {
        "run/pretraining/label/checkpoint.pt",
        "run/pretraining/label/result.json",
        "run/run_record_train.json",
        "run/run_record_eval.json",
        "provenance/store/identity.json",
        "provenance/store/extractor.json",
        "provenance/store/provenance/vllm/vllm_command.txt",
        "provenance/store/fills/t.json",
        "provenance/store/fills/t.provenance/vllm_command.txt",
    }  # no lock file, no store entries
    for name, path in files.items():
        assert manifest["files"][name]["sha256"] == push_run.sha256(path)
    assert manifest["checkpoint_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert "# label" in card
    assert "| 2 | 0.6000 | 7.600 | 0.6000 | 0.6000 |" in card
    assert "Offline evaluation" in card


def test_refusals(tmp_path):
    push_run = _module()
    run, _ = _run(tmp_path, eval_status="running")
    with pytest.raises(RuntimeError, match="offline evaluation is not completed"):
        push_run.build(run)
    (run / "run_record_eval.json").unlink()
    push_run.build(run)  # no eval: allowed unless required
    with pytest.raises(RuntimeError, match="no offline evaluation"):
        push_run.build(run, require_eval=True)
    (run / "pretraining" / "label" / "checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from result.json"):
        push_run.build(run)
    record = json.loads((run / "run_record_train.json").read_text())
    (run / "run_record_train.json").write_text(json.dumps({**record, "status": "x"}))
    with pytest.raises(RuntimeError, match="training is not completed"):
        push_run.build(run)


def test_push_is_one_commit_and_records_the_revision(tmp_path):
    push_run = _module()
    run, store = _run(tmp_path)
    (store / "hub.json").write_text('{"repo": "erdem-erdem/dyn-spec-dflash-store-x"}')
    calls = {}

    class Api:
        def create_repo(self, repo, **kwargs):
            calls["repo"] = (repo, kwargs)

        def create_commit(self, **kwargs):
            calls["commit"] = kwargs
            return SimpleNamespace(oid="c" * 40, commit_url="https://hf/commit")

    report = push_run.push(run, "erdem-erdem/dyn-spec-dflash-label", False, api=Api())
    assert calls["repo"] == (
        "erdem-erdem/dyn-spec-dflash-label",
        {"repo_type": "model", "private": False, "exist_ok": True},
    )
    names = [op.path_in_repo for op in calls["commit"]["operations"]]
    assert "MANIFEST.json" in names
    assert "README.md" in names
    assert "run/pretraining/label/checkpoint.pt" in names
    card = next(
        op.path_or_fileobj
        for op in calls["commit"]["operations"]
        if op.path_in_repo == "README.md"
    )
    assert b"erdem-erdem/dyn-spec-dflash-store-x" in card
    assert report["revision"] == "c" * 40
    assert json.loads((run / "hub.json").read_text())["revision"] == "c" * 40
    # hub.json stays local: a second build does not publish it.
    _, files, _, _ = push_run.build(run)
    assert "run/hub.json" not in files
