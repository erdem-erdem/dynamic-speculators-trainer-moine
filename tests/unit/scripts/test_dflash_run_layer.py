"""S6d: DFlash run layer (run_arm / run_matrix) and committed recipes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "examples/dflash-lora"


def _module(name):
    path = ROOT / "scripts/dflash_port" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_arm = _module("run_arm")
run_matrix = _module("run_matrix")


@pytest.mark.parametrize("matrix", sorted((RECIPES / "matrices").glob("*.yaml")))
def test_committed_matrices_and_recipes_resolve(matrix, tmp_path):
    plan = run_matrix.main([str(matrix), "--output-root", str(tmp_path), "--dry-run"])
    assert plan
    assert not any(item["completed"] for item in plan)


def test_m0_arms_are_matched():
    jobs = {}
    for arm in ("base", "generator", "static_lora"):
        cfg = run_arm.resolve(
            RECIPES / f"m0/{arm}.yaml",
            tier="tier1",
            output_root="/tmp/unused",
            wandb=True,
            store_mode=None,
        )
        jobs[arm] = run_arm.training_job(cfg)
        assert cfg.adapter.layers == "all"
        assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert cfg.wandb.enabled
        assert cfg.online.activation_store.mode == "read"
    for key in ("subsets", "validation_subsets", "updates", "training_examples"):
        assert len({json.dumps(job[key]) for job in jobs.values()}) == 1, key
    assert jobs["static_lora"]["arm"] == "fresh_lora"
    assert jobs["generator"]["arm"] == "direct"


def test_overrides(tmp_path):
    cfg = run_arm.resolve(
        RECIPES / "m0/generator.yaml",
        tier="tier9",
        output_root=tmp_path,
        wandb=False,
        store_mode="write",
    )
    assert cfg.output_root == tmp_path.resolve() / "tier9" / cfg.name
    assert cfg.input_root == cfg.output_root / "inputs"
    assert not cfg.wandb.enabled
    assert cfg.online.activation_store.mode == "write"


def test_matrix_rejects_bad_entries(tmp_path):
    bad = tmp_path / "bad.yaml"
    for content, error in (
        ({"tier": "t", "runs": []}, ValueError),
        ({"tier": "t", "runs": [{"recipe": "x.yaml", "mode": "fly"}]}, ValueError),
        (
            {"tier": "t", "runs": [{"recipe": "missing.yaml", "mode": "base"}]},
            FileNotFoundError,
        ),
    ):
        bad.write_text(yaml.safe_dump(content))
        with pytest.raises(error):
            run_matrix.load(bad)


def test_matrix_skips_completed_entries(tmp_path):
    matrix = RECIPES / "matrices/tier1.yaml"
    cfg = run_arm.resolve(
        RECIPES / "m0/base.yaml",
        tier="tier1",
        output_root=tmp_path,
        wandb=True,
        store_mode=None,
    )
    cfg.output_root.mkdir(parents=True)
    (cfg.output_root / "run_record_base.json").write_text('{"status": "completed"}')
    plan = run_matrix.main([str(matrix), "--output-root", str(tmp_path), "--dry-run"])
    assert [item["completed"] for item in plan] == [True, False, False, False, False]
