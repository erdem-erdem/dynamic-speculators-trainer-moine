"""S6d: DFlash run layer (run_arm / run_matrix) and committed recipes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "examples/dflash-lora"
SHA = "0123456789abcdef0123456789abcdef01234567"


def _module(name):
    path = ROOT / "scripts/dflash_port" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_arm = _module("run_arm")
run_matrix = _module("run_matrix")


def _machine(tmp_path):
    path = tmp_path / "machine.local.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "target_model_path": "/models/qwen3-8b",
                "drafter_revision": SHA,
                "drafter_model_path": f"/drafters/{SHA}",
            }
        )
    )
    return path


def _resolve(recipe, tmp_path, **overrides):
    options = {
        "tier": "tier1",
        "output_root": tmp_path / "runs",
        "wandb": True,
        "store_mode": None,
        "machine": tmp_path / "absent.yaml",
    }
    return run_arm.resolve(RECIPES / recipe, **{**options, **overrides})


@pytest.mark.parametrize("matrix", sorted((RECIPES / "matrices").glob("*.yaml")))
def test_committed_matrices_and_recipes_resolve(matrix, tmp_path):
    argv = [str(matrix), "--output-root", str(tmp_path), "--dry-run"]
    plan = run_matrix.main([*argv, "--machine", str(tmp_path / "absent.yaml")])
    assert plan
    assert not any(item["completed"] for item in plan)
    # Without this box's machine config the drafter pin is unresolved.
    assert all(item["unresolved"] for item in plan)
    plan = run_matrix.main([*argv, "--machine", str(_machine(tmp_path))])
    assert not any(item["unresolved"] for item in plan)


def test_m0_arms_are_matched(tmp_path):
    jobs = {}
    for arm in ("base", "generator", "static_lora"):
        cfg = _resolve(f"m0/{arm}.yaml", tmp_path).cfg
        jobs[arm] = run_arm.training_job(cfg)
        assert cfg.adapter.layers == "all"
        assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert cfg.wandb.enabled
        assert cfg.hub_push
        assert cfg.online.activation_store.mode == "read"
    for key in ("subsets", "validation_subsets", "updates", "training_examples"):
        assert len({json.dumps(job[key]) for job in jobs.values()}) == 1, key
    assert jobs["static_lora"]["arm"] == "fresh_lora"
    assert jobs["generator"]["arm"] == "direct"
    assert jobs["generator"]["hub_repo"].startswith("erdem-erdem/dflash-b16-")


def test_overrides_and_machine_config(tmp_path):
    resolved = _resolve(
        "m0/generator.yaml",
        tmp_path,
        tier="tier9",
        wandb=False,
        store_mode="write",
        machine=_machine(tmp_path),
        hub=False,
        materialize=True,
    )
    cfg = resolved.cfg
    assert cfg.output_root == (tmp_path / "runs").resolve() / "tier9" / cfg.name
    assert cfg.input_root == cfg.output_root / "inputs"
    assert not cfg.wandb.enabled
    assert cfg.online.activation_store.mode == "write"
    assert str(cfg.online.target_model_path) == "/models/qwen3-8b"
    assert str(cfg.online.drafter_model_path) == f"/drafters/{SHA}"
    assert "hub_repo" not in run_arm.training_job(cfg)
    written = yaml.safe_load(cfg.training_recipe.read_text())
    assert written["models"]["drafter_revision"] == SHA
    assert written["training"]["block_drafter"]["loss_fn"] == "ce"
    assert not resolved.unresolved


def test_real_run_refuses_unresolved_pins(tmp_path):
    with pytest.raises(ValueError, match="Unresolved"):
        _resolve("m0/generator.yaml", tmp_path, materialize=True)


def test_digest_ignores_toggles_but_not_the_drafter_pin(tmp_path):
    machine = _machine(tmp_path)
    base = run_arm.configuration_digest(
        _resolve("m0/base.yaml", tmp_path, machine=machine)
    )
    toggled = _resolve(
        "m0/base.yaml",
        tmp_path,
        machine=machine,
        wandb=False,
        hub=False,
        store_mode="write",
    )
    assert run_arm.configuration_digest(toggled) == base
    other = yaml.safe_load(machine.read_text())
    other["drafter_revision"] = "f" * 40
    machine.write_text(yaml.safe_dump(other))
    pinned = run_arm.configuration_digest(
        _resolve("m0/base.yaml", tmp_path, machine=machine)
    )
    assert pinned != base


def test_matrix_rejects_bad_entries(tmp_path):
    bad = tmp_path / "bad.yaml"
    missing = {"tier": "t", "runs": [{"recipe": "missing.yaml", "mode": "base"}]}
    for content, error in (
        ({"tier": "t", "runs": []}, ValueError),
        ({"tier": "t", "runs": [{"recipe": "x.yaml", "mode": "fly"}]}, ValueError),
        (missing, FileNotFoundError),
    ):
        bad.write_text(yaml.safe_dump(content))
        with pytest.raises(error):
            run_matrix.load(bad)


def test_matrix_skips_completed_entries_of_the_same_configuration(tmp_path):
    matrix = RECIPES / "matrices/tier1.yaml"
    machine = _machine(tmp_path)
    resolved = _resolve("m0/base.yaml", tmp_path, machine=machine)
    resolved.cfg.output_root.mkdir(parents=True)
    record = resolved.cfg.output_root / "run_record_base.json"
    digest = run_arm.configuration_digest(resolved)
    record.write_text(
        json.dumps({"status": "completed", "configuration_digest": digest})
    )
    argv = [
        str(matrix),
        "--output-root",
        str(tmp_path / "runs"),
        "--machine",
        str(machine),
        "--dry-run",
    ]
    plan = run_matrix.main(argv)
    assert [item["completed"] for item in plan] == [True, False, False, False, False]
    run_matrix.main([*argv, "--no-wandb", "--no-hub"])  # toggles: still the same
    record.write_text(json.dumps({"status": "completed", "configuration_digest": "x"}))
    with pytest.raises(ValueError, match="different configuration"):
        run_matrix.main(argv)
