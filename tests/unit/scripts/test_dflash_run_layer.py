"""S6d: DFlash run layer (run_arm / run_matrix) and committed recipes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from speculators.generator.campaign import apply_row_selection
from speculators.generator.structure_study import Pool

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
        (
            {
                "tier": "t",
                "runs": [{"recipe": "x.yaml", "mode": "train", "group_size": 2}],
            },
            ValueError,
        ),
        (
            {
                "tier": "t",
                "runs": [{"recipe": "x.yaml", "mode": "eval", "group_size": 0}],
            },
            ValueError,
        ),
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


def test_grouped_eval_entry_is_its_own_record(tmp_path):
    """M6: an eval with `group_size` passes --group-size and uses its own
    record, so it never collides with (or is skipped as) the plain eval."""
    machine = _machine(tmp_path)
    resolved = _resolve("m0/static_lora.yaml", tmp_path, machine=machine)
    output = resolved.cfg.output_root
    output.mkdir(parents=True)
    digest = run_arm.configuration_digest(resolved)
    (output / "run_record_eval.json").write_text(
        json.dumps({"status": "completed", "configuration_digest": digest})
    )
    matrix = tmp_path / "grouped.yaml"
    entries = [
        {"recipe": "examples/dflash-lora/m0/static_lora.yaml", "mode": "eval"},
        {
            "recipe": "examples/dflash-lora/m0/static_lora.yaml",
            "mode": "eval",
            "group_size": 2,
        },
    ]
    matrix.write_text(yaml.safe_dump({"tier": "tier1", "runs": entries}))
    argv = [
        str(matrix),
        "--output-root",
        str(tmp_path / "runs"),
        "--machine",
        str(machine),
        "--dry-run",
    ]
    plain, grouped = run_matrix.main(argv)
    assert plain["completed"]
    assert not grouped["completed"]
    assert "--group-size" not in plain["command"]
    position = grouped["command"].index("--group-size")
    assert grouped["command"][position + 1] == "2"
    assert run_arm.record_name("eval") == "run_record_eval.json"
    assert run_arm.record_name("eval", 2) == "run_record_eval-k2.json"
    with pytest.raises(SystemExit):  # --group-size is for eval only
        run_arm.main(["x.yaml", "--tier", "t", "--mode", "train", "--group-size", "2"])


def test_math10k_pool_job_uses_the_stored_rows_and_keeps_m0_digests(tmp_path):
    """The 10K-math pool: training subsets only for training, the stored
    evaluation rows (50 per subset) for validation / test, no donor pool, no
    evaluated subset among the training subsets; and adding `pool` leaves the
    configuration digest of a pool-less recipe unchanged (no `pool` key)."""
    resolved = _resolve("math10k/generator.yaml", tmp_path, tier="math10k")
    job = run_arm.training_job(resolved.cfg)
    assert job["subsets"] == [f"math-{i:05d}" for i in range(10)]
    assert job["training_examples"] == 10_000
    assert "donor_subsets" not in job
    evaluated = job["validation_subsets"] + job["test_subsets"]
    assert not set(evaluated) & set(job["subsets"])
    assert set(job["validation_row_ids"]) == set(evaluated)
    assert {len(v) for v in job["validation_row_ids"].values()} == {50}
    record = json.loads((RECIPES / "math10k/eval_splits.json").read_text())
    assert record["counts"]["validation"] == {
        "math": 200,
        "code": 100,
        "stem": 100,
        "chat": 100,
    }
    assert record["check"]["overlap"] == 0
    assert run_arm.eval_donors(job, ("correct", "fixed")) == ("correct",)
    m0 = _resolve("m0/generator.yaml", tmp_path)
    assert m0.cfg.pool is None
    values = m0.cfg.model_dump(mode="json")
    assert "pool" in values  # dumped as None ...
    digest = run_arm.configuration_digest(m0)
    values.pop("pool")  # ... and dropped by the digest
    assert digest == run_arm.configuration_digest(m0)


def test_pool_validation_and_stored_row_selection(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        Pool(
            train_subsets=["math-00000"],
            validation_subsets=["math-00001"],
            test_subsets=["math-00001"],
            rows=tmp_path / "rows.json",
        )
    with pytest.raises(ValueError, match="unique within"):
        Pool(
            train_subsets=["math-00000", "math-00000"],
            validation_subsets=["math-00001"],
            test_subsets=["math-00002"],
            rows=tmp_path / "rows.json",
        )
    # A train subset may also be evaluated (user 2026-11-14) when the stored
    # rows' overlap check covered the train splits of their release.
    shared = tmp_path / "shared.json"
    rec = {
        "release": "r0002-100k",
        "check": {"overlap": 0, "releases": ["r0001"]},
        "subsets": {
            "math-00001": {"role": "validation", "rows": [{"primary_id": "a"}]},
            "math-00002": {"role": "test", "rows": [{"primary_id": "d"}]},
        },
    }
    shared.write_text(json.dumps(rec))
    both = Pool(
        train_subsets=["math-00000", "math-00001", "math-00002"],
        validation_subsets=["math-00001"],
        test_subsets=["math-00002"],
        rows=shared,
    )
    assert both.shares_subsets()
    with pytest.raises(ValueError, match="did not cover"):
        both.row_ids()
    rec["check"]["releases"] = ["r0001", "r0002-100k"]
    shared.write_text(json.dumps(rec))
    assert both.row_ids() == {"math-00001": ["a"], "math-00002": ["d"]}
    with pytest.raises(ValueError, match="<category>-NNNNN"):
        Pool(
            train_subsets=["math-0"],
            validation_subsets=["code-00001"],
            test_subsets=["stem-00001"],
            rows=tmp_path / "rows.json",
        )
    rows = tmp_path / "rows.json"
    record = {
        "check": {"overlap": 0},
        "subsets": {
            "code-00001": {"role": "validation", "rows": [{"primary_id": "b"}]},
            "stem-00001": {"role": "test", "rows": [{"primary_id": "c"}]},
        },
    }
    rows.write_text(json.dumps(record))
    pool = Pool(
        train_subsets=["math-00000"],
        validation_subsets=["code-00001"],
        test_subsets=["stem-00001"],
        rows=rows,
    )
    assert pool.row_ids() == {"code-00001": ["b"], "stem-00001": ["c"]}
    swapped = pool.model_copy(update={"test_subsets": ["chat-00001"]})
    with pytest.raises(ValueError, match="differ from the pool lists"):
        swapped.row_ids()

    class Corpus:
        rows = [{"primary_id": p} for p in ("a", "b", "c", "d")]
        subsets = {"code-00001": {}, "stem-00001": {}}

        def indices(self, name, validation=False):
            return {"code-00001": [0, 1], "stem-00001": [2, 3]}[name]

    corpus = Corpus()
    job = {
        "validation_rows_per_subset": 1,
        "validation_rows_seed": 0,
        "validation_subsets": ["code-00001"],
        "test_subsets": ["stem-00001"],
        "validation_row_ids": pool.row_ids(),
    }
    apply_row_selection(corpus, job)
    assert corpus.subsets["code-00001"]["validation_indices"] == [1]
    assert corpus.subsets["stem-00001"]["validation_indices"] == [2]
    job["validation_row_ids"] = {"code-00001": ["z"], "stem-00001": ["c"]}
    with pytest.raises(ValueError, match="stored evaluation rows missing"):
        apply_row_selection(Corpus(), job)
