"""S6c: offline evaluation variants (M4, M7), similarity (D13), cost (D14)."""

import math
from types import SimpleNamespace

import pytest
import torch

from speculators.generator.offline_eval import (
    Variant,
    geometry,
    gram,
    per_row_table,
    similarity,
)
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401

SUBSETS = ("chat-00002", "chat-00003", "math-00002", "math-00003")


def _delta(pair, scale):
    a, b = pair
    return scale * b.double() @ a.double()


def test_gram_equals_materialised_frobenius_products():
    torch.manual_seed(0)
    pairs = [(torch.randn(3, 7), torch.randn(5, 3)) for _ in range(4)]
    want = torch.tensor(
        [
            [float((_delta(p, 2.0) * _delta(q, 2.0)).sum()) for q in pairs]
            for p in pairs
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(gram(pairs, 2.0), want)


def _clustered(noise, n=6):
    """Two categories around two orthogonal directions."""
    torch.manual_seed(1)
    bases = {"chat": torch.randn(4, 8), "math": torch.randn(4, 8)}
    samples = []
    for category, base in bases.items():
        for _ in range(n):
            delta = base + noise * torch.randn(4, 8)
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            samples.append((f"{category}-00002", {"m": (vh, u * s)}))
    return samples


def test_geometry_separates_categories_and_detects_dynamism():
    report = similarity(_clustered(noise=0.05), {"m": 1.0})["all"]
    assert report["cos_in"] > 0.9
    assert report["cos_gap"] > 0.5
    assert report["eta2"] > 0.9
    assert report["centroid_accuracy"] == 1.0
    assert 0 < report["dyn_ratio"] <= 1
    assert report["effective_rank"] >= 1


def test_geometry_of_static_lora_is_not_dynamic():
    pair = (torch.randn(2, 6), torch.randn(4, 2))
    g = gram([pair] * 4, 2.0)
    report = geometry(g, ["chat", "chat", "math", "math"])
    assert report["cos_in"] == pytest.approx(1.0)
    assert report["cos_cross"] == pytest.approx(1.0)
    assert report["dyn_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert report["effective_rank"] == 0.0
    assert math.isnan(report["eta2"])


def test_geometry_metrics_match_explicit_vectors():
    samples = _clustered(noise=0.5, n=4)
    labels = [s.split("-")[0] for s, _ in samples]
    flat = torch.stack([_delta(f["m"], 1.0).flatten() for _, f in samples])
    report = similarity(samples, {"m": 1.0})["modules"]["m"]
    mean = flat.mean(0)
    total = ((flat - mean) ** 2).sum()
    assert report["dyn_ratio"] == pytest.approx(float(total / (flat**2).sum()))
    between = 0.0
    for category in set(labels):
        rows = flat[[i for i, x in enumerate(labels) if x == category]]
        between += len(rows) * float(((rows.mean(0) - mean) ** 2).sum())
    assert report["eta2"] == pytest.approx(between / float(total))
    cos = torch.nn.functional.cosine_similarity(flat[:, None], flat[None], dim=-1)
    same = [[x == y for y in labels] for x in labels]
    pairs_in = [
        float(cos[i, j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and same[i][j]
    ]
    assert report["cos_in"] == pytest.approx(sum(pairs_in) / len(pairs_in))


DONORS = ("chat-00009", "math-00009")


class _Rows(dict):
    def __missing__(self, index):
        # Primary ids sort in reverse index order: selection is by primary id.
        return {"primary_id": f"p{1000 - index:04d}"}


def _runtime(scale=2.0, group_size=4):
    adapters = {"m": SimpleNamespace(scale=scale)}
    rows = {s: list(range(10 * i, 10 * i + 8)) for i, s in enumerate(SUBSETS)}
    rows |= {
        s: list(range(200 + 10 * i, 200 + 10 * i + 6)) for i, s in enumerate(DONORS)
    }
    train = {
        s: list(range(100 + 10 * i, 100 + 10 * i + 6)) for i, s in enumerate(SUBSETS)
    }
    return SimpleNamespace(
        cfg=SimpleNamespace(context=SimpleNamespace(evaluation_sizes=[group_size])),
        paths={"m": "m"},
        drafter=SimpleNamespace(get_submodule=adapters.__getitem__),
        corpus=SimpleNamespace(
            indices=lambda s, validation: rows[s] if validation else train[s],
            rows=_Rows(),
        ),
        adapter_state=lambda: {"m": (torch.ones(2, 3), torch.ones(4, 2))},
    )


def test_coefficient_replaces_training_scale():
    runtime = _runtime(scale=2.0)
    factors = {"m": (torch.ones(2, 3), torch.full((4, 2), 3.0))}
    assert Variant().factors(runtime, factors) is factors
    for coefficient in (0.0, 1.0, 2.0, 8.0):
        a, b = Variant(coefficient=coefficient).factors(runtime, factors)["m"]
        # effective ΔW = scale · b' · a = coefficient · b · a
        torch.testing.assert_close(2.0 * b @ a, coefficient * factors["m"][1] @ a)
    # Static arm (no generated factors): its trained adapter factors are scaled.
    a, b = Variant(coefficient=0.0).factors(runtime, None)["m"]
    assert not b.any()


def test_donor_conditioning_rows():
    runtime = _runtime()
    rows = [0, 1, 2, 3]
    assert Variant().conditioning(runtime, "chat-00002", rows, 0) == rows
    fixed = Variant(donor="fixed", subsets=SUBSETS)
    # Without donor subsets: the first group-size (4) training rows of the
    # first subset, by primary id: never evaluated rows, and the same set
    # (one LoRA) for every cohort and subset, short cohorts included.
    for subset, cohort in (
        ("math-00003", rows),
        ("chat-00002", [0, 1, 2, 3]),
        ("chat-00002", [5]),
    ):
        assert fixed.conditioning(runtime, subset, cohort, 5) == [105, 104, 103, 102]
    with pytest.raises(ValueError, match="off by default"):
        Variant(donor="same_category", subsets=SUBSETS)
    same = Variant(donor="same_category", subsets=SUBSETS, allow_same_category=True)
    picked = same.conditioning(runtime, "chat-00002", rows, 0)
    assert set(picked) <= set(range(4, 8)) | set(range(10, 18))  # chat, not own
    single = Variant(
        donor="same_category",
        subsets=("chat-00002", "math-00002"),
        allow_same_category=True,
    )
    assert set(single.conditioning(runtime, "chat-00002", rows, 0)) == {4, 5, 6, 7}
    assert picked == same.conditioning(runtime, "chat-00002", rows, 0)
    assert picked != same.conditioning(runtime, "chat-00002", rows, 1)
    cross = Variant(donor="cross_category", subsets=SUBSETS)
    math_rows = set(range(20, 28)) | set(range(30, 38))
    assert set(cross.conditioning(runtime, "chat-00003", rows, 0)) <= math_rows
    chat_rows = set(range(8)) | set(range(10, 18))
    # Cyclic in sorted order: the last category takes donors from the first.
    assert set(cross.conditioning(runtime, "math-00002", rows, 0)) <= chat_rows
    # Unseen donor subsets (user 2026-11-09): every donor comes from them.
    unseen = Variant(donor="fixed", subsets=SUBSETS, donor_subsets=DONORS)
    assert unseen.conditioning(runtime, "math-00003", rows, 0) == [205, 204, 203, 202]
    one = _runtime(group_size=1)
    assert unseen.conditioning(one, "math-00003", [7], 3) == [205]
    cross = Variant(donor="cross_category", subsets=SUBSETS, donor_subsets=DONORS)
    assert set(cross.conditioning(runtime, "chat-00002", rows, 0)) <= set(
        range(210, 216)
    )
    assert set(cross.conditioning(one, "math-00002", [20], 0)) <= set(range(200, 206))
    same = Variant(
        donor="same_category",
        subsets=SUBSETS,
        donor_subsets=DONORS,
        allow_same_category=True,
    )
    assert set(same.conditioning(runtime, "chat-00002", rows, 0)) <= set(
        range(200, 206)
    )
    assert Variant(group_size=4).label == "coef-train_donor-correct_k-4"
    with pytest.raises(ValueError, match="group_size"):
        Variant(group_size=0)
    with pytest.raises(ValueError, match="Unknown donor"):
        Variant(donor="random", subsets=SUBSETS)
    with pytest.raises(ValueError, match="need the evaluated subsets"):
        Variant(donor="fixed")


def _check_cohorts_pool_exactly(result):
    """Per-cohort (per-row) records pool exactly to each subset's metrics."""
    from speculators.train.utils import (  # noqa: PLC0415
        normalize_counted_metrics,
    )

    for subset in result["subsets"].values():
        pooled = {}
        for cohort in subset["cohorts"]:
            for key, value in cohort["counts"].items():
                pooled[key] = pooled.get(key, 0.0) + value
        (context,) = subset["contexts"].values()
        assert normalize_counted_metrics(pooled) == context
        assert (
            sum(len(c["rows"]) for c in subset["cohorts"])
            == (subset["validation_examples"])
        )


def test_per_row_table():
    def sweep(values):
        return {
            "subsets": {
                f"{category}-00000": {
                    "cohorts": [
                        {"primary_ids": [f"{category}{i}"], "metrics": {"eal": v}}
                        for i, v in enumerate(row)
                    ]
                }
                for category, row in values.items()
            }
        }

    table = per_row_table(
        {
            0.5: sweep({"math": [3.0, 1.0], "chat": [2.0, 2.0]}),
            2.0: sweep({"math": [2.0, 2.0], "chat": [1.0, 2.0]}),
        }
    )
    assert table["coefficients"] == ["0.5", "2"]
    assert table["rows"]["math0"] == {
        "category": "math",
        "subset": "math-00000",
        "by": {"0.5": 3.0, "2": 2.0},
        "best": ["0.5"],
    }
    assert table["rows"]["chat1"]["best"] == ["0.5", "2"]
    math = table["categories"]["math"]
    assert math["row_mean"] == {"0.5": 2.0, "2": 2.0}
    assert math["oracle"] == 2.5
    assert math["oracle_gain_over_best_fixed"] == 0.5
    assert math["best_counts"] == {"0.5": 1, "2": 1}
    with pytest.raises(ValueError, match="one row per cohort"):
        per_row_table(
            {1.0: {"subsets": {"a-0": {"cohorts": [{"primary_ids": [1, 2]}]}}}}
        )


def _tree_digest(root):
    import hashlib  # noqa: PLC0415

    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_evaluate_job_end_to_end(online_campaign):  # noqa: F811
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.offline_eval import (  # noqa: PLC0415
        campaign_job,
        evaluate_job,
    )

    cfg, _, _ = online_campaign
    with pytest.raises(FileNotFoundError, match="no completed training run"):
        evaluate_job(cfg, campaign_job(cfg, "fresh_lora"), name="unit")
    with pytest.raises(ValueError, match="No single campaign job"):
        campaign_job(cfg, "nonexistent")
    assert campaign.run_campaign(cfg, "all")["runs"] == 4
    before = _tree_digest(cfg.output_root)
    options = {
        "name": "unit",
        "coefficients": (None, 0.0),
        "cost_cohorts": 2,
    }
    generated = evaluate_job(cfg, campaign_job(cfg, "scale-1-direct"), **options)
    static = evaluate_job(cfg, campaign_job(cfg, "fresh_lora"), **options)
    assert _tree_digest(cfg.output_root / "pretraining") == {
        k.removeprefix("pretraining/"): v
        for k, v in before.items()
        if k.startswith("pretraining/")
    }
    for record in (generated, static):
        assert record["reproduction"]["reproduces_training_validation"]
        assert record["group_size"] == 4  # EAGLE-3 campaign: context-4
        assert "m4_per_row" not in record  # needs one LoRA per row
        for result in record["ablations"].values():
            assert (
                result["validation_examples"]
                == record["reproduction"]["validation_examples"]
            )
            _check_cohorts_pool_exactly(result)
    # Generator: coefficients x donors (correct, fixed, cross); static: coefficients.
    assert set(generated["ablations"]) == {
        f"coef-{c}_donor-{d}"
        for c in ("train", "0")
        for d in ("correct", "fixed", "cross_category")
    }
    assert set(static["ablations"]) == {
        "coef-train_donor-correct",
        "coef-0_donor-correct",
    }
    # Coefficient 0 is the frozen base drafter, whatever the donor. (These two
    # arms validate on different subsets, so their rows differ.)
    base = generated["ablations"]["coef-0_donor-correct"]["contexts"]
    assert base != generated["ablations"]["coef-train_donor-correct"]["contexts"]
    for donor in ("fixed", "cross_category"):
        assert generated["ablations"][f"coef-0_donor-{donor}"]["contexts"] == base
    assert generated["similarity"]["categories"] == ["chat", "code", "math", "stem"]
    assert generated["similarity"]["all"]["samples"] >= 4
    assert 1 <= generated["cost"]["batched_cohorts"] <= 2  # fixture: 4 rows/subset
    assert "similarity" not in static
    written = cfg.output_root / "offline_eval/unit/pretraining/scale-1-direct.json"
    assert written.is_file()
    cfg.optimization.lr *= 2  # not the configuration that was trained
    with pytest.raises(ValueError, match="different configuration"):
        evaluate_job(cfg, campaign_job(cfg, "fresh_lora"), **options)


@pytest.mark.parametrize("family", ["eagle3", "dflash"])
def test_base_only_equals_initial_validation_of_both_arms(
    tmp_path, monkeypatch, family
):
    """The base arm (released drafter, no adapters) is bit-identical to the
    update-0 validation of the generator and static arms (ΔW = 0 at init)."""
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.offline_eval import evaluate_base  # noqa: PLC0415
    from tests.unit.generator.test_campaign import (  # noqa: PLC0415
        build_tiny_campaign,
    )

    runtimes = []
    cfg, _ = build_tiny_campaign(tmp_path, monkeypatch, family, runtimes)
    jobs = campaign.jobs(cfg)
    generator_job = jobs[0]
    static_job = next(job for job in jobs if job["arm"] == "fresh_lora")
    # Matched arms (as EAGLE-3 GEN-009/010): same validation subsets.
    static_job = {
        **static_job,
        "validation_subsets": generator_job["validation_subsets"],
    }
    initial = []
    for job in (generator_job, static_job):
        history = campaign.run_one(cfg, job)["history"]
        assert history[0]["update"] == 0
        initial.append(history[0])
    run_dir = cfg.output_root / generator_job["stage"] / generator_job["label"]
    before = sorted((p, p.stat().st_mtime_ns) for p in run_dir.rglob("*"))
    base = evaluate_base(cfg, generator_job, name="unit")
    assert len(runtimes) == 3
    assert not runtimes[-1].paths
    assert base["arm"] == "base"
    assert sorted((p, p.stat().st_mtime_ns) for p in run_dir.rglob("*")) == before
    for row in initial:
        assert base["validation"]["contexts"] == row["contexts"]
        assert base["validation"]["score"] == row["score"]
    _check_cohorts_pool_exactly(base["validation"])
    path = cfg.output_root / "offline_eval/unit/base" / f"{generator_job['label']}.json"
    assert path.is_file()


def test_one_lora_per_row_with_fixed_rows_test_and_unseen_donors(tmp_path, monkeypatch):
    """The 2026-11-09 protocol on the DFlash family: one LoRA per row, a
    fixed seeded row selection, an untouched test split and unseen donors."""
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.offline_eval import (  # noqa: PLC0415
        evaluate_base,
        evaluate_job,
    )
    from tests.unit.generator.test_campaign import (  # noqa: PLC0415
        build_tiny_campaign,
    )

    cfg, _ = build_tiny_campaign(tmp_path, monkeypatch, "dflash")
    # As examples/dflash-lora/common.yaml: one conditioning row per LoRA.
    cfg.context.min_examples = cfg.context.max_examples = 1
    cfg.context.evaluation_sizes = [1]
    jobs = campaign.jobs(cfg)
    extra = {
        "validation_rows_per_subset": 2,
        "validation_rows_seed": 11,
        "test_subsets": [f"{d}-00004" for d in campaign.DOMAINS],
        "donor_subsets": [f"{d}-00005" for d in campaign.DOMAINS],
    }
    generator_job = {**jobs[0], **extra}
    static_job = {
        **next(job for job in jobs if job["arm"] == "fresh_lora"),
        **extra,
        "validation_subsets": generator_job["validation_subsets"],
    }
    validation_rows = 2 * len(generator_job["validation_subsets"])
    initial = []
    for job in (generator_job, static_job):
        history = campaign.run_one(cfg, job)["history"]
        assert history[0]["validation_examples"] == validation_rows
        assert set(history[0]["contexts"]) == {"1"}
        initial.append(history[0])
    base = evaluate_base(cfg, generator_job, name="unit")
    for row in initial:  # ΔW = 0 at init: the base arm, bit-identical
        assert base["validation"]["contexts"] == row["contexts"]
    assert base["group_size"] == 1
    for subset in base["validation"]["subsets"].values():
        assert all(len(c["rows"]) == 1 for c in subset["cohorts"])
    _check_cohorts_pool_exactly(base["validation"])
    base_test = evaluate_base(cfg, generator_job, name="unit", split="test")
    assert set(base_test["test"]["subsets"]) == set(extra["test_subsets"])
    assert base_test["test"]["validation_examples"] == 2 * 4
    assert (
        (cfg.output_root / "offline_eval/unit/base")
        .joinpath(f"{generator_job['label']}.test.json")
        .is_file()
    )

    record = evaluate_job(
        cfg,
        generator_job,
        name="unit",
        coefficients=(0.5, 2.0),
        cost_cohorts=2,
    )
    assert record["reproduction"]["reproduces_training_validation"]
    assert set(record["ablations"]) == {
        f"coef-{c}_donor-{d}"
        for c in ("0.5", "2")
        for d in ("correct", "fixed", "cross_category")
    } | {"coef-0.5_donor-correct_k-4", "coef-2_donor-correct_k-4"}
    k4 = record["ablations"]["coef-2_donor-correct_k-4"]
    assert set(k4["contexts"]) == {"4"}
    # K=4 on a generator trained on one row: allowed, scoped, and recorded.
    assert k4["beyond_trained_examples"] is True
    assert (
        record["ablations"]["coef-2_donor-correct"].get("beyond_trained_examples")
        is None
    )
    assert k4["validation_examples"] == validation_rows
    table = record["m4_per_row"]
    assert len(table["rows"]) == validation_rows
    assert table["coefficients"] == ["0.5", "2"]
    # Donors are unseen rows: never trained on, never evaluated.
    for label in ("coef-2_donor-fixed", "coef-2_donor-cross_category"):
        conditioning = {
            index
            for subset in record["ablations"][label]["subsets"].values()
            for cohort in subset["cohorts"]
            for index in cohort["conditioning"]
        }
        evaluated = {
            index
            for subset in record["ablations"][label]["subsets"].values()
            for cohort in subset["cohorts"]
            for index in cohort["rows"]
        }
        assert conditioning
        assert not conditioning & evaluated
    fixed = {
        tuple(cohort["conditioning"])
        for subset in record["ablations"]["coef-2_donor-fixed"]["subsets"].values()
        for cohort in subset["cohorts"]
    }
    assert len(fixed) == 1  # one LoRA for every row
    assert len(next(iter(fixed))) == 1  # from one donor row
    assert record["similarity"]["all"]["samples"] == validation_rows
    test = evaluate_job(cfg, generator_job, name="unit", split="test")
    assert test["test"]["validation_examples"] == 2 * 4
    with pytest.raises(ValueError, match="no test subsets"):
        evaluate_job(cfg, jobs[0], name="unit", split="test")
