"""S6c: offline evaluation variants (M4, M7), similarity (D13), cost (D14)."""

import math
from types import SimpleNamespace

import pytest
import torch

from speculators.generator.offline_eval import (
    Variant,
    geometry,
    gram,
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


def _runtime(scale=2.0):
    adapters = {"m": SimpleNamespace(scale=scale)}
    rows = {s: list(range(10 * i, 10 * i + 8)) for i, s in enumerate(SUBSETS)}
    return SimpleNamespace(
        paths={"m": "m"},
        drafter=SimpleNamespace(get_submodule=adapters.__getitem__),
        corpus=SimpleNamespace(indices=lambda s, validation: rows[s]),
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
    assert fixed.conditioning(runtime, "math-00003", rows, 5) == [0, 1, 2, 3]
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
    with pytest.raises(ValueError, match="Unknown donor"):
        Variant(donor="random", subsets=SUBSETS)
    with pytest.raises(ValueError, match="need the evaluated subsets"):
        Variant(donor="fixed")


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
        evaluate_job(
            cfg, campaign_job(cfg, "fresh_lora"), name="unit", rows_per_category=2
        )
    with pytest.raises(ValueError, match="No single campaign job"):
        campaign_job(cfg, "nonexistent")
    assert campaign.run_campaign(cfg, "all")["runs"] == 4
    before = _tree_digest(cfg.output_root)
    options = {
        "name": "unit",
        "coefficients": (None, 0.0),
        "rows_per_category": 2,
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
        rows = record["ablation_set"]["rows"]
        assert len(rows) == 2 * 4  # 2 per category, 4 categories
        for result in record["ablations"].values():
            assert result["validation_examples"] == len(rows)
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
    base = evaluate_base(cfg, generator_job, name="unit", rows_per_category=2)
    assert len(runtimes) == 3
    assert not runtimes[-1].paths
    assert base["arm"] == "base"
    assert sorted((p, p.stat().st_mtime_ns) for p in run_dir.rglob("*")) == before
    for row in initial:
        assert base["validation"]["contexts"] == row["contexts"]
        assert base["validation"]["score"] == row["score"]
    assert base["ablation"]["validation_examples"] == 2 * 4
    path = cfg.output_root / "offline_eval/unit/base" / f"{generator_job['label']}.json"
    assert path.is_file()
