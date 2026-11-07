"""S5 rigor suite (IMPLEMENTATION_PLAN §7 items 1-4, 6-8) on a tiny CPU DFlash.

Item 1 (zero LoRA ≡ base) is `test_dflash_generator` / `test_dflash_kv_path`;
item 5 is `test_dflash_kv_path`; item 9 (CUDA / real checkpoint) is T1/T2.
"""

import hashlib
from types import SimpleNamespace

import pytest
import torch

from speculators.generator import campaign
from speculators.generator.config import Architecture
from speculators.generator.drafters import drafter_family
from speculators.generator.engine import check_trainable
from speculators.generator.model import LoRAGenerator, install_adapters
from speculators.train.utils import normalize_counted_metrics
from tests.unit.generator.dflash_fixtures import tiny_dflash_drafter
from tests.unit.generator.test_campaign import build_tiny_campaign
from tests.unit.generator.test_dflash_generator import (
    _bank,
    _batch,
    _cfg,
    _drafter,
    _run,
)
from tests.unit.generator.test_experiments import tiny_drafter

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
ALL_LINEAR += ["down_proj"]
RANK = 4
OVERFIT_STEPS = 200  # probe (lr 1e-3): loss 2.73 -> 0.016, full_acc 1.0 by step 120


def _adapted(targets=ALL_LINEAR, fc=True, rank=RANK):
    drafter = _drafter()
    family = drafter_family(_cfg(fc=fc), _bank())
    paths = install_adapters(
        drafter,
        rank,
        2 * rank,
        0.0,
        targets=family.adapter_targets(drafter, targets),
    )
    family.check_drafter(drafter)
    family.prepare_drafter(drafter)
    return family, drafter, paths


def _shapes(drafter, paths):
    return {
        key: (
            drafter.get_submodule(path).base.in_features,
            drafter.get_submodule(path).base.out_features,
        )
        for key, path in paths.items()
    }


def _generator(family, drafter, paths, rank=RANK, seed=0):
    torch.manual_seed(seed)
    shapes = _shapes(drafter, paths)
    return LoRAGenerator(
        12,
        shapes,
        rank,
        Architecture(
            encoder_hidden=32,
            example_dim=16,
            condition_dim=16,
            descriptor_dim=8,
            decoder_width=32,
            residual_blocks=1,
        ),
        layout=family.generator_layout(list(shapes)),
        head_sharing=family.head_sharing,
    )


def _random_factors(drafter, paths, seed=3, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    return {
        key: (
            torch.randn(RANK, base.in_features, generator=g) * scale,
            torch.randn(base.out_features, RANK, generator=g) * scale,
        )
        for key, base in (
            (key, drafter.get_submodule(path).base) for key, path in paths.items()
        )
    }


def _frozen_digest(drafter):
    """Hash of every non-adapter tensor, keyed by its pre-adapter name."""
    digest = hashlib.sha256()
    state = drafter.state_dict()
    for key in sorted(state):
        if key.endswith((".a", ".b")):
            continue
        digest.update(key.replace(".base.", ".").encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Item 2/3: only the generator (or, for static LoRA, only the adapters) trains.


def test_check_trainable_rejects_trainable_drafter_and_wrong_optimizer_set():
    family, drafter, paths = _adapted(targets=["q_proj"], fc=False)
    generator = _generator(family, drafter, paths)
    drafter.requires_grad_(False)
    runtime = SimpleNamespace(drafter=drafter, paths=paths)
    runtime.adapter_state = lambda: {
        key: tuple(getattr(drafter.get_submodule(p), s) for s in ("a", "b"))
        for key, p in paths.items()
    }
    parameters = list(generator.parameters())
    check_trainable(runtime, generator, parameters)
    with pytest.raises(RuntimeError, match="Optimizer parameters differ"):
        check_trainable(runtime, generator, parameters[:-1])
    drafter.layers[0].mlp.up_proj.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Frozen drafter has 1 trainable"):
        check_trainable(runtime, generator, parameters)
    # Direct (static LoRA) arm: exactly the adapter factors.
    drafter.requires_grad_(False)
    adapters = [p for pair in runtime.adapter_state().values() for p in pair]
    for p in adapters:
        p.requires_grad_(True)
    check_trainable(runtime, None, adapters)
    drafter.fc.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="not the adapters"):
        check_trainable(runtime, None, adapters)


@pytest.mark.parametrize("family", ["eagle3", "dflash"])
def test_tiny_campaign_trains_generator_and_static_lora_end_to_end(
    tmp_path, monkeypatch, family
):
    """≈2 updates per arm through campaign.run_one with the same fake data."""
    runtimes = []
    cfg, _ = build_tiny_campaign(tmp_path, monkeypatch, family, runtimes)
    torch.manual_seed(cfg.seed)
    fresh = (
        tiny_dflash_drafter(hidden=8, vocab=16, layers=2, uncompiled=False)
        if family == "dflash"
        else tiny_drafter()
    )
    reference = _frozen_digest(fresh)
    jobs = campaign.jobs(cfg)
    generator_job = jobs[0]
    static_job = next(job for job in jobs if job["arm"] == "fresh_lora")
    for job in (generator_job, static_job):
        result = campaign.run_one(cfg, job)
        assert len(result["history"]) == 2
        final = result["history"][-1]["contexts"]["4"]
        assert all(torch.isfinite(torch.tensor(v)) for v in final.values())
        if family == "dflash":
            assert final["loss"] > 0
            assert set(final) >= {"full_acc", "eal", "position_1_acc"}
        assert _frozen_digest(runtimes[-1].drafter) == reference
    if family == "dflash":
        assert runtimes[0].family.name == "dflash"
        assert runtimes[0].family.block_size == fresh.block_size


# Item 4: functional_call LoRA ≡ explicit merge; layers are independent.


def test_functional_lora_equals_explicit_merge_for_every_target():
    family, drafter, paths = _adapted()
    factors = _random_factors(drafter, paths)
    batch = _batch()
    _, loss, metrics = _run(family, drafter, batch, factors, paths)
    merged = _drafter()
    scale = 2 * RANK / RANK
    with torch.no_grad():
        for key, path in paths.items():
            a, b = factors[key]
            merged.get_submodule(path).weight += scale * b @ a
    assert len(paths) == len(drafter.layers) * len(ALL_LINEAR) + 1
    _, merged_loss, merged_metrics = _run(family, merged, batch)
    torch.testing.assert_close(loss, merged_loss, rtol=1e-5, atol=1e-5)
    for key in metrics:
        torch.testing.assert_close(
            torch.as_tensor(metrics[key]),
            torch.as_tensor(merged_metrics[key]),
            rtol=1e-5,
            atol=1e-5,
        )


def _layer_outputs(family, drafter, batch, factors, paths):
    outputs = {}
    hooks = [
        module.register_forward_hook(
            lambda _m, _i, out, name=name: outputs.__setitem__(
                name, (out[0] if isinstance(out, tuple) else out).clone()
            )
        )
        for name, module in [("fc", drafter.fc)]
        + [(f"layer{i}", layer) for i, layer in enumerate(drafter.layers)]
    ]
    try:
        _run(family, drafter, batch, factors, paths)
    finally:
        for hook in hooks:
            hook.remove()
    return outputs


def test_perturbing_one_layer_changes_only_that_layer_and_after():
    family, drafter, paths = _adapted()
    factors = _random_factors(drafter, paths)
    batch = _batch()
    before = _layer_outputs(family, drafter, batch, factors, paths)
    perturbed = dict(factors)
    for key in [k for k in paths if k.startswith("l1_")]:
        a, b = factors[key]
        perturbed[key] = (a, b + 0.5)
    after = _layer_outputs(family, drafter, batch, perturbed, paths)
    assert torch.equal(before["fc"], after["fc"])
    assert torch.equal(before["layer0"], after["layer0"])
    assert not torch.allclose(before["layer1"], after["layer1"])
    assert not torch.allclose(before["layer2"], after["layer2"])


# Item 6: block size / mask / anchors consistent and logged.


def test_drafter_geometry_is_logged_and_shared_by_training_and_validation(
    monkeypatch,
):
    messages = []
    monkeypatch.setattr("speculators.generator.drafters.dflash.report", messages.append)
    family, drafter, paths = _adapted()
    (line,) = [m for m in messages if m.startswith("DFlash drafter:")]
    for field in (
        f"block_size={drafter.block_size}",
        f"mask={drafter.config.mask_token_id}",
        "sample_from_anchor=False",
        "attention=eager",
        "anchor_density=0.0625",
        "validation_anchor_seed=11",
        "kv_lora_path=both",
        "loss=ce/dpace",
    ):
        assert field in line
    batch = _batch()
    train, _ = family.microbatch(batch, training=True)
    valid, _ = family.microbatch(batch, training=False)
    assert train["max_anchors"] == valid["max_anchors"]
    for key in batch:
        assert torch.equal(train[key], valid[key])


def test_validation_anchors_are_identical_across_arms():
    """Anchors depend only on the bank settings and batch content."""
    batch = _batch()
    anchors = []
    for cfg in (_cfg(), _cfg(fc=True, kv="context"), _cfg(layers=[0])):
        family = drafter_family(cfg, _bank())
        family.check_drafter(_drafter())
        extra, rng = family.microbatch(batch, training=False)
        from speculators.models.dflash.utils import select_anchors  # noqa: PLC0415

        with rng:
            anchors.append(
                select_anchors(extra["loss_mask"], extra["max_anchors"], 8)[0]
            )
    assert all(torch.equal(anchors[0], other) for other in anchors[1:])


# Item 7: the tiny generator overfits one cohort.


@pytest.fixture
def few_threads():
    """Tiny CPU ops are far slower with many threads (oversubscription)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(threads)


def test_tiny_generator_overfits_one_cohort(few_threads):
    family, drafter, paths = _adapted(targets=ALL_LINEAR, fc=False, rank=8)
    generator = _generator(family, drafter, paths, rank=8)
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)  # 3e-3 oscillates
    batch = _batch()
    context, sizes = torch.randn(3, 12), torch.tensor([4, 5, 6])
    first = None
    for _ in range(OVERFIT_STEPS):
        factors = generator(context, sizes)
        # Validation-mode anchors: fixed across steps, so one cohort is fitted.
        _, loss, metrics = _run(family, drafter, batch, factors, paths)
        first = float(loss) if first is None else first
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, loss, metrics = _run(
            family, drafter, batch, generator(context, sizes), paths
        )
    final = normalize_counted_metrics({k: float(v) for k, v in metrics.items()})
    assert float(loss) <= 0.1 * first
    assert final["full_acc"] >= 0.95


# Item 8: same seed -> identical anchors, losses and factors.


def _train_steps(seed, steps=3):
    family, drafter, paths = _adapted(targets=["q_proj", "v_proj"], fc=True)
    generator = _generator(family, drafter, paths, seed=seed)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-3)
    drafter.train()
    torch.manual_seed(seed)
    context, sizes = torch.randn(3, 12), torch.tensor([4, 5, 6])
    losses, anchors = [], []
    from speculators.models.dflash import core  # noqa: PLC0415

    original = core.select_anchors

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        anchors.append(result[0].clone())
        return result

    core.select_anchors = record
    try:
        for _ in range(steps):
            _, loss, _ = _run(
                family,
                drafter,
                _batch(),
                generator(context, sizes),
                paths,
                training=True,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
    finally:
        core.select_anchors = original
    with torch.no_grad():
        factors = generator(context, sizes)
    return losses, anchors, factors


def test_same_seed_gives_identical_anchors_losses_and_factors():
    first, second, other = _train_steps(5), _train_steps(5), _train_steps(6)
    for left, right in zip(first[:2], second[:2], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(left, right, strict=True))
    for key in first[2]:
        for x, y in zip(first[2][key], second[2][key], strict=True):
            assert torch.equal(x, y)
    assert not all(torch.equal(x, y) for x, y in zip(first[1], other[1], strict=True))
