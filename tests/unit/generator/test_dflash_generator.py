"""S3: DFlash generator family on a tiny CPU DFlash drafter (fp32, eager attention)."""

from types import SimpleNamespace

import pytest
import torch
from torch.func import functional_call

from speculators.bank.config import BankConfig
from speculators.generator.config import Adapter, Architecture
from speculators.generator.drafters import drafter_family
from speculators.generator.drafters.dflash import DFlashFamily
from speculators.generator.model import (
    LoRAGenerator,
    factor_parameters,
    install_adapters,
)
from tests.unit.generator.dflash_fixtures import AUX, BLOCK_DRAFTER, tiny_dflash_drafter

HIDDEN, VOCAB, LAYERS, SEQ = 32, 64, 3, 48


def _bank(**overrides):
    return BankConfig(training={"block_drafter": {**BLOCK_DRAFTER, **overrides}})


def _cfg(*, layers="all", head_sharing="per_type", fc=False, kv="both", loss=None):
    return SimpleNamespace(
        optimization=SimpleNamespace(loss_fn=loss),
        adapter=Adapter(
            layers=layers, head_sharing=head_sharing, kv_lora_path=kv, fc=fc
        ),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        device="cpu",
    )


def _drafter(seed=0):
    torch.manual_seed(seed)
    return tiny_dflash_drafter(hidden=HIDDEN, vocab=VOCAB, layers=LAYERS)


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    loss_mask = torch.ones(1, SEQ)
    loss_mask[:, :10] = 0
    return {
        "hidden_states": torch.randn(1, SEQ, len(AUX) * HIDDEN, generator=g),
        "input_ids": torch.randint(0, VOCAB - 1, (1, SEQ), generator=g),
        "loss_mask": loss_mask,
        "verifier_last_hidden_states": torch.randn(1, SEQ, HIDDEN, generator=g),
        "document_ids": torch.zeros(1, SEQ, dtype=torch.long),
        "position_ids": torch.arange(SEQ).unsqueeze(0),
    }


def _run(family, drafter, batch, factors=None, paths=None, *, training=False):
    family.check_drafter(drafter)  # as Runtime.attach_drafter; records block_size
    extra, rng = family.microbatch(batch, training=training)
    kwargs = {**batch, **family.call_kwargs(), **extra}
    with rng:
        if factors is None:
            return drafter(**kwargs)
        return functional_call(drafter, factor_parameters(paths, factors), (), kwargs)


def test_block_drafter_selects_dflash_family_and_validates_config():
    family = drafter_family(_cfg(), _bank())
    assert isinstance(family, DFlashFamily)
    assert family.score_key == "full_acc"
    assert family.identity() == {"family": "dflash", **BLOCK_DRAFTER}
    with pytest.raises(ValueError, match="requires adapter"):
        drafter_family(SimpleNamespace(**{**vars(_cfg()), "adapter": None}), _bank())
    with pytest.raises(ValueError, match="conflicts with"):
        drafter_family(_cfg(loss="lk_hybrid"), _bank())
    with pytest.raises(ValueError, match="block drafters only"):
        drafter_family(_cfg(), BankConfig())


def test_block_drafter_fields_are_required():
    incomplete = dict(BLOCK_DRAFTER)
    incomplete.pop("anchor_density")
    with pytest.raises(ValueError, match="anchor_density"):
        BankConfig(training={"block_drafter": incomplete})


def test_check_drafter():
    family = drafter_family(_cfg(), _bank())
    family.check_drafter(_drafter())
    other = BankConfig(
        training={"block_drafter": BLOCK_DRAFTER},
        hidden_states={"layer_ids": [1, 2, 3, 36]},
    )
    with pytest.raises(ValueError, match="target layers differ"):
        drafter_family(_cfg(), other).check_drafter(_drafter())


def test_adapter_targets_layout_and_generator():
    drafter = _drafter()
    family = drafter_family(_cfg(fc=True), _bank())
    targets = family.adapter_targets(drafter, family.cfg.target_modules)
    assert len(targets) == LAYERS * 4 + 1
    assert targets["l2_v_proj"] == "layers.2.self_attn.v_proj"
    assert targets["fc"] == "fc"
    layout = family.generator_layout(sorted(targets))
    assert layout["l1_q_proj"] == ("q_proj", 1)
    assert layout["fc"] == ("fc", LAYERS)
    only = drafter_family(_cfg(layers=[0], head_sharing="per_module"), _bank())
    assert set(only.adapter_targets(drafter, ["o_proj", "down_proj"])) == {
        "l0_o_proj",
        "l0_down_proj",
    }
    with pytest.raises(ValueError, match="exceed drafter depth"):
        drafter_family(_cfg(layers=[LAYERS]), _bank()).adapter_targets(
            drafter, ["q_proj"]
        )


def _adapted(fc=False):
    drafter = _drafter()
    family = drafter_family(_cfg(fc=fc), _bank())
    paths = install_adapters(
        drafter,
        4,
        8,
        0.0,
        targets=family.adapter_targets(drafter, ["q_proj", "v_proj"]),
    )
    shapes = {
        k: (
            drafter.get_submodule(p).base.in_features,
            drafter.get_submodule(p).base.out_features,
        )
        for k, p in paths.items()
    }
    torch.manual_seed(0)
    generator = LoRAGenerator(
        12,
        shapes,
        4,
        Architecture(
            encoder_hidden=16,
            example_dim=8,
            condition_dim=8,
            descriptor_dim=4,
            decoder_width=16,
            residual_blocks=1,
        ),
        layout=family.generator_layout(list(shapes)),
        head_sharing=family.head_sharing,
    )
    return family, drafter, paths, generator


def test_zero_lora_equals_base_drafter_bit_identically():
    family, adapted, paths, generator = _adapted(fc=True)
    base = _drafter()
    batch = _batch()
    factors = generator(torch.randn(3, 12), torch.tensor([4, 5, 6]))
    assert all(torch.count_nonzero(b) == 0 for _, b in factors.values())
    _, base_loss, base_metrics = _run(family, base, batch)
    _, loss, metrics = _run(family, adapted, batch, factors, paths)
    assert torch.equal(loss, base_loss)
    assert metrics.keys() == base_metrics.keys()
    for key in metrics:
        assert torch.equal(
            torch.as_tensor(metrics[key]), torch.as_tensor(base_metrics[key])
        )


def test_generated_lora_trains_only_the_generator():
    family, drafter, paths, generator = _adapted()
    torch.manual_seed(2)
    for head in generator.heads.values():
        torch.nn.init.normal_(head["B"].weight, std=0.05)
    frozen = {k: v.clone() for k, v in drafter.state_dict().items()}
    factors = generator(torch.randn(3, 12), torch.tensor([4, 5, 6]))
    drafter.train()
    _, loss, _ = _run(family, drafter, _batch(), factors, paths, training=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in drafter.parameters())
    assert all(p.grad is not None for p in generator.parameters() if p.requires_grad)
    for key, value in drafter.state_dict().items():
        assert torch.equal(value, frozen[key])


def test_anchor_density_and_deterministic_validation_anchors():
    family = drafter_family(_cfg(), _bank())
    batch = _batch()
    drafter = _drafter()
    with pytest.raises(RuntimeError, match="check_drafter must run"):
        family.microbatch(batch, training=True)
    family.check_drafter(drafter)
    extra, _ = family.microbatch(batch, training=True)
    # Anchor count from the unpadded length; tensors right-padded by block_size.
    assert extra["max_anchors"] == round(SEQ / 16)
    assert set(extra) == {"max_anchors", *batch}
    pad = drafter.block_size
    for key, value in batch.items():
        assert extra[key].shape[1] == SEQ + pad
        assert torch.equal(extra[key][:, :SEQ], value)
        fill = -1 if key == "document_ids" else 0
        assert torch.all(extra[key][:, SEQ:] == fill)
    with pytest.raises(ValueError, match="Unexpected DFlash microbatch tensor"):
        family.microbatch({**batch, "lengths": torch.ones(1, SEQ)}, training=True)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    first = _run(family, drafter, batch)[2]
    assert torch.equal(torch.get_rng_state(), before)  # global RNG untouched
    torch.manual_seed(999)
    second = _run(family, drafter, batch)[2]
    for key in first:
        assert torch.equal(torch.as_tensor(first[key]), torch.as_tensor(second[key]))
    other = drafter_family(_cfg(), _bank(validation_anchor_seed=12))
    assert family.max_anchors(8192) == 512
    assert other.max_anchors(16) == 1


def test_row_hooks():
    family = drafter_family(_cfg(), _bank())
    row = {"seq_len": 6, "loss_mask": [0, 0, 1, 1, 0, 1]}
    assert family.trained_length(6) == 6
    assert family.supervised_tokens(row) == 3
    x = torch.randn(2, 3)
    assert family.training_noise(x) is x
    sample = {"input_ids": torch.arange(4)}
    assert family.preprocess(sample) is sample


def test_tracking_routes_block_drafter_metrics():
    from speculators.generator.tracking_layout import validation_values  # noqa: PLC0415

    metrics = {"full_acc": 0.5, "position_3_acc": 0.25, "eal": 2.0, "loss": 1.5}
    values = validation_values({"score": 0.5, "contexts": {"4": metrics}}, 7)
    assert values["validation_scores/context_4_agreement"] == 0.5
    assert values["validation_scores/context_4_agreement_position_3"] == 0.25
    assert values["validation_scores/context_4_expected_accepted_length"] == 2.0
    assert values["validation_losses/context_4_loss"] == 1.5
    assert not any(key.startswith("diagnostics/") for key in values)


def test_campaign_family_contract():
    from speculators.generator.campaign_inputs import (  # noqa: PLC0415
        check_campaign_family,
    )

    check_campaign_family(_cfg(), {"math": _bank(), "chat": _bank()})
    # A block-drafter config cannot even resolve an EAGLE-3 bank.
    with pytest.raises(ValueError, match="block drafters only"):
        check_campaign_family(_cfg(), {"math": _bank(), "chat": BankConfig()})
    eagle = SimpleNamespace(**{**vars(_cfg()), "adapter": None})
    with pytest.raises(ValueError, match="requires optimization.loss_fn lk_hybrid"):
        check_campaign_family(eagle, {"math": BankConfig()})
    eagle.optimization.loss_fn = "lk_hybrid"
    check_campaign_family(eagle, {"math": BankConfig()})


def test_serving_export_refuses_block_drafter_runs(tmp_path):
    import json  # noqa: PLC0415

    from speculators.generator.serving.bundle import export_run  # noqa: PLC0415

    (tmp_path / "resolved_experiment.json").write_text(
        json.dumps({"drafter_training": {"family": "dflash"}})
    )
    with pytest.raises(NotImplementedError, match="serving port"):
        export_run(None, tmp_path, tmp_path / "bundle")
