"""Drafter-family layer: EAGLE-3 extraction must equal the pre-port behaviour."""

from types import SimpleNamespace

import pytest
import torch

from speculators.bank.config import BankConfig
from speculators.generator.drafters import DrafterFamily, drafter_family
from speculators.generator.drafters.eagle3 import Eagle3Family
from speculators.losses import resolve_loss_config
from speculators.models.eagle3.data import shift_batch


def _cfg(loss_fn=None):
    return SimpleNamespace(optimization=SimpleNamespace(loss_fn=loss_fn), adapter=None)


def _bank(**training):
    return BankConfig(training=training) if training else BankConfig()


def test_default_bank_resolves_eagle3_family():
    family = drafter_family(_cfg(), _bank())
    assert isinstance(family, Eagle3Family)
    assert family.score_key == "full_acc_0"
    assert DrafterFamily.registry["eagle3"] is Eagle3Family


def test_unregistered_family_raises(monkeypatch):
    monkeypatch.setattr(
        "speculators.generator.drafters.base.family_name", lambda _bank: "nope"
    )
    with pytest.raises(ValueError, match="No generator drafter family"):
        drafter_family(_cfg(), _bank())


@pytest.mark.parametrize("loss_fn", [None, "lk_hybrid", "ce"])
def test_eagle3_identity_and_call_kwargs_match_legacy(loss_fn):
    bank = _bank(noise_std=0.07, ttt_steps=7, ttt_step_loss_decay=0.8)
    cfg = _cfg(loss_fn)
    family = drafter_family(cfg, bank)
    # Frozen copy of engine.run_identity["drafter_training"] before the port.
    legacy_identity = {
        **{
            key: getattr(bank.training, key)
            for key in ("noise_std", "ttt_steps", "ttt_step_loss_decay")
        },
        "loss_fn": cfg.optimization.loss_fn or bank.training.loss_fn,
    }
    assert family.identity() == legacy_identity
    assert list(family.identity()) == list(legacy_identity)
    kwargs = family.call_kwargs()
    legacy_loss = resolve_loss_config(cfg.optimization.loss_fn or bank.training.loss_fn)
    assert kwargs["ttt_steps"] == 7
    assert kwargs["ttt_step_loss_decay"] == 0.8
    assert kwargs["loss_config"] == legacy_loss
    assert set(kwargs) == {"ttt_steps", "ttt_step_loss_decay", "loss_config"}


def test_eagle3_row_hooks_match_legacy():
    family = drafter_family(_cfg(), _bank())
    row = {"seq_len": 6, "loss_mask": [0, 0, 1, 1, 0, 1]}
    assert family.trained_length(row["seq_len"]) == 5
    assert family.supervised_tokens(row) == sum(row["loss_mask"][1:])
    sample = {
        "input_ids": torch.arange(6),
        "hidden_states": torch.randn(6, 12),
        "verifier_last_hidden_states": torch.randn(6, 4),
        "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.bool),
        "lengths": torch.tensor([6]),
        "position_ids": torch.arange(6),
    }
    got, want = family.preprocess(dict(sample)), shift_batch(dict(sample))
    assert got.keys() == want.keys()
    for key in want:
        torch.testing.assert_close(got[key], want[key], atol=0, rtol=0)


@pytest.mark.parametrize("noise_std", [0.0, 0.05])
def test_eagle3_training_noise_matches_legacy(noise_std):
    family = drafter_family(_cfg(), _bank(noise_std=noise_std))
    x = torch.randn(3, 10)
    torch.manual_seed(5)
    got = family.training_noise(x)
    torch.manual_seed(5)
    want = x + 2 * (torch.rand_like(x) - 0.5) * noise_std if noise_std else x
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def _drafter(model_type, layers):
    return SimpleNamespace(
        config=SimpleNamespace(speculators_model_type=model_type),
        target_layer_ids=layers,
    )


def test_eagle3_check_drafter():
    bank = _bank()
    family = drafter_family(_cfg(), bank)
    layers = bank.hidden_states.layer_ids[:-1]
    family.check_drafter(_drafter("eagle3", list(layers)))
    with pytest.raises(ValueError, match="target layers differ"):
        family.check_drafter(_drafter("eagle3", [*layers, 99]))
    with pytest.raises(ValueError, match="bank selects family 'eagle3'"):
        family.check_drafter(_drafter("dflash", list(layers)))
