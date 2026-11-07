"""S7: fail-loud guards on the generator training path."""

from types import SimpleNamespace

import pytest
import torch

from speculators import provenance
from speculators.generator.engine import Runtime, begin_run
from tests.unit.generator.test_dflash_generator import _bank, _cfg, _drafter


def _runtime():
    runtime = Runtime.__new__(Runtime)
    runtime.cfg, runtime.bank = _cfg(), _bank()
    runtime.device, runtime.dtype = torch.device("cpu"), torch.float32
    runtime.corpus = SimpleNamespace(prepare=lambda _drafter: None)
    return runtime


@pytest.mark.parametrize("name", ["lm_head", "verifier_lm_head", "embed_tokens"])
def test_attach_drafter_rejects_non_finite_verifier_tensors(name):
    drafter = _drafter()
    with torch.no_grad():
        getattr(drafter, name).weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match=f"non-finite tensors: \\['{name}.weight'\\]"):
        _runtime().attach_drafter(drafter)


def test_attach_drafter_accepts_finite_drafter():
    runtime = _runtime()
    runtime.cfg.factor_cache_mib = 1
    runtime.attach_drafter(_drafter())
    assert set(runtime.paths) == {
        f"l{i}_{name}"
        for i in range(len(runtime.drafter.layers))
        for name in runtime.cfg.target_modules
    }


def test_begin_run_refuses_unknown_git_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "git_sha", lambda _root: "unknown")
    with pytest.raises(RuntimeError, match="Cannot resolve the git SHA"):
        begin_run(tmp_path / "run", {"identity": 1})
    assert not (tmp_path / "run").exists()


def test_require_git_sha_returns_head(monkeypatch):
    monkeypatch.setattr(provenance, "git_sha", lambda _root: "abc123")
    assert provenance.require_git_sha(None) == "abc123"


def test_unported_feedback_paths_refuse_dflash():
    from speculators.generator.drafters.base import require_family  # noqa: PLC0415
    from speculators.generator.online_feedback import (  # noqa: PLC0415
        OnlineFeedbackTrainer,
    )
    from speculators.generator.saved_response_feedback import (  # noqa: PLC0415
        feedback_batch,
    )

    runtime = _runtime()  # DFlash bank
    with pytest.raises(NotImplementedError, match="not ported to dflash"):
        feedback_batch(runtime, None, None, {}, None, training=True)
    with pytest.raises(NotImplementedError, match="not ported to dflash"):
        OnlineFeedbackTrainer(runtime, None, None)
    eagle = SimpleNamespace(family=SimpleNamespace(name="eagle3"))
    require_family(eagle, "eagle3", "anything")  # allowed
