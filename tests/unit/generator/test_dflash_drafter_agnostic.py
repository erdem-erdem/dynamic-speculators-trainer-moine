"""S6a (D12): any released DFlash drafter, prepared at a pinned SHA and verified."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.generator.drafters import prepare_dflash as prep
from speculators.generator.drafters.dflash import DFlashFamily
from speculators.generator.engine import Runtime
from speculators.models.dflash.core import DFlashDraftModel
from tests.unit.generator.test_dflash_generator import _bank, _cfg

SHA = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "some-lab/Tiny-DFlash-b4"
HIDDEN, VOCAB, VERIFIER_LAYERS = 16, 32, 6
# A non-default geometry: block 4, 2 draft layers, 2 aux layers (source ids are
# pre-decoder indices, converted +1; the final verifier layer is dropped).
SOURCE_TARGETS = [0, 3, 5]
AUX = [1, 4]


def _verifier(root):
    path = root / "verifier"
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=VERIFIER_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def _source(root):
    """An external DFlash checkpoint in the publisher's (z-lab style) format."""
    path = root / "source"
    path.mkdir()
    config = {
        "model_type": "qwen3",
        "architectures": ["DFlashDraftModel"],
        "block_size": 4,
        "num_target_layers": VERIFIER_LAYERS,
        "dflash_config": {
            "mask_token_id": VOCAB - 1,
            "target_layer_ids": SOURCE_TARGETS,
        },
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
    }
    (path / "config.json").write_text(json.dumps(config))
    from speculators.convert.dflash.converter import DFlashConverter  # noqa: PLC0415

    native = DFlashConverter()._build_config(config, str(root / "verifier"), None)
    torch.manual_seed(1)
    model = DFlashDraftModel(native)
    body = {
        key: torch.randn_like(value)
        for key, value in model.state_dict().items()
        if key.startswith(("layers.", "fc.", "hidden_norm.", "norm."))
    }
    save_file(body, path / "model.safetensors")
    return path


def _native(root, *, trained_lm_head):
    """The same drafter already in speculators format (e.g. a RedHat release);
    verifier-owned tensors are omitted unless `trained_lm_head`."""
    from speculators.convert.dflash.converter import DFlashConverter  # noqa: PLC0415

    external = _source(root)
    path = root / "native"
    DFlashConverter().convert(external, path, base_model=str(root / "verifier"))
    model = DFlashDraftModel.from_pretrained(str(path))
    state = model.state_dict()
    keep = ("layers.", "fc.", "hidden_norm.", "norm.")
    if trained_lm_head:
        keep = (*keep, "lm_head.")
    body = {key: value for key, value in state.items() if key.startswith(keep)}
    save_file(body, path / "model.safetensors", metadata={"format": "pt"})
    return path


def _prepare(tmp_path, monkeypatch, source):
    def download(repo, *, revision, allow_patterns):
        assert (repo, revision) == (REPOSITORY, SHA)
        assert allow_patterns == prep.SOURCE_PATTERNS
        return str(source)

    monkeypatch.setattr(prep, "snapshot_download", download)
    return prep.prepare_dflash(
        REPOSITORY, SHA, tmp_path / "verifier", tmp_path / "prepared"
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    _verifier(tmp_path)
    source = _source(tmp_path)
    return _prepare(tmp_path, monkeypatch, source), source


def _matching_bank():
    bank = _bank()
    bank.models.drafter, bank.models.drafter_revision = REPOSITORY, SHA
    bank.hidden_states.layer_ids = [*AUX, VERIFIER_LAYERS]
    return bank


def test_prepare_records_source_hashes_and_geometry(prepared):
    path, source = prepared
    record = json.loads((path / "preparation.json").read_text())
    assert path.name == SHA
    assert record["source_repository"] == REPOSITORY
    assert record["source_revision"] == SHA
    assert record["source_format"] == "external"
    assert record["source_weights_sha256"] == {
        "model.safetensors": prep.file_digest(source / "model.safetensors")
    }
    assert record["weights_sha256"] == prep.weight_digests(path)
    assert record["aux_hidden_state_layer_ids"] == AUX
    assert (record["block_size"], record["mask_token_id"]) == (4, VOCAB - 1)
    assert (record["draft_layers"], record["hidden_size"]) == (2, HIDDEN)
    assert json.loads((path / "original_config.json").read_text())["block_size"] == 4
    assert prep.verify_prepared_drafter(path, _matching_bank()) == record


def test_prepare_native_checkpoint_fills_missing_verifier_tensors(
    tmp_path, monkeypatch
):
    verifier_path = _verifier(tmp_path)
    source = _native(tmp_path, trained_lm_head=False)
    body = load_file(source / "model.safetensors")
    path = _prepare(tmp_path, monkeypatch, source)
    record = json.loads((path / "preparation.json").read_text())
    assert record["source_format"] == "speculators"
    prepared = DFlashDraftModel.from_pretrained(str(path)).state_dict()
    verifier = Qwen3ForCausalLM.from_pretrained(verifier_path).state_dict()
    for key, value in body.items():
        assert torch.equal(prepared[key], value), key
    for key, name in {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "lm_head.weight": "lm_head.weight",
        "verifier_norm.weight": "model.norm.weight",
    }.items():
        assert torch.equal(prepared[key], verifier[name]), key
    assert set(record["verifier_filled"]) == {
        "embed_tokens.weight",
        "lm_head.weight",
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    }
    assert prep.verify_prepared_drafter(path, _matching_bank()) == record


def test_prepare_native_refuses_lm_head_that_differs_from_verifier(
    tmp_path, monkeypatch
):
    """Upstream reloads a full-vocab DFlash lm_head from the verifier on every
    load; a stored head that differs (wrong verifier, or a trained head that
    would be silently discarded) fails loudly."""
    _verifier(tmp_path)
    source = _native(tmp_path, trained_lm_head=True)
    body = load_file(source / "model.safetensors")
    body["lm_head.weight"] = body["lm_head.weight"] + 1.0
    save_file(body, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="changed source tensor lm_head.weight"):
        _prepare(tmp_path, monkeypatch, source)


def test_prepare_requires_commit_sha(tmp_path):
    with pytest.raises(ValueError, match="must be a commit SHA"):
        prep.prepare_dflash(REPOSITORY, "main", tmp_path, tmp_path)


def test_prepare_refuses_existing_output(prepared):
    path, _ = prepared
    with pytest.raises(FileExistsError):
        prep.prepare_dflash(REPOSITORY, SHA, path, path.parent)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda bank, _p: setattr(bank.models, "drafter", "z-lab/other"), "drafter"),
        (
            lambda bank, _p: setattr(bank.models, "drafter_revision", "f" * 40),
            "drafter_revision",
        ),
        (
            lambda bank, _p: setattr(bank.hidden_states, "layer_ids", [1, 3, 6]),
            "aux layers",
        ),
        (
            lambda _b, path: (path / "model.safetensors").write_bytes(
                (path / "model.safetensors").read_bytes() + b"\0"
            ),
            "weights_sha256",
        ),
        (lambda _b, path: (path / "preparation.json").unlink(), "Missing DFlash"),
    ],
)
def test_verify_rejects_any_mismatch(prepared, change, message):
    path, _ = prepared
    bank = _matching_bank()
    change(bank, path)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        prep.verify_prepared_drafter(path, bank)


def test_non_default_geometry_attaches_and_trains(prepared):
    """A second, differently shaped DFlash drafter goes through the production path."""
    path, _ = prepared
    bank = _matching_bank()
    cfg = _cfg()
    runtime = Runtime.__new__(Runtime)
    runtime.cfg, runtime.bank = cfg, bank
    runtime.device, runtime.dtype = torch.device("cpu"), torch.float32
    runtime.corpus = SimpleNamespace(prepare=lambda _drafter: None)
    runtime.cfg.factor_cache_mib = 1
    runtime.family.verify_local_drafter(path)
    drafter = DFlashDraftModel.from_pretrained(str(path), dtype=torch.float32)
    drafter.config.transformer_layer_config._attn_implementation = "eager"
    runtime.attach_drafter(drafter)
    assert runtime.family.block_size == 4
    assert set(runtime.paths) == {
        f"l{i}_{name}" for i in range(2) for name in cfg.target_modules
    }


def test_runtime_verifies_local_drafter_before_loading(tmp_path, monkeypatch):
    """`model_paths` drafters are verified before any weights are read."""
    for role in ("target", "drafter"):
        (tmp_path / role).mkdir()
        (tmp_path / role / "config.json").write_text("{}")
    seen = []

    def refuse(_family, path):
        seen.append(path)
        raise ValueError("refused")

    monkeypatch.setattr(DFlashFamily, "verify_local_drafter", refuse)
    paths = {role: tmp_path / role for role in ("target", "drafter")}
    cfg = _cfg()
    cfg.dtype = "float32"
    with pytest.raises(ValueError, match="refused"):
        Runtime(cfg, _bank(), corpus_factory=lambda *_: None, model_paths=paths)
    assert seen == [paths["drafter"].resolve()]
