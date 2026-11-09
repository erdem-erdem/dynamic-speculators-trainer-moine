"""Offline out-of-distribution set (user 2026-11-09): selection, generation,
row format, integrity, and offline evaluation through the store path."""

import json

import pytest

from speculators.generator import ood_set
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401


class _Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "<chat>" + messages[0]["content"]

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) % 13 + 1 for c in text[:6]]


def _download(tmp_path):
    files = {
        "HumanEval": [
            {"prompt": f"def f{i}():", "task_id": f"H/{i}"} for i in range(5)
        ],
        "summarization": [
            {"prompt": f"summarise {i}", "category": "news" if i % 2 else "mail"}
            for i in range(6)
        ],
    }

    def download(name):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in files[name]))
        return path

    return download


def _post(payload):
    tokens = [7, 8, 9][: payload["max_tokens"]]
    return {
        "choices": [{"token_ids": tokens, "finish_reason": "stop"}],
        "prompt_token_ids": payload["prompt"],
    }


def test_selection_formats_and_is_seeded(tmp_path):
    tokenizer = _Tokenizer()
    selected, sources = ood_set.select_prompts(
        tokenizer, ("HumanEval", "summarization"), 3, 42, download=_download(tmp_path)
    )
    assert [len(v) for v in selected.values()] == [3, 3]
    assert {r["prompt_format"] for r in selected["HumanEval"]} == {"raw"}
    assert {r["prompt_format"] for r in selected["summarization"]} == {"chat"}
    # Chat prompts: thinking off, as the r0001 trajectories.
    assert tokenizer.calls
    assert all(call["enable_thinking"] is False for call in tokenizer.calls)
    # Round-robin over source categories: both summarization kinds appear.
    assert {r["subset"] for r in selected["summarization"]} == {
        "summarization-news",
        "summarization-mail",
    }
    again, _ = ood_set.select_prompts(
        _Tokenizer(),
        ("HumanEval", "summarization"),
        3,
        42,
        download=_download(tmp_path),
    )
    assert again == selected
    assert sources[0]["selected_rows"] == 3
    with pytest.raises(ValueError, match="eligible prompts"):
        ood_set.select_prompts(
            _Tokenizer(), ("HumanEval",), 9, 42, download=_download(tmp_path)
        )


def test_build_writes_rows_and_load_checks_hashes(tmp_path):
    output = tmp_path / "ood"
    manifest = ood_set.build(
        output,
        _Tokenizer(),
        _post,
        "Qwen/Qwen3-8B",
        benchmarks=("HumanEval", "summarization"),
        max_samples=2,
        seed=1,
        provenance={"git_sha": "x"},
        workers=2,
        download=_download(tmp_path),
    )
    assert set(manifest["subsets"]) == {
        "ood-humaneval-00000",
        "ood-summarization-00000",
    }
    assert manifest["sampling"] == ood_set.R0001_SAMPLING
    _, rows = ood_set.load(output)
    row = rows["ood-humaneval-00000"][0]
    assert row["input_ids"][row["prompt_length"] :] == [7, 8, 9]
    assert row["loss_mask"] == [0] * row["prompt_length"] + [1, 1, 1]
    assert row["seq_len"] == len(row["input_ids"])
    with pytest.raises(FileExistsError):
        ood_set.build(
            output,
            _Tokenizer(),
            _post,
            "m",
            benchmarks=("HumanEval",),
            max_samples=1,
            seed=1,
            provenance={},
            download=_download(tmp_path),
        )
    (output / "ood-humaneval-00000.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        ood_set.load(output)


def test_generation_refuses_missing_or_changed_tokens():
    record = {"row_id": "r", "prompt_token_ids": [1, 2], "max_tokens": 4}
    with pytest.raises(ValueError, match="no token ids"):
        ood_set.generate(
            lambda p: {"choices": [{"token_ids": [], "finish_reason": "stop"}]},
            "m",
            record,
        )
    with pytest.raises(ValueError, match="changed the prompt"):
        ood_set.generate(
            lambda p: {
                "choices": [{"token_ids": [3], "finish_reason": "stop"}],
                "prompt_token_ids": [9],
            },
            "m",
            record,
        )


def test_ood_rows_are_evaluated_offline_without_touching_training(
    online_campaign,  # noqa: F811
    tmp_path,
):
    """Appended OOD rows keep trajectory indices, never enter a job's identity,
    and are evaluated (base arm, per-row records) through the online path."""
    from speculators.generator import campaign  # noqa: PLC0415
    from speculators.generator.local_campaign import (  # noqa: PLC0415
        load_local_inputs,
    )
    from speculators.generator.offline_eval import evaluate_base  # noqa: PLC0415

    cfg, _, requests = online_campaign
    output = tmp_path / "ood"
    ood_set.build(
        output,
        _Tokenizer(),
        _post,
        "m",
        benchmarks=("HumanEval", "summarization"),
        max_samples=2,
        seed=1,
        provenance={},
        download=_download(tmp_path),
    )
    plain = load_local_inputs(cfg)
    merged, _ = ood_set.with_ood(plain, output)
    assert merged.rows.select(range(len(plain.rows))).to_dict() == plain.rows.to_dict()
    assert merged.identity == plain.identity
    assert merged.subsets["ood-humaneval-00000"]["validation_indices"] == [
        len(plain.rows),
        len(plain.rows) + 1,
    ]
    job = campaign.jobs(cfg)[0]
    record = evaluate_base(cfg, job, name="unit", split="ood", ood=output)
    assert set(record["ood"]["subsets"]) == {
        "ood-humaneval-00000",
        "ood-summarization-00000",
    }
    assert record["ood"]["validation_examples"] == 4
    assert record["ood_set"]["dataset"]["id"] == ood_set.REPO
    assert requests  # extracted through the (mocked) vLLM path
    with pytest.raises(ValueError, match="OOD set directory"):
        evaluate_base(cfg, job, name="unit", split="ood")
