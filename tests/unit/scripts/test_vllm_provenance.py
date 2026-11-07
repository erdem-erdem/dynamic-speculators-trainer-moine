from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from launch_vllm import (  # type: ignore[import-not-found]
    _find_vllm_repo,
    _resolve_pinned,
    _save_checkpoint_sha256,
    _save_vllm_command,
    _save_vllm_install,
    _save_vllm_patch,
    _save_vllm_provenance,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

import pytest


class TestFindVllmRepo:
    def test_returns_string_or_none(self):
        result = _find_vllm_repo()
        assert result is None or isinstance(result, str)

    def test_returns_none_when_vllm_not_importable(self):
        with patch("launch_vllm.find_package_repo", return_value=None):
            assert _find_vllm_repo() is None


class TestSaveVllmCommand:
    def test_creates_file(self, tmp_path: Path):
        _save_vllm_command(
            tmp_path,
            ["python", "-m", "vllm", "serve", "model"],
            "abc123",
            "",
            "0.1.0",
        )
        assert (tmp_path / "vllm_command.txt").exists()

    def test_contains_metadata(self, tmp_path: Path):
        _save_vllm_command(
            tmp_path,
            ["python", "-m", "vllm", "serve", "model"],
            "abc123",
            "",
            "0.1.0",
        )
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "# Timestamp:" in content
        assert "# Python:" in content
        assert "# Git SHA: abc123" in content
        assert "# vllm: 0.1.0" in content
        assert "python -m vllm serve model" in content

    def test_dirty_marker_when_diff_present(self, tmp_path: Path):
        _save_vllm_command(tmp_path, ["cmd"], "abc123", "some diff", "0.1.0")
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "abc123 (dirty)" in content

    def test_no_dirty_marker_when_clean(self, tmp_path: Path):
        _save_vllm_command(tmp_path, ["cmd"], "abc123", "", "0.1.0")
        content = (tmp_path / "vllm_command.txt").read_text()
        assert "(dirty)" not in content


class TestSaveVllmPatch:
    def test_records_repo_and_diff(self, tmp_path: Path):
        _save_vllm_patch(tmp_path, "/path/to/vllm", "abc123", "diff content", "0.1.0")
        content = (tmp_path / "vllm.patch").read_text()
        assert "# repo: /path/to/vllm (abc123)" in content
        assert "diff content" in content

    def test_header_only_for_clean_checkout(self, tmp_path: Path):
        _save_vllm_patch(tmp_path, "/path/to/vllm", "abc123", "", "0.1.0")
        content = (tmp_path / "vllm.patch").read_text()
        assert "# repo: /path/to/vllm (abc123)" in content

    def test_wheel_install_fallback(self, tmp_path: Path):
        _save_vllm_patch(tmp_path, None, "unknown", "", "0.5.0")
        content = (tmp_path / "vllm.patch").read_text()
        assert "wheel install" in content
        assert "0.5.0" in content


class TestSaveCheckpointSha256:
    def test_remote_model(self, tmp_path: Path):
        _save_checkpoint_sha256(tmp_path, "org/model-name")
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "not a local path" in content

    def test_local_model_with_safetensors(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(tmp_path, str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "model.safetensors" in content
        assert len(content.split("  ")[0]) == 64  # SHA256 hex length

    def test_no_safetensors(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        _save_checkpoint_sha256(tmp_path, str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "no .safetensors files" in content

    def test_skip_hash_records_size_and_mtime(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(tmp_path, str(model_dir), skip_hash=True)
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        assert "# hashing skipped (--no-hash-checkpoints)" in content
        assert "size=" in content
        assert "mtime=" in content
        assert "model.safetensors" in content

    def test_skip_hash_does_not_contain_sha256(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"fake weights")
        _save_checkpoint_sha256(tmp_path, str(model_dir), skip_hash=True)
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        data_lines = [line for line in content.splitlines() if not line.startswith("#")]
        for line in data_lines:
            assert line.startswith("size=")

    def test_multiple_safetensors_sorted(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        for name in ("c.safetensors", "a.safetensors", "b.safetensors"):
            (model_dir / name).write_bytes(b"data")
        _save_checkpoint_sha256(tmp_path, str(model_dir))
        content = (tmp_path / "checkpoint_sha256.txt").read_text()
        names = [line.split("  ")[1] for line in content.strip().splitlines()]
        assert names == ["a.safetensors", "b.safetensors", "c.safetensors"]


class TestSaveVllmProvenance:
    def test_creates_all_artifacts(self, tmp_path: Path):
        prov_dir = tmp_path / "provenance"
        _save_vllm_provenance(["python", "serve"], str(prov_dir), "org/model")
        assert (prov_dir / "vllm_command.txt").exists()
        assert (prov_dir / "vllm.patch").exists()
        assert (prov_dir / "checkpoint_sha256.txt").exists()
        assert not (prov_dir / "drafter_checkpoint_sha256.txt").exists()

    def test_creates_drafter_checkpoint_in_eval_mode(self, tmp_path: Path):
        prov_dir = tmp_path / "provenance"
        _save_vllm_provenance(
            ["python", "serve"],
            str(prov_dir),
            "org/model",
            spec_model="org/drafter",
        )
        assert (prov_dir / "checkpoint_sha256.txt").exists()
        assert (prov_dir / "drafter_checkpoint_sha256.txt").exists()
        content = (prov_dir / "drafter_checkpoint_sha256.txt").read_text()
        assert "org/drafter" in content

    def test_creates_provenance_dir(self, tmp_path: Path):
        prov_dir = tmp_path / "nested" / "provenance"
        _save_vllm_provenance(["cmd"], str(prov_dir), "org/model")
        assert prov_dir.is_dir()

    def test_bad_dir_warns_and_returns(self, tmp_path: Path, capsys):
        _save_vllm_provenance(["cmd"], "/dev/null/impossible", "org/model")
        captured = capsys.readouterr()
        assert "Warning:" in captured.err

    def test_skip_hash_propagated(self, tmp_path: Path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "weights.safetensors").write_bytes(b"data")
        prov_dir = tmp_path / "prov"
        _save_vllm_provenance(["cmd"], str(prov_dir), str(model_dir), skip_hash=True)
        content = (prov_dir / "checkpoint_sha256.txt").read_text()
        assert "hashing skipped" in content

    def test_individual_writer_failure_does_not_block_others(self, tmp_path: Path):
        prov_dir = tmp_path / "prov"
        with patch("launch_vllm._save_vllm_command", side_effect=RuntimeError("boom")):
            _save_vllm_provenance(["cmd"], str(prov_dir), "org/model")
        assert not (prov_dir / "vllm_command.txt").exists()
        assert (prov_dir / "vllm.patch").exists()
        assert (prov_dir / "checkpoint_sha256.txt").exists()


class TestStrictProvenance:
    """--strict-provenance: any provenance failure blocks the launch."""

    @staticmethod
    def _model(tmp_path: Path) -> str:
        model = tmp_path / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        return str(model)

    def test_bad_dir_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="could not create provenance dir"):
            _save_vllm_provenance(
                ["cmd"], "/dev/null/impossible", self._model(tmp_path), strict=True
            )

    def test_writer_failure_raises_after_writing_the_others(self, tmp_path: Path):
        prov_dir = tmp_path / "prov"
        with (
            patch("launch_vllm._save_vllm_command", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="vllm_command.txt"),
        ):
            _save_vllm_provenance(
                ["cmd"], str(prov_dir), self._model(tmp_path), strict=True
            )
        assert (prov_dir / "vllm.patch").exists()
        assert (prov_dir / "checkpoint_sha256.txt").exists()

    def test_complete_provenance_passes(self, tmp_path: Path):
        prov_dir = tmp_path / "prov"
        with (
            patch("launch_vllm._find_vllm_repo", return_value=None),
            patch("launch_vllm._save_vllm_install") as install,
        ):
            _save_vllm_provenance(
                ["cmd"], str(prov_dir), self._model(tmp_path), strict=True
            )
        assert (prov_dir / "vllm_command.txt").exists()
        assert "weights" not in (prov_dir / "checkpoint_sha256.txt").read_text()
        install.assert_called_once()  # wheel install: fingerprint recorded

    @pytest.mark.parametrize(
        ("model", "skip", "message"),
        [
            ("org/model", False, "local checkpoint"),
            (None, True, "forbids --no-hash-checkpoints"),
        ],
    )
    def test_degraded_provenance_is_refused(self, tmp_path, model, skip, message):
        prov_dir = tmp_path / "prov"
        with pytest.raises(RuntimeError, match=message):
            _save_vllm_provenance(
                ["cmd"],
                str(prov_dir),
                model or self._model(tmp_path),
                skip_hash=skip,
                strict=True,
            )
        assert not prov_dir.exists()  # refused before anything is written

    def test_wheel_install_fingerprint(self, tmp_path: Path):
        fake = type("D", (), {"read_text": lambda self, name: "a,sha256=x,1\n"})()
        with patch("importlib.metadata.distribution", return_value=fake):
            _save_vllm_install(tmp_path, "0.31.0")
        text = (tmp_path / "vllm_install.txt").read_text()
        assert text.startswith("vllm 0.31.0\nrecord_sha256 ")

    def test_requires_provenance_dir(self):
        with (
            patch(
                "sys.argv", ["launch_vllm.py", "train", "gpt2", "--strict-provenance"]
            ),
            patch("launch_vllm._build_train_cmd", return_value=["train-cmd"]),
            patch("launch_vllm.os.execvp") as execvp,
            pytest.raises(SystemExit, match="requires --provenance-dir"),
        ):
            main()
        execvp.assert_not_called()


class TestRenderSetupGating:
    """Render thread bounds apply to non-headless train mode only."""

    def _run(self, argv: list[str]) -> dict:
        called = {"render": False}
        with (
            patch("sys.argv", ["launch_vllm.py", *argv]),
            patch("launch_vllm._build_train_cmd", return_value=["train-cmd"]),
            patch("launch_vllm._build_eval_cmd", return_value=["eval-cmd"]),
            patch(
                "launch_vllm._set_render_thread_defaults",
                side_effect=lambda: called.__setitem__("render", True),
            ),
            patch("launch_vllm.os.execvp"),
        ):
            main()
        return called

    def test_train_sets_render_defaults(self):
        assert self._run(["train", "gpt2"]) == {"render": True}

    def test_eval_does_not_set_render_defaults(self):
        called = self._run(["eval", "gpt2", "--spec-model", "org/drafter"])
        assert called == {"render": False}

    def test_train_headless_skips_render_defaults(self):
        called = self._run(["train", "gpt2", "--", "--headless"])
        assert called == {"render": False}


SHA = "0123456789abcdef0123456789abcdef01234567"


class TestPinnedHubCheckpoints:
    """HF repos pinned to a commit SHA are served from that exact snapshot."""

    def test_local_path_and_unpinned_ids_pass_through(self, tmp_path: Path):
        assert _resolve_pinned(str(tmp_path), None) == (str(tmp_path), None)
        assert _resolve_pinned("org/model", None) == ("org/model", None)

    def test_revision_must_be_a_commit_sha(self):
        with pytest.raises(SystemExit, match="40-character commit SHA"):
            _resolve_pinned("org/model", "main")

    def test_revision_is_for_hub_ids_only(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not local paths"):
            _resolve_pinned(str(tmp_path), SHA)

    def test_pinned_id_resolves_to_snapshot(self, tmp_path: Path):
        with patch(
            "huggingface_hub.snapshot_download", return_value=str(tmp_path)
        ) as download:
            path, source = _resolve_pinned("org/model", SHA)
        download.assert_called_once_with("org/model", revision=SHA)
        assert path == str(tmp_path)
        assert source == {"repository": "org/model", "revision": SHA, "snapshot": path}

    def test_strict_train_from_pinned_hub_repo(self, tmp_path: Path):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "model.safetensors").write_bytes(b"weights")
        prov = tmp_path / "prov"
        argv = [
            "launch_vllm.py",
            "train",
            "org/model",
            "--model-revision",
            SHA,
            "--strict-provenance",
            "--provenance-dir",
            str(prov),
        ]
        built = {}

        def build(args, vllm_args):
            built.update(model=args.model, vllm_args=list(vllm_args))
            return ["vllm", "serve", args.model, *vllm_args]

        with (
            patch("sys.argv", argv),
            patch("huggingface_hub.snapshot_download", return_value=str(snapshot)),
            patch("launch_vllm._build_train_cmd", side_effect=build),
            patch("launch_vllm._set_render_thread_defaults"),
            patch("launch_vllm._save_vllm_install"),
            patch("launch_vllm.os.execvp") as execvp,
        ):
            main()
        assert built["model"] == str(snapshot)  # vLLM serves the hashed files
        assert built["vllm_args"][-2:] == ["--served-model-name", "org/model"]
        source = (prov / "checkpoint_source.txt").read_text()
        assert SHA in source
        assert "org/model" in source
        assert "weights" not in (prov / "checkpoint_sha256.txt").read_text()
        execvp.assert_called_once()

    def test_strict_refuses_unpinned_hub_repo(self, tmp_path: Path):
        argv = [
            "launch_vllm.py",
            "train",
            "org/model",
            "--strict-provenance",
            "--provenance-dir",
            str(tmp_path / "prov"),
        ]
        with (
            patch("sys.argv", argv),
            patch("launch_vllm._build_train_cmd", return_value=["vllm"]),
            patch("launch_vllm._set_render_thread_defaults"),
            patch("launch_vllm.os.execvp") as execvp,
            pytest.raises(RuntimeError, match="pinned with --model-revision"),
        ):
            main()
        execvp.assert_not_called()
