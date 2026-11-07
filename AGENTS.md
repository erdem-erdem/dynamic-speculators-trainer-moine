# Speculators — Agent Guide

## DFlash generator port (active work)

Porting the fork's LoRA generator from EAGLE-3 to DFlash. Start at `dflash_port/PROGRESS.md`. It covers status
and rules and links to `FINDINGS.md`, `GUIDE_CORRECTED.md`, `PLAN.md` and the detailed `notes/`.

- **To set up a GPU box and run** (environments, data, models, store fill, smoke, training, evaluation): `examples/dflash-lora/QUICKSTART.md`.
- **Planned experiments and ablations, their order and status** (living document): `dflash_port/EXPERIMENT_MATRIX.md`.

## Provenance

Training, eval, and vLLM-launch scripts write reproducibility artifacts so runs can be recreated later.

- **Training** (`src/speculators/train/utils.py`): `save_train_command()` writes `train_command.txt` (timestamp, git SHA, world size, package versions, full argv) and `speculators.patch` (uncommitted diff) into the checkpoint directory.
- **Eval** (`scripts/evaluate/evaluate.py`): `save_eval_provenance()` writes `eval_command.txt` (timestamp, git SHA, package versions, full argv) into the output directory.
- **vLLM launch** (`scripts/launch_vllm.py`): `_save_vllm_provenance()` writes `vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt` (plus `drafter_checkpoint_sha256.txt` in eval mode) — but **only when `--provenance-dir` is passed**. Three subcommands:
  - `launch_vllm.py responses MODEL`: target response generation without hidden-state extraction.
  - `launch_vllm.py train MODEL` (default): hidden-states extraction for training data generation.
  - `launch_vllm.py eval MODEL --spec-model DRAFTER`: speculative decoding serving for evaluation.

**Always pass `--provenance-dir` when running `scripts/launch_vllm.py`** — point it at the associated run's output/checkpoint directory (e.g. `--provenance-dir <output_dir>`) so the vLLM provenance is co-located with the training/eval it belongs to.

When publishing a model or eval results (HuggingFace, GitHub comments, etc.), always include the provenance artifacts (`train_command.txt`, `eval_command.txt`, `vllm_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, patches) so the run can be reproduced.

## Reporting

Keep Markdown experiment records as the working documentation. Update the
LaTeX report under `reporting/` only when the user explicitly requests a report
or an extension; do not generate daily or automatic reports. Follow
`reporting/README.md`: distinguish measurements, implementation status, and open
questions, preserve evidence limitations, and avoid final conclusions.
