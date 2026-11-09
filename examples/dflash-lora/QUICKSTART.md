# DFlash generator: from clone to results

This page takes a fresh GPU box from a clone to finished Tier-1 runs. Everything runs from the repository root.

- **Why each step exists:** `dflash_port/PROGRESS.md`.
- **What each experiment is:** `dflash_port/EXPERIMENT_MATRIX.md`.

## 0. Requirements

- **Hardware:** Linux, NVIDIA GPUs (tested on 2× H100 80GB) and Python 3.12.
- **Disk:** about 0.3 TB for the 4K activation store, 1.7 TB for 24K and 6.5 TB for 100K. Add the model, drafter and checkpoints.
- **Accounts:**
  - Hugging Face: `hf auth login`, with read access and write access to `erdem-erdem` for checkpoint pushes.
  - W&B: `wandb login`. Optional; `--no-wandb` turns it off.

```bash
git clone <this repository> && cd dynamic-speculators-trainer
```

## 1. Setup (`scripts/dflash_port/setup.sh`)

Each step runs only with its own flag, and the steps are idempotent. With no flags, the script checks every item and prints the command for anything missing.

```bash
scripts/dflash_port/setup.sh --fetch-wheels      # HF erdem-erdem/dyn-spec-train-vllm-wheels-fixed-vllm -> ./wheels
scripts/dflash_port/setup.sh --create-envs       # .venv (training) + .venv-vllm (extraction), offline
scripts/dflash_port/setup.sh --fetch-data r0001  # 4K and 24K pools (HF azeeer/eagle3-qwen3-8b-trajectories-v1)
scripts/dflash_port/setup.sh --target /path/to/Qwen3-8B   # an existing copy at b968826d…
#   or: scripts/dflash_port/setup.sh --download-target
scripts/dflash_port/setup.sh --prepare-drafter <commit SHA of z-lab/Qwen3-8B-DFlash-b16>
scripts/dflash_port/setup.sh --fetch-store <commit SHA>   # optional: the 4K store from HF, skipping P0
scripts/dflash_port/setup.sh                     # check: everything [ok]
```

> **Installation caveat (environments):**
> - **`.venv` (training)** installs directly from the wheels.
> - **`.venv-vllm`** must install **guidellm first, then vLLM**, in that order:
>   1. `.venv-vllm/bin/pip install --no-index --find-links wheels/vllm guidellm==0.7.1`
>   2. `.venv-vllm/bin/pip install --no-index --find-links wheels/vllm vllm==0.31.0`
>
> `--create-envs` runs exactly these commands. The up-to-date wheels are in the HF repo
> `erdem-erdem/dyn-spec-train-vllm-wheels-fixed-vllm` (pinned in `setup.sh`).

Notes:

- **Getting the drafter SHA:** `git ls-remote https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16 HEAD`.
- **What preparation does:** it downloads that exact commit and converts it. It records the source SHA, the weight SHA256s and the drafter's layer settings in `preparation.json`, and the runtime checks these on every load. It also writes `examples/dflash-lora/machine.local.yaml`, the per-box target path and drafter pin. That file is git-ignored, so the recipes are never edited by hand.
- **Other DFlash drafters:** these work too (`scripts/dflash_port/prepare_dflash_drafter.py`). The recipe's `models.drafter` and aux layers must match.
- **100K pool:** `--fetch-data r0002-100k`, needed only for the scale tier.
- **Activation store from HF:** `--fetch-store <SHA>` downloads the published 4K store (`erdem-erdem/dflash-qwen3-8b-activations-4k`, pinned commit) and re-hashes every file against its `MANIFEST.json`. With it, step 2 (P0) is not needed on that box.

## 2. P0: fill the 4K activation store (one GPU)

```bash
scripts/dflash_port/fill_store.sh examples/dflash-lora/m0/generator.yaml --gpu 0 --compare 32
```

This does four things:
1. Starts the vLLM extractor with `--strict-provenance`. Unhashed weights, or an unidentified vLLM, stop the launch.
2. Extracts every training and validation row of the M0 job into `output/dflash-lora/activation-store-4k`, plus the validation rows of the test (`*-00001`) and donor (`*-00002`) subsets. The store is content-addressed, so a rerun only adds missing rows, and entries from a different extractor are never mixed in.
3. Re-extracts 32 rows directly through vLLM, bypassing the store. This is the **online ≡ offline check**: read `comparison` in `<store>/fills/<time>.json`. `bit_identical` or a tiny `max_abs_diff` is expected; anything else is a finding to report.
4. Stops the server.

After this step, no run needs vLLM: every M0 arm reads the store offline.

**Publish the store** (once, from the box that filled it). Visibility is a required choice:

```bash
.venv/bin/python scripts/dflash_port/push_store.py push output/dflash-lora/activation-store-4k \
  --repo erdem-erdem/dflash-qwen3-8b-activations-4k --visibility public|private
```

It writes `MANIFEST.json` (every file's SHA256), uploads with resumable `upload_large_folder`, and prints the commit SHA for `setup.sh --fetch-store`. Record that SHA in `RUNLOG.md`.

## 3. P2: smoke runs, then STOP

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dflash_port/run_matrix.py \
  examples/dflash-lora/matrices/p2_smoke.yaml --no-hub [--no-wandb]
```

The smoke matrix runs every code path once on the real models, with 100 updates per trained arm:

1. base evaluation;
2. generator and static-LoRA training;
3. both offline evaluations.

Then check:

- **Base vs initial validation:** base equals each trained arm's update-0 validation (`history[0]`).
- **Losses:** losses are finite and decreasing.
- **Reproduction:** `reproduction.reproduces_training_validation` in the offline evaluations. On GPU a tiny difference is possible; it is reported, not enforced.
- **Run records:** `run_record_*.json` all say `completed`.

**Stop here and report before Tier 1** (budget approval).

## 4. Tier 1 (M0): base, generator, static LoRA

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/dflash_port/run_matrix.py \
  examples/dflash-lora/matrices/tier1.yaml [--dry-run] [--output-root DIR] [--no-wandb] [--no-hub]
```

- **Order:**
  1. base;
  2. generator (1,500 updates);
  3. static LoRA (1,500 updates);
  4. offline evaluation of both trained arms on the 200 fixed validation rows: reproduction; M4 coefficients 0.5/1/2/4/8 with per-row results (`m4_per_row`: each row's accept length per coefficient, best coefficient, per-row oracle) and K=4 means; M7 donors correct/fixed/cross-category from the unseen `*-00002` rows; D13 similarity; D14 cost.
- **Evaluation protocol:** `reporting/dflash/README.md#evaluation-protocol` (50 rows per category, one LoRA per row, untouched test set).
- **Untouched test set (final numbers only):** `run_arm.py RECIPE --tier tier1 --mode test` (trained arm) or `--mode base_test` (released drafter). Not part of the matrix.
- **Resuming:** rerunning the same command resumes from checkpoints and skips completed entries. A completed entry whose configuration changed is refused, never silently reused.
- **One arm at a time:** `.venv/bin/python scripts/dflash_port/run_arm.py RECIPE --tier tier1 --mode train|base|eval|test|base_test`.
- **Two GPUs:** run one matrix entry per GPU with `run_arm.py` and separate `CUDA_VISIBLE_DEVICES`.

## Where things go

| What | Where |
|---|---|
| Runs | `<output-root>/<tier>/<recipe name>/` (default output root `/workspace/yusuf/runs/dflash`). Training job: `pretraining/<name>/` (checkpoints, `result.json`, provenance) |
| Run records | `run_record_<mode>.json`: command, git SHA, host, GPUs, configuration digest, drafter pin, status, outcome |
| Offline evaluation | `<output-root>/<tier>/<name>/offline_eval/<tier>/...json` |
| HF checkpoints | `erdem-erdem/dflash-b16-qwen3-8b-<arm>-attn-r32-4k-1500` (public), pushed every 500 updates and at the end |
| W&B | project `dflash-generator-v1` |
| Box setup record | `output/dflash-lora/env/env-<time>.json`; extractor logs: `output/dflash-lora/logs/` |

Following the repository rules (`dflash_port/SESSION_PROMPT.md`), add an entry to `dflash_port/RUNLOG.md` before and after every run, failures included.
