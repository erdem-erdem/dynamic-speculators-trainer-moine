# DFlash generator recipes

The fork's LoRA generator on a frozen DFlash drafter. These recipes mirror the EAGLE-3
experiments; the agreed scope and decisions are in `dflash_port/EXPERIMENT_MATRIX.md`.

## Layout

| Path | Contents |
|---|---|
| `training_recipe.yaml` | Frozen models and revisions; extraction layers (the drafter's aux layers + 36); DFlash loss settings (`training.block_drafter`) |
| `common.yaml` | Shared protocol, mirroring `eagle3-lora/deepseek_ttt7_generator_v1`: 4K pool, 1,500 updates, rank 32 / α 64, q/k/v/o on all drafter layers, generator 512/128/64/32/512/2, W&B `dflash-generator-v1`, offline activation store |
| `m0/{base,generator,static_lora}.yaml` | Tier 1, matched arms. These are structure-study jobs, so all arms use the same training and validation subsets; only `arm` differs (as in EAGLE-3 GEN-009/010) |
| `matrices/tier1.yaml` | Tier 1, in run order: base → generator → static LoRA → offline evaluations |

Recipes for other tiers are added once their settings are confirmed, tier by tier.

## Before running (P0, `dflash_port/PROGRESS.md`)

1. Check the box with `scripts/dflash_port/remote_setup.sh`. Then run its `--install-vllm`, `--download-target` and `--prepare-drafter <SHA>` steps.
2. Replace both `PIN_AT_P0` values with the prepared drafter's commit SHA and directory:
   - `training_recipe.yaml`: `models.drafter_revision`;
   - `common.yaml`: `online.drafter_model_path`.
3. Fill the 4K activation store once:
   - launch `scripts/launch_vllm.py train ...` with `--strict-provenance --provenance-dir output/dflash-lora/activation-store-4k/provenance/vllm` (strict mode fails the launch if any provenance artifact is missing; the fill refuses to start without it);
   - run `scripts/dflash_port/fill_activation_store.py` on a recipe with `--store-mode write`, adding `--compare N` for the online ≡ offline check.
4. Log in with `hf auth login` (repos are created under `erdem-erdem`, public) and `wandb login`.

## Running

```bash
# Whole tier, in order; skips entries whose run record says completed
python scripts/dflash_port/run_matrix.py examples/dflash-lora/matrices/tier1.yaml [--dry-run] [--no-wandb] [--output-root DIR]

# One arm
python scripts/dflash_port/run_arm.py examples/dflash-lora/m0/generator.yaml --tier tier1 --mode train|base|eval [--dry-run]
```

Output goes to `<output-root>/<tier>/<recipe name>/`; the default output root is `/workspace/yusuf/runs/dflash`.

- **Training** (`--mode train`) saves local checkpoints every 500 updates and pushes to the HF Hub at the same points.
- **Run record:** `run_record_<mode>.json` holds the command, git SHA, host, GPUs, config digest, seed, status and outcome.
- **Offline evaluation** (`--mode eval`) covers:
  - a full-size reproduction check;
  - the M4 coefficients 0.5/1/2/4/8 and the M7 donors correct/fixed/cross-category, both on a fixed set of 24 rows per category;
  - D13 similarity and D14 cost.

  Results go to `offline_eval/<tier>/`.
- **Base** (`--mode base`) evaluates the released drafter with no adapters, on the same split.
