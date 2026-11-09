# DFlash generator recipes

The fork's LoRA generator on a frozen DFlash drafter. These recipes mirror the EAGLE-3
experiments; the agreed scope and decisions are in `dflash_port/EXPERIMENT_MATRIX.md`.

## Layout

| Path | Contents |
|---|---|
| `training_recipe.yaml` | Frozen models and revisions; extraction layers (the drafter's aux layers + 36); DFlash loss settings (`training.block_drafter`) |
| `common.yaml` | Shared protocol, mirroring `eagle3-lora/deepseek_ttt7_generator_v1`: 4K pool, 1,500 updates, rank 32 / α 64, q/k/v/o on all drafter layers, generator 512/128/64/32/512/2, W&B `dyn-spec-dflash-v1`, offline activation store |
| `m0/{base,generator,static_lora}.yaml` | Tier 1, matched arms. These are structure-study jobs, so all arms use the same training and validation subsets; only `arm` differs (as in EAGLE-3 GEN-009/010) |
| `matrices/tier1.yaml` | Tier 1, in run order: base → generator → static LoRA → offline evaluations |
| `p2/`, `matrices/p2_smoke.yaml` | P2 smoke: the M0 arms with 100 updates |
| `machine.local.yaml` | Per box, git-ignored, written by `setup.sh`: target path, drafter revision and dir |

Recipes for other tiers are added once their settings are confirmed, tier by tier.

## Running

**Start with [QUICKSTART.md](QUICKSTART.md)**: clone → `scripts/dflash_port/setup.sh` → P0 store fill → P2 smoke → Tier 1.

- **Per-box values:** the target path and the drafter commit/dir come from the git-ignored `machine.local.yaml`, which `setup.sh --prepare-drafter` writes. The recipes keep `PIN_AT_P0`, and real runs refuse to start while it is unresolved.
- **One arm:** `scripts/dflash_port/run_arm.py RECIPE --tier T --mode train|base|eval`. Flags: `--dry-run`, `--no-wandb`, `--no-hub`, `--output-root`, `--machine`.
- **One tier:** `scripts/dflash_port/run_matrix.py matrices/<tier>.yaml`. It runs entries in order, resumes, skips completed entries, and refuses completed entries whose configuration changed.
