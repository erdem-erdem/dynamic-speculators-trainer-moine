# ABL24K: loss, LoRA rank and generator size on 24K mixed data (2 epochs)

- **Status:** done (2026-11-14). One seed per arm.
- **Requested by the user (2026-11-13):** an end-to-end batch, run unattended over 15 hours on 2 GPUs, with trainings, evals, reports, W&B, HF and git:
  - ablations of the loss, static LoRA rank from 32 to 1024, and generator size through head sharing;
  - 24K data, 2 epochs.
- **Plan:** [experiment_suggestions/plan_24k_2026-11-13.md](experiment_suggestions/plan_24k_2026-11-13.md) (the user chose the "Balanced" allocation).
- **Previous:** [MIX24K.md](MIX24K.md) (the same data, 3 epochs, static r32 vs 1.1B generator), [PERTURB24K.md](PERTURB24K.md) (where the drafter can be moved).
- **Interactive page:** https://claude.ai/artifact/GJDcjp7Wd4K2Egvb22mHxQ ("ABL24K Ablations"; private until shared)

## TL;DR

Measurements first; interpretation is marked. Test = 500 rows; Δτ = paired difference in category-mean τ, 95% bootstrap CI.

- **Static LoRA rank is the largest lever here.** Against static r32, test Δτ:

  | Rank | Δτ (test) | 95% CI |
  |---|---|---|
  | 64 | +0.017 | [+0.010, +0.026] |
  | 128 | +0.027 | [+0.017, +0.037] |
  | 256 | +0.035 | [+0.022, +0.048] |
  | 512 | +0.046 | [+0.032, +0.061] |
  | 1024 | +0.050 | [+0.034, +0.066] |

  - Validation agrees: +0.007 → +0.045.
  - Gains flatten after r512, and OOD peaks at r512 (+0.036 [+0.008, +0.066]); r1024 OOD is +0.016.
  - Static r1024 (136M parameters) beats the 1.1B generator on test (+0.014 [+0.002, +0.026]) and on validation (+0.020 [+0.009, +0.031]).
- **Every generator lands at about +0.03 τ over static r32, whatever its size or rank:**

  | Generator | Params | Δτ (test) | 95% CI |
  |---|---|---|---|
  | Per-module heads | 1.1B | +0.035 | [+0.019, +0.052] |
  | Heads shared across layers | 0.23B | +0.026 | [+0.009, +0.042] |
  | Shared heads, r64 | 0.45B | +0.032 | [+0.017, +0.048] |
  | Shared heads, decoder 128, r128 | 0.45B | +0.026 | [+0.009, +0.042] |
  | __GEN_R512__ | | | |

  - Shared vs per-module heads: −0.010 [−0.022, +0.001] on test, −0.001 on validation.
  - **Sharing heads cuts the generator 5× with no resolvable loss,** and halves the time to make a LoRA (4.0 vs 7.6 ms).
  - OOD, no generator beats static r32 (all intervals include zero).
- **The losses tie in distribution.** Against KL (static r32), test Δτ:
  - CE: +0.006 [−0.005, +0.018];
  - KL with the new label-probability gate (α 0.1): −0.003 [−0.015, +0.008].

  The label gate is the only loss change with a resolvable OOD gain, +0.028 [+0.005, +0.051]. CE's OOD gain, +0.024 [−0.008, +0.057], is not resolvable.
- *Interpretation:* at 24K rows and 2 epochs, extra parameters do more as a higher-rank static LoRA than as a generator. A 34M static (r256) equals the 1.1B generator, and 68–136M statics pass it. The generator's advantage over r32 (MIX24K) is about the size of what r128–r256 adds to a static LoRA. Fixing the D-PACE gate does not move in-distribution τ.

## What was run

Every arm uses the same frozen drafter (`z-lab/Qwen3-8B-DFlash-b16`):
- LoRA on q, k, v, o of all 5 layers, coefficient α/r = 2;
- 24,000 rows (r0002-100k subsets 00000..00005 × math / code / stem / chat);
- 2 epochs = 2,667 updates × 18 rows, AdamW 1e-4, seed 42;
- validation every 250 updates;
- the MIX24K eval sets: validation and test 500 rows each (200 math + 100 code / stem / chat), and OOD v3 with 240 rows.

| Axis | Arms | Fixed |
|---|---|---|
| Static rank | r32, 64, 128, 256, 512, 1024 (α = 2r) | KL + D-PACE |
| Loss (static r32) | KL + D-PACE (reference); CE + D-PACE; KL + label-probability gate α 0.1 (new) | rank 32 |
| Generator size and rank | per-module heads decoder 256 r32 (1.1B); heads shared across layers decoder 256 r32 (0.23B), r64 (0.45B); shared decoder 128 r128 (0.45B); shared decoder 32 r512 (0.46B) | KL + D-PACE; encoder, condition 32 |

**The label gate (new code, `per_position_loss_weight: dpace-label`).**
- D-PACE weights slot j by wⱼ = Σₖ≥ⱼ Πᵢ≤ₖ sᵢ with sᵢ = (1 − α) qᵢ + α. With qᵢ = the drafter's probability of the target's token and α → 0, this is exactly the gradient weight of expected accepted length. The test is `tests/unit/models/test_dpace_label_gate.py`.
- The existing code computes q = exp(−loss) from whatever loss it weights, so the KL runs gate on exp(−KL), which is not an acceptance probability.
- `dpace-label` takes q from the drafter's probability of the target's argmax (stop-gradient), whatever the loss term.

**Changes against the plan (logged in RUNLOG):**
- The planned "best-loss generator" was replaced by a rank-128 shared-head generator, because the three losses tied.
- An extra rank-512 shared-head generator used the GPU that freed up at 03:55.

## Results

__TABLES__

### Direct comparisons (paired, 95% CI)

| Comparison | validation | test | OOD |
|---|---|---|---|
| Shared heads (0.23B) − per-module heads (1.1B) | −0.001 [−0.013, +0.010] | −0.010 [−0.022, +0.001] | +0.000 [−0.029, +0.031] |
| Shared heads r64 (0.45B) − per-module heads (1.1B) | −0.006 [−0.018, +0.007] | −0.003 [−0.015, +0.008] | −0.006 [−0.031, +0.021] |
| Static r256 − 1.1B generator | +0.014 [+0.001, +0.027] | −0.000 [−0.013, +0.012] | +0.017 [−0.011, +0.044] |
| Static r512 − 1.1B generator | +0.017 [+0.005, +0.029] | +0.011 [−0.001, +0.023] | +0.025 [−0.003, +0.051] |
| Static r1024 − 1.1B generator | +0.020 [+0.009, +0.031] | +0.014 [+0.002, +0.026] | +0.005 [−0.024, +0.035] |

__EXTRA_DIRECT__

### Against the 3-epoch runs (MIX24K)
- **Static r32:** 2 epochs scores 0.010 τ below 3 epochs on test (CI [+0.002, +0.019]).
- **1.1B generator:** 2 epochs (7.472) ≈ 3 epochs (7.469).
- The 2-epoch static ranking would not change with a third epoch of that size: r64 alone already exceeds it.

## Interpretation and open questions
Interpretation, not measured directly:
- **Rank, not dynamism, explains most of what extra parameters buy here.**
  - PERTURB24K found r512 ≈ r32 after 600 updates. At 2,667 updates, rank pays monotonically up to ≈ r512. Its capacity ceiling was a short-training effect.
  - The generator's effective rank of ΔW across rows is 2–3. It acts as a category router on top of a rank-32 LoRA, and its gain matches roughly r128–r256 of extra static rank.
- **Generator size is not the bottleneck.** 0.23B and 1.1B are indistinguishable, and every generator's LoRAs vary along only 2–3 directions across rows (D13). A generator should be judged against a static LoRA of matched parameter count, not against r32.
- **Natural next arms (not run):**
  - a generator on top of a high-rank static path (static r512 + a small dynamic residual);
  - generators emitting r256–r512 with even narrower codes;
  - the label gate on high-rank statics (its OOD gain stacked on rank);
  - seeds for the top three arms.
- **For the 100–150K run:** static r512 is the strongest arm per parameter. If a generator is trained, its matched control should be static r512, not r32.

## Limits
- One seed per arm; the smaller differences (±0.01 τ) are within typical seed drift.
- Offline, greedy, teacher-forced τ on stored responses; no serving measurement. High-rank statics also cost more to apply per token than r32, which is unmeasured.
- ms / LoRA was measured while the other GPU trained; the 1.1B generator's 7.6 ms vs 4.2 ms in MIX24K shows the noise.
- Training loss values are not comparable across loss arms (each gate weights differently); compare τ.
- The base drafter likely saw prompts from the same Nemotron V2 source (see the plan's §2).

## Artifacts
- **Recipes:** `examples/dflash-lora/abl24k/`; matrices `examples/dflash-lora/matrices/abl24k_*.yaml`; lanes `scripts/dflash_port/run_lane.sh`.
- **Runs:** `/workspace/yusuf/runs/dflash/abl24k/<run>/`. Provenance per run: `pretraining/<run>/train_command.txt`, `speculators.patch`, and `run_record_*.json` (git SHA, command, configuration digest).
- **HF (public):** `erdem-erdem/dyn-spec-dflash-abl24k-*` (__HF_LIST__).
- **W&B:** project `dyn-spec-dflash-v1`, runs `dflash-abl24k-*-gseed42`.
- **Results JSON:** `scripts/dflash_port/abl24k_collect.py`; tables: `scripts/dflash_port/abl24k_page/tables.py`; page: `scripts/dflash_port/abl24k_page/build.py`.
