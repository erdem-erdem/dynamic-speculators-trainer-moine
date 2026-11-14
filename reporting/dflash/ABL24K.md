# ABL24K: loss, LoRA rank and generator size on 24K mixed data (2 epochs)

- **Status:** done (2026-11-14), plus the hybrid test of the same day. One seed per arm.
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
  | Shared heads, decoder 32, r512 | 0.46B | +0.024 | [+0.008, +0.040] |
  | Hybrid: per-layer static r512 + shared heads, decoder 32 (2026-11-14) | 0.51B | +0.021 | [+0.004, +0.038] |

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

| Arm | trainable | train min | val τ | test τ | OOD τ | test Δτ vs static r32 | val Δτ | OOD Δτ |
|---|---|---|---|---|---|---|---|---|
| base drafter | 0 | – | 7.225 | 7.251 | 5.936 | – | – | – |
| static r32, KL | 4.3M | 111 | 7.398 | 7.436 | 6.139 | ref | ref | ref |
| static r64, KL | 8.5M | 104 | 7.405 | 7.454 | 6.154 | +0.017 [+0.010, +0.026] | +0.007 [+0.000, +0.014] | +0.015 [+0.000, +0.032] |
| static r128, KL | 17.0M | 99 | 7.420 | 7.463 | 6.167 | +0.027 [+0.017, +0.037] | +0.022 [+0.013, +0.030] | +0.028 [+0.006, +0.050] |
| static r256, KL | 34.1M | 99 | 7.437 | 7.471 | 6.168 | +0.035 [+0.022, +0.048] | +0.039 [+0.028, +0.050] | +0.029 [+0.004, +0.054] |
| static r512, KL | 68.2M | 105 | 7.441 | 7.482 | 6.175 | +0.046 [+0.032, +0.061] | +0.043 [+0.030, +0.055] | +0.036 [+0.008, +0.066] |
| static r1024, KL | 136.3M | 108 | 7.443 | 7.486 | 6.156 | +0.050 [+0.034, +0.066] | +0.045 [+0.032, +0.058] | +0.016 [-0.015, +0.048] |
| static r32, CE | 4.3M | 103 | 7.397 | 7.443 | 6.163 | +0.006 [-0.005, +0.018] | -0.001 [-0.013, +0.012] | +0.024 [-0.008, +0.057] |
| static r32, KL + label gate α 0.1 | 4.3M | 101 | 7.399 | 7.433 | 6.167 | -0.003 [-0.015, +0.008] | +0.000 [-0.010, +0.011] | +0.028 [+0.005, +0.051] |
| generator per-module heads, r32 (1.1B) | 1.11B | 118 | 7.423 | 7.472 | 6.151 | +0.035 [+0.019, +0.052] | +0.025 [+0.012, +0.039] | +0.012 [-0.019, +0.041] |
| generator shared heads, r32 | 229.8M | 103 | 7.422 | 7.462 | 6.151 | +0.026 [+0.009, +0.042] | +0.024 [+0.010, +0.039] | +0.012 [-0.020, +0.044] |
| generator shared heads, r64 | 448.8M | 110 | 7.418 | 7.468 | 6.145 | +0.032 [+0.017, +0.048] | +0.020 [+0.005, +0.036] | +0.006 [-0.029, +0.042] |
| generator shared heads, r128, decoder 128 | 450.3M | 110 | 7.421 | 7.462 | 6.148 | +0.026 [+0.009, +0.042] | +0.022 [+0.007, +0.038] | +0.009 [-0.025, +0.043] |
| generator shared heads, r512, decoder 32 | 460.4M | 110 | 7.408 | 7.460 | 6.152 | +0.024 [+0.008, +0.040] | +0.010 [-0.005, +0.026] | +0.013 [-0.025, +0.049] |
| hybrid: per-layer static r512 + shared heads (decoder 32) | 515.0M | 113 | 7.410 | 7.457 | 6.139 | +0.021 [+0.004, +0.038] | +0.012 [-0.005, +0.029] | -0.000 [-0.039, +0.039] |
| static r32, KL, 3 epochs (MIX24K) | 4.3M | 162 | 7.405 | 7.447 | 6.144 | +0.010 [+0.002, +0.019] | +0.007 [-0.002, +0.015] | +0.005 [-0.011, +0.022] |
| generator 1.1B, KL, 3 epochs (MIX24K) | 1.11B | 168 | 7.426 | 7.469 | 6.148 | +0.032 [+0.014, +0.052] | +0.028 [+0.012, +0.044] | +0.008 [-0.030, +0.047] |

Test τ per category (Δ vs static r32 in parentheses):

| Arm | math | code | stem | chat |
|---|---|---|---|---|
| base drafter | 10.022 | 7.590 | 6.728 | 4.666 |
| static r32, KL | 10.216 (+0.000) | 7.778 (+0.000) | 6.922 (+0.000) | 4.829 (+0.000) |
| static r64, KL | 10.227 (+0.011) | 7.794 (+0.015) | 6.950 (+0.028) | 4.844 (+0.015) |
| static r128, KL | 10.244 (+0.028) | 7.792 (+0.014) | 6.960 (+0.038) | 4.858 (+0.029) |
| static r256, KL | 10.257 (+0.041) | 7.808 (+0.030) | 6.958 (+0.036) | 4.861 (+0.032) |
| static r512, KL | 10.274 (+0.058) | 7.823 (+0.045) | 6.972 (+0.050) | 4.860 (+0.031) |
| static r1024, KL | 10.275 (+0.059) | 7.821 (+0.043) | 6.985 (+0.063) | 4.863 (+0.034) |
| static r32, CE | 10.210 (-0.006) | 7.761 (-0.018) | 6.952 (+0.029) | 4.849 (+0.020) |
| static r32, KL + label gate α 0.1 | 10.196 (-0.020) | 7.764 (-0.015) | 6.939 (+0.016) | 4.835 (+0.006) |
| generator per-module heads, r32 (1.1B) | 10.245 (+0.029) | 7.803 (+0.025) | 6.994 (+0.071) | 4.845 (+0.016) |
| generator shared heads, r32 | 10.239 (+0.023) | 7.802 (+0.023) | 6.968 (+0.046) | 4.839 (+0.010) |
| generator shared heads, r64 | 10.252 (+0.036) | 7.803 (+0.025) | 6.964 (+0.042) | 4.855 (+0.026) |
| generator shared heads, r128, decoder 128 | 10.237 (+0.021) | 7.805 (+0.027) | 6.979 (+0.056) | 4.827 (-0.001) |
| generator shared heads, r512, decoder 32 | 10.238 (+0.022) | 7.794 (+0.015) | 6.968 (+0.045) | 4.843 (+0.014) |
| hybrid: per-layer static r512 + shared heads (decoder 32) | 10.238 (+0.022) | 7.789 (+0.011) | 6.965 (+0.042) | 4.837 (+0.009) |
| static r32, KL, 3 epochs (MIX24K) | 10.223 (+0.007) | 7.786 (+0.008) | 6.944 (+0.021) | 4.835 (+0.006) |
| generator 1.1B, KL, 3 epochs (MIX24K) | 10.242 (+0.026) | 7.812 (+0.034) | 6.970 (+0.047) | 4.852 (+0.023) |

Generators (D13 similarity on 500 validation rows, D14 cost):

| Arm | params | cross-row cosine | within-category cosine | dyn_ratio | effective rank | η² | centroid acc | ms / LoRA |
|---|---|---|---|---|---|---|---|---|
| generator per-module heads, r32 (1.1B) | 1.11B | 0.894 | 0.987 | 0.081 | 2.14 | 0.80 | 0.970 | 7.58 |
| generator shared heads, r32 | 229.8M | 0.862 | 0.981 | 0.107 | 2.97 | 0.78 | 0.964 | 3.99 |
| generator shared heads, r64 | 448.8M | 0.908 | 0.987 | 0.072 | 2.64 | 0.78 | 0.970 | 6.97 |
| generator shared heads, r128, decoder 128 | 450.3M | 0.970 | 0.996 | 0.024 | 2.00 | 0.76 | 0.962 | 4.23 |
| generator shared heads, r512, decoder 32 | 460.4M | 0.984 | 0.998 | 0.014 | 2.27 | 0.73 | 0.942 | 4.17 |
| hybrid: per-layer static r512 + shared heads (decoder 32) | 515.0M | 0.988 | 0.998 | 0.009 | 1.77 | 0.70 | 0.894 | 5.13 |
| generator 1.1B, KL, 3 epochs (MIX24K) | 1.11B | 0.795 | 0.972 | 0.155 | 2.48 | 0.78 | 0.972 | 4.15 |

LoRA coefficient at inference (validation full_acc; training scale 2):

| Arm | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| static r32, KL | 0.5296 | 0.5320 | **0.5335** | 0.5263 | 0.4818 |
| static r64, KL | 0.5299 | 0.5323 | **0.5340** | 0.5273 | 0.4865 |
| static r128, KL | 0.5303 | 0.5327 | **0.5346** | 0.5278 | 0.4871 |
| static r256, KL | 0.5307 | 0.5335 | **0.5351** | 0.5276 | 0.4835 |
| static r512, KL | 0.5309 | 0.5338 | **0.5353** | 0.5269 | 0.4762 |
| static r1024, KL | 0.5312 | 0.5339 | **0.5356** | 0.5250 | 0.4634 |
| static r32, CE | 0.5299 | 0.5319 | **0.5330** | 0.5230 | 0.3860 |
| static r32, KL + label gate α 0.1 | 0.5298 | **0.5315** | 0.5296 | 0.5090 | 0.4063 |
| generator per-module heads, r32 (1.1B) | 0.5308 | 0.5336 | **0.5347** | 0.5232 | 0.4579 |
| generator shared heads, r32 | 0.5307 | 0.5339 | **0.5345** | 0.5229 | 0.4590 |
| generator shared heads, r64 | 0.5312 | 0.5338 | **0.5347** | 0.5223 | 0.4500 |
| generator shared heads, r128, decoder 128 | 0.5311 | 0.5341 | **0.5350** | 0.5226 | 0.4481 |
| generator shared heads, r512, decoder 32 | 0.5314 | 0.5342 | **0.5348** | 0.5198 | 0.4334 |
| hybrid: per-layer static r512 + shared heads (decoder 32) | 0.5314 | 0.5342 | **0.5347** | 0.5188 | 0.4279 |
| static r32, KL, 3 epochs (MIX24K) | 0.5299 | 0.5323 | **0.5338** | 0.5264 | 0.4767 |
| generator 1.1B, KL, 3 epochs (MIX24K) | 0.5314 | 0.5343 | **0.5347** | 0.5189 | 0.4297 |

### Direct comparisons (paired, 95% CI)

| Comparison | validation | test | OOD |
|---|---|---|---|
| Shared heads (0.23B) − per-module heads (1.1B) | −0.001 [−0.013, +0.010] | −0.010 [−0.022, +0.001] | +0.000 [−0.029, +0.031] |
| Shared heads r64 (0.45B) − per-module heads (1.1B) | −0.006 [−0.018, +0.007] | −0.003 [−0.015, +0.008] | −0.006 [−0.031, +0.021] |
| Static r256 − 1.1B generator | +0.014 [+0.001, +0.027] | −0.000 [−0.013, +0.012] | +0.017 [−0.011, +0.044] |
| Static r512 − 1.1B generator | +0.017 [+0.005, +0.029] | +0.011 [−0.001, +0.023] | +0.025 [−0.003, +0.051] |
| Static r1024 − 1.1B generator | +0.020 [+0.009, +0.031] | +0.014 [+0.002, +0.026] | +0.005 [−0.024, +0.035] |

**Hybrid test (user 2026-11-14):** a per-layer static r512 path plus shared heads that add the prompt-dependent part (`head_sharing: per_type_bias`). Its only difference from the r512 shared-head generator is the per-module static part. Paired vs static r512:

| Arm − static r512 | validation | test | OOD |
|---|---|---|---|
| Hybrid (0.51B) | −0.031 [−0.046, −0.017] | −0.025 [−0.039, −0.012] | −0.037 [−0.070, −0.002] |
| Shared-head r512 generator (0.46B) | −0.033 [−0.046, −0.019] | −0.022 [−0.035, −0.009] | −0.024 [−0.058, +0.008] |

- **The hybrid does not pass.** Its static path learns much less than a standalone static r512 trained with the same rank, data and learning rate.
- Its LoRAs are nearly constant across rows: cross-row cosine 0.988, dyn_ratio 0.009.
- *Hypothesis (unverified):* the head weights make the generated factors move with a much larger effective step than a static LoRA's. Check: a lower learning rate on the head weights than on the per-module static factors, or a warm start from the trained static r512.

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
- **HF (public):** `erdem-erdem/dyn-spec-dflash-abl24k-*` (prefix `dyn-spec-dflash-`): `abl24k-generator-attn-r128-2ep-kl-pt128` @ a3084add, `abl24k-generator-attn-r32-2ep-kl-pm256` @ e55ad756, `abl24k-generator-attn-r32-2ep-kl-pt256` @ d47aaa3f, `abl24k-generator-attn-r512-2ep-kl-hybrid32` @ 989cb612, `abl24k-generator-attn-r512-2ep-kl-pt32` @ 773a2dd3, `abl24k-generator-attn-r64-2ep-kl-pt256` @ 1224bcd3, `abl24k-static-lora-attn-r1024-2ep-kl` @ 72cd3213, `abl24k-static-lora-attn-r128-2ep-kl` @ 2d3d3f76, `abl24k-static-lora-attn-r256-2ep-kl` @ cacd18fa, `abl24k-static-lora-attn-r32-2ep-ce` @ 35b8c8f2, `abl24k-static-lora-attn-r32-2ep-kl` @ 64bbbb5c, `abl24k-static-lora-attn-r32-2ep-kllabel` @ efc320cb, `abl24k-static-lora-attn-r512-2ep-kl` @ b57a9150, `abl24k-static-lora-attn-r64-2ep-kl` @ 8797d11e.
- **W&B:** project `dyn-spec-dflash-v1`, runs `dflash-abl24k-*-gseed42`.
- **Results JSON:** `scripts/dflash_port/abl24k_collect.py`; tables: `scripts/dflash_port/abl24k_page/tables.py`; page: `scripts/dflash_port/abl24k_page/build.py`.
