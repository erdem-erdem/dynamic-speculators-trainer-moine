# 10K-math: static LoRA vs small generator

- **Status:** done (2026-11-10)
- **Requested by the user (2026-11-10):** train a static LoRA and a small generator on 10K math rows for 3 epochs, then evaluate on enlarged sets: math and the other categories (in distribution), plus out of distribution.

## TL;DR

- **Both trained arms beat the base drafter on every split and category, and the two arms are tied.** On the 500-row test set:
  - Base drafter: 721 accepted draft tokens per 100 blocks (all blocks pooled; each block drafts 15).
  - Static LoRA: 737 per 100 blocks (+0.140 τ category mean, 95% CI [+0.12, +0.16]).
  - Generator: 738 per 100 blocks (+0.138 τ).
  - Generator vs static: −0.002 τ, 95% CI [−0.017, +0.012]. No difference.
- **Math, the trained domain, gains most:** +0.20 τ for static and +0.22 τ for the generator, about +2 accepted draft tokens per 10 blocks. Code, stem and chat gain +0.06 to +0.15 τ.
- **Out of distribution, static is better than the generator:**
  - Static +0.205 τ over base; generator +0.154 τ.
  - Generator − static: −0.051 τ, CI [−0.092, −0.010].
- **The generator no longer overfits** (3 epochs here vs 6.75 on 4K), **but it collapsed to an almost fixed LoRA.**
  - Its generated ΔW are nearly identical across rows: cross-row cosine 0.965, dyn_ratio 0.03, effective rank 1.04. On 4K these were 0.67, 0.27 and 3.6.
  - So it behaves like a static LoRA with 1.1B parameters instead of 4.3M.

## Setup

| | Static LoRA | Generator |
|---|---|---|
| Training data | 10,000 math rows: r0002-100k math-00000..00009 train splits | same |
| Schedule | 1,667 updates × 18 rows = 3 epochs; validation every 250 updates | same |
| Adapter | attention q/k/v/o on all 5 draft layers, rank 32, α 64 (coefficient 2) | same; one LoRA generated per row |
| Parameters | 4,259,840 | 1,105,658,048 (decoder width 256, condition dim 32; per-module heads) |
| Loss | CE + D-PACE (M0) | same |
| Train time (H100) | 84 min | 88 min |
| Recipe | `examples/dflash-lora/math10k/static_lora.yaml` | `.../generator.yaml` |

**Evaluation sets.** Every row is stored in `examples/dflash-lora/math10k/eval_splits.json` and was checked never to occur in training:
- **Validation** (scored during training; used to choose the scale) and **test** (final numbers): 500 rows each, 200 math + 100 code + 100 stem + 100 chat, from different subsets.
- **Eligibility:** a row qualifies only if its primary id, prompt tokens and full tokens occur in no train split of any local release (124,000 rows, r0001 + r0002-100k). The data repeats prompts across rows, so subset membership alone would have leaked; 305 candidates were excluded.
- **OOD v3:** HumanEval, math_reasoning and summarization, 80 each = 240 (`erdem-erdem/dyn-spec-dflash-ood-v3`). None shares a prompt with training.
- **No donor control** (by choice).

**Metrics.** Offline, teacher-forced, greedy:
- **τ:** accepted length per 16-token block, including the verifier's bonus token.
- **Accepted draft tokens per block:** τ − 1, out of 15 drafted.
- **Acceptance rate:** (τ − 1) / 15.
- **Category pooling:** within a category, values are pooled over blocks. "all" is the category mean, matching earlier reports. Totals count every block of the split.
- **CIs:** paired 95% bootstrap over rows, stratified by category, 5,000 resamples (`scripts/dflash_port/compare_arms.py`).

## Results

### Test (500 rows; final numbers)

**test** (500 rows)

| Arm | | math | code | stem | chat | all (category mean) |
|---|---|---|---|---|---|---|
| base | full_acc | 0.7029 | 0.5552 | 0.4981 | 0.3708 | 0.5317 |
| | τ | 10.022 | 7.590 | 6.728 | 4.666 | 7.251 |
| | accepted draft tokens / block | 9.02 of 15 | 6.59 of 15 | 5.73 of 15 | 3.67 of 15 | 6.25 of 15 |
| | acceptance rate | 60.1% | 43.9% | 38.2% | 24.4% | 41.7% |
| static | full_acc | 0.7086 | 0.5597 | 0.5023 | 0.3753 | 0.5365 |
| | τ | 10.220 | 7.699 | 6.882 | 4.765 | 7.391 |
| | accepted draft tokens / block | 9.22 of 15 | 6.70 of 15 | 5.88 of 15 | 3.76 of 15 | 6.39 of 15 |
| | acceptance rate | 61.5% | 44.7% | 39.2% | 25.1% | 42.6% |
| generator | full_acc | 0.7088 | 0.5593 | 0.4999 | 0.3748 | 0.5357 |
| | τ | 10.243 | 7.702 | 6.845 | 4.766 | 7.389 |
| | accepted draft tokens / block | 9.24 of 15 | 6.70 of 15 | 5.85 of 15 | 3.77 of 15 | 6.39 of 15 |
| | acceptance rate | 61.6% | 44.7% | 39.0% | 25.1% | 42.6% |


Total accepted draft tokens over the whole test set:

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 49,021 | 353,580 | 721 |
| static | 49,021 | 361,286 | 737 |
| generator | 49,021 | 361,686 | 738 |

Versus base:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| static | all | +0.140 | [+0.120, +0.161] | +0.0047 | [+0.0037, +0.0059] | +14.0 |
| static | math | +0.198 | [+0.171, +0.226] | +0.0057 | [+0.0045, +0.0070] | +19.8 |
| static | code | +0.109 | [+0.073, +0.144] | +0.0045 | [+0.0028, +0.0062] | +10.9 |
| static | stem | +0.154 | [+0.114, +0.194] | +0.0042 | [+0.0022, +0.0063] | +15.4 |
| static | chat | +0.099 | [+0.050, +0.158] | +0.0045 | [+0.0018, +0.0079] | +9.9 |
| generator | all | +0.138 | [+0.116, +0.159] | +0.0040 | [+0.0027, +0.0053] | +13.8 |
| generator | math | +0.221 | [+0.193, +0.251] | +0.0059 | [+0.0044, +0.0074] | +22.1 |
| generator | code | +0.112 | [+0.070, +0.151] | +0.0041 | [+0.0019, +0.0062] | +11.2 |
| generator | stem | +0.118 | [+0.069, +0.163] | +0.0018 | [-0.0010, +0.0046] | +11.8 |
| generator | chat | +0.100 | [+0.052, +0.154] | +0.0041 | [+0.0011, +0.0078] | +10.0 |

Generator versus static:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| generator | all | -0.002 | [-0.017, +0.012] | -0.0008 | [-0.0015, -0.0000] | -0.2 |
| generator | math | +0.023 | [+0.000, +0.047] | +0.0001 | [-0.0008, +0.0011] | +2.3 |
| generator | code | +0.003 | [-0.027, +0.033] | -0.0004 | [-0.0019, +0.0009] | +0.3 |
| generator | stem | -0.036 | [-0.072, +0.000] | -0.0024 | [-0.0043, -0.0006] | -3.6 |
| generator | chat | +0.001 | [-0.024, +0.027] | -0.0004 | [-0.0019, +0.0010] | +0.1 |

### Validation (500 rows)

**validation** (500 rows)

| Arm | | math | code | stem | chat | all (category mean) |
|---|---|---|---|---|---|---|
| base | full_acc | 0.6777 | 0.5953 | 0.4863 | 0.3475 | 0.5267 |
| | τ | 9.626 | 8.373 | 6.589 | 4.312 | 7.225 |
| | accepted draft tokens / block | 8.63 of 15 | 7.37 of 15 | 5.59 of 15 | 3.31 of 15 | 6.23 of 15 |
| | acceptance rate | 57.5% | 49.2% | 37.3% | 22.1% | 41.5% |
| static | full_acc | 0.6845 | 0.6016 | 0.4899 | 0.3499 | 0.5315 |
| | τ | 9.846 | 8.524 | 6.686 | 4.387 | 7.361 |
| | accepted draft tokens / block | 8.85 of 15 | 7.52 of 15 | 5.69 of 15 | 3.39 of 15 | 6.36 of 15 |
| | acceptance rate | 59.0% | 50.2% | 37.9% | 22.6% | 42.4% |
| generator | full_acc | 0.6842 | 0.6019 | 0.4881 | 0.3486 | 0.5307 |
| | τ | 9.866 | 8.516 | 6.664 | 4.375 | 7.355 |
| | accepted draft tokens / block | 8.87 of 15 | 7.52 of 15 | 5.66 of 15 | 3.37 of 15 | 6.36 of 15 |
| | acceptance rate | 59.1% | 50.1% | 37.8% | 22.5% | 42.4% |


| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 45,566 | 324,187 | 711 |
| static | 45,566 | 331,716 | 728 |
| generator | 45,566 | 331,867 | 728 |

Versus base:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| static | all | +0.136 | [+0.117, +0.154] | +0.0048 | [+0.0039, +0.0057] | +13.6 |
| static | math | +0.219 | [+0.189, +0.248] | +0.0068 | [+0.0055, +0.0081] | +21.9 |
| static | code | +0.151 | [+0.107, +0.195] | +0.0063 | [+0.0044, +0.0082] | +15.1 |
| static | stem | +0.097 | [+0.059, +0.134] | +0.0036 | [+0.0014, +0.0058] | +9.7 |
| static | chat | +0.075 | [+0.037, +0.112] | +0.0025 | [+0.0006, +0.0043] | +7.5 |
| generator | all | +0.130 | [+0.108, +0.152] | +0.0040 | [+0.0029, +0.0051] | +13.0 |
| generator | math | +0.240 | [+0.204, +0.274] | +0.0065 | [+0.0049, +0.0081] | +24.0 |
| generator | code | +0.143 | [+0.097, +0.188] | +0.0066 | [+0.0045, +0.0088] | +14.3 |
| generator | stem | +0.075 | [+0.020, +0.130] | +0.0018 | [-0.0010, +0.0046] | +7.5 |
| generator | chat | +0.063 | [+0.024, +0.103] | +0.0011 | [-0.0007, +0.0029] | +6.3 |

Generator versus static:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| generator | all | -0.005 | [-0.022, +0.011] | -0.0008 | [-0.0015, -0.0000] | -0.5 |
| generator | math | +0.020 | [-0.005, +0.046] | -0.0003 | [-0.0013, +0.0008] | +2.0 |
| generator | code | -0.008 | [-0.038, +0.019] | +0.0003 | [-0.0009, +0.0015] | -0.8 |
| generator | stem | -0.022 | [-0.063, +0.021] | -0.0018 | [-0.0036, +0.0001] | -2.2 |
| generator | chat | -0.012 | [-0.046, +0.022] | -0.0013 | [-0.0029, +0.0003] | -1.2 |

### Out of distribution (OOD v3, 240 rows)

**ood** (240 rows)

| Arm | | ood-humaneval | ood-math-reasoning | ood-summarization | all (category mean) |
|---|---|---|---|---|---|
| base | full_acc | 0.6023 | 0.5504 | 0.2007 | 0.4511 |
| | τ | 8.095 | 7.039 | 2.676 | 5.936 |
| | accepted draft tokens / block | 7.10 of 15 | 6.04 of 15 | 1.68 of 15 | 4.94 of 15 |
| | acceptance rate | 47.3% | 40.3% | 11.2% | 32.9% |
| static | full_acc | 0.6125 | 0.5604 | 0.2037 | 0.4589 |
| | τ | 8.353 | 7.332 | 2.739 | 6.141 |
| | accepted draft tokens / block | 7.35 of 15 | 6.33 of 15 | 1.74 of 15 | 5.14 of 15 |
| | acceptance rate | 49.0% | 42.2% | 11.6% | 34.3% |
| generator | full_acc | 0.6093 | 0.5557 | 0.2037 | 0.4562 |
| | τ | 8.271 | 7.255 | 2.747 | 6.091 |
| | accepted draft tokens / block | 7.27 of 15 | 6.26 of 15 | 1.75 of 15 | 5.09 of 15 |
| | acceptance rate | 48.5% | 41.7% | 11.6% | 33.9% |


| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 8,993 | 35,048 | 390 |
| static | 8,993 | 36,462 | 405 |
| generator | 8,993 | 36,179 | 402 |

Versus base:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| static | all | +0.205 | [+0.163, +0.246] | +0.0077 | [+0.0057, +0.0098] | +20.5 |
| static | ood-humaneval | +0.258 | [+0.179, +0.337] | +0.0102 | [+0.0066, +0.0141] | +25.8 |
| static | ood-math-reasoning | +0.293 | [+0.206, +0.382] | +0.0100 | [+0.0054, +0.0147] | +29.3 |
| static | ood-summarization | +0.064 | [+0.044, +0.086] | +0.0030 | [+0.0016, +0.0044] | +6.4 |
| generator | all | +0.154 | [+0.105, +0.203] | +0.0051 | [+0.0027, +0.0076] | +15.4 |
| generator | ood-humaneval | +0.175 | [+0.078, +0.275] | +0.0070 | [+0.0023, +0.0117] | +17.5 |
| generator | ood-math-reasoning | +0.216 | [+0.109, +0.327] | +0.0054 | [+0.0000, +0.0107] | +21.6 |
| generator | ood-summarization | +0.072 | [+0.047, +0.097] | +0.0030 | [+0.0008, +0.0051] | +7.2 |

Generator versus static:

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| generator | all | -0.051 | [-0.092, -0.010] | -0.0026 | [-0.0044, -0.0007] | -5.1 |
| generator | ood-humaneval | -0.083 | [-0.158, -0.006] | -0.0033 | [-0.0067, +0.0003] | -8.3 |
| generator | ood-math-reasoning | -0.077 | [-0.173, +0.012] | -0.0046 | [-0.0087, -0.0007] | -7.7 |
| generator | ood-summarization | +0.008 | [-0.012, +0.027] | +0.0000 | [-0.0019, +0.0019] | +0.8 |

### Training curves (validation, 500 rows; full_acc / τ averaged over subsets as logged)

| Update | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1667 |
|---|---|---|---|---|---|---|---|---|
| Static | 0.5265 / 7.703 | 0.5302 / 7.826 | 0.5308 / 7.838 | 0.5312 / 7.847 | 0.5311 / 7.849 | 0.5312 / 7.850 | 0.5314 / 7.850 | 0.5313 / 7.856 |
| Generator | 0.5265 / 7.703 | 0.5301 / 7.823 | 0.5312 / 7.853 | 0.5310 / 7.863 | 0.5313 / 7.869 | 0.5310 / 7.866 | 0.5306 / 7.850 | 0.5305 / 7.855 |

Both curves rise by update 500 and then stay flat. The generator peaks around update 1000 and gives back about 0.01 τ by the end. On 4K it fell below base; here it does not.

### LoRA coefficient (validation)

| Coefficient | 0.5 | 1 | 2 (training scale) | 4 | 8 |
|---|---|---|---|---|---|
| Static full_acc / τ | 0.5292 / 7.772 | 0.5307 / 7.818 | **0.5313 / 7.856** | 0.5233 / 7.704 | 0.3976 / 4.513 |
| Generator full_acc / τ | 0.5299 / 7.793 | 0.5317 / 7.842 | 0.5305 / **7.855** | 0.5140 / 7.574 | 0.4005 / 5.093 |

(τ averaged over subsets, as logged.) Both arms reach their highest τ at the training scale. The generator's full_acc is slightly higher at coefficient 1, but its τ is lower there (paired vs static at coefficient 2: −0.015 τ, CI [−0.029, −0.001]). Its test and OOD numbers therefore use the training scale, chosen on validation.

### What the generator learned (D13, D14)

| | 10K math | 4K (M0, for reference) |
|---|---|---|
| cross-row cosine of generated ΔW | 0.965 | 0.667 |
| dyn_ratio (row-specific share) | 0.032 | 0.268 |
| effective rank of the ΔW set | 1.04 | 3.60 |
| category centroid accuracy | 0.644 | 0.965 |

The generated LoRAs are almost the same for every row: one direction, little row or category information. The generator found a near-constant ΔW, which explains why it ties the static LoRA in distribution. It is slightly worse OOD, possibly because the small row-dependent part it does produce doesn't transfer. Cost: 4.36 ms per generated LoRA, merge 0.67 ms, 1.1B parameters, vs none for the static LoRA.

## Interpretation and limits

- On 10K math, a per-prompt generator gives no acceptance gain over a static LoRA in distribution, and loses some OOD. It also learns almost no per-prompt variation. Three epochs on 10K removed the overfitting seen at 4K, but did not produce useful dynamism.
- The static LoRA's gain over base is small but robust: every category's CI excludes zero on test, and math gains most. That is about 1.6 more accepted draft tokens per 10 blocks on test, from 721 to 737 per 100 blocks.
- **Limits:**
  - One seed per arm.
  - Offline, greedy, teacher-forced metrics; no vLLM serving measurement yet.
  - The OOD set is 240 rows with responses capped at 256 tokens.
  - Validation was used only to check the coefficient; no checkpoint selection (both arms use the final checkpoint).

## Artifacts

- Runs: `/workspace/yusuf/runs/dflash/math10k/{math10k-base, math10k-static-lora-attn-r32-3ep, math10k-generator-attn-r32-3ep}`.
- W&B project `dyn-spec-dflash-v1`; run names `dflash-math10k-*-gseed42`.
- HF models: `erdem-erdem/dyn-spec-dflash-math10k-static-lora-attn-r32-3ep` (@ 7ab9ffc1) and `erdem-erdem/dyn-spec-dflash-math10k-generator-attn-r32-3ep`.
- HF store: `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-math10k` (upload in progress). HF OOD set: `erdem-erdem/dyn-spec-dflash-ood-v3` @ bcba573c.
- Code: 80f9ad2 (pools, stored eval rows), 08935cb (summaries for pool jobs), f249588 (`compare_arms.py`). RUNLOG entries of 2026-11-10.
