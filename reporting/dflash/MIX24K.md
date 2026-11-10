# 24K mixed: static LoRA vs generator, both KL

- **Status:** done (2026-11-10)
- **Requested by the user (2026-11-10):** a static LoRA and the generator trained on 24K rows, 6K from each of math / code / stem / chat, **both with the KL loss**, evaluated on the 10K-math sets. The full pipeline: store fill and HF upload, trainings (W&B), evals, HF models, report.
- **Previous:** [MATH10K.md](MATH10K.md) (10K math, CE), [MATH10K_LOSSES.md](MATH10K_LOSSES.md) (10K math, lk_hybrid / KL generators).

## TL;DR

- **With the same loss, the generator beats the static LoRA in distribution, by a small margin. It is the first matched comparison where it does.**
  - Test (500 rows): Δτ +0.022, 95% CI [+0.004, +0.040]; accepted draft tokens 743 vs 741 per 100 blocks (base 721).
  - Validation (500 rows) agrees: +0.021, CI [+0.006, +0.037].
  - Every category is positive (+0.017 to +0.026 τ on test), but no single category's interval excludes zero. One seed per arm.
- **OOD, they tie:** Δτ +0.003, CI [−0.034, +0.041] (static and generator both 407 accepted draft tokens per 100 blocks; base 390). At 10K the generators lost OOD.
- **The generator is dynamic again.** Its LoRAs vary by row and separate by category: cross-row cosine 0.795 (10K-math KL: 0.989), dyn_ratio 0.155 (0.014), effective rank 2.48 (1.02), category centroid accuracy 0.972. Mixed domains give it something to route on; one domain did not.
- **More, mixed data helps both arms over every 10K-math arm, mostly outside math.** 24K static vs 10K-math static (CE): +0.055 τ on test (CI [+0.037, +0.076]); math itself is unchanged. Both arms gain over base in every category (+0.17 to +0.22 τ for static).
- **Neither arm overfits:** validation rises to the last checkpoint (4,000 updates, 3 epochs).
- **The gap is small for the cost:** 1.1B generator parameters and ≈ 4.2 ms per generated LoRA (+0.7 ms merge) buy about +2.2 accepted draft tokens per 100 blocks (≈ 0.3% τ).

## Setup

| | Static LoRA | Generator |
|---|---|---|
| Training data | 24,000 rows: r0002-100k train splits of subsets 00000..00005 in each of math, code, stem, chat (6,000 per category) | same |
| Schedule | 4,000 updates × 18 rows = 3 epochs; validation every 250 updates | same |
| Loss | KL to the verifier distribution (`kl_div`) + D-PACE position weights | same |
| Adapter | attention q/k/v/o on all 5 draft layers, rank 32, α 64 (coefficient 2) | same; one LoRA generated per row |
| Parameters | 4,259,840 | 1,105,658,048 (decoder width 256, condition dim 32, per-module heads) |
| Train time (H100, both GPUs busy) | 172 min | 183 min |
| Recipe | `examples/dflash-lora/mix24k/static_lora.yaml` | `.../mix24k/generator.yaml` |
| Run | `mix24k-static-lora-attn-r32-3ep-kl` | `mix24k-generator-attn-r32-3ep-kl` |

- **Evaluation sets:** the 10K-math stored sets (`examples/dflash-lora/math10k/eval_splits.json`): validation and test 500 rows each (200 math + 100 code / stem / chat, from different subsets), plus OOD v3 (HumanEval, math_reasoning, summarization × 80 = 240). Every eval row's prompt and sequence occur in no train split of any local release; the eval subsets (math-00010.., code-00012.., stem/chat-00010..) are also disjoint from the training subsets. The base drafter's numbers are the math10k tier's base run (same rows, same drafter).
- **Activation store:** `output/dflash-lora/activation-store-mix24k`, 25,000 rows + OOD v3; online ≡ offline on 32 rows (full hidden states max |diff| 0.0). Published as `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-mix24k` @ d638d564.
- **Metrics:** offline, teacher-forced, greedy. τ is the accepted length per 16-token block including the verifier's bonus token; accepted draft tokens per block = τ − 1 of 15; acceptance rate = (τ − 1) / 15. Within a category values are pooled over blocks; "all" is the category mean; totals count every block of the split. CIs: paired 95% bootstrap over rows, stratified by category, 5,000 resamples (`scripts/dflash_port/compare_arms.py --reference static`).
- **Why both arms use KL:** at 10K the KL / lk_hybrid generators edged a CE static but their LoRAs were nearly constant, so loss and adapter type were confounded. With one loss for both arms, the difference here is the adapter type alone.

## Results

### Test (500 rows; final numbers)

| Arm | | math | code | stem | chat | all (category mean) |
|---|---|---|---|---|---|---|
| base | full_acc | 0.7029 | 0.5552 | 0.4981 | 0.3708 | 0.5317 |
| | τ | 10.022 | 7.590 | 6.728 | 4.666 | 7.251 |
| | accepted draft tokens / block | 9.02 of 15 | 6.59 of 15 | 5.73 of 15 | 3.67 of 15 | 6.25 of 15 |
| | acceptance rate | 60.1% | 43.9% | 38.2% | 24.4% | 41.7% |
| static | full_acc | 0.7086 | 0.5640 | 0.5054 | 0.3778 | 0.5390 |
| | τ | 10.223 | 7.786 | 6.944 | 4.835 | 7.447 |
| | accepted draft tokens / block | 9.22 of 15 | 6.79 of 15 | 5.94 of 15 | 3.83 of 15 | 6.45 of 15 |
| | acceptance rate | 61.5% | 45.2% | 39.6% | 25.6% | 43.0% |
| generator | full_acc | 0.7089 | 0.5641 | 0.5068 | 0.3795 | 0.5398 |
| | τ | 10.242 | 7.812 | 6.970 | 4.852 | 7.469 |
| | accepted draft tokens / block | 9.24 of 15 | 6.81 of 15 | 5.97 of 15 | 3.85 of 15 | 6.47 of 15 |
| | acceptance rate | 61.6% | 45.4% | 39.8% | 25.7% | 43.1% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 49,021 | 353,580 | 721 |
| static | 49,021 | 363,202 | 741 |
| generator | 49,021 | 364,225 | 743 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.196 | [-0.226, -0.170] | -0.0072 | [-0.0092, -0.0056] | -19.6 |
| base | math | -0.201 | [-0.235, -0.170] | -0.0057 | [-0.0076, -0.0041] | -20.1 |
| base | code | -0.196 | [-0.237, -0.157] | -0.0088 | [-0.0116, -0.0064] | -19.6 |
| base | stem | -0.216 | [-0.251, -0.180] | -0.0073 | [-0.0099, -0.0046] | -21.6 |
| base | chat | -0.169 | [-0.279, -0.095] | -0.0070 | [-0.0139, -0.0024] | -16.9 |
| generator | all | +0.022 | [+0.004, +0.040] | +0.0009 | [-0.0001, +0.0018] | +2.2 |
| generator | math | +0.019 | [-0.003, +0.041] | +0.0004 | [-0.0006, +0.0013] | +1.9 |
| generator | code | +0.026 | [-0.012, +0.063] | +0.0001 | [-0.0014, +0.0015] | +2.6 |
| generator | stem | +0.026 | [-0.014, +0.065] | +0.0014 | [-0.0006, +0.0034] | +2.6 |
| generator | chat | +0.017 | [-0.019, +0.056] | +0.0017 | [-0.0006, +0.0046] | +1.7 |

### Validation (500 rows)

| Arm | | math | code | stem | chat | all (category mean) |
|---|---|---|---|---|---|---|
| base | full_acc | 0.6777 | 0.5953 | 0.4863 | 0.3475 | 0.5267 |
| | τ | 9.626 | 8.373 | 6.589 | 4.312 | 7.225 |
| | accepted draft tokens / block | 8.63 of 15 | 7.37 of 15 | 5.59 of 15 | 3.31 of 15 | 6.23 of 15 |
| | acceptance rate | 57.5% | 49.2% | 37.3% | 22.1% | 41.5% |
| static | full_acc | 0.6844 | 0.6054 | 0.4934 | 0.3526 | 0.5339 |
| | τ | 9.823 | 8.586 | 6.781 | 4.428 | 7.405 |
| | accepted draft tokens / block | 8.82 of 15 | 7.59 of 15 | 5.78 of 15 | 3.43 of 15 | 6.40 of 15 |
| | acceptance rate | 58.8% | 50.6% | 38.5% | 22.9% | 42.7% |
| generator | full_acc | 0.6847 | 0.6070 | 0.4940 | 0.3536 | 0.5348 |
| | τ | 9.843 | 8.621 | 6.810 | 4.430 | 7.426 |
| | accepted draft tokens / block | 8.84 of 15 | 7.62 of 15 | 5.81 of 15 | 3.43 of 15 | 6.43 of 15 |
| | acceptance rate | 59.0% | 50.8% | 38.7% | 22.9% | 42.8% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 45,566 | 324,187 | 711 |
| static | 45,566 | 332,725 | 730 |
| generator | 45,566 | 333,712 | 732 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.180 | [-0.199, -0.160] | -0.0072 | [-0.0084, -0.0061] | -18.0 |
| base | math | -0.197 | [-0.225, -0.168] | -0.0067 | [-0.0081, -0.0053] | -19.7 |
| base | code | -0.213 | [-0.255, -0.171] | -0.0101 | [-0.0121, -0.0079] | -21.3 |
| base | stem | -0.191 | [-0.233, -0.148] | -0.0071 | [-0.0098, -0.0044] | -19.1 |
| base | chat | -0.117 | [-0.156, -0.078] | -0.0051 | [-0.0075, -0.0028] | -11.7 |
| generator | all | +0.021 | [+0.006, +0.037] | +0.0009 | [+0.0001, +0.0017] | +2.1 |
| generator | math | +0.020 | [-0.002, +0.041] | +0.0003 | [-0.0007, +0.0013] | +2.0 |
| generator | code | +0.035 | [-0.006, +0.075] | +0.0016 | [+0.0001, +0.0033] | +3.5 |
| generator | stem | +0.029 | [-0.004, +0.063] | +0.0006 | [-0.0012, +0.0023] | +2.9 |
| generator | chat | +0.002 | [-0.028, +0.034] | +0.0011 | [-0.0008, +0.0030] | +0.2 |

### Out of distribution (OOD v3, 240 rows)

| Arm | | ood-humaneval | ood-math-reasoning | ood-summarization | all (category mean) |
|---|---|---|---|---|---|
| base | full_acc | 0.6023 | 0.5504 | 0.2007 | 0.4511 |
| | τ | 8.095 | 7.039 | 2.676 | 5.936 |
| | accepted draft tokens / block | 7.10 of 15 | 6.04 of 15 | 1.68 of 15 | 4.94 of 15 |
| | acceptance rate | 47.3% | 40.3% | 11.2% | 32.9% |
| static | full_acc | 0.6155 | 0.5609 | 0.2085 | 0.4616 |
| | τ | 8.373 | 7.277 | 2.783 | 6.144 |
| | accepted draft tokens / block | 7.37 of 15 | 6.28 of 15 | 1.78 of 15 | 5.14 of 15 |
| | acceptance rate | 49.2% | 41.8% | 11.9% | 34.3% |
| generator | full_acc | 0.6129 | 0.5583 | 0.2098 | 0.4603 |
| | τ | 8.337 | 7.324 | 2.781 | 6.148 |
| | accepted draft tokens / block | 7.34 of 15 | 6.32 of 15 | 1.78 of 15 | 5.15 of 15 |
| | acceptance rate | 48.9% | 42.2% | 11.9% | 34.3% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 8,993 | 35,048 | 390 |
| static | 8,993 | 36,631 | 407 |
| generator | 8,993 | 36,621 | 407 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.208 | [-0.245, -0.172] | -0.0105 | [-0.0127, -0.0084] | -20.8 |
| base | ood-humaneval | -0.277 | [-0.359, -0.199] | -0.0132 | [-0.0175, -0.0091] | -27.7 |
| base | ood-math-reasoning | -0.239 | [-0.309, -0.168] | -0.0105 | [-0.0151, -0.0057] | -23.9 |
| base | ood-summarization | -0.107 | [-0.129, -0.087] | -0.0078 | [-0.0094, -0.0062] | -10.7 |
| generator | all | +0.003 | [-0.034, +0.041] | -0.0013 | [-0.0035, +0.0009] | +0.3 |
| generator | ood-humaneval | -0.035 | [-0.112, +0.038] | -0.0026 | [-0.0061, +0.0009] | -3.5 |
| generator | ood-math-reasoning | +0.047 | [-0.034, +0.128] | -0.0025 | [-0.0080, +0.0028] | +4.7 |
| generator | ood-summarization | -0.002 | [-0.025, +0.021] | +0.0013 | [-0.0006, +0.0033] | -0.2 |

### Against the 10K-math arms (same test / OOD rows)

Paired differences vs **24K static (KL)**, category mean, 95% bootstrap CI (5,000 resamples). The 10K-math arms were trained on math only, so their non-math categories are out of domain.

Test (500 rows):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| 24K generator (KL) | all | +0.022 | [+0.004, +0.040] | +0.0009 | [-0.0001, +0.0018] | +2.2 |
| 10K-math static (CE) | all | -0.055 | [-0.076, -0.037] | -0.0025 | [-0.0037, -0.0014] | -5.5 |
| 10K-math generator (KL) | all | -0.043 | [-0.061, -0.025] | -0.0022 | [-0.0032, -0.0012] | -4.3 |
| 10K-math generator (lk_hybrid) | all | -0.039 | [-0.058, -0.021] | -0.0020 | [-0.0030, -0.0010] | -3.9 |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| 24K static (KL) | 49,021 | 363,202 | 741 |
| 24K generator (KL) | 49,021 | 364,225 | 743 |
| 10K-math static (CE) | 49,021 | 361,286 | 737 |
| 10K-math generator (KL) | 49,021 | 362,365 | 739 |
| 10K-math generator (lk_hybrid) | 49,021 | 362,505 | 739 |

OOD v3 (240 rows):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| 24K generator (KL) | all | +0.003 | [-0.034, +0.041] | -0.0013 | [-0.0035, +0.0009] | +0.3 |
| 10K-math static (CE) | all | -0.003 | [-0.034, +0.027] | -0.0028 | [-0.0043, -0.0012] | -0.3 |
| 10K-math generator (KL) | all | -0.030 | [-0.064, +0.006] | -0.0037 | [-0.0056, -0.0017] | -3.0 |
| 10K-math generator (lk_hybrid) | all | -0.042 | [-0.080, -0.002] | -0.0035 | [-0.0055, -0.0014] | -4.2 |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| 24K static (KL) | 8,993 | 36,631 | 407 |
| 24K generator (KL) | 8,993 | 36,621 | 407 |
| 10K-math static (CE) | 8,993 | 36,462 | 405 |
| 10K-math generator (KL) | 8,993 | 36,339 | 404 |
| 10K-math generator (lk_hybrid) | 8,993 | 36,246 | 403 |

- On test, the 24K static beats every 10K-math arm by 0.04–0.06 τ (all CIs exclude zero). The gain is in code, stem and chat (the 10K-math arms never saw them); math is unchanged or slightly favours the math-only generators (+0.03 τ).
- OOD, the 24K static ties the 10K-math static (CE). The 10K-math generators trail it by 0.03 (KL) and 0.04 τ (lk_hybrid); only lk_hybrid's interval excludes zero.
- These are cross-run comparisons with different data, losses and seeds; the matched comparison is the one above.

### Training curves (validation, 500 rows; full_acc / τ averaged over subsets as logged)

| Update | 0 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|---|
| Static (KL) | 0.5265 / 7.703 | 0.5319 / 7.854 | 0.5328 / 7.867 | 0.5329 / 7.868 | 0.5333 / 7.883 | 0.5335 / 7.879 | 0.5335 / 7.882 | 0.5338 / 7.888 | 0.5338 / 7.886 |
| Generator (KL) | 0.5265 / 7.703 | 0.5328 / 7.856 | 0.5336 / 7.878 | 0.5338 / 7.888 | 0.5337 / 7.892 | 0.5342 / 7.895 | 0.5344 / 7.896 | 0.5345 / 7.905 | 0.5347 / 7.907 |

Every 250 updates is in the per-run details below. Both arms keep improving slowly to the end; the generator is ahead from update 250 on (by 0.001–0.002 full_acc, 0.01–0.02 τ). Neither declines, unlike the 4K generators.

### LoRA coefficient (validation; full_acc / τ averaged over subsets)

| Coefficient | 0.5 | 1 | 2 (training scale) | 4 | 8 |
|---|---|---|---|---|---|
| Static (KL) | 0.5299 / 7.777 | 0.5323 / 7.827 | **0.5338 / 7.886** | 0.5264 / 7.842 | 0.4767 / 7.138 |
| Generator (KL) | 0.5314 / 7.807 | 0.5343 / 7.870 | **0.5347 / 7.908** | 0.5189 / 7.696 | 0.4297 / 6.125 |

Both arms are best at the training scale, so test and OOD use coefficient 2. The per-row oracle coefficient would add only +0.06 to +0.10 τ per category (per-run details).

### What the generator learned (D13, D14)

| | 24K mixed (KL) | 10K math (KL) | 10K math (CE) | 4K M0 (CE) |
|---|---|---|---|---|
| cross-row cosine of generated ΔW | 0.795 | 0.989 | 0.965 | 0.667 |
| within-category cosine | 0.972 | 0.998 | 0.993 | 0.925 |
| dyn_ratio (row-specific share) | 0.155 | 0.014 | 0.032 | 0.268 |
| η² (category share of variation) | 0.776 | 0.741 | 0.768 | 0.744 |
| effective rank of the ΔW set | 2.48 | 1.02 | 1.04 | 3.60 |
| category centroid accuracy | 0.972 | 0.574 | 0.644 | 0.965 |
| ms per LoRA / merge | 4.15 / 0.67 | 4.02 / 0.67 | 4.36 / 0.67 | 4.52 / 0.68 |

- The generated LoRAs are again different per row and strongly category-structured, between the 10K-math collapse and 4K. η² 0.78: most of the variation is between categories.
- So the generator mostly acts as a category router plus a small per-row part. A routed set of per-category static LoRAs (suggestion A1) is the natural next control.

## Interpretation and limits

- **The matched comparison favours the generator, slightly.** With the same loss, data and schedule, the generator leads the static LoRA on validation and test by about 0.02 τ (≈ +2 accepted draft tokens per 100 blocks, ≈ 0.3%), with both intervals excluding zero, and ties OOD. On 4K and 10K-math no generator did this against a matched static.
- **Data diversity is what changed.** The same generator architecture collapsed to one LoRA on 10K math and became category-dynamic on 24K mixed. Whether the gain comes from the category routing or the per-row part is not separated here.
- **The effect is small relative to its cost:** 1.1B generator parameters (4.26M for the static), ≈ 4.2 ms per generated LoRA plus 0.7 ms merge, and per-request LoRAs that do not batch in serving. 24K data itself was worth more (+0.055 τ for the static over its 10K-math counterpart).
- **Limits:**
  - One seed per arm; the generator's margin is about the size of seed-to-seed drift seen elsewhere.
  - Offline, greedy, teacher-forced metrics; no vLLM serving measurement.
  - The eval sets weight math 2× (200 of 500 rows); "all" is the category mean.
  - OOD is 240 rows with responses capped at 256 tokens.
  - The base numbers come from the math10k tier's base run (same drafter and rows).
- **Next controls (see [experiment_suggestions/future_suggestions.md](experiment_suggestions/future_suggestions.md)):** per-category static LoRAs (A1), more seeds (A4), the basis-LoRA generator (G1) at a fraction of the size.

## Artifacts

- Runs: `/workspace/yusuf/runs/dflash/mix24k/{mix24k-static-lora-attn-r32-3ep-kl, mix24k-generator-attn-r32-3ep-kl}` (records `run_record_{train,eval,test,ood}.json`); git f4f6385.
- W&B project `dyn-spec-dflash-v1`: `dflash-mix24k-static-lora-attn-r32-3ep-kl-gseed42` (823435ba41b9fecf), `dflash-mix24k-generator-attn-r32-3ep-kl-gseed42` (e98e30a73e80293c).
- HF models (public): `erdem-erdem/dyn-spec-dflash-mix24k-static-lora-attn-r32-3ep-kl` @ 36c816e5; `erdem-erdem/dyn-spec-dflash-mix24k-generator-attn-r32-3ep-kl` @ 62d897d4.
- HF store (public dataset): `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-mix24k` @ d638d564 (50,480 entries, 1.73 TB, MANIFEST sha256 cf4f7876…; fetch with `setup.sh --fetch-store`).
- Recipes: `examples/dflash-lora/mix24k/`; matrices `examples/dflash-lora/matrices/mix24k_gpu{0,1}.yaml`. RUNLOG entries of 2026-11-10.

## Per-run details (generated by `scripts/dflash_port/summarize_runs.py`)

### mix24k-generator-attn-r32-3ep-kl

Arm `direct`, 4000 updates, git `f4f63857a6`, trainable parameters 1,105,658,048.

**Validation during training** (200 fixed rows):

| Update | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| update 0 | 0.5265 | 7.703 | 0.6856 / 9.73 | 0.6035 / 8.48 | 0.4790 / 6.47 | 0.3162 / 3.88 |
| update 250 | 0.5319 | 7.845 | 0.6898 / 9.90 | 0.6084 / 8.62 | 0.4847 / 6.63 | 0.3211 / 3.98 |
| update 500 | 0.5328 | 7.856 | 0.6908 / 9.91 | 0.6099 / 8.64 | 0.4854 / 6.65 | 0.3213 / 3.97 |
| update 750 | 0.5333 | 7.868 | 0.6912 / 9.93 | 0.6113 / 8.66 | 0.4846 / 6.64 | 0.3230 / 3.99 |
| update 1000 | 0.5336 | 7.878 | 0.6904 / 9.93 | 0.6117 / 8.68 | 0.4841 / 6.63 | 0.3233 / 4.01 |
| update 1250 | 0.5336 | 7.881 | 0.6910 / 9.95 | 0.6118 / 8.68 | 0.4848 / 6.67 | 0.3218 / 3.98 |
| update 1500 | 0.5338 | 7.888 | 0.6908 / 9.95 | 0.6124 / 8.68 | 0.4854 / 6.68 | 0.3221 / 3.98 |
| update 1750 | 0.5338 | 7.888 | 0.6902 / 9.93 | 0.6129 / 8.72 | 0.4862 / 6.67 | 0.3232 / 3.99 |
| update 2000 | 0.5337 | 7.892 | 0.6913 / 9.96 | 0.6129 / 8.72 | 0.4842 / 6.65 | 0.3233 / 4.00 |
| update 2250 | 0.5340 | 7.898 | 0.6903 / 9.94 | 0.6145 / 8.73 | 0.4832 / 6.64 | 0.3236 / 4.03 |
| update 2500 | 0.5342 | 7.895 | 0.6909 / 9.96 | 0.6125 / 8.71 | 0.4858 / 6.69 | 0.3236 / 4.02 |
| update 2750 | 0.5343 | 7.904 | 0.6913 / 9.95 | 0.6133 / 8.71 | 0.4876 / 6.70 | 0.3225 / 4.02 |
| update 3000 | 0.5344 | 7.896 | 0.6905 / 9.92 | 0.6123 / 8.71 | 0.4857 / 6.69 | 0.3251 / 4.01 |
| update 3250 | 0.5348 | 7.910 | 0.6909 / 9.95 | 0.6135 / 8.74 | 0.4864 / 6.70 | 0.3245 / 4.04 |
| update 3500 | 0.5345 | 7.905 | 0.6907 / 9.94 | 0.6130 / 8.73 | 0.4862 / 6.70 | 0.3225 / 4.03 |
| update 3750 | 0.5342 | 7.910 | 0.6914 / 9.95 | 0.6141 / 8.75 | 0.4839 / 6.68 | 0.3247 / 4.05 |
| update 4000 | 0.5347 | 7.907 | 0.6908 / 9.94 | 0.6141 / 8.75 | 0.4862 / 6.73 | 0.3240 / 4.03 |

**Offline reproduction:** 0.534689 vs training 0.534700 (not bitwise).

**Offline variants** (M4 coefficient × M7 donor; `_k-4`: 4 rows per LoRA):

| Variant | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| coef-0.5_donor-correct | 0.5314 | 7.807 | 0.6900 / 9.83 | 0.6097 / 8.61 | 0.4835 / 6.57 | 0.3204 / 3.95 |
| coef-0.5_donor-correct_k-4 | 0.5278 | 7.802 | 0.6919 / 9.88 | 0.6100 / 8.57 | 0.4864 / 6.73 | 0.3120 / 3.88 |
| coef-1_donor-correct | 0.5343 | 7.870 | 0.6928 / 9.93 | 0.6129 / 8.69 | 0.4864 / 6.62 | 0.3222 / 4.00 |
| coef-1_donor-correct_k-4 | 0.5301 | 7.861 | 0.6946 / 9.96 | 0.6123 / 8.64 | 0.4888 / 6.80 | 0.3141 / 3.91 |
| coef-2_donor-correct | 0.5347 | 7.908 | 0.6909 / 9.94 | 0.6141 / 8.75 | 0.4864 / 6.74 | 0.3240 / 4.03 |
| coef-2_donor-correct_k-4 | 0.5306 | 7.905 | 0.6927 / 10.01 | 0.6129 / 8.66 | 0.4905 / 6.85 | 0.3149 / 3.94 |
| coef-4_donor-correct | 0.5189 | 7.696 | 0.6728 / 9.67 | 0.5955 / 8.47 | 0.4717 / 6.56 | 0.3132 / 3.95 |
| coef-4_donor-correct_k-4 | 0.5171 | 7.716 | 0.6758 / 9.74 | 0.5959 / 8.43 | 0.4802 / 6.72 | 0.3068 / 3.91 |
| coef-8_donor-correct | 0.4297 | 6.125 | 0.5689 / 7.81 | 0.4919 / 6.65 | 0.3854 / 5.09 | 0.2565 / 3.20 |
| coef-8_donor-correct_k-4 | 0.4360 | 6.241 | 0.5724 / 7.93 | 0.4974 / 6.69 | 0.3936 / 5.20 | 0.2694 / 3.45 |

**M4 per-row τ** (best fixed coefficient vs per-row oracle):

| Category | best fixed s | τ at best fixed | per-row oracle τ | oracle gain | rows best at each s |
|---|---|---|---|---|---|
| chat | 2 | 3.972 | 4.046 | +0.074 | 0.5: 23, 1: 35, 2: 39, 4: 31, 8: 3 |
| code | 2 | 8.223 | 8.306 | +0.082 | 0.5: 17, 1: 26, 2: 47, 4: 13, 8: 0 |
| math | 2 | 9.483 | 9.547 | +0.064 | 0.5: 31, 1: 64, 2: 93, 4: 19, 8: 0 |
| stem | 2 | 6.669 | 6.747 | +0.078 | 0.5: 27, 1: 18, 2: 46, 4: 18, 8: 0 |

**D13 similarity of the generated ΔW** (all modules pooled): 500 LoRAs; cos within 0.9718, across 0.7946 (gap 0.1772); dyn_ratio 0.1554; η² 0.776; effective rank 2.48; centroid accuracy 0.972.

**D14 cost** (cuda): 1,105,658,048 generator parameters; 20 modules, rank 32; factors 8.5 MB per LoRA; 4.15 ms per call; 5.10 ms for 8 batched; merge 0.67 ms.

**test:**

| Arm | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| mix24k-generator-attn-r32-3ep-kl | 0.5396 | 8.021 | 0.7095 / 10.23 | 0.5447 / 7.55 | 0.4914 / 6.70 | 0.3721 / 4.76 |

**ood:**

| Arm | score | τ | ood-humaneval (acc / τ) | ood-math-reasoning (acc / τ) | ood-summarization (acc / τ) |
|---|---|---|---|---|---|
| mix24k-generator-attn-r32-3ep-kl | 0.4603 | 6.148 | 0.6129 / 8.34 | 0.5583 / 7.32 | 0.2098 / 2.78 |

### mix24k-static-lora-attn-r32-3ep-kl

Arm `fresh_lora`, 4000 updates, git `f4f63857a6`, trainable parameters 4,259,840.

**Validation during training** (200 fixed rows):

| Update | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| update 0 | 0.5265 | 7.703 | 0.6856 / 9.73 | 0.6035 / 8.48 | 0.4790 / 6.47 | 0.3162 / 3.88 |
| update 250 | 0.5309 | 7.831 | 0.6896 / 9.87 | 0.6073 / 8.62 | 0.4835 / 6.61 | 0.3201 / 3.97 |
| update 500 | 0.5319 | 7.854 | 0.6896 / 9.89 | 0.6077 / 8.63 | 0.4840 / 6.64 | 0.3210 / 3.97 |
| update 750 | 0.5322 | 7.861 | 0.6903 / 9.89 | 0.6088 / 8.64 | 0.4847 / 6.64 | 0.3210 / 3.98 |
| update 1000 | 0.5328 | 7.867 | 0.6903 / 9.90 | 0.6102 / 8.65 | 0.4850 / 6.65 | 0.3217 / 3.99 |
| update 1250 | 0.5329 | 7.865 | 0.6909 / 9.91 | 0.6102 / 8.65 | 0.4853 / 6.66 | 0.3216 / 3.98 |
| update 1500 | 0.5329 | 7.868 | 0.6906 / 9.92 | 0.6100 / 8.65 | 0.4847 / 6.64 | 0.3224 / 3.99 |
| update 1750 | 0.5332 | 7.873 | 0.6909 / 9.91 | 0.6112 / 8.68 | 0.4847 / 6.67 | 0.3227 / 4.01 |
| update 2000 | 0.5333 | 7.883 | 0.6914 / 9.93 | 0.6121 / 8.70 | 0.4847 / 6.68 | 0.3226 / 3.99 |
| update 2250 | 0.5333 | 7.883 | 0.6910 / 9.94 | 0.6116 / 8.70 | 0.4846 / 6.66 | 0.3218 / 3.99 |
| update 2500 | 0.5335 | 7.879 | 0.6912 / 9.93 | 0.6121 / 8.70 | 0.4841 / 6.66 | 0.3224 / 3.99 |
| update 2750 | 0.5335 | 7.885 | 0.6913 / 9.94 | 0.6127 / 8.70 | 0.4849 / 6.67 | 0.3218 / 3.99 |
| update 3000 | 0.5335 | 7.882 | 0.6916 / 9.94 | 0.6121 / 8.69 | 0.4835 / 6.65 | 0.3225 / 4.00 |
| update 3250 | 0.5337 | 7.888 | 0.6919 / 9.95 | 0.6124 / 8.71 | 0.4844 / 6.67 | 0.3225 / 3.99 |
| update 3500 | 0.5338 | 7.888 | 0.6918 / 9.94 | 0.6120 / 8.70 | 0.4846 / 6.67 | 0.3231 / 4.01 |
| update 3750 | 0.5337 | 7.888 | 0.6920 / 9.94 | 0.6121 / 8.71 | 0.4841 / 6.67 | 0.3226 / 4.01 |
| update 4000 | 0.5338 | 7.886 | 0.6917 / 9.93 | 0.6126 / 8.70 | 0.4840 / 6.67 | 0.3223 / 3.99 |

**Offline reproduction:** 0.533787 vs training 0.533787 (bit-exact).

**Offline variants** (M4 coefficient × M7 donor; `_k-4`: 4 rows per LoRA):

| Variant | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| coef-0.5_donor-correct | 0.5299 | 7.777 | 0.6886 / 9.80 | 0.6079 / 8.58 | 0.4814 / 6.55 | 0.3193 / 3.92 |
| coef-0.5_donor-correct_k-4 | 0.5267 | 7.778 | 0.6905 / 9.85 | 0.6076 / 8.54 | 0.4853 / 6.72 | 0.3113 / 3.86 |
| coef-1_donor-correct | 0.5323 | 7.827 | 0.6909 / 9.86 | 0.6105 / 8.62 | 0.4836 / 6.59 | 0.3211 / 3.96 |
| coef-1_donor-correct_k-4 | 0.5286 | 7.834 | 0.6929 / 9.92 | 0.6091 / 8.58 | 0.4876 / 6.77 | 0.3134 / 3.90 |
| coef-2_donor-correct | 0.5338 | 7.886 | 0.6917 / 9.93 | 0.6126 / 8.70 | 0.4840 / 6.67 | 0.3223 / 3.99 |
| coef-2_donor-correct_k-4 | 0.5303 | 7.889 | 0.6933 / 9.99 | 0.6097 / 8.62 | 0.4905 / 6.83 | 0.3145 / 3.95 |
| coef-4_donor-correct | 0.5264 | 7.842 | 0.6828 / 9.89 | 0.6032 / 8.60 | 0.4760 / 6.65 | 0.3202 / 4.03 |
| coef-4_donor-correct_k-4 | 0.5238 | 7.852 | 0.6870 / 9.96 | 0.6025 / 8.54 | 0.4869 / 6.80 | 0.3093 / 3.92 |
| coef-8_donor-correct | 0.4767 | 7.138 | 0.6312 / 9.12 | 0.5466 / 7.82 | 0.4221 / 5.88 | 0.2882 / 3.73 |
| coef-8_donor-correct_k-4 | 0.4751 | 7.175 | 0.6342 / 9.20 | 0.5458 / 7.77 | 0.4345 / 6.07 | 0.2803 / 3.68 |

**M4 per-row τ** (best fixed coefficient vs per-row oracle):

| Category | best fixed s | τ at best fixed | per-row oracle τ | oracle gain | rows best at each s |
|---|---|---|---|---|---|
| chat | 4 | 3.978 | 4.056 | +0.077 | 0.5: 16, 1: 21, 2: 35, 4: 47, 8: 10 |
| code | 2 | 8.202 | 8.270 | +0.069 | 0.5: 14, 1: 19, 2: 46, 4: 30, 8: 0 |
| math | 2 | 9.445 | 9.530 | +0.085 | 0.5: 27, 1: 47, 2: 74, 4: 66, 8: 2 |
| stem | 2 | 6.638 | 6.733 | +0.095 | 0.5: 18, 1: 18, 2: 37, 4: 39, 8: 1 |

**test:**

| Arm | score | τ | math (acc / τ) | code (acc / τ) | stem (acc / τ) | chat (acc / τ) |
|---|---|---|---|---|---|---|
| mix24k-static-lora-attn-r32-3ep-kl | 0.5387 | 8.000 | 0.7085 / 10.20 | 0.5447 / 7.52 | 0.4898 / 6.69 | 0.3689 / 4.74 |

**ood:**

| Arm | score | τ | ood-humaneval (acc / τ) | ood-math-reasoning (acc / τ) | ood-summarization (acc / τ) |
|---|---|---|---|---|---|
| mix24k-static-lora-attn-r32-3ep-kl | 0.4616 | 6.144 | 0.6155 / 8.37 | 0.5609 / 7.28 | 0.2085 / 2.78 |
