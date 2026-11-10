# 10K-math: generator training loss (CE vs lk_hybrid vs KL)

- **Status:** done (2026-11-10)
- **Requested by the user (2026-11-10):** two more generators on the same 10K math data: (1) `lk_hybrid`, the soft KL + TV loss against the verifier's full distribution used in the EAGLE-3 trainings; (2) KL (`kl_div`). Compared with the CE generator and the static LoRA of [MATH10K.md](MATH10K.md).

## TL;DR

- **The soft-loss generators edge the static LoRA in distribution; CE does not.**
  - Test (500 rows), Δτ vs static: lk_hybrid +0.016, 95% CI [+0.001, +0.032]; KL +0.013, CI [−0.003, +0.028]; CE −0.002, CI [−0.017, +0.012].
  - That is +1.6 / +1.3 / −0.2 accepted draft tokens per 100 blocks (static: 737).
  - lk_hybrid's interval just excludes zero on test; it does not on validation (+0.006, CI [−0.010, +0.021]). One seed each.
  - The gains sit in math (+0.036 to +0.038 τ, CIs exclude zero) and code (+0.027 to +0.032).
- **OOD, static stays best.** Every generator trails it: lk_hybrid −0.039 τ (CI [−0.078, +0.001]), KL −0.027 (CI [−0.066, +0.012]), CE −0.051 (CI [−0.092, −0.010]). The soft losses roughly halve the CE generator's OOD loss.
- **The soft losses make the generated LoRAs even more constant.** Cross-row cosine: CE 0.965, lk_hybrid 0.976, KL 0.989. Effective rank about 1.02. So the small gain over static comes from a better near-constant ΔW (the soft-loss target), not from per-prompt dynamism.
- **No overfitting:** both soft-loss generators are flat or rising at the end of training. The CE generator gave back about 0.01 τ after update 1,000.
- **Open control:** a static LoRA trained with lk_hybrid or KL. Without it, the soft-loss gain cannot be attributed to the generator.

## Setup

Same as [MATH10K.md](MATH10K.md) (data, 1,667 updates = 3 epochs, eval rows, store, generator: 1,105,658,048 parameters, decoder width 256, condition dim 32, attention targets, rank 32, α 64, LoRA coefficient 2). Only `training.block_drafter.loss_fn` differs:

| Arm | Loss | Recipe | git | Train time |
|---|---|---|---|---|
| Static LoRA | CE + D-PACE | `math10k/static_lora.yaml` | 2ccc5e1 | 84.1 min |
| Generator CE | CE + D-PACE | `math10k/generator.yaml` | 08935cb | 88.2 min |
| Generator lk_hybrid | `lk_hybrid` + D-PACE | `math10k/generator_lkhybrid.yaml` | 99bf143 | 90.7 min |
| Generator KL | `kl_div` + D-PACE | `math10k/generator_kl.yaml` | 99bf143 | 89.2 min |

- The loss functions are the same `speculators.losses` implementations the EAGLE-3 path uses (`lk_hybrid` with η = 3). D-PACE per-position weights stay on, as in 4K M5; EAGLE-3 instead weights its 3 TTT steps.
- Metrics, CIs and pooling as in MATH10K.md: offline, teacher-forced, greedy; paired bootstrap over rows stratified by category, 5,000 resamples (`compare_arms.py --reference static`); "all" is the category mean.

## Results

### Test (500 rows; final numbers)

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
| generator CE | full_acc | 0.7088 | 0.5593 | 0.4999 | 0.3748 | 0.5357 |
| | τ | 10.243 | 7.702 | 6.845 | 4.766 | 7.389 |
| | accepted draft tokens / block | 9.24 of 15 | 6.70 of 15 | 5.85 of 15 | 3.77 of 15 | 6.39 of 15 |
| | acceptance rate | 61.6% | 44.7% | 39.0% | 25.1% | 42.6% |
| generator lk_hybrid | full_acc | 0.7098 | 0.5603 | 0.5020 | 0.3758 | 0.5370 |
| | τ | 10.258 | 7.731 | 6.890 | 4.753 | 7.408 |
| | accepted draft tokens / block | 9.26 of 15 | 6.73 of 15 | 5.89 of 15 | 3.75 of 15 | 6.41 of 15 |
| | acceptance rate | 61.7% | 44.9% | 39.3% | 25.0% | 42.7% |
| generator KL | full_acc | 0.7096 | 0.5606 | 0.5019 | 0.3750 | 0.5368 |
| | τ | 10.256 | 7.726 | 6.879 | 4.755 | 7.404 |
| | accepted draft tokens / block | 9.26 of 15 | 6.73 of 15 | 5.88 of 15 | 3.76 of 15 | 6.40 of 15 |
| | acceptance rate | 61.7% | 44.8% | 39.2% | 25.0% | 42.7% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 49,021 | 353,580 | 721 |
| static | 49,021 | 361,286 | 737 |
| generator CE | 49,021 | 361,686 | 738 |
| generator lk_hybrid | 49,021 | 362,505 | 739 |
| generator KL | 49,021 | 362,365 | 739 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.140 | [-0.161, -0.120] | -0.0047 | [-0.0059, -0.0037] | -14.0 |
| base | math | -0.198 | [-0.226, -0.171] | -0.0057 | [-0.0070, -0.0045] | -19.8 |
| base | code | -0.109 | [-0.144, -0.073] | -0.0045 | [-0.0062, -0.0028] | -10.9 |
| base | stem | -0.154 | [-0.194, -0.114] | -0.0042 | [-0.0063, -0.0022] | -15.4 |
| base | chat | -0.099 | [-0.158, -0.050] | -0.0045 | [-0.0079, -0.0018] | -9.9 |
| generator CE | all | -0.002 | [-0.017, +0.012] | -0.0008 | [-0.0015, -0.0000] | -0.2 |
| generator CE | math | +0.023 | [+0.000, +0.047] | +0.0001 | [-0.0008, +0.0011] | +2.3 |
| generator CE | code | +0.003 | [-0.027, +0.033] | -0.0004 | [-0.0019, +0.0009] | +0.3 |
| generator CE | stem | -0.036 | [-0.072, +0.000] | -0.0024 | [-0.0043, -0.0006] | -3.6 |
| generator CE | chat | +0.001 | [-0.024, +0.027] | -0.0004 | [-0.0019, +0.0010] | +0.1 |
| generator lk_hybrid | all | +0.016 | [+0.001, +0.032] | +0.0005 | [-0.0002, +0.0012] | +1.6 |
| generator lk_hybrid | math | +0.038 | [+0.013, +0.065] | +0.0012 | [+0.0000, +0.0023] | +3.8 |
| generator lk_hybrid | code | +0.032 | [-0.000, +0.066] | +0.0006 | [-0.0008, +0.0020] | +3.2 |
| generator lk_hybrid | stem | +0.008 | [-0.031, +0.049] | -0.0003 | [-0.0021, +0.0014] | +0.8 |
| generator lk_hybrid | chat | -0.012 | [-0.039, +0.014] | +0.0006 | [-0.0007, +0.0019] | -1.2 |
| generator KL | all | +0.013 | [-0.003, +0.028] | +0.0003 | [-0.0004, +0.0010] | +1.3 |
| generator KL | math | +0.036 | [+0.009, +0.064] | +0.0009 | [-0.0002, +0.0021] | +3.6 |
| generator KL | code | +0.027 | [-0.008, +0.061] | +0.0009 | [-0.0005, +0.0023] | +2.7 |
| generator KL | stem | -0.002 | [-0.038, +0.034] | -0.0004 | [-0.0020, +0.0011] | -0.2 |
| generator KL | chat | -0.009 | [-0.039, +0.019] | -0.0002 | [-0.0018, +0.0014] | -0.9 |

### Validation (500 rows)

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
| generator CE | full_acc | 0.6842 | 0.6019 | 0.4881 | 0.3486 | 0.5307 |
| | τ | 9.866 | 8.516 | 6.664 | 4.375 | 7.355 |
| | accepted draft tokens / block | 8.87 of 15 | 7.52 of 15 | 5.66 of 15 | 3.37 of 15 | 6.36 of 15 |
| | acceptance rate | 59.1% | 50.1% | 37.8% | 22.5% | 42.4% |
| generator lk_hybrid | full_acc | 0.6846 | 0.6026 | 0.4902 | 0.3511 | 0.5321 |
| | τ | 9.864 | 8.521 | 6.700 | 4.380 | 7.366 |
| | accepted draft tokens / block | 8.86 of 15 | 7.52 of 15 | 5.70 of 15 | 3.38 of 15 | 6.37 of 15 |
| | acceptance rate | 59.1% | 50.1% | 38.0% | 22.5% | 42.4% |
| generator KL | full_acc | 0.6843 | 0.6028 | 0.4912 | 0.3514 | 0.5324 |
| | τ | 9.863 | 8.525 | 6.689 | 4.378 | 7.364 |
| | accepted draft tokens / block | 8.86 of 15 | 7.53 of 15 | 5.69 of 15 | 3.38 of 15 | 6.36 of 15 |
| | acceptance rate | 59.1% | 50.2% | 37.9% | 22.5% | 42.4% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 45,566 | 324,187 | 711 |
| static | 45,566 | 331,716 | 728 |
| generator CE | 45,566 | 331,867 | 728 |
| generator lk_hybrid | 45,566 | 332,123 | 729 |
| generator KL | 45,566 | 332,064 | 729 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.136 | [-0.154, -0.117] | -0.0048 | [-0.0057, -0.0039] | -13.6 |
| base | math | -0.219 | [-0.248, -0.189] | -0.0068 | [-0.0081, -0.0055] | -21.9 |
| base | code | -0.151 | [-0.195, -0.107] | -0.0063 | [-0.0082, -0.0044] | -15.1 |
| base | stem | -0.097 | [-0.134, -0.059] | -0.0036 | [-0.0058, -0.0014] | -9.7 |
| base | chat | -0.075 | [-0.112, -0.037] | -0.0025 | [-0.0043, -0.0006] | -7.5 |
| generator CE | all | -0.005 | [-0.022, +0.011] | -0.0008 | [-0.0015, -0.0000] | -0.5 |
| generator CE | math | +0.020 | [-0.005, +0.046] | -0.0003 | [-0.0013, +0.0008] | +2.0 |
| generator CE | code | -0.008 | [-0.038, +0.019] | +0.0003 | [-0.0009, +0.0015] | -0.8 |
| generator CE | stem | -0.022 | [-0.063, +0.021] | -0.0018 | [-0.0036, +0.0001] | -2.2 |
| generator CE | chat | -0.012 | [-0.046, +0.022] | -0.0013 | [-0.0029, +0.0003] | -1.2 |
| generator lk_hybrid | all | +0.006 | [-0.010, +0.021] | +0.0007 | [+0.0000, +0.0013] | +0.6 |
| generator lk_hybrid | math | +0.019 | [-0.004, +0.041] | +0.0001 | [-0.0009, +0.0012] | +1.9 |
| generator lk_hybrid | code | -0.003 | [-0.039, +0.033] | +0.0010 | [-0.0003, +0.0023] | -0.3 |
| generator lk_hybrid | stem | +0.013 | [-0.021, +0.048] | +0.0003 | [-0.0013, +0.0020] | +1.3 |
| generator lk_hybrid | chat | -0.007 | [-0.035, +0.018] | +0.0012 | [-0.0000, +0.0023] | -0.7 |
| generator KL | all | +0.003 | [-0.013, +0.019] | +0.0009 | [+0.0003, +0.0016] | +0.3 |
| generator KL | math | +0.017 | [-0.006, +0.042] | -0.0002 | [-0.0014, +0.0009] | +1.7 |
| generator KL | code | +0.002 | [-0.031, +0.034] | +0.0012 | [+0.0000, +0.0025] | +0.2 |
| generator KL | stem | +0.002 | [-0.034, +0.037] | +0.0013 | [-0.0003, +0.0029] | +0.2 |
| generator KL | chat | -0.009 | [-0.035, +0.018] | +0.0014 | [+0.0001, +0.0028] | -0.9 |

### Out of distribution (OOD v3, 240 rows)

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
| generator CE | full_acc | 0.6093 | 0.5557 | 0.2037 | 0.4562 |
| | τ | 8.271 | 7.255 | 2.747 | 6.091 |
| | accepted draft tokens / block | 7.27 of 15 | 6.26 of 15 | 1.75 of 15 | 5.09 of 15 |
| | acceptance rate | 48.5% | 41.7% | 11.6% | 33.9% |
| generator lk_hybrid | full_acc | 0.6114 | 0.5575 | 0.2056 | 0.4582 |
| | τ | 8.290 | 7.271 | 2.746 | 6.102 |
| | accepted draft tokens / block | 7.29 of 15 | 6.27 of 15 | 1.75 of 15 | 5.10 of 15 |
| | acceptance rate | 48.6% | 41.8% | 11.6% | 34.0% |
| generator KL | full_acc | 0.6109 | 0.5582 | 0.2048 | 0.4580 |
| | τ | 8.340 | 7.256 | 2.746 | 6.114 |
| | accepted draft tokens / block | 7.34 of 15 | 6.26 of 15 | 1.75 of 15 | 5.11 of 15 |
| | acceptance rate | 48.9% | 41.7% | 11.6% | 34.1% |

| Arm | draft blocks | accepted draft tokens (total) | per 100 blocks |
|---|---|---|---|
| base | 8,993 | 35,048 | 390 |
| static | 8,993 | 36,462 | 405 |
| generator CE | 8,993 | 36,179 | 402 |
| generator lk_hybrid | 8,993 | 36,246 | 403 |
| generator KL | 8,993 | 36,339 | 404 |

Paired differences vs **static** (all = category mean; 95% bootstrap interval, 5,000 resamples of rows, stratified by category):

| Arm | scope | Δ τ | 95% CI | Δ full_acc | 95% CI | Δ accepted tokens / 100 blocks |
|---|---|---|---|---|---|---|
| base | all | -0.205 | [-0.246, -0.163] | -0.0077 | [-0.0098, -0.0057] | -20.5 |
| base | ood-humaneval | -0.258 | [-0.337, -0.179] | -0.0102 | [-0.0141, -0.0066] | -25.8 |
| base | ood-math-reasoning | -0.293 | [-0.382, -0.206] | -0.0100 | [-0.0147, -0.0054] | -29.3 |
| base | ood-summarization | -0.064 | [-0.086, -0.044] | -0.0030 | [-0.0044, -0.0016] | -6.4 |
| generator CE | all | -0.051 | [-0.092, -0.010] | -0.0026 | [-0.0044, -0.0007] | -5.1 |
| generator CE | ood-humaneval | -0.083 | [-0.158, -0.006] | -0.0033 | [-0.0067, +0.0003] | -8.3 |
| generator CE | ood-math-reasoning | -0.077 | [-0.173, +0.012] | -0.0046 | [-0.0087, -0.0007] | -7.7 |
| generator CE | ood-summarization | +0.008 | [-0.012, +0.027] | +0.0000 | [-0.0019, +0.0019] | +0.8 |
| generator lk_hybrid | all | -0.039 | [-0.078, +0.001] | -0.0007 | [-0.0027, +0.0014] | -3.9 |
| generator lk_hybrid | ood-humaneval | -0.063 | [-0.138, +0.013] | -0.0011 | [-0.0043, +0.0021] | -6.3 |
| generator lk_hybrid | ood-math-reasoning | -0.061 | [-0.150, +0.024] | -0.0029 | [-0.0079, +0.0020] | -6.1 |
| generator lk_hybrid | ood-summarization | +0.007 | [-0.013, +0.025] | +0.0019 | [+0.0004, +0.0034] | +0.7 |
| generator KL | all | -0.027 | [-0.066, +0.012] | -0.0009 | [-0.0027, +0.0008] | -2.7 |
| generator KL | ood-humaneval | -0.013 | [-0.094, +0.069] | -0.0016 | [-0.0049, +0.0015] | -1.3 |
| generator KL | ood-math-reasoning | -0.076 | [-0.158, +0.005] | -0.0022 | [-0.0062, +0.0019] | -7.6 |
| generator KL | ood-summarization | +0.007 | [-0.014, +0.029] | +0.0011 | [-0.0006, +0.0026] | +0.7 |

### Training curves (validation, 500 rows; full_acc / τ averaged over subsets as logged)

| Update | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1667 |
|---|---|---|---|---|---|---|---|---|
| Static | 0.5265 / 7.703 | 0.5302 / 7.826 | 0.5308 / 7.838 | 0.5312 / 7.847 | 0.5311 / 7.849 | 0.5312 / 7.850 | 0.5314 / 7.850 | 0.5313 / 7.856 |
| Generator CE | 0.5265 / 7.703 | 0.5301 / 7.823 | 0.5312 / 7.853 | 0.5310 / 7.863 | 0.5313 / 7.869 | 0.5310 / 7.866 | 0.5306 / 7.850 | 0.5305 / 7.855 |
| Generator lk_hybrid | 0.5265 / 7.703 | 0.5308 / 7.833 | 0.5319 / 7.847 | 0.5318 / 7.858 | 0.5318 / 7.867 | 0.5318 / 7.852 | 0.5321 / 7.862 | 0.5319 / 7.864 |
| Generator KL | 0.5265 / 7.703 | 0.5305 / 7.828 | 0.5315 / 7.840 | 0.5315 / 7.859 | 0.5318 / 7.858 | 0.5319 / 7.855 | 0.5325 / 7.866 | 0.5322 / 7.862 |

- The CE generator declines after update 1,000, mostly in stem and chat (out of domain for math training): stem 0.4902 → 0.4884.
- lk_hybrid and KL are flat or rising to the end; the 4K M5 runs also overfit least with these losses.

### LoRA coefficient (validation; full_acc / τ averaged over subsets)

| Coefficient | 0.5 | 1 | 2 (training scale) | 4 | 8 |
|---|---|---|---|---|---|
| Static | 0.5292 / 7.772 | 0.5307 / 7.818 | **0.5313 / 7.856** | 0.5233 / 7.704 | 0.3976 / 4.513 |
| Generator CE | 0.5299 / 7.793 | 0.5317 / 7.842 | 0.5305 / **7.855** | 0.5140 / 7.574 | 0.4005 / 5.093 |
| Generator lk_hybrid | 0.5297 / 7.783 | 0.5319 / 7.836 | **0.5319 / 7.864** | 0.5197 / 7.681 | 0.4469 / 6.268 |
| Generator KL | 0.5296 / 7.781 | 0.5317 / 7.834 | **0.5322 / 7.862** | 0.5199 / 7.678 | 0.4475 / 6.312 |

Every arm reaches its highest τ at the training scale, so test and OOD use coefficient 2. The soft-loss generators degrade more slowly at large coefficients (s = 8: τ 6.3 vs 5.1 for CE and 4.5 for static).

### What the generators learned (D13, D14)

| | Generator CE | Generator lk_hybrid | Generator KL | 4K M0 (CE, reference) |
|---|---|---|---|---|
| cross-row cosine of generated ΔW | 0.965 | 0.976 | 0.989 | 0.667 |
| within-category cosine | 0.993 | 0.996 | 0.998 | 0.925 |
| dyn_ratio (row-specific share) | 0.032 | 0.021 | 0.014 | 0.268 |
| η² (category share of variation) | 0.768 | 0.804 | 0.741 | 0.744 |
| effective rank of the ΔW set | 1.04 | 1.02 | 1.02 | 3.60 |
| category centroid accuracy | 0.644 | 0.670 | 0.574 | 0.965 |
| ms per generated LoRA / merge | 4.36 / 0.67 | 6.74 / 0.82 | 4.02 / 0.67 | 4.52 / 0.68 |

The same 1.1B architecture throughout; the lk_hybrid latency difference is from a shared box (same network, same shapes), not the loss.

## Interpretation and limits

- **The loss matters more than the per-prompt mechanism here.** Switching CE → lk_hybrid / KL turns a tie with static into a small in-distribution lead and halves the OOD deficit. At the same time the generators get *less* dynamic. The most direct reading: the soft targets give a better adapter, and the generator ends up as a near-constant one.
- **Needed control:** a static LoRA trained with lk_hybrid (and KL). If it matches the soft-loss generators, the gain is the loss, not the generator.
- **Limits:**
  - One seed per arm. lk_hybrid's test interval barely excludes zero, and validation does not confirm it.
  - Offline, greedy, teacher-forced metrics; no serving measurement.
  - OOD is 240 rows with responses capped at 256 tokens.

## Artifacts

- Runs: `/workspace/yusuf/runs/dflash/math10k/math10k-generator-attn-r32-3ep-{lkhybrid,kl}` (train, eval, test, OOD records `run_record_*.json`).
- W&B project `dyn-spec-dflash-v1`; runs `dflash-math10k-generator-attn-r32-3ep-lkhybrid-gseed42`, `dflash-math10k-generator-attn-r32-3ep-kl-gseed42`.
- HF models (public): `erdem-erdem/dyn-spec-dflash-math10k-generator-attn-r32-3ep-lkhybrid` @ 152a8e40, `erdem-erdem/dyn-spec-dflash-math10k-generator-attn-r32-3ep-kl` @ fdf8bec6.
- Recipes: `examples/dflash-lora/math10k/{generator,training_recipe}_{lkhybrid,kl}.yaml`; matrices `math10k_loss_gpu{0,1}.yaml`. Code 99bf143. RUNLOG 2026-11-10.
