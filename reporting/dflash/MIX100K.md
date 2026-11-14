# MIX100K: static LoRA r512 vs generator r64 on 100K mixed rows (3 epochs)

- **Status:** done (2026-11-14). One seed per arm.
- **Requested by the user (2026-11-14):** after ABL24K, one 100K comparison of the two strongest designs:
  - static r512 vs generator r64;
  - every other setting as in ABL24K, but 3 epochs;
  - logging, HF, report and results page as usual.
- **Previous:** [ABL24K.md](ABL24K.md) (24K ablations; hybrid test), [MIX24K.md](MIX24K.md).
- **Interactive page:** https://claude.ai/artifact/7B8e3ojcYJCL5KA2eB6JDB ("MIX100K Static vs Generator"; private until shared)

## TL;DR

Test = 500 rows; Δτ = paired difference in category-mean τ, 95% bootstrap CI.

- **The static LoRA wins at 100K.** Generator − static:

  | Split | Δτ | 95% CI |
  |---|---|---|
  | Validation | −0.019 | [−0.031, −0.005] |
  | Test | −0.013 | [−0.026, +0.001] |
  | OOD | −0.049 | [−0.081, −0.015] |

  - Test is borderline; validation and OOD exclude zero.
  - The generator loses on math, code and stem and leads only on chat (+0.011 [−0.009, +0.030]; not resolvable).
- **Data scale helps both arms by the same amount.** 100K (3 epochs) vs 24K (2 epochs) on test:

  | Arm | Δτ | 95% CI |
  |---|---|---|
  | Static r512 | +0.060 | [+0.043, +0.078] |
  | Generator r64 | +0.061 | [+0.040, +0.083] |

  - 100K static r512 vs base: +0.291 τ on test (751 vs 721 accepted draft tokens per 100 blocks), and +0.106 over the 24K static r32.
- **The generator became more dynamic with more data, but that did not turn into accuracy:**

  | Data | Cross-row cosine | dyn_ratio | Effective rank | Centroid accuracy |
  |---|---|---|---|---|
  | 24K | 0.908 | 0.072 | 2.64 | 0.970 |
  | 100K | 0.727 | 0.214 | 3.47 | 0.982 |

- *Interpretation:* at 100K the per-prompt LoRA of a 0.45B generator does not beat one 68M rank-512 LoRA. The ABL24K finding, that extra parameters do more as static rank than as a generator, holds at 4× the data.

## Setup

| | Static r512 | Generator r64 |
|---|---|---|
| Run | `mix100k-static-lora-attn-r512-3ep-kl` | `mix100k-generator-attn-r64-3ep-kl-pt256` |
| Training rows | 100,000: train splits of all 25 subsets of r0002-100k × math / code / stem / chat (25K each) | same |
| Schedule | 3 epochs = 16,667 updates × 18 rows; validation every 500 updates | same |
| Loss | KL to the target + D-PACE (α 0.5) | same |
| Optimizer | AdamW lr 1e-4, wd 0.01, clip 0.85, warmup 100; seed 42 | same |
| Adapter | q/k/v/o of all 5 layers, rank 512, α 1024 (coefficient 2) | q/k/v/o of all 5 layers, rank 64, α 128 (coefficient 2); one LoRA generated per prompt |
| Trainable parameters | 68,157,440 | 448,790,720 (heads shared across layers, decoder 256, condition 32) |
| Train time (H100, both GPUs busy) | 11.6 h | 12.0 h |
| Eval rows | validation / test 500 each (200 math + 100 code / stem / chat), OOD v3 240 | same (same rows as every 24K run) |

- **Store:** `output/dflash-lora/activation-store-mix100k`: 101,000 rows + OOD v3, 6.3 TB. Online ≡ offline on 32 rows (full states max |diff| 0.0).
- **Pool rule:** the train and eval subsets overlap at the subset level, allowed row-disjoint by the user on 2026-11-14. No eval row occurs in any train split by id, prompt hash or full-sequence hash (124 train splits checked).

## Results

| Arm | val τ | test τ | OOD τ | test τ math / code / stem / chat |
|---|---|---|---|---|
| base drafter | 7.225 | 7.251 | 5.936 | 10.022 / 7.590 / 6.728 / 4.666 |
| 24K static r32 (2 ep) | 7.398 | 7.436 | 6.139 | 10.216 / 7.778 / 6.922 / 4.829 |
| 24K static r512 (2 ep) | 7.441 | 7.482 | 6.175 | 10.274 / 7.823 / 6.972 / 4.860 |
| 24K generator r64 (2 ep) | 7.418 | 7.468 | 6.145 | 10.252 / 7.803 / 6.964 / 4.855 |
| **100K static r512 (3 ep)** | **7.488** | **7.542** | **6.231** | 10.325 / 7.896 / 7.022 / 4.925 |
| **100K generator r64 (3 ep)** | 7.470 | 7.530 | 6.182 | 10.312 / 7.873 / 6.996 / 4.936 |

**Direct comparisons** (paired, 95% CI):

| Comparison | validation | test | OOD |
|---|---|---|---|
| 100K generator r64 − 100K static r512 | −0.019 [−0.031, −0.005] | −0.013 [−0.026, +0.001] | −0.049 [−0.081, −0.015] |
| 100K static r512 − 24K static r512 | +0.047 [+0.035, +0.059] | +0.060 [+0.043, +0.078] | +0.056 [+0.021, +0.090] |
| 100K generator r64 − 24K generator r64 | +0.052 [+0.036, +0.069] | +0.061 [+0.040, +0.083] | +0.037 [−0.000, +0.075] |
| 100K static r512 − 24K static r32 | +0.090 [+0.076, +0.105] | +0.106 [+0.084, +0.130] | +0.092 [+0.056, +0.128] |

**Generator − static per category (test):**

| Category | Δτ | 95% CI |
|---|---|---|
| math | −0.013 | [−0.033, +0.010] |
| code | −0.023 | [−0.051, +0.006] |
| stem | −0.026 | [−0.064, +0.010] |
| chat | +0.011 | [−0.009, +0.030] |

OOD: HumanEval −0.071 [−0.135, −0.007], math reasoning −0.037 [−0.110, +0.038], summarization −0.038 [−0.060, −0.016].

**Validation over training** (category-mean τ, last checkpoints):

| Arm | 14,500 | 15,000 | 15,500 | 16,000 | 16,500 | 16,667 |
|---|---|---|---|---|---|---|
| static r512 | 7.478 | 7.483 | 7.475 | 7.478 | 7.482 | 7.486 |
| generator r64 | 7.457 | 7.464 | 7.463 | 7.459 | 7.467 | 7.468 |

Both arms are close to a plateau in the third epoch.

**LoRA coefficient at inference** (validation full_acc): both arms are best at the training scale 2.

| Coefficient | 0.5 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Static | 0.5319 | 0.5357 | 0.5380 | 0.5275 | 0.4583 |
| Generator | 0.5321 | 0.5355 | 0.5368 | 0.5218 | 0.4334 |

**Generator dynamism (D13, 500 validation rows):**

| Metric | Value |
|---|---|
| Cross-row cosine | 0.727 |
| Within-category cosine | 0.953 |
| dyn_ratio | 0.214 |
| Effective rank | 3.47 |
| η² | 0.72 |
| Category centroid accuracy | 0.982 |
| Generation time | 4.3 ms per LoRA |

## Interpretation and open questions
Interpretation, not measured directly:
- **Where the generator stands.**
  - At 24K the r64 generator trailed static r512 by about 0.014 τ on test.
  - At 100K it trails by 0.013 on test, and its OOD deficit grows to 0.049.
  - More data made its LoRAs more prompt-specific (cosine 0.91 → 0.73), but that specificity does not raise accuracy over a single high-rank LoRA.
- **Rank remains the best use of parameters** for this drafter and data. A static r1024 (136M) at 100K was not run; at 24K it added +0.004 over r512.
- **Ideas not yet tried that could change the picture:**
  - generators whose static path matches a strong static adapter and whose per-prompt part is small and slowly trained: the hybrid with a lower head learning rate, or the core-matrix generator in [experiment_suggestions/future_core_matrix_generator.md](experiment_suggestions/future_core_matrix_generator.md);
  - data where the right adaptation is prompt-specific (thinking mode, other sources);
  - seeds.

## Limits
- One seed per arm; the test difference is borderline.
- Offline, greedy, teacher-forced τ; no serving measurement. The generator's per-request cost (4.3 ms generation + merge) is not included.
- The 24K and 100K points differ in both data and epochs.
- The base drafter likely saw prompts from the same Nemotron V2 source.

## Artifacts
- **Recipes:** `examples/dflash-lora/mix100k/`; matrices `examples/dflash-lora/matrices/mix100k_*.yaml`.
- **Runs:** `/workspace/yusuf/runs/dflash/mix100k/<run>/` (provenance: `train_command.txt`, `speculators.patch`, `run_record_*.json`).
- **HF (public):** `erdem-erdem/dyn-spec-dflash-mix100k-static-lora-attn-r512-3ep-kl` @ bedb1f51, `erdem-erdem/dyn-spec-dflash-mix100k-generator-attn-r64-3ep-kl-pt256` @ b4270ecc.
- **W&B:** project `dyn-spec-dflash-v1`, runs `dflash-mix100k-*-gseed42`.
- **Page builder:** `scripts/dflash_port/mix100k_page/build.py` (results JSON from `scripts/dflash_port/abl24k_collect.py`).
