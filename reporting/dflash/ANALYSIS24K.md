# Where DFlash moves: error, gradient, prompt, knock-out and representation analysis of the 24K checkpoints

- **Status:** done (2026-11-11). Analysis only: no training, no change to any run, store or library code.
- **Question (user 2026-11-11):** every ablation moves the DFlash drafter by tiny amounts (best static LoRA ≈ +0.2 τ over base; generator − static +0.022 τ on 24K). Where, and how much, can the drafter be influenced, before designing new generators?
- **Models:** base drafter `z-lab/Qwen3-8B-DFlash-b16@9b41424b`; 24K static LoRA (KL) `mix24k-static-lora-attn-r32-3ep-kl`; 24K generator (KL) `mix24k-generator-attn-r32-3ep-kl` ([MIX24K.md](MIX24K.md)). Attention q/k/v/o, rank 32, coefficient 2.
- **Interactive page:** https://claude.ai/artifact/8PFHPXYS74DCCMv4QX2x2R ("Where DFlash Moves"; private until shared)
- **Code:** `scripts/dflash_port/analysis/` (harness `common.py`; tests `tests/unit/scripts/test_dflash_analysis24k.py`). **Raw results:** `/workspace/yusuf/runs/dflash/analysis24k/` (`slots/`, `grads/`, `prompt/`, `knockouts/`, `hidden/`, `results/*.json`). **Figures:** [analysis24k/](analysis24k/).

## TL;DR

Measurements first; interpretation in italics. Offline, teacher-forced, greedy, one seed; τ pooled over blocks unless noted.

- **How much of the remaining error looks fixable.** 82.9% of the base drafter's first misses are near-misses (the target's token is in the drafter's top 5) and 84.4% happen while the data is still on the target's greedy path. Fixing just the on-path, target-confident (p ≥ 0.5) near-misses (53.2% of misses) would add up to 0.90 τ; the strict version (p ≥ 0.9; 24.3% of misses) 0.42 τ; every first miss 1.45 τ. The static LoRA gains +0.20 τ, almost all from target-confident near-misses, while shortening 6.5% of blocks (it lengthens 11.0%). Target-uncertain (p < 0.5) misses are 27.6% and off-path misses 15.6%. *The error left is mostly of a kind a small correction could flip; the adapters reach a fraction of it and churn blocks in both directions.*
- **Where a rank-32 LoRA could act, where categories disagree.** At the base drafter the first-order rank-32 gain is largest in `fc` (75 ×10⁻³) and the K/V paths (context 31, block 29; q/o 18; MLP 16). Categories agree in the K/V paths and deep q (noise-corrected cosine 0.71 context, 0.81 block) and disagree most in the MLPs (0.48) and o_proj. Held out, a shared rank-32 direction reaches 0.98 / 0.98 / 0.99 / 0.98 (context / block / q,o / MLP) of a category-specific one's first-order gain. At the static-adapted point no module has a reproducible mean gradient left (split-half cosine ≤ 0.11, median 0.03; `fc` 0.09, vs up to 0.98 at the base). *Categories' gradients differ most in the MLP and o_proj, but at this data scale (500 rows per category half) a category-specific rank-32 step is barely better on held-out rows than the shared one, and after the static LoRA no reproducible direction is left. At first order there is little room for category-level adaptation.*
- **Is the prompt redundant for the drafter?** The drafter reads the prompt directly: removing it from the injected context costs -1.08 τ (category mean), about as much as removing as many random response tokens (-1.03). Category is linearly decodable from every drafter layer (≥ 0.96). The generator, which still sees the prompt summary, does not make up for a prompt-less context (generator − static +0.013 without the prompt vs +0.021 with it). *Prompt-level information is largely redundant with what the drafter already attends to; the generator's gain is a small category-level adjustment (swapping in another category's prompt moves its effect on the L4 states by 0.45 of its size, a same-category prompt by 0.21).*
- **Sensitivity.** The drafter is flat: a random rank-32 update with the static LoRA's norm changes τ by at most 0.003 on any attention or MLP module (0.008 at 4×); only `fc` reacts (-0.074 at 4×). Knocking out the context path in layer 0 alone costs -5.91 τ (all layers -6.27). The 20 single-module static transplants sum to +0.22 τ. *Gains come from a few trained directions spread thinly over many modules; no single module holds them.*
- **For the generator design** (see [Implications](#implications-for-the-generator-design)): prompt-summary conditioning (C1–C3) has little new information to add; block- or response-conditioned adaptation (G6, C4) and objectives aimed at the confident near-miss first misses without breaking accepted blocks (D3) target what is left; per-category controls (A1) should include the MLP / o_proj, where categories disagree most, though the held-out gradient test predicts little gain; cheap low-dimensional generators (G1, G3) match the category-level structure that the 24K generator actually uses.

## Method common to every step

- **Forward pass:** the production one. One `Runtime` built from the 24K generator recipe through `campaign.run_one(..., evaluate=)` (read-only; the run identity is checked against the trained run), adapters on q/k/v/o of all 5 layers, `Runtime.forward` → `functional_call` with per-arm factors: zero factors (base; bit-identical to the released drafter), the static checkpoint's factors, or the factors the generator makes from each row's own prompt. The drafter's `forward` is replaced by the upstream `_backbone_forward` + `compute_metrics` plus a capture of logits, targets, mask and block indices. Validation anchors (content-seeded, density 1/16), eval mode, bf16 autocast, FlexAttention.
- **Reproduction check (passed).** Category-mean full_acc, validation (500 rows): base 0.526526 (recorded 0.526529; the record is the math10k base run without adapters), static 0.533787 (recorded 0.533787), generator 0.534689 (recorded 0.534689). Test: base 0.531536 (0.531532), static 0.538721 (0.538721), generator 0.539586 (0.539586). Category-mean τ, validation: base 7.223, static 7.403, generator 7.424. A first attempt did not reproduce the LoRA arms: newly installed adapters start in training mode (LoRA dropout on), and production validation calls `drafter.eval()` first; the harness now does too. The block-level τ of step 1 equals the metric counts per subset (max |Δ| 0).
- **τ in this report:** per-block accepted run + 1 (bonus), **pooled over blocks** unless marked "category mean" (the evaluation headline is the category mean; pooled τ = 1 + accepted draft tokens per block, e.g. base test 8.213 = 721 per 100 blocks + 1).

## Step 1: error decomposition (is the remaining error fixable?)

**Measured.** For every block of the test split (49,021 blocks, 500 rows; validation in `results/errors_validation.json`, same picture) and each arm: the first slot that fails; at that slot the target's top-1 probability `p_top` and entropy, the rank the drafter gave the target's top-1 token, the drafter's probability of it, and whether the data stayed on the target's greedy path inside the block before the miss (**on-path**: every earlier slot's data token equals the target's argmax label). The labels are the target's argmax along the sampled response; after a sampled non-argmax token, the label follows a token the drafter cannot know (**off-path** miss). Rank 1 at a miss is an exact bf16 logit tie that argmax broke the other way. The **fix gain** of a miss is the τ gained if only that first miss were corrected (the block then runs to its next failing slot). Code: `collect_slots.py`, `analyze_slots.py`.

**Results (test, 49,021 blocks; pooled τ).** Shares of misses / τ a perfect fix of that class would add:

| arm | τ | full blocks | blocks ending in a miss | truncated | fix every first miss (Δτ) | misses on-path | addressable (p≥0.5, rank≤5, on-path) | strict (p≥0.9, rank≤5, on-path) | target uncertain (p<0.5) | off-path | bf16 ties (rank 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 8.213 | 24.2% | 75.1% | 0.8% | 1.446 | 84.4% | 53.2% / 0.900 | 24.3% / 0.422 | 27.6% / 0.341 | 15.6% / 0.159 | 4.3% / 0.081 |
| static | 8.409 | 25.9% | 73.3% | 0.8% | 1.429 | 83.2% | 52.0% / 0.869 | 22.6% / 0.387 | 28.3% / 0.354 | 16.8% / 0.167 | 3.7% / 0.070 |
| generator | 8.430 | 26.1% | 73.1% | 0.8% | 1.432 | 82.9% | 51.6% / 0.863 | 22.5% / 0.384 | 28.5% / 0.361 | 17.1% / 0.169 | 3.5% / 0.066 |

Base drafter, first misses by target confidence × drafter rank of the target's token:

| target confidence | drafter rank | share of misses | Δτ if fixed | on-path share |
|---|---|---|---|---|
| p_top >=0.9 | rank 1-5 | 29.4% | 0.478 | 82.7% |
| p_top >=0.9 | rank >5 | 3.8% | 0.045 | 71.7% |
| p_top 0.5-0.9 | rank 1-5 | 33.0% | 0.522 | 87.5% |
| p_top 0.5-0.9 | rank >5 | 6.1% | 0.061 | 72.7% |
| p_top <0.5 | rank 1-5 | 20.5% | 0.271 | 90.7% |
| p_top <0.5 | rank >5 | 7.1% | 0.071 | 76.2% |

Rank of the target's token at the first miss (base): 1 (bf16 tie) 4.3%, 2 51.2%, 3–5 27.4%, 6–20 12.7%, >20 4.4%. Medians at the miss: target top-1 probability 0.688, target entropy 0.733 nats, drafter probability of the target's token 0.132, drafter's own top-1 probability 0.400, target probability of the drafter's pick 0.0019; the drafter's pick is the target's second choice in 34.5% of misses.

By category (base, test):

| category | τ | blocks ending in a miss | fix every first miss | addressable | strict | target uncertain | off-path |
|---|---|---|---|---|---|---|---|
| math | 10.022 | 64.5% | 1.369 | 56.2% / 0.903 | 28.3% / 0.462 | 25.7% / 0.294 | 18.3% / 0.162 |
| code | 7.590 | 81.1% | 1.539 | 50.5% / 0.912 | 20.4% / 0.370 | 30.6% / 0.408 | 18.9% / 0.209 |
| stem | 6.728 | 83.1% | 1.482 | 53.2% / 0.915 | 19.0% / 0.352 | 27.7% / 0.348 | 13.0% / 0.144 |
| chat | 4.666 | 93.2% | 1.533 | 50.1% / 0.865 | 23.7% / 0.419 | 28.2% / 0.392 | 8.0% / 0.095 |

Slot level (every valid slot, base): 72.7% of slots have target p_top ≥ 0.9 and the drafter matches 70.3% of them; 0.5–0.9: 19.5% of slots, 34.2% matched; < 0.5: 7.8%, 17.9%. On-path accuracy falls with depth: slot 1 90.9%, slot 8 71.0%, slot 15 58.7% (the data is still on the greedy path at slot 15 in 55.7% of blocks). Figures: [block outcomes](analysis24k/block_outcomes.png), [first-miss slot](analysis24k/first_miss_slot.png).

**What the LoRAs fix** (same blocks and anchors in every arm; a block is *fixed* when the adapted arm's first failure comes later, *broken* when earlier):

| base block class | share of blocks | static: fixed / broken | static Δτ contribution | generator: fixed / broken | generator Δτ contribution | generator − static, Δτ contribution (classes of the static arm) |
|---|---|---|---|---|---|---|
| full blocks | 24.2% | 0.0% / 4.0% | -0.039 | 0.0% / 5.8% | -0.063 | -0.071 |
| miss p≥0.9, rank≤5 | 22.1% | 25.7% / 10.2% | +0.150 | 29.8% / 13.8% | +0.167 | +0.047 |
| miss p 0.5–0.9, rank≤5 | 24.8% | 14.0% / 5.7% | +0.081 | 18.5% / 7.5% | +0.111 | +0.055 |
| miss p<0.5, rank≤5 | 15.4% | 11.4% / 4.0% | +0.025 | 14.6% / 5.0% | +0.034 | +0.019 |
| miss p≥0.9, rank>5 | 2.9% | 2.5% / 11.3% | -0.006 | 3.9% / 15.8% | -0.007 | -0.008 |
| miss p 0.5–0.9, rank>5 | 4.6% | 0.5% / 9.6% | -0.008 | 0.8% / 13.8% | -0.013 | -0.010 |
| miss p<0.5, rank>5 | 5.3% | 0.2% / 8.0% | -0.008 | 0.3% / 11.3% | -0.012 | -0.009 |
| miss on-path (any) | 63.4% | 16.4% / 6.3% | +0.250 | 20.4% / 8.5% | +0.305 | +0.117 |
| miss off-path (any) | 11.7% | 5.3% / 12.8% | -0.015 | 5.8% / 17.0% | -0.025 | -0.025 |

Totals: static − base Δτ +0.196 (11.0% of blocks longer, 6.5% shorter); generator − base +0.217 (13.6% / 8.8%); generator − static +0.021 (8.3% / 7.7%). Figure: [contributions](analysis24k/transition_contrib.png).

**Interpretation.**
- *Most first misses look like near-misses, not ignorance.* In 82.9% of base misses the drafter has the target's token in its top 5, and 84.4% of misses happen while the data is still on the target's greedy path. The target is confident (p_top ≥ 0.5) in 72.4% of them.
- *Addressable share.* Our working definition (on-path, target p_top ≥ 0.5, drafter rank ≤ 5) covers 53.2% of base misses; fixing only those first misses would add 0.90 τ (pooled), against 1.45 for fixing every first miss and +0.20 actually gained by the static LoRA. The strict version (p_top ≥ 0.9) is 24.3% of misses and 0.42 τ. These are first-order ceilings with later misses unchanged; they say the drafter's errors are mostly of a kind a small correction could flip, not that a LoRA can reach them.
- *Less fixable part.* Target-uncertain misses (p_top < 0.5) are 27.6% of misses (0.34 τ); off-path misses (a sampled non-argmax token earlier in the block) 15.6% (0.16 τ). Neither can be fixed by better prediction of the target's argmax from the context alone. bf16 logit ties are 4.3% of misses (0.08 τ), an artefact of greedy matching on bf16 logits.
- *What the LoRAs fix.* Both adapters gain almost entirely on target-confident near-misses (p ≥ 0.9, rank ≤ 5: static +0.150 τ, generator +0.167) and lose a little on blocks the base already accepted in full (static -0.039, generator -0.063) and on far misses. The changes are churn: the static lengthens 11.0% of blocks and shortens 6.5%. The generator moves more blocks in both directions than the static; its net +0.02 over static is the difference of 8.3% improved and 7.7% worsened blocks.

## Step 2: gradient maps (where could a rank-32 LoRA help, where do categories disagree?)

**Measured.** The training loss (KL to the target distribution + D-PACE position weights; the 24K recipe's `block_drafter` settings) back-propagated row by row through 1,000 training rows per category (4,000 rows; seeded sample from the 24K pool r0002-100k subsets 00000..00005; never evaluation rows; validation anchors, eval mode), at the base drafter and at the static-adapted point (trained static factors installed). Gradients w.r.t. `fc` and every q/k/v/o and gate/up/down of the 5 layers; K and V split by path by giving the context path and the block path separate copies of the weight (their sum is the ordinary gradient). Per module and category, on the category's mean gradient G: the **rank-32 gain** ‖P₃₂(G)‖_F (the first-order loss decrease per unit Frobenius step of the best rank-32 update), the **top-32 share** of the gradient energy, the **split-half cosine** (two disjoint halves of 500 rows: how reproducible G is), the **noise-corrected cross-category cosine** (mean cosine between half-sample gradients of two categories divided by √ of their split-half cosines), and the **held-out capture**: the first-order gain of the best *shared* rank-32 direction as a share of the best *category-specific* rank-32 direction, both fitted on one half and scored on the other (1 = a per-category adapter gains nothing over a static one at first order). Code: `grad_maps.py`. Figures: [base](analysis24k/grad_maps_base.png), [static point](analysis24k/grad_maps_static.png).

Mean training loss per row at the base: math 0.346, code 0.391, stem 0.418, chat 0.503; at the static point: math 0.350, code 0.403, stem 0.434, chat 0.521.

Rank-32 gain at the base drafter (mean over categories; ×10⁻³):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1.23 | 2.64 | 1.36 | 18.04 | 4.40 | 7.53 | 2.14 | 1.80 | 2.33 |
| L1 | 4.21 | 2.65 | 2.47 | 13.57 | 5.74 | 4.53 | 3.61 | 3.44 | 2.83 |
| L2 | 8.52 | 3.60 | 6.51 | 13.02 | 10.28 | 4.49 | 3.83 | 4.63 | 3.43 |
| L3 | 7.89 | 5.87 | 10.63 | 9.39 | 11.42 | 3.54 | 3.82 | 4.84 | 4.07 |
| L4 | 7.83 | 7.62 | 13.33 | 7.43 | 13.67 | 3.48 | 6.00 | 6.43 | 5.82 |

`fc`: 75.3 ×10⁻³ (one 4096 × 20,480 matrix)

Grouped (root of summed squared gains over layers): attention q/o 18.4, K/V context path 30.7, K/V block path 28.6, MLP 16.1, fc 75.3 (×10⁻³). At the static point: attention q/o 7.7, K/V context path 10.1, K/V block path 5.3, MLP 7.5, fc 19.7.

Top-32 share of the gradient energy (base):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.63 | 0.82 | 0.98 | 0.93 | 0.96 | 0.74 | 0.53 | 0.57 | 0.42 |
| L1 | 0.88 | 0.70 | 0.89 | 0.94 | 0.95 | 0.62 | 0.63 | 0.72 | 0.50 |
| L2 | 0.95 | 0.80 | 0.98 | 0.96 | 0.98 | 0.71 | 0.62 | 0.78 | 0.57 |
| L3 | 0.96 | 0.93 | 0.99 | 0.97 | 0.99 | 0.74 | 0.63 | 0.77 | 0.64 |
| L4 | 0.97 | 0.97 | 1.00 | 0.98 | 1.00 | 0.83 | 0.78 | 0.83 | 0.78 |

`fc`: 0.99

Split-half cosine (base; reliability of the 500-row mean gradient):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.17 | 0.37 | 0.57 | 0.71 | 0.65 | 0.42 | 0.14 | 0.20 | 0.20 |
| L1 | 0.48 | 0.33 | 0.56 | 0.78 | 0.79 | 0.36 | 0.30 | 0.44 | 0.31 |
| L2 | 0.76 | 0.55 | 0.88 | 0.87 | 0.90 | 0.49 | 0.38 | 0.60 | 0.39 |
| L3 | 0.82 | 0.81 | 0.96 | 0.89 | 0.96 | 0.54 | 0.42 | 0.61 | 0.46 |
| L4 | 0.86 | 0.89 | 0.98 | 0.90 | 0.98 | 0.64 | 0.58 | 0.64 | 0.55 |

`fc`: 0.89

Noise-corrected cross-category cosine (base; mean of the 6 category pairs; values above 1 are noise):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.60 | 0.62 | 0.81 | 0.69 | 0.71 | 0.53 | 0.37 | 0.41 | 0.43 |
| L1 | 0.57 | 0.48 | 0.64 | 0.65 | 0.74 | 0.42 | 0.46 | 0.51 | 0.47 |
| L2 | 0.74 | 0.64 | 0.80 | 0.71 | 0.79 | 0.49 | 0.49 | 0.61 | 0.44 |
| L3 | 0.83 | 0.86 | 0.89 | 0.77 | 0.87 | 0.56 | 0.53 | 0.60 | 0.46 |
| L4 | 0.83 | 0.85 | 0.89 | 0.86 | 0.91 | 0.61 | 0.44 | 0.50 | 0.44 |

`fc`: 0.72

Held-out capture of the shared rank-32 direction (base; mean over categories):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1.21 | 1.13 | 1.12 | 0.96 | 1.01 | 0.97 | 1.03 | 1.01 | 1.05 |
| L1 | 0.99 | 1.01 | 0.98 | 0.92 | 0.96 | 0.92 | 1.00 | 0.94 | 1.02 |
| L2 | 0.97 | 0.98 | 0.96 | 0.91 | 0.95 | 0.93 | 0.99 | 0.95 | 0.97 |
| L3 | 0.99 | 1.00 | 0.97 | 0.94 | 0.96 | 0.98 | 1.07 | 0.97 | 0.98 |
| L4 | 0.98 | 0.97 | 0.96 | 0.98 | 0.97 | 0.96 | 0.87 | 0.89 | 0.90 |

`fc`: 0.92

Per category, mean over the 45 modules: math 0.88, code 0.94, stem 0.90, chat 1.21.

At the static point, rank-32 gain (×10⁻³; includes the noise floor of a 1,000-row mean):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 1.30 | 2.12 | 0.90 | 7.41 | 2.45 | 4.34 | 1.82 | 1.43 | 1.52 |
| L1 | 2.70 | 1.89 | 1.47 | 4.36 | 1.97 | 2.60 | 2.38 | 1.80 | 1.50 |
| L2 | 3.31 | 1.74 | 1.82 | 3.01 | 2.24 | 2.03 | 2.01 | 1.65 | 1.57 |
| L3 | 2.18 | 1.31 | 1.52 | 1.72 | 1.44 | 1.29 | 1.75 | 1.51 | 1.63 |
| L4 | 1.66 | 1.14 | 1.04 | 1.22 | 1.13 | 1.06 | 2.50 | 2.37 | 2.79 |

`fc`: 19.7 ×10⁻³

Split-half cosine at the static point (reliability; compare with the base table above):

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o | gate | up | down |
|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.016 | 0.028 | 0.060 | 0.055 | 0.031 | 0.025 | 0.009 | 0.011 | 0.013 |
| L1 | 0.030 | 0.016 | 0.034 | 0.043 | 0.027 | 0.017 | 0.014 | 0.020 | 0.019 |
| L2 | 0.048 | 0.023 | 0.090 | 0.054 | 0.043 | 0.022 | 0.018 | 0.026 | 0.030 |
| L3 | 0.026 | 0.014 | 0.055 | 0.071 | 0.072 | 0.028 | 0.024 | 0.036 | 0.038 |
| L4 | 0.028 | 0.023 | 0.073 | 0.099 | 0.103 | 0.056 | 0.097 | 0.112 | 0.097 |

`fc`: 0.094

At the static point the mean gradients are not reproducible between halves, so the gains above are mostly the noise floor, and cross-category cosines and held-out captures are not interpretable there (not shown; they are in `grads/static/summary.json`).

Cosine between the trained static ΔW and −G (base, all categories pooled), attention modules:

| layer | q | k_ctx | k_blk | v_ctx | v_blk | o |
|---|---|---|---|---|---|---|
| L0 | 0.132 | 0.083 | 0.046 | 0.053 | 0.022 | 0.100 |
| L1 | 0.057 | 0.078 | 0.039 | 0.061 | 0.038 | 0.169 |
| L2 | 0.041 | 0.029 | 0.041 | 0.033 | 0.048 | 0.184 |
| L3 | 0.043 | 0.013 | 0.050 | 0.003 | 0.058 | 0.146 |
| L4 | 0.042 | 0.002 | 0.044 | -0.007 | 0.071 | 0.169 |

**Interpretation.**
- *Where a rank-32 step has first-order leverage:* `fc` has the largest single gain (75.3 ×10⁻³), then the K/V paths: the context V of the first layers and the block K/V of the deep layers. Grouped, the K/V paths (context 30.7, block 28.6) exceed q/o (18.4) and the MLPs (16.1). The static LoRA (q/k/v/o) can act on all but `fc` and the MLPs; it uses one K/V weight for both paths.
- *Rank 32 is enough for attention, not for the MLPs.* In the L3–L4 q and K/V modules (and the block path from L0) the top-32 singular directions hold ≥ 0.89 of the gradient energy; in the MLPs and early o/q the share is 0.4–0.8, so these gradients are spread and noisier (split-half cosines 0.14–0.64).
- *Conflict map:* categories agree most in the K/V paths and deep q (noise-corrected cosine: context path 0.71, block path 0.81, q/o 0.62) and least in the MLPs (0.48) and o_proj. The held-out test tempers this: a shared rank-32 direction fitted on all categories reaches 0.98 (context path), 0.98 (block path), 0.99 (q/o) and 0.98 (MLP) of the held-out first-order gain of a category-specific direction fitted on 500 rows of that category; the lowest values are in the L4 MLP (≈ 0.87–0.90) and `fc` (0.92). Category disagreement exists, but with this much data a per-category rank-32 step would add at most ≈ 10% first-order gain over the shared one, mostly in modules the current generator does not adapt (L4 MLP, `fc`).
- *After static training* the grouped gains drop (q/o 18.4 → 7.7, context path 30.7 → 10.1, block path 28.6 → 5.3, MLP 16.1 → 7.5, `fc` 75.3 → 19.7 ×10⁻³), and what is left is not reproducible between halves of the same category (split-half cosine ≤ 0.11 in every module, `fc` 0.09). The attention-only static LoRA also removed the consistent first-order signal in the MLPs and `fc`, which it does not adapt: there is no shared or per-category direction left at first order, only per-row variation.
- The trained static ΔW is nearly orthogonal to the base gradient (cosine 0.00–0.18; highest for o_proj). The static LoRA does not simply follow the initial first-order direction; first-order gains are local and do not say how far a direction can be followed.

## Step 3: prompt usage (is the generator's prompt information redundant?)

**Measured.** (a) Attention weights of draft-slot queries (slots 1–15 of valid blocks) on 200 validation rows (50 per category, rows ≤ 4,096 tokens so the dense weights fit), for base, static and generator. FlexAttention returns no weights, so this pass switches the drafter to the eager path (`eager_attention_forward` with the float mask of the same mask mod). Equivalence on 5 rows per arm: eager vs flex argmax agreement 0.982–0.996, mean |logit difference| 0.025, τ per row within 0.08 (bf16 kernel differences). Mass is split into: prompt tokens (positions before the first supervised token; the first 4 of them separately, as attention-sink candidates), response tokens of the injected context, the block's anchor token, and its mask tokens; plus the 32 most recent context tokens. (b) τ on all 500 validation rows with the prompt removed from the injected context (an extra FlexAttention mask term: block queries never attend to prompt positions), the same keeping the first 4 prompt tokens (attention-sink control), and two controls removing n = min(prompt length, response length / 2) response tokens: random response positions, and the earliest response positions. Median prompt length 138 tokens, median response 978; n equals the prompt length in 85.4% of rows. The generator still receives its prompt summary in every condition (only the drafter's context is masked). Code: `prompt_usage.py`. Figures: [attention](analysis24k/attention.png), [masking](analysis24k/masking.png).

(a) Mean attention mass per draft query (mean over heads):

| arm | layer | prompt | first 4 prompt tokens | rest of prompt | prompt if uniform | response context | 32 most recent | anchor | block masks |
|---|---|---|---|---|---|---|---|---|---|
| base | L0 | 24.6% | 10.2% | 14.4% | 37.4% | 58.9% | 44.8% | 6.8% | 9.7% |
| base | L1 | 11.5% | 2.0% | 9.6% | 37.4% | 50.7% | 22.7% | 3.0% | 34.8% |
| base | L2 | 15.8% | 2.4% | 13.5% | 37.4% | 51.3% | 19.3% | 1.5% | 31.4% |
| base | L3 | 23.0% | 4.5% | 18.6% | 37.4% | 54.0% | 18.3% | 1.4% | 21.6% |
| base | L4 | 21.4% | 4.1% | 17.3% | 37.4% | 54.9% | 19.1% | 1.6% | 22.2% |
| static | L0 | 24.7% | 10.3% | 14.5% | 37.4% | 56.7% | 43.4% | 7.2% | 11.3% |
| static | L1 | 11.9% | 2.2% | 9.8% | 37.4% | 48.7% | 22.0% | 3.1% | 36.3% |
| static | L2 | 15.9% | 2.6% | 13.3% | 37.4% | 50.2% | 18.9% | 1.5% | 32.4% |
| static | L3 | 23.6% | 5.3% | 18.3% | 37.4% | 52.9% | 18.2% | 1.4% | 22.1% |
| static | L4 | 21.8% | 4.8% | 17.0% | 37.4% | 53.8% | 19.1% | 1.4% | 23.0% |
| generator | L0 | 25.4% | 10.9% | 14.6% | 37.4% | 56.8% | 43.3% | 7.0% | 10.7% |
| generator | L1 | 11.8% | 2.2% | 9.6% | 37.4% | 48.6% | 22.0% | 3.1% | 36.6% |
| generator | L2 | 15.8% | 2.6% | 13.2% | 37.4% | 50.0% | 18.9% | 1.5% | 32.7% |
| generator | L3 | 23.1% | 5.0% | 18.1% | 37.4% | 53.0% | 18.2% | 1.4% | 22.5% |
| generator | L4 | 21.7% | 4.7% | 17.0% | 37.4% | 54.0% | 19.2% | 1.5% | 22.9% |

Base drafter by category, prompt mass per layer (in brackets: without the first 4 tokens):

| category | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| math | 20.0% (10.2%) | 7.2% (5.4%) | 9.2% (7.0%) | 14.5% (9.9%) | 12.1% (8.7%) |
| code | 25.4% (16.5%) | 12.7% (11.1%) | 18.1% (16.1%) | 25.2% (21.3%) | 22.8% (19.8%) |
| stem | 28.3% (15.9%) | 16.9% (14.5%) | 23.0% (19.9%) | 33.1% (28.6%) | 31.7% (26.4%) |
| chat | 27.6% (16.7%) | 11.7% (9.4%) | 16.3% (13.5%) | 24.0% (18.8%) | 24.5% (18.6%) |

Heads with the most prompt mass outside the first 4 tokens (base): L0 h18 33.2%, L4 h3 28.8%, L4 h2 28.1%, L3 h30 27.7%, L0 h27 27.4%, L0 h2 26.8%; 41 of 160 heads put more than 20% there.

(b) Category-mean τ with the full context, and Δτ when part of the context is removed (500 validation rows; 95% paired bootstrap):

| arm | τ (full context) | prompt removed | prompt removed except its first 4 tokens | random response tokens removed | earliest response tokens removed |
|---|---|---|---|---|---|
| base | 7.225 | -1.082 [-1.166, -1.007] | -0.962 [-1.043, -0.887] | -1.029 [-1.140, -0.933] | -1.422 [-1.563, -1.288] |
| static | 7.405 | -1.096 [-1.179, -1.020] | -0.983 [-1.067, -0.906] | -1.048 [-1.157, -0.947] | -1.486 [-1.628, -1.352] |
| generator | 7.426 | -1.105 [-1.189, -1.029] | -0.992 [-1.074, -0.917] | -1.064 [-1.176, -0.963] | -1.489 [-1.631, -1.356] |

Δτ with the prompt removed, by category:

| arm | math | code | stem | chat |
|---|---|---|---|---|
| base | -0.369 | -0.885 | -2.564 | -0.509 |
| static | -0.366 | -0.904 | -2.591 | -0.522 |
| generator | -0.377 | -0.907 | -2.624 | -0.511 |

Generator − static Δτ under each condition: none +0.021 [+0.006, +0.037]; prompt +0.013 [-0.002, +0.026]; resp_random +0.005 [-0.010, +0.020]; resp_early +0.018 [+0.005, +0.031].

**Interpretation.**
- *The drafter reads the prompt, but less than its share of the context.* Draft queries put 9.6%–18.6% of their attention on prompt tokens beyond the first 4 (plus 2.0%–10.2% on the first 4), against 37.4% if attention were uniform over visible tokens; about half goes to response context and, from L1 on, 22–35% to the block's own mask tokens. The adapters hardly change this distribution (all three arms within ≈ 0.02 per field).
- *Removing the prompt from the context costs as much as removing as many response tokens:* base -1.08 τ [-1.17, -1.01] vs -1.03 [-1.14, -0.93] for random response tokens and -1.42 for the earliest response tokens. Keeping the first 4 prompt tokens (attention-sink control) changes this to -0.96 [-1.04, -0.89]: the sink tokens account for 0.12 τ, the rest is prompt content. It hurts most in stem (-2.56) and least in math (-0.37). The prompt is therefore information the drafter already obtains directly through the injected target features.
- *The generator does not substitute for the prompt.* With the prompt removed from the drafter's context, the generator (which still conditions on the prompt summary) loses as much as the static LoRA: generator − static is +0.013 [-0.002, +0.026] without the prompt vs +0.021 [+0.006, +0.037] with it, and +0.005 [-0.010, +0.020] with random response tokens removed. Its prompt-derived LoRA does not carry the content the drafter lost; its small edge needs the full context, as an adjustment on top of a drafter that reads the prompt anyway.

## Step 4: knock-outs and sensitivity

**Measured.** 200 validation rows (50 per category, seeded), cached on the GPU; every variant is one production pass with forward hooks; Δτ is the category-mean τ difference against the same arm unmodified, with 95% paired bootstrap intervals over rows (stratified, 2,000 resamples). Knock-outs: one attention head (its slice of the o_proj input zeroed), a layer's attention output, a layer's MLP output, and K and V of one path (zeros) in one layer or all layers. Arms base, static, generator (heads: base and static). Sensitivity (base): a random rank-32 ΔW (Gaussian factors) on one module, Frobenius norm equal to the trained static ΔW of that module (MLP and `fc`: the static's mean relative norm ‖ΔW‖/‖W‖ = 0.0270 times ‖W‖), at 1× and 4×, 3 seeds; and the trained static ΔW of a single module alone (transplant). Code: `knockouts.py`, `analyze_maps.py`.

Reference τ (category mean, 200 rows): base 7.292, static 7.461, generator 7.494.

| knocked out | arm | L0 | L1 | L2 | L3 | L4 | all layers |
|---|---|---|---|---|---|---|---|
| attention output | base | -6.07 | -1.95 | -1.91 | -2.07 | -2.51 | – |
| attention output | static | -6.24 | -1.95 | -1.85 | -1.89 | -2.37 | – |
| attention output | generator | -6.27 | -2.00 | -1.85 | -1.94 | -2.44 | – |
| MLP output | base | -2.49 | -0.65 | -0.89 | -1.43 | -1.34 | – |
| MLP output | static | -2.30 | -0.64 | -0.83 | -1.05 | -1.26 | – |
| MLP output | generator | -2.39 | -0.67 | -0.88 | -1.14 | -1.30 | – |
| K/V context path | base | -5.91 | -1.98 | -1.99 | -1.61 | -1.24 | -6.27 |
| K/V context path | static | -6.08 | -2.00 | -1.79 | -1.47 | -1.23 | -6.44 |
| K/V context path | generator | -6.10 | -1.99 | -1.87 | -1.48 | -1.24 | -6.48 |
| K/V block path | base | -0.90 | -0.41 | -1.48 | -0.66 | -1.47 | -5.32 |
| K/V block path | static | -0.83 | -0.40 | -1.48 | -0.62 | -1.29 | -5.46 |
| K/V block path | generator | -0.82 | -0.41 | -1.45 | -0.63 | -1.30 | -5.49 |

(Intervals are tight; e.g. base attention L0 [-6.47, -5.67], MLP L1 [-0.71, -0.60], block path L1 [-0.44, -0.37].)

Heads (base): mean Δτ -0.035; 31 of 160 heads cost more than 0.05 τ, 5 more than 0.1; 2 heads improve τ when removed. Largest: L4 h11 -0.295, L4 h8 -0.258, L4 h10 -0.157, L1 h28 -0.144, L1 h30 -0.125. Per layer, mean Δτ: L0 -0.037, L1 -0.031, L2 -0.034, L3 -0.026, L4 -0.047.

Heads (static): mean Δτ -0.023; 18 of 160 heads cost more than 0.05 τ, 4 more than 0.1; 12 heads improve τ when removed. Largest: L4 h11 -0.261, L4 h8 -0.180, L1 h28 -0.116, L1 h30 -0.115, L4 h10 -0.097. Per layer, mean Δτ: L0 -0.024, L1 -0.021, L2 -0.022, L3 -0.015, L4 -0.032.

Sensitivity (base, τ 7.292); random: mean [min, max] of 3 seeds; transplant: Δτ [95% CI]:

| module | ‖ΔW‖_F | random rank-32, 1× | random rank-32, 4× | trained static ΔW alone |
|---|---|---|---|---|
| fc | 47.07 | -0.0086 [-0.0151, +0.0014] | -0.0736 [-0.0909, -0.0548] | – |
| l0_q | 18.83 | -0.0008 [-0.0030, +0.0019] | -0.0017 [-0.0058, +0.0009] | +0.0153 [+0.002, +0.029] |
| l0_k | 4.74 | +0.0014 [-0.0005, +0.0040] | -0.0011 [-0.0054, +0.0013] | +0.0069 [-0.003, +0.017] |
| l0_v | 3.73 | -0.0003 [-0.0014, +0.0003] | -0.0006 [-0.0037, +0.0032] | +0.0112 [+0.001, +0.021] |
| l0_o | 7.61 | -0.0016 [-0.0031, -0.0005] | -0.0027 [-0.0058, +0.0004] | +0.0059 [-0.005, +0.016] |
| l0_gate | 22.03 | +0.0011 [-0.0017, +0.0032] | -0.0049 [-0.0065, -0.0023] | – |
| l0_up | 23.06 | +0.0026 [+0.0004, +0.0054] | +0.0002 [-0.0016, +0.0024] | – |
| l0_down | 23.03 | +0.0012 [+0.0004, +0.0024] | -0.0009 [-0.0012, -0.0006] | – |
| l1_q | 15.41 | -0.0001 [-0.0021, +0.0014] | -0.0007 [-0.0035, +0.0043] | +0.0106 [+0.001, +0.020] |
| l1_k | 5.05 | +0.0005 [-0.0032, +0.0050] | -0.0001 [-0.0025, +0.0039] | +0.0034 [-0.003, +0.010] |
| l1_v | 4.39 | -0.0009 [-0.0022, +0.0008] | -0.0028 [-0.0061, +0.0017] | +0.0129 [+0.005, +0.021] |
| l1_o | 11.18 | +0.0004 [-0.0018, +0.0034] | -0.0048 [-0.0120, +0.0016] | +0.0293 [+0.017, +0.041] |
| l1_gate | 22.33 | -0.0007 [-0.0032, +0.0011] | -0.0018 [-0.0042, -0.0005] | – |
| l1_up | 23.72 | +0.0005 [-0.0018, +0.0022] | -0.0009 [-0.0038, +0.0010] | – |
| l1_down | 23.54 | +0.0012 [-0.0011, +0.0039] | -0.0025 [-0.0049, +0.0002] | – |
| l2_q | 16.22 | -0.0005 [-0.0018, +0.0016] | -0.0047 [-0.0092, +0.0006] | +0.0243 [+0.016, +0.033] |
| l2_k | 5.81 | +0.0005 [-0.0014, +0.0035] | +0.0011 [-0.0059, +0.0060] | +0.0077 [+0.000, +0.016] |
| l2_v | 5.39 | +0.0005 [-0.0020, +0.0030] | -0.0043 [-0.0068, -0.0016] | +0.0159 [+0.008, +0.023] |
| l2_o | 12.76 | -0.0013 [-0.0049, +0.0037] | -0.0039 [-0.0067, +0.0003] | +0.0231 [+0.013, +0.033] |
| l2_gate | 22.35 | -0.0005 [-0.0019, +0.0014] | +0.0004 [-0.0038, +0.0066] | – |
| l2_up | 23.84 | -0.0008 [-0.0014, -0.0001] | -0.0008 [-0.0054, +0.0045] | – |
| l2_down | 23.72 | +0.0002 [-0.0024, +0.0028] | -0.0002 [-0.0039, +0.0040] | – |
| l3_q | 16.25 | +0.0015 [-0.0021, +0.0068] | +0.0003 [-0.0014, +0.0028] | +0.0112 [+0.001, +0.021] |
| l3_k | 8.35 | +0.0016 [-0.0003, +0.0037] | -0.0020 [-0.0042, +0.0011] | +0.0076 [-0.002, +0.017] |
| l3_v | 7.49 | +0.0002 [-0.0011, +0.0021] | +0.0010 [-0.0032, +0.0037] | +0.0140 [+0.006, +0.022] |
| l3_o | 15.95 | +0.0000 [-0.0015, +0.0008] | -0.0025 [-0.0052, -0.0005] | +0.0017 [-0.006, +0.010] |
| l3_gate | 22.34 | -0.0018 [-0.0074, +0.0013] | -0.0023 [-0.0064, +0.0008] | – |
| l3_up | 23.84 | +0.0003 [-0.0010, +0.0022] | -0.0024 [-0.0069, +0.0019] | – |
| l3_down | 23.74 | -0.0019 [-0.0077, +0.0028] | -0.0025 [-0.0043, +0.0009] | – |
| l4_q | 17.94 | +0.0003 [-0.0013, +0.0022] | -0.0058 [-0.0103, -0.0016] | +0.0047 [-0.004, +0.013] |
| l4_k | 8.65 | -0.0001 [-0.0019, +0.0028] | -0.0022 [-0.0039, +0.0007] | -0.0007 [-0.008, +0.007] |
| l4_v | 11.11 | -0.0016 [-0.0020, -0.0010] | -0.0066 [-0.0074, -0.0057] | +0.0074 [-0.000, +0.015] |
| l4_o | 18.49 | -0.0022 [-0.0045, +0.0004] | -0.0054 [-0.0077, -0.0035] | +0.0097 [-0.000, +0.020] |
| l4_gate | 22.09 | -0.0014 [-0.0037, +0.0030] | -0.0054 [-0.0105, -0.0006] | – |
| l4_up | 23.39 | +0.0013 [+0.0003, +0.0028] | -0.0072 [-0.0117, -0.0049] | – |
| l4_down | 22.94 | +0.0009 [-0.0020, +0.0029] | -0.0085 [-0.0106, -0.0068] | – |

Sum of the 20 single-module transplants: +0.222 τ; the full static LoRA on the same rows: +0.169 τ. Figures: [heads](analysis24k/head_knockouts.png), [modules](analysis24k/module_knockouts.png), [sensitivity](analysis24k/sensitivity.png).

**Interpretation.**
- *The drafter depends heavily on every block it has* (removing any layer's attention costs 1.9–6.1 τ, any MLP 0.65–2.5 τ) and above all on the context: zeroing the context K/V in layer 0 alone costs -5.91 τ, nearly as much as in all layers (-6.27). The block path matters less per layer (-0.41 to -1.48) but removing it everywhere costs -5.32. Single heads matter little (mean −0.03); a handful in L4 and L1 carry 0.1–0.3 τ each.
- *Adapted drafters have the same dependency map:* static and generator knock-out effects differ from the base by at most 0.38 τ (largest in the MLPs of L3), and per-head effects correlate across base and static (Pearson 0.96); the static drafter is slightly more robust to single-head removal. Adapters re-weight existing computation; they do not reroute it.
- *Flat to random directions:* a random rank-32 update with the trained static's norm changes τ by at most 0.003 on any attention or MLP module, and by at most 0.008 even at 4× that norm; only `fc` is sensitive (4×: -0.074). The trained ΔW of single modules gives up to +0.03 τ each, and their sum (+0.222) exceeds the full static gain on the same rows (+0.169): the gain is spread thinly over many modules, and the single-module effects overlap (sub-additive). At the static's step size the drafter is flat in random directions and responds only to a few specific ones. Sensitivity is not improvability: a flat random response says random directions are harmless, not that a trained direction cannot help.

## Step 5: representations (probes and CKA)

**Measured.** For 6 seeded fully valid blocks per validation row (44,805 draft-slot positions; the same blocks for every arm), the drafter's state after each of the 5 layers (the layer-0 input at draft slots is the constant mask embedding, skipped) for base, static, generator, and the generator conditioned on another row's prompt (same category; next category). Linear probes (multinomial logistic regression, standardized, L2 1e-3, LBFGS; 80/20 split by row; 9,000 test positions): category (accuracy; majority class 40.0%) and "this slot is accepted" (in the leading correct run of its block, for that arm; AUC), with slot index as an extra feature, against the drafter's own top-1 probability as a baseline. Linear CKA and relative change ‖X_a − X_b‖/‖X_a‖ between arms on the same positions; CKA of base drafter layers vs the target layers fed in, at the position each slot predicts from. Code: `collect_slots.py --hidden`, `analyze_repr.py`. Figure: [representations](analysis24k/representations.png).

Relative change / linear CKA per layer:

| pair | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| static vs base | 0.049 / 0.9905 | 0.071 / 0.9922 | 0.105 / 0.9853 | 0.144 / 0.9764 | 0.202 / 0.9437 |
| generator vs base | 0.052 / 0.9905 | 0.089 / 0.9874 | 0.136 / 0.9769 | 0.185 / 0.9639 | 0.246 / 0.9234 |
| generator vs static | 0.038 / 0.9962 | 0.068 / 0.9945 | 0.102 / 0.9910 | 0.138 / 0.9864 | 0.174 / 0.9794 |
| generator: own vs same-category prompt | 0.012 / 0.9992 | 0.021 / 0.9991 | 0.032 / 0.9987 | 0.043 / 0.9984 | 0.053 / 0.9977 |
| generator: own vs other-category prompt | 0.026 / 0.9973 | 0.047 / 0.9967 | 0.072 / 0.9941 | 0.093 / 0.9921 | 0.115 / 0.9896 |

Changes relative to base (Δ = X − X_base): mean per-position cosine between the static and generator changes, between the generator's change with its own prompt and with a donor prompt, and the size of the donor-induced difference relative to the generator's change:

| layer | cos(Δ static, Δ generator) | cos(Δ own, Δ same-cat donor) | cos(Δ own, Δ other-cat donor) | ‖own − same-cat‖ / ‖Δ own‖ | ‖own − other-cat‖ / ‖Δ own‖ |
|---|---|---|---|---|---|
| L0 | 0.688 | 0.971 | 0.857 | 0.229 | 0.513 |
| L1 | 0.633 | 0.968 | 0.840 | 0.236 | 0.538 |
| L2 | 0.635 | 0.967 | 0.839 | 0.239 | 0.534 |
| L3 | 0.638 | 0.971 | 0.856 | 0.229 | 0.500 |
| L4 | 0.671 | 0.972 | 0.867 | 0.207 | 0.452 |

Probes (test positions): category accuracy L0–L4 | accept AUC L0–L4 | baselines: drafter top-1 probability AUC, slot-only AUC:

| arm | cat L0 | cat L1 | cat L2 | cat L3 | cat L4 | accept L0 | accept L1 | accept L2 | accept L3 | accept L4 | p_top AUC | slot AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.977 | 0.977 | 0.969 | 0.963 | 0.971 | 0.845 | 0.877 | 0.892 | 0.909 | 0.915 | 0.960 | 0.718 |
| static | 0.980 | 0.979 | 0.968 | 0.964 | 0.972 | 0.849 | 0.869 | 0.876 | 0.901 | 0.906 | 0.938 | 0.714 |
| generator | 0.982 | 0.978 | 0.971 | 0.968 | 0.975 | 0.846 | 0.864 | 0.877 | 0.902 | 0.907 | 0.935 | 0.710 |

Category accuracy from the target layers fed in: t2 0.956, t10 0.975, t18 0.988, t26 0.983, t34 0.972, t36 0.975.

Linear CKA, base drafter layer vs target layer (rows L0–L4; columns target 2, 10, 18, 26, 34, 36):

| drafter | t2 | t10 | t18 | t26 | t34 | t36 |
|---|---|---|---|---|---|---|
| L0 | 0.035 | 0.077 | 0.001 | 0.010 | 0.047 | 0.028 |
| L1 | 0.075 | 0.162 | 0.002 | 0.022 | 0.111 | 0.062 |
| L2 | 0.145 | 0.271 | 0.003 | 0.039 | 0.220 | 0.124 |
| L3 | 0.161 | 0.291 | 0.003 | 0.044 | 0.265 | 0.177 |
| L4 | 0.179 | 0.294 | 0.003 | 0.047 | 0.297 | 0.303 |

**Interpretation.**
- *Category is already in the drafter's states* at every layer (probe accuracy 0.96–0.98 against a majority baseline of 40.0%) and in every target layer fed in (0.96–0.99). A prompt-derived category signal from the generator adds information the drafter already has; whether it uses it is another question (steps 1 and 3).
- *Adapters change the late layers most:* relative change grows from ≈ 0.05 at L0 to 0.20 (static) and 0.25 (generator) at L4; CKA stays ≥ 0.92. Static and generator change the states in partly shared directions (per-position cosine 0.63–0.69).
- *Prompt dependence of the generator's change:* swapping the prompt for another category's moves the states by about half the generator's own change (L4: 0.45); a same-category prompt by about a fifth (0.21). The generator's effect is mostly category-level, matching the D13 picture (η² 0.78).
- *Acceptance is readable but not better than the drafter's own confidence:* the linear accept-probe reaches AUC 0.915 at L4 against 0.960 for the drafter's top-1 probability. Both adapters lower that confidence AUC a little (0.938 static, 0.935 generator). The drafter-vs-target CKA is low everywhere (≤ 0.30, highest for target layers 10, 34 and 36 in the late drafter layers); linear CKA is dominated by a few large-norm residual dimensions, so these values are indicative only.

## Implications for the generator design

Interpretation (not measured directly; single seed, offline metrics):

- **Conditioning (C1–C5).** The prompt is already in the drafter's context and is used (step 3), and category is decodable from every drafter layer (step 5). Normalizing or pooling the prompt summary (C1, C2, C3) can sharpen the routing signal but adds no information the drafter lacks. Response-so-far or per-block conditioning (C4, G6) addresses a different quantity: misses are local (near-misses on the next tokens of the response), and they are where the remaining error sits. C5a (oracle category one-hot) remains the cheapest test of the category-routing ceiling.
- **Parameterization (G1–G6).** The 24K generator's effect is mostly category-level (step 5; D13 η² 0.78) and small. A basis mixture (G1) or diagonal core (G3) can represent that at a fraction of the 1.1B parameters. Because the drafter is flat to random rank-32 directions (step 4), larger per-prompt ΔW are not dangerous at this norm, but they also do not help on their own; what helps is a few specific directions.
- **Where to adapt.** First-order leverage is highest in `fc` and the K/V paths, where categories agree. Category disagreement is largest in the MLP and o_proj, but held out a shared rank-32 direction captures ≈ 0.9–1.0 of a category-specific one's gain everywhere (lowest in the L4 MLP and `fc`). A dynamic part therefore has little first-order room in any module at this data scale; if one is tried, the L4 MLP and `fc` are the places, which are outside the current q/k/v/o targets. The K/V path split (D1) showed no difference; the gradient maps agree that both paths carry similar, category-aligned signal.
- **Objective (D3, D-PACE).** The adapters gain on confident near-miss first misses and lose on blocks the base already accepted (step 1). A τ-aligned loss (D3) that weights the first failing slot and penalizes breaking accepted prefixes targets exactly this trade-off, for static and generator alike. Note that D-PACE weights depend on the drafter's own confidence, so training losses are not comparable across arms (the static point has a higher weighted KL than the base on the same rows although its τ is higher).
- **Evaluation hygiene.** 15.6% of first misses follow a sampled non-argmax token earlier in the block, and about 4% are exact bf16 logit ties. Both inflate the apparent error; greedy-path labels and tie-aware matching would make small arm differences easier to read.

### Most promising next experiments

1. **τ-aligned loss (D3) on the static LoRA, then the generator.** Directly targets the confident near-miss first misses (≈ 0.4–0.9 τ addressable) and penalizes breaking accepted blocks (the adapters' main loss). Needs a loss term; one 24K run each, ≈ 3 h.
2. **Per-category static LoRAs including MLP/o (A1+).** Bounds category routing where the gradient maps show the most disagreement (the held-out first-order test predicts ≤ ≈ 10% more gain than a shared adapter); 4 static runs on the 24K category splits. If they do not beat the global static, prompt-level dynamism is not worth its cost on this drafter.
3. **Block-conditioned adapter (G6 with G1).** A small basis of LoRAs mixed per anchor block from the drafter's own context features, since the prompt is redundant and misses are local.
4. **Static `fc` (+ MLP) adapter.** `fc` has the largest first-order gain and is the only module sensitive to random updates; D4 tested `fc` only inside the generator.
5. **Greedy-path, tie-aware evaluation.** Re-score existing arms with labels from the target's greedy path (or regenerated greedy responses) and tie-aware matching, so that the ≈ 20% of first misses that are off-path or bf16 ties stop counting as drafter errors.

## Limits

- One seed per arm; offline, teacher-forced, greedy matching on bf16 logits; no serving measurement.
- Fix gains are first-order ceilings (one corrected slot per block, later misses unchanged), not achievable targets.
- "On-path" is judged against the target's argmax along the sampled response; responses were sampled, so the labels themselves are argmaxes along a sampled path.
- Gradients: first-order, mean over 1,000 rows per category with validation anchors; at the static point the mean gradients are noise-dominated (split-half cosine ≤ 0.11), so their gains, cosines and captures are not interpretable there. Held-out capture can exceed 1 by noise. D-PACE weights depend on the drafter's confidence (loss values are not comparable across arms; gradients are the training gradients).
- Attention weights come from the eager path (98–99% argmax agreement with FlexAttention); the attention study uses 200 rows of at most 4,096 tokens (longer rows excluded).
- Masking removes positions from the attention of all layers; the prompt-length-matched controls remove at most half of the response. The generator's own prompt summary is never masked.
- Knock-outs and sensitivity use 200 validation rows; zeroing K/V is one choice of ablation (mean-ablation would differ). Random-direction sensitivity says nothing about trained directions ("sensitivity ≠ improvability").
- Linear CKA is dominated by a few large-norm residual dimensions; the drafter-vs-target CKA values are indicative only. Probes are linear with fixed L2.

## Reproduce

```bash
export CUDA_VISIBLE_DEVICES=0
# one-time: the generator run's inputs copied into the analysis directory (not part of the run identity)
cp -r /workspace/yusuf/runs/dflash/mix24k/mix24k-generator-attn-r32-3ep-kl/inputs /workspace/yusuf/runs/dflash/analysis24k/inputs/generator
A=scripts/dflash_port/analysis
.venv/bin/python $A/collect_slots.py --split validation --hidden && .venv/bin/python $A/collect_slots.py --split test
.venv/bin/python $A/grad_maps.py --point base && .venv/bin/python $A/grad_maps.py --point static
.venv/bin/python $A/knockouts.py --part all
.venv/bin/python $A/prompt_usage.py --part all && .venv/bin/python $A/prompt_usage.py --part sink
.venv/bin/python $A/analyze_slots.py --split test && .venv/bin/python $A/analyze_slots.py --split validation
.venv/bin/python $A/analyze_maps.py && .venv/bin/python $A/analyze_repr.py
.venv/bin/python $A/make_figures.py && .venv/bin/python $A/write_report.py
```

GPU 0 only; peak ≈ 58 GB GPU (gradient maps), ≈ 30 GB host RSS per process.
