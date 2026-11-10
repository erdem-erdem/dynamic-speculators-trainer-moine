# DFlash generator: first evaluation graphs

What each figure in this folder compares. Results are discussed in the experiment reports (`reporting/dflash/*.md`), not here.

## Common to every figure

All figures are W&B validation panels from training runs of a LoRA adapter on the DFlash drafter `z-lab/Qwen3-8B-DFlash-b16` (target model Qwen/Qwen3-8B). Every run uses seed 42.

Two kinds of adapter appear:
- **Static LoRA:** one LoRA trained directly and used for every prompt.
- **Generator:** a hypernetwork that reads a summary of the prompt (the target model's hidden states at the last prompt token) and produces a separate LoRA for that prompt.

Each line is one training run, named `dflash-<experiment>-<arm>-<targets>-r<rank>-<data>-<updates>[-<variant>]-gseed42`. Here `attn` means q/k/v/o projections, `r32` means LoRA rank 32, `4k` means the 4K training pool and `1500` means 1,500 optimizer updates.

**Training and validation data:**
- **4K figures (all except the first):** 4,000 rows (1,000 each of math, code, stem and chat) and 1,500 updates. Validation uses 200 fixed rows (50 per category) every 250 updates.
- **10K-math figure:** 10,000 math rows for 3 epochs (1,667 updates). Validation uses 500 rows (200 math plus 100 each of code, stem and chat).

Validation rows are never trained on. The x-axis is `optimizer_step`, except in `overall_agreement_by_training_seconds`, where it is cumulative training time in seconds.

**The six panels (the same in every figure):**

| Panel | Meaning |
|---|---|
| `overall_agreement` | Headline score: block accuracy (`full_acc`), averaged over the 4 categories with each weighed equally. `full_acc` is the fraction of draft positions 1–15 of a 16-token block whose top-1 prediction matches the target model's. |
| `overall_agreement_by_training_seconds` | The same score plotted against training time. |
| `math_agreement`, `stem_agreement` | The same score for one category only. |
| `context_1_expected_accepted_length` | τ: the average number of tokens accepted per drafted block (the run of correct draft tokens up to the first miss, plus the target's bonus token). Higher means more speedup. `context_1` means each LoRA is generated from one row (its own prompt). |
| `context_1_agreement_position_9` | Accuracy at draft position 9 alone, judged independently of the other positions. |

All metrics are offline and teacher-forced: the drafter predicts over saved target-model responses, with no live generation.

---

## `4k-static-vs-generator-girst.png` — headline: static LoRA vs generator (M0)

- **Investigates:** whether a per-prompt LoRA from the generator beats a single static LoRA.
- **Runs:**
  - `m0-static-lora-attn-r32-4k-1500` (static LoRA)
  - `m0-generator-attn-r32-4k-1500` (generator)
- **What varies:** only the adapter type.
- **Shared settings:** q/k/v/o in all 5 drafter layers, rank 32, α 64 (scale 2), CE loss with D-PACE per-position weights. The generator has decoder width 512 and one output head per adapted module.

## `static-vs-generator-on-attn.png` — static vs generator, LoRA on layer 0 only (D2)

- **Investigates:** the static-vs-generator comparison when only the first drafter layer is adapted. EAGLE-3 has one drafter layer; DFlash has five.
- **Runs:**
  - `d2-static-lora-attn-r32-4k-1500-layer0`
  - `d2-generator-attn-r32-4k-1500-layer0`
- **What varies:** only the adapter type.
- **Shared settings:** q/k/v/o of layer 0 only; everything else as M0.

## `which-part-do-we-put-lora-to.png` — where the LoRA is applied (D1 and D2)

- **Investigates:** two placement questions together.
- **D2 (which layer):** the LoRA is on the last drafter layer (layer 4) only.
  - `d2-static-lora-attn-r32-4k-1500-layer4`
  - `d2-generator-attn-r32-4k-1500-layer4`
- **D1 (which input path of k/v):** in DFlash attention, k_proj and v_proj process two inputs: the target-model context features, and the block's draft tokens being denoised. The generated k/v LoRA is applied to one path only, on all 5 layers. q/o adapters are unchanged.
  - `d1-generator-attn-r32-4k-1500-kvctx`: context path only.
  - `d1-generator-attn-r32-4k-1500-kvblock`: block path only.
  - The M0 default applies it to both paths.
- **Shared settings:** as M0 (rank 32, attention targets, 4K pool, 1,500 updates).

## `mlp-vs-attn.png` — target modules (M1)

- **Investigates:** which drafter modules the LoRA should adapt.
- **Runs (each target set has a static LoRA and a generator):**
  - `mlp`: gate/up/down projections only.
    - `m1-static-lora-mlp-r32-4k-1500`
    - `m1-generator-mlp-r32-4k-1500`
  - `all`: attention plus MLP (q/k/v/o, gate/up/down).
    - `m1-static-lora-all-r32-4k-1500`
    - `m1-generator-all-r32-4k-1500`
- **What varies:** the target module set, and static vs generator within each set.
- **Shared settings:** all 5 layers, rank 32, α 64, everything else as M0. The attention-only pair is M0 (first headline figure).

## `generator-rank.png` — LoRA rank (M2)

- **Investigates:** how the generated LoRA's rank affects the generator.
- **Runs:**
  - `m2-generator-attn-r8-4k-1500`
  - `m2-generator-attn-r16-4k-1500`
  - `m2-generator-attn-r64-4k-1500`
  - Rank 32 is M0.
- **What varies:** only the rank r, with α = 2r so the LoRA scale α/r = 2 stays fixed.
- **Shared settings:** generators only, attention targets, everything else as M0.

## `decoder-width.png` — generator capacity (M3)

- **Investigates:** whether a larger generator helps.
- **Runs:**
  - `m3-generator-attn-r32-4k-1500-decoder256`
  - `m3-generator-attn-r32-4k-1500-decoder1024`
  - Width 512 is M0.
- **What varies:** only the generator's decoder width. The output heads map the decoder output to the LoRA factors, so the parameter count scales almost linearly with width: about 1.1B at 256, 2.2B at 512 and 4.4B at 1024.
- **Shared settings:** generators only, rank 32, attention targets, everything else as M0.

## `ce-vs-kl-vs-lk-losses.png` — training loss (M5)

- **Investigates:** which training loss suits the generator.
- **Runs:**
  - `m5-generator-attn-r32-4k-1500-ce-expdecay7`: cross-entropy against the target's top-1 token, with a fixed exponential per-position decay (γ = 7, the DFlash paper's value for block size 16) instead of D-PACE weights.
  - `m5-generator-attn-r32-4k-1500-kl`: KL divergence to the target model's full next-token distribution (D-PACE weights).
  - `m5-generator-attn-r32-4k-1500-lkhybrid`: `lk_hybrid`, the EAGLE-3 generator's loss, a soft blend of KL and total-variation terms against the target's full distribution (D-PACE weights).
  - The M0 default is CE with D-PACE weights.
- **What varies:** only the loss function / per-position weighting.
- **Shared settings:** generators only, everything else as M0.

## `anchors-256-vs-1024-in-generator.png` — anchor density (D5)

- **Investigates:** whether the number of training blocks per sequence matters for the generator. DFlash trains on 16-token blocks that start at sampled anchor positions in each sequence.
- **Runs:**
  - `d5-generator-attn-r32-4k-1500-anchors256`: 256 anchors per 8,192 tokens.
  - `d5-generator-attn-r32-4k-1500-anchors1024`: 1,024 anchors per 8,192 tokens.
  - The M0 default is 512.
- **What varies:** only the anchor density.
- **Caveat:** validation uses the same density, so these two runs are scored on fewer or more blocks than the other runs. Their starting (update 0) values differ for that reason.

## `alpha-for-attn-generator.png` — D-PACE α (D5)

- **Investigates:** whether the strength of the per-position loss weighting matters for the generator. D-PACE weights each draft position's loss; α sets how strongly the weighting depends on position.
- **Runs:**
  - `d5-generator-attn-r32-4k-1500-alpha025`: α = 0.25.
  - `d5-generator-attn-r32-4k-1500-alpha1`: α = 1.0.
  - The M0 default is α = 0.5.
- **What varies:** only α.
- **Shared settings:** generators only, everything else as M0.

## `math10k-static-vs-gen.png` — 10K math: static LoRA vs generator

- **Investigates:** the static-vs-generator comparison with more, single-domain training data, trained for longer.
- **Runs:**
  - `math10k-static-lora-attn-r32-3ep`
  - `math10k-generator-attn-r32-3ep`
- **What varies:** only the adapter type.
- **Shared settings:**
  - Training on 10,000 math rows for 3 epochs (1,667 updates). Validation on 500 rows (200 math, 100 each of code, stem and chat) whose prompts occur in no training split.
  - Attention targets, rank 32, α 64, CE + D-PACE.
  - A smaller generator than M0: decoder width 256, condition dim 32 (M0: 512 and 64).
