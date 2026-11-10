# Critique of the evidence so far, and paths worth exploring

*2026-11-10. Written after the 4K ablations (Tiers 1–4), the 10K-math static vs generator comparison, and the first ~1,500 updates of the 10K-math lk_hybrid / KL generators. Brainstorm and planning document: suggestions, not results. Numbers are cited from the reports in `reporting/dflash/`.*

## 1. What the 10K-math validation curves show (4 runs)

Validation, 500 rows (200 math + 100 code / stem / chat), category-mean full_acc / τ:

| Update | CE generator | Static | lk_hybrid generator | KL generator |
|---|---|---|---|---|
| 0 (base) | 0.5265 / 7.703 | same | same | same |
| 500 | 0.5312 / 7.853 | 0.5308 / 7.838 | 0.5319 / 7.847 | 0.5315 / 7.840 |
| 1000 | 0.5313 / 7.869 | 0.5311 / 7.849 | 0.5318 / 7.867 | 0.5318 / 7.858 |
| 1500 | 0.5306 / 7.850 | 0.5314 / 7.850 | 0.5321 / 7.862 | 0.5325 / 7.866 |

(lk_hybrid and KL were still training when this table was taken; finals, test and OOD with paired CIs go in MATH10K.md.)

- **The soft losses keep rising; CE does not.** After update 1,000 the CE generator slides back, mostly in stem and chat (stem 0.4902 → 0.4884), while math holds. Those categories are out of domain for math-only training. KL and lk_hybrid are still climbing in stem and code at update 1,500. The 4K M5 result agrees: the soft losses overfit least and act as a regularizer.
- **The gaps are tiny.** At update 1,500 the four arms span 0.002 full_acc and 0.016 τ. The paired 95% CI on test was about ±0.015 τ, so these arms are not yet distinguishable.
- **The four curves have nearly the same shape.** This fits a generator that emits about one ΔW. Collapse is the working assumption until the similarity check (D13) on the lk_hybrid and KL checkpoints says otherwise.

## 2. Critique of the evidence so far

1. **The headroom is small, and we never measured the ceiling.** The best static LoRA adds about +0.15 τ (≈ 2%) over base, at 4K and at 10K. Every ablation lands within about ±0.005 full_acc of the others. "The generator doesn't beat static" is therefore weak evidence: there may be almost nothing per-prompt left to gain on this drafter and data.
2. **The M7 donor result is the strongest negative evidence.** A fixed or other-category LoRA is never worse than the row's own. So either the per-prompt information is not useful, or the drafter has no per-prompt slack to use it.
3. **The conditioning signal is weak.**
   - It is the hidden state at the last prompt token, which is the same chat-template token for every prompt.
   - It is unnormalized and dominated by the deepest layer (summary norms 12 / 56 / 85 / 279 / 805 for layers 2 / 10 / 18 / 26 / 34).
   - Within math, summaries sit at cosine 0.97 to their mean.
   - The audit traced the collapse to this input: the decoder squeezes the per-row variation further, to cosine ≈ 0.998 between generated factors.
   - EAGLE-3 also used last-prompt conditioning, but it pooled 1–8 rows and adapted a weaker drafter.
4. **The generator is over-parameterized for the data.** It has 1.1–2.2B parameters and trains on 4K–10K rows. A constant ΔW is the simplest low-loss solution, and nothing in training discourages it.
5. **No positive control.** We have never shown the DFlash generator pipeline producing useful per-prompt dynamism on a task that requires it. Without that, "no gain" cannot be separated from "cannot learn it".
6. **Single seed.** The 4K conclusions rest on differences of about 0.003 measured on 200 rows. EAGLE-3 records show about 5% run-to-run drift.
7. **The headline metric does not track speedup.** `full_acc` counts slots after the first miss, which never affects acceptance. τ is the speedup proxy and should be the headline; `compare_arms.py` already reports it.
8. **No serving measurement.** Offline τ is a good proxy for DFlash block drafting. Real tok/s and the generator's overhead (≈ 4.4 ms per call, ≈ 0.7 ms merge) are unmeasured (P5). At +2% τ, the overhead could cancel the gain.

## 3. Experiments, by value

### A. Bound the problem (cheap, decisive)
1. **Category-routed static LoRAs.** Train one static per category and evaluate each row with its own category's LoRA. This is the oracle for "a dynamic LoRA chosen by category" (4 × ~1 h at 4K, or splits of the 24K mix). If it does not beat the global static, category-level dynamism is worthless here.
2. **Per-prompt oracle LoRA.** Fit a LoRA on each row's own response for ~50 rows. This is in-sample (cheating) and gives the absolute ceiling for per-prompt adaptation. If it gains less than about +0.3 τ over static, no generator can matter on this drafter. Needs a small script.
3. **Prompt-only test-time training.** Fit a LoRA on the prompt's own tokens, which are available when serving. This is the realistic competitor to a hypernetwork.
4. **Three seeds** for static and the best generator at 10K, to establish the noise floor.

### B. Positive control
A set where the right draft depends on the prompt and one static LoRA must compromise. For example, a system-prompt tag forcing mutually inconsistent styles or formats, or different languages, built with our vLLM responses pipeline. The generator should clearly win there. If it does not, the problem is the machinery or the conditioning, not the premise.

### C. Conditioning fixes (one 4K or 10K run each)
- Per-layer RMS or z-score normalization of the summary.
- Mean or attention pooling over prompt tokens instead of the last token.
- Condition on the response so far, regenerating the LoRA every N tokens; long responses drift from what the prompt suggests.
- K same-category rows pooled, as in EAGLE-3 (M6 gave a small positive hint).

### D. Parameterization and objective
- **Static plus residual:** ΔW = W_static + g(x). Train the static path with a higher LR (or pretrain it), and regularize or dropout g, so the generator only learns the per-prompt residual.
- **A smaller generator** with low-rank heads, which fits 10K rows better.
- **KL or lk_hybrid by default:** both 4K and 10K favour them over CE.
- **Optimize τ directly:** expected accepted length = Σₖ Πⱼ≤ₖ p(slot j correct) is differentiable and aligns training with speedup.

### E. Data regime
- The planned 24K mix (4 domains), later 100K. More domains mean more for the generator to route on.
- Target where base is weak: chat τ ≈ 4.1 and summarization OOD ≈ 2.8, against math ≈ 10. The upside lies there.

### F. Reality check
P5 vLLM serving for base vs static: does +2% τ become real tok/s? Needed before any generator claim.

## 4. Suggested order (2–3 GPU-days)
1. Finish lk_hybrid / KL 10K (test and OOD with paired CIs; D13 similarity).
2. A1 + A2 (bounds) alongside C-normalization + D-residual on 10K-math with the KL loss.
3. The 24K mix with the winning generator configuration, alongside the static.
4. B if the bounds show headroom but the generator still collapses.

## 5. Open questions for the user
1. Goal: show a generator beats static at all (research claim), or maximize real speedup for this drafter? It decides whether B or F comes first.
2. May we create new data (e.g. multilingual or system-prompt-conditioned Qwen3-8B responses) for the positive control?
3. Are code changes OK (oracle / TTT scripts, normalized or pooled conditioning, static-plus-residual generator, τ loss)?
4. Keep the 24K run on the 10K settings with CE, or switch it to KL or lk_hybrid?
5. Compute: still 2 GPUs; any wall-clock limit?

---

# Part 2 (2026-11-10): the generator itself, and B / C / D in depth

*Added after a closer look at `src/speculators/generator/model.py`. Suggestions only; the user decides.*

## 6. Where the generator's parameters and latency go

```
summary (5 layers × 4096 = 20,480, unnormalized) + log length
  → encoder  Linear 20,481→512, GELU, Linear→128, LayerNorm        ≈ 10.5M
  → pool over K rows: [mean, std, log K]                             (K = 1 here, so std = 0)
  → condition  Linear 257→128, GELU, Linear→64 (math10k: 32)          tiny
  → for each of 20 modules: [condition, module emb 32, layer emb 32]
  → decoder  Linear→W, GELU, 2 × ResidualMLP(W)                       ~1–2M
  → heads  A: Linear(W → r·in), B: Linear(W → out·r)                 ≈ (W+1) × 4.26M
```

- **The output heads are about 99% of the generator:** 2.18B parameters at width 512, 1.09B at 256. The rest is about 12M.
- **The heads are one linear map.** Every generated A/B lies in a (W+1)-dimensional affine subspace of the 4.26M-dimensional factor space, and ΔW = B(z)·A(z) is bilinear in the decoder code z. We pay 2–4 GB of weights; the trained generators use about one direction of that subspace (effective rank 1.04 at 10K, 3.6 at 4K).
- **Cost:**
  - Training: generator forward + backward ≈ 0.1 s against ≈ 1.2 s for the drafter per update, so training is not the bottleneck.
  - Serving: ≈ 4.4 ms per call, mostly reading the head weights, plus 2–4 GB of GPU memory next to the target.
  - Batching: a different ΔW per request cannot be merged into shared weights, so it needs multi-LoRA batched kernels. This is likely the real deployment cost.
- **With K = 1 the std branch of the pooling is always zero:** wasted input, harmless.

**Conclusion:** the generator is large because it emits 4.26M free numbers per prompt, while the data supports only a handful of useful degrees of freedom. The main lever is what the generator outputs, not its width.

## 7. Generator architecture paths (smallest to largest change)

| # | Design | Generator outputs | Params (attention, r32) | Serving | Why |
|---|---|---|---|---|---|
| G1 | **Basis-LoRA mixture** | Coefficients c(x) over a bank of N learned LoRAs; c₀ = 1 fixed (LoRA 0 = static part) | N × 4.26M (N = 16 → 68M) plus a tiny router | ΔW = Σ cᵢBᵢAᵢ is one rank-N·r LoRA with per-request scales; batches cleanly, no merge | Static is the N = 1 case, so it should match static at minimum. Dynamism is explicit and measurable (coefficient entropy and spread). |
| G2 | **Dynamic core** ΔW = B·M(x)·A | A, B static; an r×r core per module | 4.26M + ≈ W × 20K (≈ 5–10M) | Cheap: tiny output, one small matmul | The generator only changes how a shared subspace is used, like the M4 scale effect but per rank direction. |
| G3 | **Diagonal core** ΔW = B·diag(s(x))·A | 32 scales per module (640 numbers) | ≈ 4.3M + under 1M | Trivial | The smallest possible dynamism; an ablation floor. Generalizes M4's per-row oracle coefficient. |
| G4 | **One factor generated** | A static (or B static); generate the other | ≈ half of now | as now | Cheap capacity cut. |
| G5 | **Low-rank or chunked heads** (as in hypernetwork-adapter work such as Text-to-LoRA) | Full A/B, with head = U·V, U shared across modules and chunks | 10–100× smaller | as now | Keeps the full-factor generator but removes most of its parameters. D3 (shared heads, 1/5 the size) already showed no cost. |
| G6 | **Token- or block-level routing** (G1 conditioned per anchor) | Coefficients per DFlash block from the current context | as G1 | per-block scales | The adapter follows the response as it drifts from the prompt: a mixture-of-LoRA-experts drafter. |

**Suggested first:** G1, then G2 / G3 as ablations. G1 addresses size, latency and serving at once and contains the static-plus-residual idea (D). Its failure mode is easy to read: if it learns c ≈ constant, the data has no use for dynamism.

## 8. C. Conditioning, in depth

The audit found that the input is already near-identical across prompts (cosine 0.97 to the mean) and that the network compresses it further. In order:

- **C1 Normalize:** per-layer RMS norm or a learned LayerNorm at the encoder input, so layer 34 (norm 805) no longer dominates. One line of code; a 10K run is ≈ 1.5 h.
- **C2 Pool over the prompt:** mean or attention pooling over all prompt tokens instead of the last (template) token. The store already holds full per-token hidden states, so this needs a new summary type, not new extraction.
- **C3 Layer choice:** middle layers only, or one layer. Cheap to test offline first: a linear probe from the summary to category, or to the per-row best LoRA coefficient (M4 per-row oracle).
- **C4 Response-so-far conditioning:** natural with G6; too expensive without it (one full LoRA per block).
- **C5 Oracle conditioning (diagnostic):**
  - (a) Feed the true category as a one-hot. If even that generator ≤ static, category routing has no value here.
  - (b) Feed a summary of the row's own response (cheating). If that generator does not beat static either, the generator machinery is the bottleneck, not the input.

  Both are small code changes and need no new data.

## 9. D. Objective and parameterization, in depth

- **D1 Static plus residual:** ΔW = W_static + g(x). Warm-start the static path from the trained static LoRA; g starts at zero with its own LR and is regularized (dropout or weight decay on g only). G1 contains this; it also works with the current generator.
- **D2 KL or lk_hybrid as the default loss:** they overfit least at 4K and 10K, and at 10K they edge the static LoRA in distribution ([MATH10K_LOSSES.md](../MATH10K_LOSSES.md)).
- **D3 τ-aligned loss:** E[accepted length] = Σₖ Πⱼ≤ₖ pⱼ(correct) over a block's slots. It rewards fixing the first mistake over later slots that are never accepted. This applies to static too.
- **D4 Keep the information, don't force diversity:** an auxiliary head predicting category (or the M4 best coefficient) from the condition. It stops the condition from collapsing without forcing useless ΔW diversity.
- **D5 Learned per-prompt scale:** M4 found the 4K generator best at half its training scale. Let the generator output its scale (G3 is the general case).
- **D6 Static LoRA with lk_hybrid / KL (new control, from the 10K loss runs).** The soft-loss generators beat the CE static by +0.013 to +0.016 τ on test, but their ΔW are even more constant (cross-row cosine 0.976 / 0.989). A static LoRA with the same loss is needed before any of that gain is credited to the generator. Two runs of ≈ 1.5 h, no code.

## 10. B. Positive controls, in depth (a setting where dynamism must pay)

- **B1 Thinking vs non-thinking (favourite).** Regenerate some responses with Qwen3-8B `/think` on. The prompt flag fully determines the response distribution (long reasoning traces vs direct answers), so one static LoRA must compromise. Realistic, and uses existing prompts and the pipeline: new responses plus a store fill. If G1 or the generator does not beat static here, something is broken.
- **B2 Style or format tags:** system prompts demanding mutually exclusive formats (JSON, numbered steps, one paragraph, another language). Synthetic, so we control exactly how much dynamism pays.
- **B3 Multilingual prompts,** where base is weak, so headroom is large.
- **First check the bound exists:** per-category static LoRAs should clearly beat the global static (A1 on the control data).

## 11. A possible first batch (≈ 1 GPU-day on both GPUs)

1. D6: static LoRA with lk_hybrid and with KL on 10K-math (no code).
2. C5 oracle conditioning (one-hot category; response-summary cheat) with the current generator and KL loss.
3. C1 normalized summary + D1 static-plus-residual, KL loss, on 10K-math.
4. G1 basis-LoRA mixture (N = 16 plus static, KL loss) on 4K and 10K.
5. Generate B1 thinking / non-thinking data on the other GPU between runs.

## 12. Open questions for the user (part 2)

1. Code changes: G1, C1, C2, C5 and D1 need new code (a generator variant, a summary type, a loss term). OK to build them, with tests, on main?
2. New data for B1: Qwen3-8B thinking-mode responses for a subset of prompts (e.g. 2K rows per category); new store fills of about 1–2 TB.
3. Serving target: must the generator eventually run in vLLM with batching? If yes, G1 / G2 should be the main line.
4. Latency and memory budget next to the 8B target (e.g. under 100M parameters, under 1 ms per call)? It decides how far to push G2 / G3 / G5.
5. Sequencing: wait with the 24K mix until a generator design wins (G1, or C1 + D1)?
