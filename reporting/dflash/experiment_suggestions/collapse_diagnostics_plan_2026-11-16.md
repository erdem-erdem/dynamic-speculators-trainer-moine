# Does the generator read its prompt? A collapse-diagnostics plan for the DFlash LoRA generator

*2026-11-16. Suggestions only. Nothing here has been run. It adapts the method of two external write-ups on a KV-cache LoRA generator ("The Generator That Won't Read" and the "Memory Bypass Brief", zeroKV, 23 Oct – 15 Nov 2026) to our generator.*

## 1. What the zeroKV work did, and what carries over

**Their setting.** A hypernetwork reads the KV entries that SnapKV evicted and writes a LoRA meant to put the lost facts back. Two generators were built independently. Both raised the score (+1.5–4 RULER points), and neither read the document.

**Their method in four steps.** Each step is a kind of check we can reuse.

| Step | Their question | Their tools | Their answer |
|---|---|---|---|
| 1. Controls | Does the gain depend on the input? | own vs **swapped** patch; **average** patch; **context-free twin** (same generator trained without input); writes disabled; input **zeroed / shuffled**; forced eviction (only reading can help) | own ≈ swap ≈ mean ≈ twin. The generator learned one fixed LoRA. |
| 2. Oracle | Can the format hold the signal at all? | LoRA **fitted directly** on one document, tested on a reworded question; fitted vs **unseen** stretches | Retrieval: yes (86%, 30/30). Book text: memorizes the fitted part (80–97%), recovers ≤ 8% on unseen parts. |
| 3. Signal budget | How much of the loss and gradient depends on the input? | loss split into "same for every document / differs between documents / token noise"; gradient share by token type | 1.7–6% of the loss depends on the document. |
| 4. Data-path autopsy | Where does the signal die? | RMS gradient per parameter by component; parameter movement from init; spread across documents at each stage (input → memory → ΔW); gate statistics; norm growth | Content path gets 10³–10⁵× less gradient than the heads. Inputs differ (1−cos 0.14), memories do not (10⁻⁶). Gates became a constant mask. |

**Their rules.** Every run reports base, static, own and swap on the same rows; own vs swap is the success metric. They use paired bootstrap, measure seed noise (a second seed moved results 2.5 points), and report answer-token metrics separately. A metric only counts if some control can make it fail. Their October proxy said "92.8% repair", and it was measuring a fixed correction.

**Their fixes.** Static + centered residual, swap-margin hinge, answer-only loss, a writer fix, a 30–100× encoder LR, strict eviction. None opened an own-vs-swap gap. Their next step is a **learnability ladder**: the easiest task where reading is the only route, with the pass rule fixed before the run.

### How our situation differs (this changes which checks matter most)

| | zeroKV | Our DFlash generator |
|---|---|---|
| Is ΔW numerically constant? | Yes: 1−cos ≈ 10⁻⁷ between documents | **No**, in most arms: cross-row cosine 0.73 (100K), 0.79–0.91 (24K), effective rank 2.6–3.6, category recoverable at 0.96–0.98 (D13). **Near-constant** in some arms: 10K-math (cos 0.965–0.989, eff. rank 1.04), shared-head r128 (0.970), r512 (0.984). |
| Does the input carry information the target lacks? | Yes by construction: the facts were evicted | **Doubtful.** The drafter already reads the prompt through the injected context (ANALYSIS24K step 3: removing the prompt costs −1.08 τ; category is decodable from every drafter layer at ≥ 0.96). This is closer to their **PG-19 case** (the signal is mostly generic) than to RULER. |
| What the controls show so far | own = swap = mean = twin | M7 (4K): own is the **worst** donor at s = 2 (0.5220 vs 0.5240 fixed / 0.5245 cross-category). Not repeated at 24K / 100K. Mean, twin, centroid and same-category controls were **never run**. |
| Gain over static | +1.5–4 points over SnapKV, all content-free | +0.022 τ over static r32 (MIX24K). **Below** static r512 at 24K and 100K (−0.013 test, −0.049 OOD). |
| Bypass path | reader's learned queries and biases | **head biases** (A/B head bias = a built-in static LoRA; `per_type_bias` makes it per layer) plus W·z̄ for the mean condition. |

So our main question is not "is ΔW constant?" but three narrower ones:
1. **Is the per-prompt variation useful, or is it noise / memorization?** At 4K it was harmful (M7).
2. **Is the generator's gain over static r32 a reading effect or a parameterization effect?** The ABL24K hybrid note proposes that the generator trains like a static LoRA at a much higher effective LR.
3. **Is there per-prompt headroom at all on this drafter?** This is the H2 "no signal" question, and we have never measured it.

## 2. Hypotheses

| ID | Hypothesis | What would show it |
|---|---|---|
| **H1 Bypass / collapse** | Per-prompt signal exists, but optimization routes everything through the biases and z̄; the content path starves | oracle shows headroom (B1/B3) **and** own ≈ mean ≈ twin (A) **and** content path gets little gradient (C) |
| **H2 No signal** | This drafter + data has no per-prompt slack beyond what its context already provides | per-row oracle LoRAs do not beat the global static on **unseen** tokens of the same row (B1), and the row-specific gradient is not reproducible (B4) |
| **H3 Router only** | The dynamism is category routing; within-category variation is noise | own ≈ category centroid (A4); PCA-k saturates at k ≈ 3 (A7); η² ≈ 0.72–0.74 already hints at this |
| **H4 Input bottleneck** | Useful per-prompt information exists but the last-template-token summary does not carry it | oracle inputs (F1: true response summary, category one-hot) work, but the real summary does not; probes from the summary fail (D4) |
| **H5 Parameterization effect** | The gain over static r32 comes from the generator's head parameterization (an over-parameterized static, effectively higher LR), not from reading | context-free twin ≈ generator (A6); the ΔW step size per update is larger than the static's (C5) |
| **H6 Memorization** | The per-prompt part fits training rows and does not transfer | own > swap on **training** rows, own ≤ swap on held-out rows (E5) |

These hypotheses are not exclusive. H3 + H5 + H6 together would explain every result so far.

## 3. The plan

Most of it is forward and backward passes on existing checkpoints, and it reuses existing harnesses:
- `offline_eval.Variant` (donor, coefficient, rows; `allow_same_category` exists but is off);
- `offline_eval.similarity` (the D13 Gram);
- `scripts/dflash_port/analysis/common.py` + `grad_maps.py` (per-row gradients at a given adapter point; split-half);
- `scripts/dflash_port/perturbation/` (churn, slot profiles).

**Checkpoints to analyze** (attention q/k/v/o, all KL):
- `mix24k-generator-attn-r32-3ep-kl` (1.1B; the only resolvable win over static r32);
- `abl24k` shared-head r64 (0.45B) and hybrid r512;
- `mix100k-generator-attn-r64-3ep-kl-pt256` (most dynamic, loses to static r512);
- `m0` 4K generator (the overfitting case).

Matched statics: r32 and r512 at 24K, r512 at 100K.

**Caveat.** Run dirs hold only the final `checkpoint.pt`. Trajectory checks (C, D3) need intermediate snapshots. First check the HF repos' commit history. If there are none, add a "save generator state every N updates" option and rerun one 24K arm (≈ 2 h).

### Phase A: reading controls (forward only, ≈ 2–3 GPU-h for all checkpoints)

Every variant is evaluated on the same rows and anchors as the headline numbers: 500 validation rows plus test. Report per-row paired Δτ (category mean), 95% bootstrap CI stratified by category, and the share of rows where own wins.

| ID | Control | How (in our code) | A reading generator would… |
|---|---|---|---|
| A1 | **Swapped donors**: same category (new), other category (M7), random | `Variant(donor="same_category", allow_same_category=True)`; donors from the unseen `*-00002` pool | beat same-category swap. This is the key test, since other-category swap mostly tests routing |
| A2 | **Average patch**: the exact mean ΔW over the 500 rows | Mean ΔW has rank ≤ n·r. Apply it exactly as a concatenated-factor LoRA (stack the scaled factors, rank n·r, chunked), plus its rank-r SVD truncation as a deployable version | beat it. This was their strongest control: the average patch was usually the *best* bar |
| A3 | **Mean-condition patch** (decoder fed z̄) and **bias-only patch** (head weights zeroed, so only head biases remain) | hook in `LoRAGenerator.decode_factors` | beat both. Bias-only shows how much of ΔW never depended on the prompt (R15 in the roadmap) |
| A4 | **Category-centroid patch** (mean ΔW per category, applied to that category's rows) | as A2, per category | beat it (else H3: a 4-way router) |
| A5 | **Input perturbation**: summary zeroed, replaced by the training mean, shuffled across rows, Gaussian noise at the between-row scale, one layer dropped at a time (layers 2/10/18/26/34) | hook at the summary input | change ΔW by about the between-row distance and lose τ. Report ‖ΔW(perturbed) − ΔW‖ / ‖ΔW_i − ΔW_j‖, their D5 measure |
| A6 | **Context-free twin**: the same generator recipe trained with a constant input (zeros or the training-mean summary) | one config flag + one 24K 2-epoch run (≈ 2 h); the matched pair is ABL24K shared-head r64 | beat the twin. **This is the decisive test for H5**: if twin ≈ generator, the +0.02–0.03 over static r32 is parameterization, not reading |
| A7 | **PCA-k truncation** of the generated set: ΔW = mean + top-k principal components of the per-row deviation, k = 0, 1, 2, 4, 8, 16, all | from the Gram (exact) | need k > 3 to keep its margin (else H3) |
| A8 | **Dynamic-part coefficient**: ΔW = ΔW̄ + c·(ΔW_i − ΔW̄), c ∈ {−1, 0, 0.5, 1, 1.5, 2} | generalizes M4 | be best at c ≈ 1, and c = −1 should hurt. If c* ≈ 0, the dynamic part is noise. If c = −1 ≈ c = +1, it is symmetric noise. **Cheapest single test of usefulness** |

**Pass rule** (fixed before the run): own > same-category swap, own > average patch, and own > twin, each with the CI excluding zero, on validation and test. Otherwise the gain is content-free.

### Phase B: is there a per-prompt signal? (oracles; ≈ 4–8 GPU-h)

This bounds H2. It is the analog of their R1 / PG-19 oracle, and it is the step our roadmap has listed as "A2 per-prompt oracle LoRA" without running it.

| ID | Test | How | Reading |
|---|---|---|---|
| B1 | **Per-row oracle LoRA, fitted vs unseen**. For ~64 validation rows (16 per category), fit a rank-32 LoRA on the first half of the row's response tokens, starting from (a) base and (b) the static r512 point, and evaluate τ on the **second half** | the static-training loop on one row, ≈ 100–300 steps, early-stopped on a third, disjoint slice. Also fit on all tokens for the in-sample ceiling | **Headroom = unseen-half Δτ over the global static.** If ≈ 0 (their PG-19: ≤ 8% unseen vs 80–97% fitted), H2 holds and no generator design will help on this data. If clearly > 0.1 τ, H1/H4 are live |
| B2 | **Oracle transfer matrix**: row i's oracle LoRA applied to row j's unseen half, for j in the same / other category | reuses B1 LoRAs; 64 × 64 forwards on short halves | own-oracle ≫ same-category-oracle means the signal is row-specific. ≈ means it is category-level, and a router suffices |
| B3 | **Category-routed static** (one static per category on the 24K splits) vs the global static | 4 short static runs (r32 and r512) | the ceiling for routing. If it equals the global static, category dynamism is worth nothing here (roadmap A1) |
| B4 | **Gradient signal budget** (their 4a): at the static r512 point, per-row adapter gradients g_i on 1,000 training rows. Split the total energy into ‖mean g‖² (shared), between-category, and within-category; make each **noise-corrected** by computing g_i twice per row on disjoint anchor halves (split-half covariance) | `grad_maps.py` already computes per-row gradients and split-half cosines | share of *reproducible* gradient energy that is row-specific. ANALYSIS24K found no reproducible **mean** gradient left at the static point; the per-row reproducible part is the open number |
| B5 | **Where the row-specific signal sits**: the B4 split restricted to τ-relevant positions (the first-miss slot of each block, on-path, target-confident) vs all positions | position masks from `analyze_slots.py` | their answer-token lesson: they found 1.7% of the loss weight on what matters. If the row-specific part lives on positions that D-PACE down-weights, the loss is hiding it |

### Phase C: where does the gradient stop? (their 4b; ≈ 1–2 GPU-h, plus snapshots)

| ID | Measure | How |
|---|---|---|
| C1 | **RMS gradient per parameter by component**, in data-flow order: encoder L1/L2/LN, pool, condition MLP, module/layer embeddings, decoder trunk, head **weights**, head **biases** | one backward pass over ~64 cohorts from the checkpoint (no optimizer state needed); at init, mid and end if snapshots exist |
| C2 | **Gradient that reaches the condition**: ‖∂L/∂z_i‖ vs ‖∂L/∂bias‖, and the per-row variance of ∂L/∂z_i (the useful part) vs its mean (the part that only moves z̄) | autograd on z |
| C3 | **Parameter movement from init** per component, ‖θ − θ₀‖/‖θ₀‖ (their "encoder moved 0.35%") | θ₀ comes from the seeded init (seed 42; check that the init is reproducible) |
| C4 | **Head output decomposition**: for each head, ‖bias + W z̄‖ (static part) vs E‖W(z_i − z̄)‖ (dynamic part), per module; the ratio over training | forward only |
| C5 | **Effective step size (H5)**: change of ΔW per update for the generator (mean and per-row parts) vs the static LoRA, at matched points | a few optimizer steps from the final checkpoints with optimizer state, or consecutive snapshots. Tests the ABL24K hybrid hypothesis directly |

### Phase D: representation spread along the data path (their 4c, V1–V4; forward only, < 1 GPU-h)

| ID | Measure | Why |
|---|---|---|
| D1 | **Between-row spread at every stage**: raw summary (per layer and concatenated), after the input norm (if any), encoder output, pooled, condition z, decoder hidden per module, A, B, ΔW. Report the 1−cos matrix median and var/‖mean‖² per stage, on 500 rows | finds the stage that squeezes. Earlier audit: summaries have cosine 0.97 to their mean, generated factors 0.998 (10K-math). Their analog: 0.14 → 10⁻⁶ |
| D2 | **Effective rank / participation ratio** of the row set at each stage; dead or saturated units (GELU outputs constant across rows; LayerNorm with one dominant direction) | analog of their binary gates. A constant mask of live units looks the same in an MLP |
| D3 | **D13 + D1 over training** (needs snapshots) | P2 smoke had dyn_ratio 0.0005 at update 100 and 0.27 at the end (4K). Did dynamism grow while held-out τ fell? That is when memorization would have set in |
| D4 | **Linear probes** from each stage to: category, prompt length, response length, row's base τ, **row's oracle-LoRA direction** (from B1), row's best dynamic coefficient (A8) | information present but unused (probes OK, A-controls fail) vs information squeezed out (probes fail at a stage) |
| D5 | **Spectrum of the head weight matrices** and of the per-layer summary norms (12 / 56 / 85 / 279 / 805: layer 34 dominates) | how many z directions are read; whether one aux layer drives the condition |

### Phase E: is the dynamism functional? (specific to us, since our ΔW varies; ≈ 1–2 GPU-h)

| ID | Test | Reading |
|---|---|---|
| E1 | **Noise floor**: the same LoRA evaluated with two anchor seeds; one static arm retrained with 2 more seeds (24K, 2 ep, ≈ 2 h each) | the size of a per-row own-vs-swap difference that means anything. Their second seed moved 2.5 points |
| E2 | **Directional test**: does ΔW_i − ΔW̄ point along the row's own descent direction −g_i (taken at ΔW̄) more than along −g_j of other rows (same category, other category)? Report cosine and the first-order gain ⟨−g_i, ΔW_i − ΔW̄⟩, own vs swapped | the cheapest mechanistic test of reading. Positive own > swap = the generator writes what the row needs. ≈ 0 = it writes noise. Negative = it writes harm (M7 at 4K) |
| E3 | **Who benefits**: correlate own − swap (A1) and own − mean (A2) with category, prompt length, the summary's distance from the mean, base τ, and OOD-ness | dynamism might help atypical rows and hurt typical ones, or the reverse |
| E4 | **Churn decomposition** (PERTURB style): own vs mean patch, blocks fixed and broken, by first-miss class | is the generator's extra +0.02 a net of large flows? (ANALYSIS24K: generator vs static 8.3% better / 7.7% worse blocks) |
| E5 | **Training vs held-out rows**: A1/A2/A8 on 500 **training** rows | own ≫ swap on training rows with own ≈ swap held out = memorization (H6). Their shortcut 2 was exactly this: EM 0.94 with own = swap |

### Phase F: positive controls and a learnability ladder (only after A–E; each rung has a pass rule fixed in advance)

The point is to separate "nothing to learn" (H2) from "cannot learn it" (H1/H4). We have never shown that our pipeline can learn useful per-prompt dynamism on any task.

| Rung | Task | Pass rule | If it fails |
|---|---|---|---|
| F0 | **Distillation sanity**: train the generator by regression to reproduce a known set of distinct LoRAs from the summary, e.g. the 4 category statics (B3), then the per-row oracle LoRAs (B1) on held-out rows | category: near-perfect; per-row: held-out R² clearly above the mean-LoRA baseline | the architecture/input cannot represent the mapping (H4) |
| F1 | **Oracle conditioning**, end to end: (a) category one-hot as the input; (b) a summary of the row's own **response** (cheating) | (b) beats static by more than the generator does | if even (b) ≤ static, the machinery or the drafter (not the input) is the bottleneck; if (b) works but (a) does not, the signal is per-row, not per-category |
| F2 | **Synthetic conflict**: training data where the best adaptation provably differs by prompt and is cued in the prompt. Options: mix thinking-mode and non-thinking responses for the same prompt pool, cued by the system prompt; or two target styles. Show first that category-routed statics (B3) beat the global static on it | generator ≥ routed static > global static, own > swap | the end-to-end gradient does not route the signal (H1) even when it clearly exists |
| F3 | **Real data again**, only with the fixes that passed F0–F2 | Phase A pass rule | |

## 4. Fixes to consider later (mapped from their table; do not start before A, B and E)

Their main lesson: fixes chosen before the diagnosis all failed, because the optimizer always found the easier route.

| Their fix | Our version | Status here |
|---|---|---|
| F1 static + centered residual | ΔW = ΔW_static + (G(x_i) − EMA mean of G), static warm-started from static r512, residual with its own (lower) LR and decay | partly tried: the hybrid `per_type_bias` has no centering and no warm start, and lost to static r512 by −0.025 |
| F2 swap margin | hinge: L_own + λ·max(0, m + L_own − L_swap(same category)) | not tried |
| F4 answer-token loss | weight the loss toward τ-relevant positions (first-miss, target-confident), or the τ-aligned loss (roadmap D3) | D-PACE / label gate tried; tied |
| F5 faster encoder | separate LRs for encoder vs heads, **lower head-weight LR** (H5) | proposed in MIX100K, not run |
| Conditioning dropout | train with z replaced by z̄ for p = 0.3 of cohorts, so the bias path must not carry everything | not tried |
| Input fixes (ours) | per-layer RMS normalization of the summary; pooling over all prompt tokens instead of the last template token | roadmap C1/C2, not run |
| Reconstruction pretraining (their F3) | pretrain the generator by F0 regression onto oracle LoRAs, then fine-tune end to end | not tried |

## 5. Suggested order and cost

| Step | Content | GPU | Agent work |
|---|---|---|---|
| 1 | A1, A2, A3, A7, A8, E2, E5 on mix24k-1.1B and mix100k-r64 (forward + one gradient pass) | ≈ 3 h | one script on `offline_eval` + `analysis/common.py`; hooks for z̄ / bias-only / mean ΔW |
| 2 | C1, C2, C4, D1, D2, D5 on the same checkpoints | ≈ 1 h | same script, + per-stage hooks |
| 3 | A6 twin (24K, 2 ep) and E1 seeds (2 static runs), in parallel on 2 GPUs | ≈ 2 + 4 h | one config flag (constant input) |
| 4 | B1, B2 per-row oracles on 64 rows; B4/B5 gradient budget | ≈ 6 h | a one-row LoRA fitting loop; reuse `grad_maps.py` |
| 5 | B3 routed statics (8 short runs), plus snapshot rerun for C3/C5/D3 | ≈ 8 h | snapshot option |
| 6 | Decide: H2 confirmed → stop generator design on this data, move to data where adaptation is per-prompt (F2-style). H1/H4/H5 → ladder F0–F2, then targeted fixes from §4 | | |

**Deliverable per step:** one report in `reporting/dflash/` (measurements, controls and their pass rules, limits), following `reporting/README.md`.

## 6. What would change our current reading

- **Twin ≈ generator (A6), or c* ≈ 0 (A8):** the 24K "generator beats static r32" result is a parameterization effect. The fair baseline is then a higher-rank or higher-LR static, which ABL24K already shows wins.
- **Unseen-half oracle headroom ≈ 0 (B1):** per-prompt LoRAs cannot help on this drafter and data (their PG-19 verdict). Generator work should move to settings with real per-prompt variation (thinking mode, other sources, other drafters) or to cheaper non-generator adaptation.
- **Oracle headroom > 0, own ≈ same-category swap, starved content gradient (B1 + A1 + C):** classic bypass collapse (their H1). The fixes in §4 then have a target, and the ladder (F) tells which one to try first.
