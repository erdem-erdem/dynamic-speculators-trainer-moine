# Research note: the "core matrix" generator, and what Text-to-LoRA / Doc-to-LoRA offer

*2026-11-14. Research and idea document only: no code was edited, no job was run. Provenance tags: **[repo]** = read from our code/reports; **[paper]** = stated in the paper (section/table given); **[mine]** = my own reasoning or arithmetic, unverified by experiment. Paper numbers were obtained through an HTML-to-text fetch of arXiv, not by reading the PDFs myself; treat exact figures as "as extracted" and re-check before citing in a report.*

## 0. Summary

- **Core-matrix idea (ΔW = B·M(x)·A).** It is a sound, cheap design (generator outputs 20·r² numbers instead of 4.26M·(r/32)·2). Its closest relatives are LoRA-XS (trains only an r×r matrix between frozen SVD factors) and AdaLoRA's P·Λ·Q. [mine] The key structural fact: with a linear head, M(x) is *affine in the decoder code z*, so ΔW(x) is affine in z. A core generator with code width W is therefore exactly a **mixture of W+1 LoRAs that all share one (A,B) subspace**. That matches what we observe (2–3 effective dynamic directions) and makes it a restricted version of G1.
- **Expressivity limit [mine]:** M can never change the row/column space of ΔW. It can only choose which rank-r subspace directions are used and how they are mixed. So a core is only as good as the shared subspace. This is why option (b)/(c), building the subspace from the strongest static result (r512), is the interesting arm: it is the only formulation that can *combine* the static-rank win (+0.046) with the dynamic win (about +0.03 at r32).
- **Recommended first experiments (each ≈ 2 h on one H100, 24K, 2 epochs, KL + D-PACE):** (1) CORE-S: static r512 A,B warm-started and frozen, M = I + g(x) with a shared head, r512 (generator ≈ 34M with W=128); control = the same run with g forced constant and static-r512 itself. (2) CORE-J: joint training at r128, M = I + g(x), vs static r128 (+0.027) and shared-head generator r128 (+0.026). (3) Category one-hot and mean-M controls fall out of the same code path. Expected outcome (my guess, not a measurement): CORE-S ≈ static r512 + 0.00 to +0.02 τ; large uncertainty, because the dynamic gain at r32 may already be absorbed by what r512 provides (ABL24K says the generator gain matches r128 to r256 of extra static rank).
- **Papers.** The user's two URLs are identical (https://arxiv.org/pdf/2506.06105, Text-to-LoRA). The second paper is almost certainly **Doc-to-LoRA: Learning to Instantly Internalize Contexts**, arXiv 2602.15902 (Charakorn, Cetin, Uesaka, Lange; ICML 2026), https://arxiv.org/abs/2602.15902. **Flag: duplicated URL; I assumed Doc-to-LoRA.**
- **Most transferable ideas:** T2L's *one shared backbone emitting one r×d matrix per call, selected by A/B / module / layer embeddings* ("M" variant) and its *rank-wise "S" variant* (head is d_emb×d, 5M params total). Our shared-head runs already took the first step. D2L's *Perceiver with r latent queries, one latent per rank* is the natural way to give a generator a token-level input, but our bottleneck is not input richness per se (see Open questions).

---

## 1. Core-matrix design

### 1.1 Notation and our conventions [repo]
ΔW = scale · B · A, A is r×in, B is out×r, scale = α/r = 2. Adapters: q,k,v,o on 5 layers = 20 modules. Σ(in+out) per layer = 8192 (q) + 5120 (k) + 5120 (v) + 8192 (o) = 26,624, ×5 layers = **133,120 per unit rank** (matches 0.133M). Core version: ΔW = scale · B · M · A, M is r×r per module.

### 1.2 Formulations
| id | Description | A,B | M | Comment [mine] |
|---|---|---|---|---|
| (a) joint | A,B static, learned jointly with a generator emitting M(x) | trained | g(x) | Gauge freedom (below) means M's scale/rotation and B's are tangled; needs M0 = I or fixed-norm A,B. Cheapest to code. |
| (b) two-stage | Train static A,B (e.g. from the static r512 LoRA, or a truncated SVD of it), freeze; generator emits M | frozen | g(x) (+I) | Cleanest: subspace known good; no gauge. Exactly LoRA-XS style, but with M generated. Loses the ability of A,B to adapt to dynamics. |
| (c) residual | M = I + g(x), or M0 + g(x) with learned M0 | trained or frozen | static + dynamic | With M = I the static LoRA is *inside* the model (this is "static + residual", suggestion D1 in future_suggestions.md). M0 learned jointly with trainable A,B is gauge-redundant: absorb into B, so use I. With frozen A,B, a learned M0 is useful (cheap fine-tune of the static LoRA). |
| (d1) diagonal | M = diag(s(x)) | trained/frozen | r scales | = G3. 20·r numbers. Smallest dynamism; equals "per rank-direction scale". After SVD of the static ΔW, the scales are singular-value modulations (AdaLoRA-like). |
| (d2) low-rank | M = I + U(x)V(x)ᵀ, rank k | | 2·r·k numbers | Interpolates between diagonal-ish and full. With 2–3 observed dynamic directions, k = 2–4 may suffice. |
| (d3) block-diag | M = blockdiag(M_1..M_g), blocks r/g | | r²/g numbers | Cuts head by g. Plausible if A,B are SVD-ordered (blocks = singular-value bands). |
| (e) sharing | per-module head (20 heads) vs one head for all 20 modules, vs per-type | | | **All M are r×r regardless of module shape**, so *one* head can serve all 20 modules (module/layer embeddings tell them apart). This is a real advantage over the current A/B heads (in/out differ per type). Per-module heads are only worth it if shared is shown to lose. |
| (f) granularity | one M per module (20), per module type shared across layers, or per layer | | | Per-module is the default. Layer-shared M is a strong, tiny ablation. |

### 1.3 Parameter counts [mine, arithmetic]
Static A,B = 133,120·r. Core output per prompt = 20·r². Linear-head size = (W+1)·20·r² if one head per module (full per-module), or (W+1)·r² if a single head is shared by all 20 modules (module/layer embeddings feed the decoder, as in `head_sharing="per_type"` today). Existing generator rows from ABL24K [repo] for comparison.

| rank r | static A,B (= static LoRA) | M numbers / prompt (20·r²) | per-module heads, W=128 | per-module heads, W=512 | single shared head, W=128 | single shared head, W=512 |
|---|---|---|---|---|---|---|
| 32 | 4.26M | 20.5K | 2.64M | 10.5M | 0.13M | 0.53M |
| 64 | 8.52M | 81.9K | 10.6M | 42.0M | 0.53M | 2.1M |
| 128 | 17.0M | 328K | 42.3M | 168M | 2.1M | 8.4M |
| 256 | 34.1M | 1.31M | 169M | 673M | 8.5M | 33.7M |
| 512 | 68.2M | 5.24M | 676M | 2.69B | 33.8M | 135M |

(+ ≈ 12M for encoder/condition/decoder at current widths [repo, future_suggestions §6]; decoder size adds little.) For reference [repo]: current generators 1.1B / 0.23B / 0.45B; static r512 = 68M, r1024 = 136M. So a core generator on a r512 subspace with a *shared* head is **68M (A,B) + ≈ 34M to 135M ≈ 100–200M total** versus the 1.1B current generator; under a 500M cap every shared-head configuration fits, per-module heads fit up to r256/W=128.

Serving cost [mine]: M(x) is r² per module per request; applying it per token is r² multiply-adds against r·(in+out) for A and B, i.e. r²/(r·(in+out)) = r/(in+out): 6% of the A,B cost at r512 for 8192-dim modules (12% for k,v). Cheap. Memory for the generator head is small for shared heads (≤ 0.27 GB fp16 at W=512, r512), versus 2–4 GB for today's.

### 1.4 Expressivity
- **What M can do:** for fixed A,B, ΔW ∈ {B·M·A}: the r²-dim space of matrices with column space ⊂ col(B), row space ⊂ row(A). It can rotate, rescale, mix and (if M singular) rank-reduce within that subspace. Full M can represent every ΔW inside that subspace. Diagonal M can only rescale directions *as defined by the pairing of A and B rows/columns*, which is arbitrary unless A,B come from an SVD of a static ΔW; so diagonal is only meaningful with (b)-style SVD-aligned factors.
- **What it cannot do:** move outside the subspace. If the best dynamic LoRA for a medical prompt needs input directions the static A never reads, no M can produce it. The generated-A,B model has no such restriction (but evidence says it only uses 2–3 directions anyway).
- **Relation to the observed 2–3 effective dimensions [repo: ABL24K, D13; mine: reading]:** M(x) = M_0 + Σ_k z_k(x) M_k with W+1 terms is linear in z. If the learned z only varies along 2–3 directions (category routing), then the core generator needs only 2–3 matrices M_k, i.e. ≈ 3·20·r² numbers: at r512 that is 15.7M. All remaining head capacity is idle in today's generators too. This argues for (d2)/low-W cores: W = 4–16 is enough to represent what is observed, and the head becomes tiny (W+1)·r² ≈ 4.5M at r512, W=16, shared. It also means the core design *as a tool* is equivalent to G1 (basis mixture) with the bases sharing A,B; if G1 with free bases learns a constant router, the core will too.
- **Static r512 plus per-prompt M:** with A,B ← r512 static and M = I + g, ΔW(x) = ΔW_static + B g(x) A. The static part keeps the +0.046 gain; g adds a category-level correction with 2–3 degrees of freedom of size r². Whether g's correction adds on top is exactly the open question (ABL24K: "gain matches r128–r256 of extra static rank", i.e. they may be the same gain obtained two ways; additive vs redundant is unknown).

### 1.5 Gauge / identifiability [mine]
ΔW = B M A is invariant under B→B P⁻¹, M→P M Q, A→Q⁻¹ A for any invertible P,Q. Consequences:
- Joint training (a): M and B are redundant in the *mean*: M0 can be absorbed into B and A. Parametrize M = I + g(x) (so the static part is B·A) and the redundancy is limited to the invariances of the static part itself (already present in plain LoRA, harmless). But the *scale* of g(x) relative to B's scale is a gauge: gradients to g scale with ‖A‖‖B‖; with B zero-init (standard LoRA) g gets zero gradient at step 0 (B=0 ⇒ ∂L/∂g = 0): **B zero-init + M generated is a saddle**. Init must break this: either B nonzero small, or M0 learned (static path) so the static LoRA warms up first (the LoRA-XS paper avoids it by freezing both).
- With frozen factors orthonormalized (A rows orthonormal, B columns orthonormal, e.g. from SVD of the trained static ΔW with singular values put into M0 = diag(S)), M's entries are in a canonical basis; this gives a well-conditioned, regularizable parameter (‖g‖_F relative to ‖M0‖_F is meaningful; weight decay is meaningful).
- If A,B are trained jointly and not normalized, consider fixing ‖rows of A‖ by renormalization, or just parameterize M in SVD-aligned form post-hoc by retraining stage 2.

### 1.6 Training dynamics, pitfalls
1. **Collapse to constant M** is the default outcome here (precedent [repo]: collapse at 10K math; 24K only 2–3 directions). A core generator reduces capacity but does not fix the input signal problem. Controls (mean-M, one-hot) are essential.
2. **LR separation:** static A,B at the static-LoRA LR; generator at its own. With M = I + g, scale g's output (init head weights at zero as now, so g(x)=0 at start and the model starts as the static LoRA). The head bias is a free static M offset: that is the learned M0, so (c) comes for free by not zeroing... but note bias at zero ⇒ M = I exactly.
3. **Initialization for (a) joint:** B ≠ 0 (e.g. B ~ small Kaiming) or two-phase schedule (static warm-up 300 updates, then enable g) to avoid the saddle in 1.5.
4. **Regularizers:** weight decay on the head only; optional ‖g‖² penalty relative to ‖M0‖; dropout on z; optional auxiliary category-prediction head from z (suggestion D4, to retain information without forcing diversity).
5. **Precision / scale:** M is applied inside `scale·B(M(Ax))`; α/r scaling at r512 gives scale = 2 regardless of r, as in our static r512 runs [repo]; fine.
6. **Merging and serving:** ΔW = B·M·A is mergeable per request (one matmul r×r then two big ones; or merge costs out·r·in flops: too heavy per request). Unmerged per-request path: Ax (shared across requests only in the input, but per-request x anyway), then M_i (r×r) per request, then B shared: **requests in a batch share A and B, only the small M_i differs**, which batches as a bmm of r×r matrices. This is much friendlier than multi-LoRA serving with distinct A_i,B_i (suggestion §6 in future_suggestions notes that as the real deployment cost). Static part B·A (the M=I part) can be merged into the base weights once; only B·g·A remains dynamic, rank ≤ r.

### 1.7 Prior work for this exact structure
- **LoRA-XS** (Bałazy et al., 2024, arXiv 2405.17604) [from memory, not re-fetched]: W + U_r Σ_r R V_rᵀ, where U_r,Σ_r,V_r are the top-r SVD of W (frozen) and only the r×r matrix R is trained. Closest structure to our (b): frozen SVD-derived subspace, trainable r×r core. Difference: R is static per task; we generate it per prompt, and our subspace could come from a trained static LoRA rather than from SVD of W (a pretrained-weight SVD is not obviously aligned with the *drafter-adaptation* direction; ΔW-derived subspace is better matched).
- **AdaLoRA** (Zhang et al., 2023, arXiv 2303.10512): ΔW = P Λ Q with orthogonality regularizer on P,Q and importance-based pruning of Λ. Relevant for: (1) orthogonality regularizer as the gauge fix, (2) diagonal M (d1) = Λ. Not generated.
- **VeRA** (Kopiczko et al., 2023, arXiv 2310.11454): frozen shared random A,B; trains two scaling vectors (d, b) per layer: ΔW = Λ_b B Λ_d A. A diagonal-core special case with *random* frozen subspace; shows tiny numbers of trainable parameters can work at high shared rank. Our diagonal (d1) with a *trained* r512 subspace is the VeRA idea with a better subspace.
- **Tied-LoRA** (Renduchintala et al., 2023, arXiv 2311.09578): A,B tied across layers, with per-layer trainable scaling vectors; relevant for the layer-shared-subspace variant (e): sharing A,B across the 5 layers is an option to cut static params 5× (q,o same shapes across layers).
- **DoRA** (Liu et al., 2024): magnitude/direction decomposition of weights. Related only loosely (per-column magnitude vector); could be the dynamic part (generate magnitudes), but adds a different, non-low-rank dynamic path. Low priority.
- **HyperLoRA / HyperNetworks in general:** T2L cites Lv et al. 2024 (HyperLoRA); Hyperdecoders (Ivison & Peters 2022) generate adapters from encoder states and are a T2L baseline [paper, Table 2]; "generate only a small core" is the standard way to shrink the output layer of a hypernetwork. I did not verify the details of these beyond T2L's citations.
- **Mixture-of-LoRA families (LoRA-Mixer, MoLE, LoRAHub/Arrow routing):** per-token or per-prompt routing weights over a bank of LoRAs. Core-with-W-codes is the same family with a *shared subspace* bank. T2L compares against Arrow Routing (70.7 on the 8-task average, from Ostapenko et al. 2024) [paper, Table 2]. I did not verify LoRA-Mixer/MoLE details; cite from memory only.

### 1.8 Proposed experiments (24K, 2 epochs ≈ 2 h each, one seed each; add a second seed for the 2–3 most interesting arms)
Common: KL + D-PACE, same eval sets as ABL24K, report test and OOD Δτ vs static r32 and vs the matched static of the same rank, with paired CIs, plus D13 (cross-row cosine / effective rank of ΔW).

| Arm | Formulation | Rank | Trainable | Generator size | Matched / control |
|---|---|---|---|---|---|
| C0 | (c) joint, M = I + g(x), shared head, W=128; B nonzero init or 300-update warm-up | 128 | A,B + generator | 17M + 2.1M + 12M | static r128 (+0.027), shared-head generator r128 (+0.026) [repo] |
| C1 | (b)/(c) two-stage: A,B = frozen r512 static (from the ABL24K checkpoint, orthonormalized via SVD: M0 = diag(S) learned, g generated) | 512 | head only (+M0) | 34M (W=128 shared) | static r512 (+0.046) as floor; "mean-M": evaluate the arm with g replaced by its mean over the eval set (no extra training) |
| C2 | C1 with static retrained jointly (A,B unfrozen, small LR) | 512 | all | 68M + 34M | tests whether A,B adapting beats frozen |
| C3 | diagonal core on C1 subspace (d1) | 512 | head | 20·512 = 10K numbers, head ≈ 0.07M at W=128 | C1; isolates "rescale only" |
| C4 | low-rank core (d2) k = 4, W=16 | 512 | head | ≈ 1M | C1 |
| C5 | C1 with generator input replaced by category one-hot | 512 | head | same | upper bound on category routing with this machinery (suggestion C5a) |
| C6 | C1 with layer-shared M (one M per module type, not per module) | 512 | head | smaller | ablation |
| C7 | C1 trained, then evaluate with each row's category-mean M (the "category-mean M" control) vs per-row M | - | none | - | tells whether per-row information beyond category matters |

**Primary reads:** (i) C1 − static r512 on test and OOD: does a dynamic core add to the high-rank static? (ii) C0 − static r128: does the core beat a matched static at a rank where capacity is still limiting? (iii) C5 vs C1: how much is category routing. **Expected outcomes [mine, speculative]:** C1 ≈ static r512 within ±0.01 (if the dynamic gain overlaps with rank gain), perhaps +0.01 to +0.02 if additive; C0 ≈ static r128 + 0.005 to 0.01; C3 ≈ C1 minus most of the dynamic effect; C5 ≈ C1. A result where C1 beats static r512 by > 0.02 with CIs excluding zero would be the first evidence that dynamism is additive with rank. A null C1 would say, consistent with ABL24K, that "router on top of a rank-32 LoRA" is just a substitute for rank.
Cost: C1/C3/C4/C5/C6 train only a small head over frozen A,B, so memory and time are lower than a full generator run (drafter forward/backward dominates, ≈ 1.2 s per update [repo]); budget ≈ 2 h each. Order: C1, C0, C5, C3, then C2, C4, C6, C7.

### 1.9 Code sketch (model.py, no edits made) [mine]
Goal: emit **(A, B·M)** so `factor_parameters`, `detached_factors`, `backward_factors` and `AdapterLinear` (which computes b @ (a @ x)) stay unchanged, since B_eff = B_static · M(x) is an out×r matrix per row.
1. New `CoreHeads` in `LoRAGenerator.__init__` (flag `parameterization="core"`): per-module `nn.Parameter A_s[name]` (r×in) and `B_s[name]` (out×r) (init: A kaiming, B small random for joint; loaded from a static checkpoint for two-stage and `requires_grad_(False)` or lower LR param group). One head `nn.Linear(decoder_width, r*r)` shared across all modules (or `ModuleDict` per module for the per-module variant), weight zero-init, bias zero so M = I + view(r,r) (add `torch.eye(r)` in forward; do not put eye in bias, to keep weight decay meaningful).
2. `decode_factors`: for each module, `g = head(decoded).view(*leading, r, r)`; `M = eye + g`; `A_out = A_s.expand(*leading, r, in)` (an expand view, no copy); `B_out = B_s @ M` via `torch.matmul(B_s, M)` → shape (*leading, out, r). Return `(A_out, B_out)`. Gradient from `backward_factors` flows through the expand and matmul into A_s, B_s, head with no other changes (torch.autograd.backward handles views).
3. Diagonal variant: `s = head(decoded)` of size r; `B_out = B_s * (1 + s).unsqueeze(-2)`. Low-rank variant: `U,V` heads of size r·k each; `M = I + U Vᵀ`; compute `B_s @ U` first (out×k) then `@ Vᵀ` to keep cost low.
4. Optimizer groups in `campaign_train.py`: static A_s,B_s as a separate param group (LR from the static-LoRA config); heads/decoder as now. Checkpoint save/load must include A_s,B_s (they are generator parameters, so state_dict covers them). For two-stage: a loader that takes a static adapter state_dict (`...q_proj.a/b` keys), optionally SVD-orthonormalizes `B·A` per module to (B_s = U√S, A_s = √S Vᵀ or the orthonormal form with M0 = diag(S)).
5. Eval/merge: `ΔW(x) = scale · B_s M(x) A_s`; the existing merge path should work as it consumes the generated pair. For a batched serving prototype, keep `A_s,B_s` outside the generator output and emit only M.
6. Tests worth adding when implemented: g=0 reproduces the static model exactly; shapes with `leading=(n,)` cohorts; `functional_call` gradient reaches A_s.

---

## 2. Text-to-LoRA (T2L)

**Paper used:** "Text-to-LoRA: Instant Transformer Adaption", Charakorn, Cetin, Tang, Lange (Sakana AI), arXiv 2506.06105 (v2), https://arxiv.org/abs/2506.06105, code https://github.com/SakanaAI/text-to-lora. (This is the paper behind *both* URLs the user pasted.)

*Source of everything below unless tagged [mine]: the arXiv HTML text, via fetch. Section/table refs are as the fetch labelled them.*

**Problem setting.** Generate a task-specific LoRA for an LLM from a natural-language task description, in a single forward pass, zero-shot for unseen tasks (Abstract, Sec. 1).

**Generator formulation.**
- *Input* (Sec. 3, Eq. 3–4; App. F, Fig. 8–9): for each (module type m, layer l) the concatenation of a task-description embedding f(z), a module-type embedding and a layer-index embedding (both 32-D, learnable). Task embedding from gte-large-en-v1.5 (1024-D) (Sec. 4, 5.2); a Mistral-7B last-token activation gives comparable results (Sec. 5.2, Table 4). Task encoder: Linear 1024→64 + LayerNorm.
- *Backbone* (App. F, Fig. 8): MLP mixer 128→512→128 (SiLU, dropout 0.05), two residual MLP blocks, then a 128→512→512 MLP; d_out = 512. It is an MLP, not a transformer.
- *Batching:* all (m,l) pairs are batched, so all modules and layers' LoRAs come out of one forward pass; the same backbone and head are shared across all layers/modules.
- *Targets:* LoRA rank 8, α = 16, dropout 0.05, rsLoRA, on q_proj and v_proj of every block; Mistral-7B-Instruct (main), also Llama-3.1-8B-Instruct and Gemma-2-2B-Instruct (App. A). Target LoRA size 3,407,872 params (App. F).
- *Output variants (Sec. 3.1, App. F):*
  | Variant | Output per forward call | Head | Total hypernetwork |
  |---|---|---|---|
  | **L** (large) | A and B together (2·r·d) | d_out × 2·r·d | 55,252,992 (≈55M) |
  | **M** (medium) | one r×d matrix; an A/B embedding selects A or B | d_out × r·d | 34,282,240 (≈34M) |
  | **S** (small) | one length-d vector = one rank of one matrix; A/B embedding + rank embedding (8 ranks) select it | d_emb × d | 4,923,392 (≈5M) |
  M and S produce all r ranks and both factors by calling the backbone repeatedly with different embeddings (so the batch is (m,l,A/B,rank)). Output bias initialization: "Bias-HyperInit" for L; scaled bias ranges for M and S; without this training was unstable (App. G).
- *Interesting structural detail for us:* T2L's S variant builds every rank vector from the same 512→d head, differing only by embeddings. Their "output = one rank at a time" is the chunking idea. Counting the v_proj 4096→1024 shape under their single-head design is not explained in the text I retrieved; I did not resolve it.

**Training objectives (Sec. 3.2–3.3).**
- *Reconstruction* (Eq. 6): L1 between generated and pre-trained LoRA weights (targets z-scored per entry, App. H). Needs a LoRA library: 9 benchmark LoRAs (≈10 min), or 479 SNI LoRAs (≈10 h on one H100, 100K epochs).
- *SFT* (Eq. 5): end-to-end through the frozen base LLM on task data; no LoRA library.
- Data: 500 Super-Natural-Instructions tasks (Lots-of-LoRAs), 479 after validation hold-out and contamination removal; GPT-4o mini descriptions, 200 per task (128 used for SFT) (Sec. 4, App. J, L).

**Inference cost (App. I).** About 0.856 TFLOPs per instance for T2L (encoder 0.029, hypernetwork ≈ 5e-6, base LLM 0.827) vs 4.177 TFLOPs with 3-shot ICL; the paper reports a >4× FLOP reduction. Hypernetwork cost is negligible.

**Main results.**
- Table 1 (9 benchmarks, reconstruction): base 55.8; task-specific LoRAs 73.3; T2L L/M/S ≈ 73.0–73.5, sometimes above the oracle, attributed to lossy compression acting as regularization.
- Table 2 (zero-shot on 10 unseen tasks, SFT-trained): Mistral base 55.8, 3-shot ICL 61.0, multi-task LoRA 66.3, Hyperdecoders 67.3; T2L S 65.9, M 67.5, L 67.7. Oracle task-specific LoRAs 75.8 (8-task average); Arrow routing 70.7. Zero-shot T2L does not reach task-specific LoRAs.
- Other bases (App. A): T2L L beats multi-task LoRA by 0.4 (Llama-3.1-8B) and 1.2 (Gemma-2-2B w/ ICL) points.

**Ablations relevant to us.**
- *Size vs quality:* L ≥ M > S at 479 tasks (67.7 / 67.5 / 65.2, Table 3); S does not benefit from extended training. So very small heads lose ≈ 2 points where the generator must separate hundreds of tasks. We have ≈ 4 categories, a much weaker demand.
- *Reconstruction vs SFT (Sec. 5.4, Table 6):* reconstruction averages 61.8 vs SFT 66.3 zero-shot. Parameter-space similarity of A/B does not track task similarity (App. D, Fig. 6), which the authors give as the reason reconstruction transfers poorly. Reconstruction in ΔW space did not help (App. K).
- *Compression vs reconstruction error (Sec. 4.1, Fig. 3):* matching oracles when per-element L1 < 1e-4, ≈ 65% of oracle at higher error.
- *Scaling tasks:* helpful only if compute scales too (Sec. 5.1, Table 3; App. C, Table 10).
- *Description alignment (Sec. 5.3, Table 5):* wrong-task descriptions 51.4, random strings 63.5 (vs 72–73 correct): the generator really does condition on the input, unlike ours on 10K math.
- *Embedding model (Sec. 5.2):* little difference between gte and Mistral activations.

**Limitations (Sec. 7).** Description quality; LoRA only as output space; zero-shot gap to oracle LoRAs; cross-size transfer unexplored; far-from-SNI distributions untested.

**Caveat for transfer [mine]:** T2L's targets are *low-rank-8 LoRAs on 479 distinct tasks with supervised diversity*; their dynamism is the whole point. Our target (a drafter's LoRA varying along 2–3 directions across rows) has a much smaller dynamic range, and we start from a strong static adapter.

---

## 3. Doc-to-LoRA (D2L)

**Paper used:** "Doc-to-LoRA: Learning to Instantly Internalize Contexts", Charakorn, Cetin, Uesaka, Lange (Sakana AI), arXiv 2602.15902 (v1; ICML 2026), https://arxiv.org/abs/2602.15902, code https://github.com/SakanaAI/doc-to-lora. Identified by search; **the second URL the user pasted was a duplicate of the T2L link, so this selection is my inference.**

**Problem setting (Sec. 1, 3.1).** ICL is transient and its KV-cache grows with context; context distillation (CD) internalizes a document in weights but costs a fine-tune per document (≈ 40 s even with oracle queries, Table 1). D2L meta-learns a hypernetwork that emits, from the document, a LoRA approximating CD in one forward pass.

**Generator formulation (Sec. 3.2, App. B.2, H).**
- *Input/encoder:* the frozen target LLM (gemma-2-2b-it, also Mistral-7B-v0.2 and Qwen3-4B-Instruct-2507 in App. E) provides per-layer token activations Z ∈ R^{L×N×D}; the hypernetwork reads layer l−1 activations to produce layer l's LoRA.
- *Architecture:* a **Perceiver** (cross-attention only, 8 blocks, hidden 512; modality-projection MLP 2304→9216 in the printout) with **r learnable latent queries** Q_m (r×d_q) attending to K(Z), V(Z), giving U_l (r×d_u) (Eq. 6): **number of latents = LoRA rank**. Two per-layer heads then map each latent to one row of A_l and one column of B_l. A learnable scalar α_l per layer scales the update (Eq. 5). I did not find in the text a layer embedding; the heads are per-layer.
- *Targets:* LoRA on the **down-projection of every MLP block**, rank 8 per chunk. ≈ 309M hypernetwork parameters (Sec. 5). Target LoRA size not stated; my estimate for gemma-2-2b (26 layers, 9216→2304, r=8) ≈ 2.4M. [mine, derived]
- *Chunking (Sec. 3.2, 4; Eq. 9):* long contexts split into K chunks (8K-token chunks in the main setup; 1024-token chunks at NIAH inference), each processed independently; A_l stacked vertically and B_l horizontally so total rank = r·K. This is rank-concatenation, not averaging.

**Objective and data.**
- *Query-independent CD* (Eq. 3–4): KL(p_θ(y|x,c) ‖ p_{θ+H(c)}(y|x)) over generated queries and self-responses, with top-16 logits stored as targets. KL beat next-token prediction on SQuAD (0.819 vs 0.763; Table 6, early checkpoint). Compare: our losses tie in distribution [repo].
- *Data:* ≈ 900M-token FineWeb-Edu subset plus PwC, SQuAD, ROPES, DROP; ≈ 3.2M unique contexts; 10 queries per sample from gemma-3-12b-it. 80K steps single-chunk then 20K steps with random chunking. About **5 days on 8 H200** (Sec. 6). NIAH variant trained with cross-entropy, 640K short samples, ≈ 33 h on one H200 (App. A).

**Inference cost (Table 1, 2WikiMultihopQA).** D2L batched 0.209 s, 11.5 GB peak vs CD with oracle query 40.2 s, 7.8 GB (normalized performance 0.857 vs 0.901), D2L iterative 0.551 s, 3.8 GB. The internalized model adds < 50 MB at 128K NIAH tokens versus > 12 GB of KV cache.

**Main results.** SQuAD 82.5% of the ICL upper bound (Sec. 5.1.1); near-perfect NIAH to ≈ 40 chunks (40K tokens) despite training on 32–256 token inputs (Sec. 4); a VLM→LLM transfer (Imagenette 75% accuracy; Table 2).

**Ablations relevant to us.** Rank 16 > rank 8 on SQuAD/DROP (0.896/0.711 vs 0.814/0.655; Table 7), i.e. rank helps even here; the KL objective helps (Table 6); removing QA data hurts DROP (Table 5); knowledge interference (Table 8).

**Limitations (Sec. 6, App. G, C.4).** One hypernetwork per target LLM; ≈ 5 GPU-days to train; gap to ICL; interference with unrelated queries; only LoRA studied; Mistral-7B struggles with long contexts (unexplained).

**Caveat [mine]:** D2L's input is a *document* with rich per-token content and 3.2M contexts of supervision; our input is the last prompt token's hidden state (5 layers × 4096), with 24K rows. D2L's latent-per-rank design is a clean precedent for "rank = number of latents", i.e. the generator's capacity is placed in cross-attention, not in a giant linear head.

---

## 4. What we can adopt for DFlash (own mapping, with estimates)

Our shapes: 20 modules, Σ(in+out) = 133,120·r, current head (W+1)·20·... Heads: per-module A+B head size (W+1)·r·Σ_modules(in+out) = (W+1)·133,120·r. At W=256, r=32: 1.09B, matching [repo].

| # | Idea | Source | Mapping onto our generator | Params (our shapes) | Under 500M? | Comment |
|---|---|---|---|---|---|---|
| 1 | **Core output (M only, shared head)** | Our own idea; LoRA-XS-like | Section 1 | 34M (W=128, r=512) generator head + 68M static | yes | Highest priority. |
| 2 | **T2L "M" variant: one backbone call → one matrix, A/B embedding selects factor** | T2L Sec. 3.1, App. F | One head per distinct matrix shape, with an A/B flag plus our module/layer embeddings. All A are r×4096; B is 4096×r (q,o) or 1024×r (k,v). Three shape classes: A(4096), B(4096), B(1024). | (W+1)·r·(4096+4096+1024); W=256, r=32: 257·32·9216 ≈ 76M (r=64: 152M) | yes | Versus 0.23B today (heads shared across layers only); the further 3x cut comes from sharing across q/o/k/v and A/B. [mine, arithmetic] |
| 3 | **T2L "S" variant / rank-wise output** | T2L Sec. 3.1 | Emit one rank vector at a time: input adds a rank embedding (r entries), head d_emb×in. Batch dimension becomes (20 modules × 2 factors × r ranks) = 1,280 calls at r=32; 20·2·512 = 20,480 at r512. | head ≈ (W+1)·4096 ≈ 1M at W=256; total generator ≈ 12M + embeddings | yes | T2L: S loses about 2 points at 479 tasks vs L. With our 2–3 effective dims the capacity cut is probably harmless. Compute grows with r (20K calls), but each call is tiny. Makes generated-A,B at r512 affordable, which the full head cannot be (r512 W=32 is already 0.46B [repo]). |
| 4 | **D2L Perceiver, rank = latents** | D2L Sec. 3.2 | Replace encoder+condition by cross-attention from r latent queries to the prompt's token hidden states, two small per-layer heads mapping latent → row of A / column of B. | Perceiver ≈ 10–50M + heads (r latents × (in or out)) ≈ r·4096·W' ≈ 4M for r=32, W'=32 | yes | Needs per-token stored hidden states for prompts; suggestion C2 notes the store already holds them. Only worthwhile if conditioning, not output parameterization, is the limitation. |
| 5 | **Chunking = rank concatenation** | D2L Eq. 9 | Generate K independent rank-r LoRAs and concatenate (A stacked vertically, B horizontally) to get rank K·r; or per-chunk of the *response* (G6). | K·(head for r) | yes | Gives a route to rank 512 from rank-32 generator calls: 16 chunks × r32. Equivalent to rank-wise generation (item 3) in spirit. |
| 6 | **Reconstruction pretraining** | T2L Sec. 3.2 | Pretrain on a bank of statics: per-category static r32/r512 LoRAs (the A1 oracle in suggestions) as targets, generator learns to output them from category-typical rows; then fine-tune end-to-end. | no new params | yes | T2L evidence is mixed: reconstruction generalizes worse than SFT (Table 6: 61.8 vs 66.3) because parameter space similarity is poorly aligned with task similarity (App. D). Useful only as an *initialization / warm start*, not as the main objective; our `factor_loss` exists [repo]. With 4 categories the bank is tiny. |
| 7 | **Bias-HyperInit** | T2L App. G | Our heads already init bias = kaiming for A, 0 for B (static LoRA in the bias) [repo]; this is the same idea. Check scaled-bias ranges for r512 heads (variance grows with r). | none | yes | Nothing to adopt except to verify the init at larger ranks. |
| 8 | **Per-layer learnable scalar α_l** | D2L Eq. 5 | Add per-module learnable scalar on ΔW (init 1 or scale); also lets the generator not need magnitude through the head. | 20 numbers | yes | Trivial; unlikely to move results; could make per-layer effect size inspectable. |
| 9 | **KL objective** | D2L Eq. 3, Table 6 | Already our default. | | | Confirms our choice; no new action. |
| 10 | **Description-alignment sanity test** | T2L Sec. 5.3, Table 5 | Positive control for conditioning: swap/randomize the generator input at eval (wrong category, random vector) and measure τ loss. | none | | We have the donor control (M7) already [repo]; cheap to extend. |

---

## 5. Ranked adoptable ideas (expected value / cost)

1. **Core generator on a frozen r512 subspace (C1) plus mean-M / one-hot controls.** One short code addition (~60 lines), small head, answers "is dynamism additive with rank". Highest information per GPU-hour.
2. **Core joint at r128 (C0).** Tests the matched-static comparison at a rank where our generators sat (+0.026 to +0.032) with ~15–30M generator instead of 0.45B.
3. **Single shared head over all 20 modules in the core design** (free, because M is r×r) and layer-shared M ablation.
4. **T2L S/rank-wise generator** if one still wants *generated A,B* at high rank: 12M generator vs 0.46B at r512. Moderate code change (rank embedding, batch of rank vectors); tests whether generated full factors at r512 can ever approach static r512.
5. **Diagonal / low-rank core** (C3/C4): ablation floors; cheap given C1's code.
6. **Reconstruction warm start from category statics** (T2L): low value per T2L's own Table 6; only if joint training fails to leave the constant solution.
7. **D2L Perceiver / token-level conditioning:** the highest-cost idea; only after controls (C5) say input information is the bottleneck.
8. **Per-layer α_l, Bias-HyperInit scale check:** trivial but tiny expected value.

## 6. Open questions

- Is dynamism additive with static rank at all? ABL24K says the generator gain ≈ the gain of r128–r256 static; the core on r512 is the direct test. If redundant, the whole generator line may be a rank-substitute and the remaining case is OOD/prompt-specific tails.
- The 2–3 observed directions are mostly category routing [repo]. A core generator needs only ~3 M_k; does that mean a **category-routed static r512** (4 LoRAs, 0 generator) matches everything? (Suggestion A1 oracle; C5/C7 give a partial answer.)
- Gauge/saddle risk for joint training (B = 0 and M generated): untested; choose the warm-up variant if C0 stalls at zero gradient.
- T2L's v_proj shape handling for GQA and the exact S/M batch layout were not resolved from the extracted text; read App. F / the code at github.com/SakanaAI/text-to-lora before copying their head layout.
- All paper numbers here came through an HTML-to-text fetch; the PDFs were not read directly; verify figures before quoting externally.
- Serving: no measurement of per-request M with shared A,B at r512 (extra r² per token per module); an estimate of 6–12% of the A,B cost is arithmetic only. High-rank static adapters themselves are unmeasured for latency [repo, ABL24K].
- One seed per arm; CIs at this scale (about ±0.015 τ) cannot resolve effects below ≈ 0.02, so expected "+0.005 to +0.01" effects will not be resolvable without extra seeds or a larger eval set.
