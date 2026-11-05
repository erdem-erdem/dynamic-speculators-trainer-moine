# Corrected guide — porting the dynamic-speculators-trainer LoRA generator from EAGLE-3 to DFlash (deliverable 2)

This is a corrected version of the secondary guide (`../../Porting the dynamic-speculators-trainer LoRA Generator
from EAGLE-3 to a DFlash Drafter.md`). It was checked against the source at HEAD `b17276c`.

Legend:
- **[V]** verified in code or records
- **[R]** refuted, with the correction stated
- **[U]** still unverified, with what is needed to verify it
- **[N]** new information the guide lacked

Evidence lives in `FINDINGS.md` and `notes/01..06`.

## TL;DR (corrected)

- **What the fork is:** a speculators fork, 825 commits total, of which **213 are the fork's** [N]. The README is unchanged upstream text [V].
  - The fork adds the `speculators generator` package and CLI [V], plus `bank/`, `benchmark/` and a vLLM serving patch [N].
  - The hypernetwork reads **1–8 prompt examples** [V] and **only the last prompt token** of each [N]. The input is 3 Qwen3-8B layers concatenated (12288-d, +log1p(prompt length)) [V].
  - It emits rank-32 LoRA factors for the frozen `RedHatAI/Qwen3-8B-speculator.eagle3` [V]. It was also run on `deepseek-ai/eagle3_qwen3_8b_ttt7` with 5 aux layers [N].
- **Evidence source:** the source tree *is* available locally. Everything below is from code and in-repo records. The author name is **`AhmetZeer`**, not "AhmedZeer" [R].
- **How to port:** rewritten in §4 below. The generator architecture is kept, with **heads shared across the 5 layers** and a real layer embedding.
  - Base drafter: z-lab/DFlash-b16 converted (aux ids `[2,10,18,26,34]`, same as the DeepSeek recipe), or RedHat `.dflash` after config check.
  - Loss: DFlash anchored-block loss.
  - One LoRA per cohort, as in the fork.
  - Serving: merged per request in a new vLLM DFlash patch.

---

## 1. What could / couldn't be verified (corrected)

| Item | Status |
|---|---|
| Fork of vllm-project/speculators, 825 commits, listed top-level dirs | [V] |
| README describes the generator | [V] It does not; the README is the upstream text. Docs live in `examples/eagle3-lora/README.md` and `docs/experiments/**` [N] |
| Commit list / diff | [V] Available locally. Fork point `e3041dd` (2026-09-26); first fork commit `910dbb2` "eagle3 lora" (09-29). `git log --author=AhmetZeer`. No upstream remote, so the diff is vs the fork point only |
| Repo path `/content/dynamic-speculators-trainer`, SHAs `6cbbb93…`, `3ca2a6b…` | [V] Colab `/content/...` paths are hard-coded in ~8 files. `3ca2a6b` = "ttt7" (11-04). HEAD is now `b17276c` (11-05) |

## 2. What the fork adds (corrected reconstruction)

**Package and CLI** [V]
- `src/speculators/generator/` (~45 modules) with `serving/bundle.py` (`bundle_state`).
- CLI `speculators generator {campaign, train-full-data, …}` and `speculators lora-bank …`.
- Main 100K launcher: `scripts/train_attention_100k.py`.
- Recorded env [V]: vLLM **0.31.0** (patched; the 100K extractor used 0.32.0 [N]), transformers 5.x, torch 2.9–2.13 pins.
  - The guide's exact versions "transformers 5.16.1, torch 2.13.0, speculators 0.8.0.dev0" are [U]: they come from HF artifacts, not repo pins.

**Setup** [V]
- Target Qwen3-8B (pinned rev `b968826d`); drafter `RedHatAI/Qwen3-8B-speculator.eagle3` (rev `08610ffa`, 1 layer, draft vocab 32000).
- Hidden-state layers `[2,18,33]` (+ final 36 for verifier targets).

**Generator architecture** [V] (`generator/model.py`, `generator/config.py:275-281`)
- Dims 512/128/64/32/512/2 [V]. `condition: last` = the **last prompt token's raw hidden states** [V; the guide's guess was right].
- Correction to "my reading":
  - Pooling is `[mean, std, log1p(K)]` over example encodings.
  - The descriptor is `[condition 64 | module embedding 32 | layer embedding 32]`. The layer embedding has **one row**, so there is no layer index [N].
  - The decoder is shared, but **each module has its own full-size output heads** (`512→r·in`, `512→out·r`). These hold ~97 % of the params: 293 M total for o+v, 646 M for q/k/v/o [N].
  - B-heads are zero-initialized, so ΔW = 0 at step 0 [N].
- `min_examples 1`, `max_examples 8`, `conditioning.states: prompt` [V].

**LoRA settings and targets**
- rank 32, alpha 64, dropout 0.05 [V].
- Code default targets o_proj+v_proj [V]. **Latest production runs use q/k/v/o** [N]. "modules-all" = all 7 [V].
- `v_proj` (in,out) = (8192, 1024) [V]: Linear weight `[1024, 8192]` because EAGLE-3's first layer takes `cat[embed, hidden]`.
- No `fc`/`lm_head` LoRA [V]: they are not in `ADAPTER_MODULES`. The base drafter is frozen [V], by call order with no runtime assert [N].
- Serving caveat [V]: per-request native hook = exactly `{o_proj, v_proj}` (`serving/controller.py:57-62`). The shared merged path supports all 7 but only **one draft layer** [N].
- LoRA application during training [N]: `AdapterLinear` + `torch.func.functional_call` with **one LoRA per cohort (episode)**, 4 episodes per update.

**Objective and hyperparameters**
- `lk_hybrid` [V] = `λ·KL(p‖q) + (1−λ)·TV`, `λ = exp(−3·stopgrad(overlap))`, overlap = Σ min(p,q) [N].
  - The campaign requires it; `ce` exists for studies.
- TTT steps: **3** in the main recipes; **7 only** in `training_recipe_ttt7.yaml` (10k run, no results yet) [R: the guide implied 7 was standard]. Decay 1.0 [V]. The loss is *summed* over steps [N].
- lr 1e-4, wd 0.01, clip 0.85, 100 warmup then **constant** LR, 4 episodes/update, token budget 8192 per microbatch [V].
- Noise is **uniform ±0.05** (std 0.029), on drafter inputs only [V+N].
- Generator trained in **pure bf16 without fp32 master weights** [N].
- Updates [V]: 1,500 (4K pilots), 2,000 (24K/100K), 6,000 (24K cap), 10,000 (TTT7 run). Conditioning 5 epochs and adaptation 1 epoch belong to the **legacy** workflow only.
- `draft_attn_impl: simple_flex_attention` [V].

**Pipeline stages**
- [R] Pretraining/conditioning/heatmap/adaptation mixes two systems:
  - Current campaign: prepare → pilot → pretraining → adaptation.
  - Legacy workflow: conditioning → heatmap (strides 10/20/30 × 6 seed groups, L1 regression to bank LoRAs) → adaptation. Retired after GEN-004 showed a fresh generator catches up.
- Arms `direct` (generator, scales 1/2/4 = **data** scale) and `fresh_lora` (matched static LoRA) [V].

**Data** [V]
- Nemotron-v2-decontaminated → math/code/stem/chat; regenerated with T 0.7, top-p 0.8, top-k 20, 4096 tokens, thinking off.
- 1000 train + 100 val per subset, frozen catalogs, overlap asserted [N].
- HF datasets `azeeer/eagle3-qwen3-8b-{trajectories,lora-bank-r32,…}-v1` are referenced in code [V]. Hub existence [U] (no network used).
- The LoRA bank = per-domain/seed o+v snapshots used **only** by the retired legacy regression path [N].
- **Activations are not loaded from a dataset.** They are re-extracted online by a vLLM process and cached in 96–120 GiB of CPU RAM [N].

**Reported numbers**
- `eagle3-generator-direct-4k-v3` setup (1,500 updates, 4k, context 4, 100 val/subset) [V].
- Per-position accuracies 0.737/0.535/0.362, math 0.793, chat 0.680: **[U] not in the repo**. If they were serving numbers, chat 0.680 would be inconsistent: serving chat pos-0 ≤ 0.62.
- "No matched static comparison / no speedup" **[R]**. The repo has many; see the table below and `notes/04`.

| Key result (C1, single run, ~5 % drift) | Value |
|---|---|
| DeepSeek ttt7 drafter → +QKVO generator (paired) | +6.6..+16.1 % tok/s ID, +1.3..+5.3 % OOD; τ math 3.22→3.45 |
| Matched static QKVO LoRA → QKVO generator | math +3.3 % tok/s / +5.6 pp acc; chat −1.9 % / −0.1 pp |
| Matched static all-module → all-module generator | generator slower (−1.5 % math, −3.9 % chat) |
| End-to-end vs target-only | **never measured** |

**Ablations found** [V, with outcomes now known — `notes/04` §3]
- Modules: QKVO > VO; all-module best offline but loses on cost.
- Generator size: decoder 256 ≈ 512; depth and condition dim no effect; encoder 1024 weak +.
- Context K: K=1/2/4 ≈; K=8 never reported.
- Conditioning source: replay / prefix / feedback are **negative**.
- Scale: 4K→16K and 24K→100K show no gain (equal exposure, per audit).
- `deepseek-ttt7-attention` = clean positive.

## 3. DFlash: what you're porting to (corrected)

**How it works**
- Block drafter, one forward per block [V].
- `fc` over concatenated target layers → `hidden_norm`, injected as extra K/V into **every** draft layer [V]. The code calls the **same `k_proj`/`v_proj` modules** separately on context and block tokens (`models/dflash/model_definitions.py:116-119`) [N].
- `q_proj`, `o_proj`, the MLP and the layernorms see block tokens only [V].
- Bidirectional within block **in the paper / z-lab / vLLM**. The repo's **CLI default is sliding-window with causal in-block attention** [N — hazard].
- Embeddings and LM head shared and frozen [V].

**Training**
- Random sorted anchors (re-sampled every forward) [V]. Cross-block attention masked [V].
- Loss: CE vs target argmax [V], weighted by `exp(−(k−1)/γ)` (repo default γ=4; paper γ=7 for bs16) or **D-PACE (the repo's default for dflash)** [N].
- Paper: 5 layers, bs16, 5 target layers, lr 6e-4, 512 anchors [V]. Repo CLI: 5 layers, bs16, `max_anchors` 512, default target layers **3** = `[2,18,33]` [N].

**Results that matter** [V, paper]
- Qwen3-8B T=0: DFlash(16) τ 6.49 / 4.86× vs EAGLE-3 2.96 / 1.76×.
- KV injection > input fusion. 5 layers best speedup.
- Train block size ≥ inference block size.
- Without target features: 2.8–3.7× (so conditioning matters).

**Upstream speculators support**
- [V] `DFlashDraftModel(DraftVocabMixin, SpeculatorModel)`, `layers` = `Qwen3DFlashDecoderLayer`s, `fc` (no bias), `hidden_norm`, `norm`, frozen `verifier_norm`.
- `speculative_tokens = block_size − 1` [V].
- [R] `--block-size` default is **16 for dflash** (8 only in the bare config class).
- `--dflash-decay-gamma` default 4.0 [V]. `--max-anchors` default 512 [V].
- The fork **did not modify** DFlash files [N].
- Serving: the repo itself has no DFlash draft loop; that is vLLM only [N]. Whether **vLLM 0.31.0** has the DFlash proposer is [U]; check on the remote box.

## 4. Port plan (rewritten from the code)

Design principle: **reuse the fork's generator/campaign machinery and change only the drafter-specific seams**.
These seams are the ones `notes/01` §7 and `notes/05` §7 identify.

**4.1 Base drafter (decision D1)**
- Recommended: **`z-lab/Qwen3-8B-DFlash-b16`** auto-converted via `SpeculatorModel.from_pretrained(..., verifier="Qwen/Qwen3-8B")`.
  - 5 layers, bs16, aux layers `[2,10,18,26,34]`, full bidirectional attention, mask id 151669 — paper-faithful.
  - Its aux layer ids are **identical to the fork's DeepSeek recipe**, so the trajectories and the extraction recipe are reused unchanged via `trajectory_recipe`.
  - It also gives a clean cross-drafter comparison: same generator input on EAGLE-3 (DeepSeek) vs DFlash.
- Alternative: `RedHatAI/Qwen3-8B-speculator.dflash`, after checking its config (layers, aux ids, window/causality, vocab) [U].
- Freeze it, and assert it is frozen (see §4.7).

**4.2 Make adapters multi-layer** (`generator/model.py`)
- `install_adapters`: select by qualified path pattern (`layers.*.self_attn.{q,k,v,o}_proj`, optional `mlp.*`, optional `fc`). Return sanitized keys (`l{i}.q_proj`).
- `LoRAGenerator`:
  - `layer_embedding = Embedding(num_layers, 32)`.
  - `module_embedding` indexed by **module type**, not dict position.
  - **Output heads shared per module type across layers** (descriptor carries the layer).
  - Head params are then independent of layer count: ≈437 M for QKVO at decoder 512, 4096 hidden. Per-layer heads would be ≈2.2 B and are an ablation only.
  - Keep B-head zero-init. Keep the existing EAGLE-3 behavior bit-identical: a regression test checks that 1-layer configs reproduce old outputs.

**4.3 K/V path switch** (DFlash layer factory)
- Subclass the attention through `DFlashDraftModel._make_decoder_layer` (`models/dflash/core.py:62-70`; dflash2/dspark use the same seam).
- Add `kv_lora_path ∈ {both, context, block}` that applies the adapter delta only at the chosen `k_proj`/`v_proj` call sites.
- **`both` is the default because it is the only variant mergeable into a single vLLM weight.** `context`/`block` are offline research arms unless the vLLM DFlash attention is also patched.
- `fc` LoRA is optional. Its A-head is large (`r·20480` outputs), so use a smaller rank or a dedicated head.

**4.4 Runtime/loss** (`generator/engine.py`, `campaign_train.py`)
- Take call kwargs from `DFlashDraftModel.get_trainer_kwargs` (block_size, max_anchors, per-position weighting, γ, loss_config) instead of the TTT kwargs.
- Allow `loss_implementation=eager` (needed for CPU and local tests).
- Score with `full_acc` / `eal` / `position_i_acc` instead of `full_acc_0`.
- **Loss (decision D2)**: primary = the DFlash-native CE + D-PACE (the repo default for dflash). Ablations: CE + exp-decay γ=7 (z-lab/paper recipe) and `lk_hybrid` + exp-decay (the fork's objective).
  - D-PACE requires `ce`.
  - Relax `campaign_config.py:130`, which hard-requires `lk_hybrid`.

**4.5 Data / batching** (`generator/data.py`)
- New DFlash microbatch builder: no `shift_batch`. `hidden_states` = aux layers concatenated, `verifier_last_hidden_states` = final layer, plus `document_ids`, `loss_mask`, `position_ids`.
- Rows are packed **only from the current cohort's documents**, so all tokens share the cohort LoRA. No per-token LoRA routing is needed.
- **Anchors**: training draws them under the per-episode seed (the fork already calls `torch.manual_seed` per episode). **Validation uses fixed seeded anchors**, so every arm sees identical blocks.
- Anchor density is a parameter: `max_anchors` per row vs tokens per row. Log the realized anchor count.
- Use the full-attention mask (match the checkpoint). Assert that the config's `layer_types`/`sliding_window` equal the checkpoint's.

**4.6 Conditioning (unchanged concept)**
- Last prompt token of the 5 aux layers (20480-d) + log1p(len), via the existing `trajectory_recipe` + online extractor at `[2,10,18,26,34]`+36. The DeepSeek recipe already does this.
- Training cohort K ~ U{1..8} is kept as the fork default. Serving uses K=1, so K=1 training is an ablation.

**4.7 Rigor harness (before any real run; all as unit tests on a tiny DFlash)**
1. Zero LoRA (B=0, or the generator at init) gives **bit-identical** logits, loss and metrics to the base drafter with the same anchors.
2. Drafter state_dict hash identical before and after N updates. A runtime assert in `run_training` checks that no drafter param has `requires_grad`.
3. Only generator params have non-None grads. The optimizer param set equals the generator params.
4. A LoRA forward through `functional_call` equals an explicit merge `W + (α/r)·B·A`, for every targeted module and each of the 5 layers. Shapes and α/r are checked, and the layer↔key mapping is checked by perturbing one layer at a time.
5. `kv_lora_path` = context / block / both changes exactly the expected tensors.
6. Block size, mask token, anchor set and mask type are identical between training-time validation and evaluation, and are logged.
7. The generator overfits one tiny batch: loss → near 0 and acc → 1 on the tiny model.
8. Determinism: same seed gives identical anchors and losses across two runs.

**4.8 Serving**
- First verify that vLLM has a DFlash proposer and pick its version.
- Port the patch: hook the DFlash proposer, merge the generated per-request adapter into the **5 layers' packed** qkv/o weights, and restore after the request.
  - At C1 per request; K=1.
  - Remove the EAGLE-only asserts (`method`, single layer).
- Spec tokens = block_size − 1 = 15.
- Measure **target-only**, base DFlash, static LoRA and generator. Report generator + merge time separately and **net it out**.
- Higher concurrency needs either cohort merging (known TTFT blow-up) or a batched-LoRA kernel. This is a research item, not a blocker.

**Pitfalls (corrected / added)**
- [V] Sliding window plus causal in-block is the CLI default: use full attention to match the checkpoint and vLLM.
- [V] Mask token, block size and `sample_from_anchor` must match between training, eval and serving.
- [V] Checkpoints omit verifier weights. `load_verifier_weights` **returns silently and leaves NaNs** if the verifier path is unset: assert finite.
- [V] Merged per-query weights break cross-request batching.
- [V] Offline gains did not predict serving ranking in the fork (GEN-009→010).
- [N] Anchor RNG makes the DFlash loss stochastic per call, so fix the seed for matched comparisons.
- [N] bf16 generator without master weights: use fp32 master, or test both.
- [N] The bank validator in `train/config/schema.py` requires eagle3; the campaign requires `lk_hybrid` and `full_acc_0`. All three need relaxing.
- [N] Silent fallbacks in the eval/serving path (`notes/03` §8) must be turned into hard errors on any path we use.

## 5. Ablation plan (rewritten)

**Base config**: z-lab DFlash-b16, frozen; QKVO × 5 layers, `kv_lora_path=both`, shared heads, rank 32, α 64, dropout 0.05; generator 512/128/64/32/512/2.
- Loss CE + D-PACE; lr 1e-4, wd 0.01, clip 0.85, warmup 100 + constant; fp32 master weights.
- 4 episodes/update, K~U{1..8}, `max_anchors` 512 per 8192-token row.
- 4K pool (1K/domain), 1,500 updates. Same splits as the fork.

**Every arm changes exactly one variable from the base.** Seeds:
- Phase 0–1: **3 seeds**.
- Later ablations: 1 seed, plus 3 seeds for any arm within 2× the seed std of the base.

Metrics (offline and vLLM):
- per-domain τ and per-position acceptance;
- C1 tok/s;
- generator + merge latency.

| Phase | Axis | Arms | Why |
|---|---|---|---|
| 0 | Baselines | target-only; base DFlash; matched static LoRA (same modules, updates, data) | The decisive control; the fork's matched gap was small |
| 0 | Generator | base config | Core question: does it help DFlash at all? |
| 0 | Conditioning controls (inference) | correct / same-category donor / cross-category donor / fixed donor | Separates prompt-specific from domain routing (the fork never ran same-category) |
| 1 | Modules | {o,v}; QKVO; MLP; all; +`fc` | Replicates the fork's main structural finding |
| 1 | K/V path | both / context-only / block-only | DFlash-specific lever; context path ≈ "how target features are read" |
| 1 | Layer scope | all 5; layer 0 only; last layer only | Where adaptation matters in a 5-layer stack |
| 1 | Heads | shared-across-layers vs per-layer | Capacity vs params (437 M vs 2.2 B) |
| 2 | Loss | CE+D-PACE / CE+decay γ7 / lk_hybrid+decay | The objective matters for block acceptance |
| 2 | Rank | 8 / 16 / 32 / 64 | Fork: monotone, small |
| 2 | Generator size | decoder 256 (fork's efficient pick) | Latency matters more with 20 module instances |
| 2 | Train K | K=1 (matches serving) vs U{1..8} | Train/serve mismatch in the fork |
| 3 | Anchors | 512 / 1024 / 3072 per row | RFC #979: more anchors helped DFlash |
| 3 | Scale | 4K/1.5k → 24K/2k → 100K/8.3k+ updates | Fork's 100K run was under-exposed |
| 3 | Block size | bs16 vs eval at bs8 | Train ≥ inference generalization with the LoRA |
| 3 | Precision | bf16-pure vs fp32-master generator | Fork risk item |
| 4 | Serving | merged C1/C4/C8; cohort vs per-request | Deployment cost |

Report everything per domain (math, code, stem, chat, plus OOD), as mean ± std over seeds.
Treat differences below the measured noise floor as null.

## Caveats (updated)

- Internals are now verified from source. The remaining [U] items are external:
  - vLLM DFlash support and version;
  - RedHat DFlash config;
  - Hub dataset availability;
  - upstream changes after `e3041dd`.
- The fork's own results are single-seed; ~5 % run-to-run drift was observed. The generator's advantage over matched static LoRA on EAGLE-3 is **small and domain-dependent**, so the DFlash study must be powered accordingly.
- Paper speedups are greedy and engine-specific; ours will be measured on our own Nemotron/OOD suite.
