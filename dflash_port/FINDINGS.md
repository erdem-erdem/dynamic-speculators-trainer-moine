# Findings report — what the fork actually adds (deliverable 1)

Repo at HEAD `b17276c` (2026-11-05). Fork point upstream `e3041dd` (2026-09-26). Fork author Ahmed Zeer: GitHub
`AhmedZeer/dynamic-speculators-trainer`, git author name `AhmetZeer` (+1 commit `azeer`), 213 commits 2026-09-29..2026-11-05, `git diff e3041dd HEAD`: 534 files, +110k / −302.
No `upstream` remote, so this is "fork vs fork point", not vs current upstream main.
Paths are relative to `src/speculators/` unless they start with a top-level dir.
Detailed evidence with file:line references is in `notes/01..06`. This file is the synthesis.
Every claim below was read in code or in the author's records. Items marked **UNVERIFIED** are not confirmed.

---

## 0. One-paragraph summary

A hypernetwork (`generator/model.py:LoRAGenerator`, ~0.3–0.65 B params, almost all in output heads) reads
the **last prompt token's** concatenated target hidden states (Qwen3-8B layers 2/18/33 → 12288-d, plus
`log1p(prompt_len)`) for a cohort of 1–8 prompts. It emits rank-32 LoRA A/B factors (α=64 → scale 2) for
the **single decoder layer** of a frozen EAGLE-3 drafter. It is trained **end-to-end through the drafter's own TTT loss**
(`lk_hybrid`, 3 TTT steps; 7 in the latest, unfinished run). Training substitutes the factors with `torch.func.functional_call`.
The baseline is a matched static LoRA (`fresh_lora`). Serving uses a source-patched vLLM 0.31.0. The adapter is generated once per
request (or cohort) from the prefill hidden states and merged into the draft weights.

Best evidence:
- Generator on DeepSeek's EAGLE-3 ttt7 drafter: +6.6..+16.1 % C1 tok/s in-domain and +1.3..+5.3 % OOD vs the same drafter.
- Vs a **matched** static LoRA: math acceptance +4–6 pp, throughput only −1.5..+3.3 %; chat shows **no** gain.
- All of it is single seed, single run, with ~5 % run-to-run drift observed. There is no target-only baseline.

The fork **never touched DFlash code**, but changed shared training infra that DFlash uses.

---

## 1. Features, grouped by phase (see `notes/06` for the full commit table)

| Phase (dates) | What was built | Key files |
|---|---|---|
| 1. EAGLE-3 static LoRA (09-29..30) | PEFT LoRA on the drafter (`--lora-r`) | `train/lora.py`, `train/cli.py`, `train/config/schema.py` |
| 2. LoRA bank (10-03..06) | per-domain/seed/snapshot static LoRAs as regression targets; Hub publishing | `bank/*`, `train/bank.py`, `cli/lora_bank.py` |
| 3. Generator v1 (10-06..07) | hypernetwork + L1 regression onto bank (legacy) → later "direct" objective | `generator/model.py`, `engine.py`, `workflow.py` |
| 4–5. vLLM adapter + Hub/Modal (10-07..16) | serial adapter serving, artifact publishing | `generator/serving/*`, `scripts/modal/*` |
| 6. Online campaigns (10-16..19) | vLLM-extracted activations cached in CPU RAM; prepare/pilot/pretraining/adaptation | `generator/campaign*.py`, `online_activations.py` |
| 7–8. HTTP serving + GuideLLM + report (10-19..26) | serving controller, benchmark suite, LaTeX report | `generator/serving/*`, `benchmark/*`, `reporting/` |
| 9–11. Dynamism / feedback / coefficient (10-26..28) | replay, prefix refresh, acceptance feedback, scale sweep | `generator/feedback*.py`, `study*.py` — **all negative/neutral** |
| 12. Structure ablations (10-30..31) | target modules, generator size, matched static LoRA | `examples/eagle3-lora/structure_*` |
| 13. DeepSeek ttt7 (10-31) | **same generator on a different (5-aux-layer) EAGLE-3 drafter** via `trajectory_recipe` | `examples/eagle3-lora/deepseek_ttt7_generator_v1/` |
| 14–15. 100K scale, audit, TTT7 (11-01..05) | 100K pool, exposure audit, 10k-update TTT7 run (**no results yet**) | `scripts/train_attention_100k.py`, `attention_100k_2k_v1/` |

About half of the commits deal with Hub and extraction robustness.

## 2. Generator: conditioning, output, injection (`notes/01` §1–3)

- **Two code paths.** The legacy path is `engine.py` `_run_job`, L1 regression onto bank snapshots, and is retired.
  The production path is `generator/campaign_train.py`, objective `"direct"`: generator → LoRA → frozen drafter loss.
- **Architecture** (`generator/model.py:28-151`, defaults `generator/config.py:275-281` = 512/128/64/32/512/2):
  - encoder `Linear(D+1→512)→GELU→Linear(→128)→LN`.
  - pool `[mean, std, log1p(K)]` (257-d), condition MLP → 64.
  - descriptor `[condition 64 | module_emb 32 | layer_emb 32]`. `layer_embedding = nn.Embedding(1, 32)` (`model.py:56`), so there is **no layer index**.
  - decoder 512 + 2 residual MLP blocks, run once per module.
  - **per-module full heads**: `A: Linear(512→r·in)`, `B: Linear(512→out·r)`. The heads hold ~97 % of params (286 M for o+v, 639 M for q/k/v/o).
  - Init: B-head weight and bias = 0, so **ΔW = 0 exactly at step 0**. A = Kaiming matrix stored in the A-head *bias*.
  - No output scaling. LoRA scale = α/r = 2.
- **Conditioning** = the raw last-prompt-token states of the 3 aux layers, with no normalization and no noise (`online_activations.py`).
  - Prompt-only extraction, so no response leakage.
  - Cohort K ~ U{1..8} in training. Validation uses K=4. Serving uses K=1 per request.
  - By default the conditioning cohort **is** the loss cohort (in-cohort adaptation).
- **Injection** (`model.py:154-219`, `engine.py:176-186`):
  - `install_adapters` replaces Linears by `AdapterLinear` (`y = base(x) + s·B(A(dropout(x)))`, same math as PEFT).
  - The generated factors are substituted via `functional_call`.
  - Granularity is **one LoRA per episode (cohort)**: 4 episodes per update, each run as separate drafter forwards. There is no per-token or per-sequence LoRA inside a packed row.
  - Supported modules are `q,k,v,o,gate,up,down` (`config.py:264-272`). There is **no `fc`, no `lm_head`**.
  - Default targets `o_proj,v_proj`. Latest production uses **q/k/v/o**.
- **EAGLE-3 lock-ins** (12 listed in `notes/01` §3). The ones that matter:
  - `install_adapters` demands exactly one match per leaf name, i.e. a single layer (`model.py:193-197`).
  - The single-row layer embedding.
  - `Runtime.call_kwargs` hard-codes `ttt_steps` and `ttt_step_loss_decay` and forces the fused Triton loss (`engine.py:115-121`).
  - The metric key `full_acc_0` (`engine.py:221-226`).
  - `Corpus.microbatches` uses EAGLE `shift_batch`, padding to 128 (`data.py:271-321`).

## 3. Loss and what is frozen (`notes/01` §4)

- `lk_hybrid` (`losses/eager.py:174-213`), per position:
  `α=Σ min(p,q)`; `λ=exp(−3·sg(α))`; `loss = λ·KL(p‖q) + (1−λ)·(1−α)`.
  - Summed (not averaged) over TTT steps, with decay 1.0.
  - `ce` = hard-label CE vs target argmax. The campaign *requires* `lk_hybrid` (`campaign_config.py:130`).
- **Optimization**:
  - AdamW lr 1e-4, wd 0.01 (also on biases and the A-prior), linear warmup 100 then **constant**.
  - Global clip 0.85.
  - 4 episodes/update; token budget 8192 *per microbatch*.
  - Noise is **uniform ±0.05** on drafter inputs only (std ≈ 0.029).
  - Seed 42 everywhere.
- **Generator dtype**: pure **bf16** params, grads and Adam states with **no fp32 master weights**; a test asserts this. This is a risk of stalled updates and is noted for the port.
- **Freezing**: the drafter is frozen by call order (`requires_grad_(False)` at load, re-applied at `campaign_train.py:400`). The optimizer only sees generator params.
  - Tests check state_dict equality on a tiny CPU drafter.
  - **No runtime assertion** in production.
- **No regularizer** on ΔW.
- **Matched static LoRA** (`lora_baseline.py`): same schedule, loss, optimizer, data, updates and targets; one shared adapter.
  - Precision differs between harnesses: bf16 in `structure_study`, fp32 in `train_matched_lora`.

## 4. Data pipeline and campaign stages (`notes/02`)

- **Data**: `Nemotron-Post-Training-v2-decontaminated` → math/code/stem/chat.
  - Responses regenerated by Qwen3-8B: T 0.7, top-p 0.8, top-k 20, 4096 max tokens, thinking off, seed 0.
  - Prompts silently left-truncated to 4096 tokens.
  - Each subset is 1000 train + 100 val, picked by seed-0 reservoir sampling; catalog frozen and overlap asserted (`local_campaign.py:497-503`).
  - r0001: 6 subsets/category (24K). r0002: 25 subsets/category (100K).
- **Local copy** (`datasets/`, git-excluded): `r0001` complete. `r0002-100k` has **chat only**. There are no activations; the EAGLE-3 dump (layers `[2,18,33,36]`, ≈1.24 TB) is unusable for DFlash. See `notes/07` §B.
- **Activations**: online. A separate vLLM process extracts hidden states for the saved token ids (layers `[2,18,33]`+final 36). They are cached in a CPU-RAM LRU of 96–120 GiB, about 256 MiB per full example. The trainer GPU holds only the drafter and generator.
  - Linux-only: `/tmp` shared dirs, `fcntl`.
- **Stages**:
  - Current campaign (`campaign.py:44-77`): prepare → pilot → pretraining → adaptation. Arms `scale-{1,2,4}-direct` + `fresh_lora`.
  - Legacy workflow (`workflow.py`): conditioning (5 epochs) → heatmap (strides 10/20/30 × 6 seed groups, L1 to bank LoRAs) → adaptation (1 epoch).
- **Drafter-swap precedent**: `deepseek_ttt7_generator_v1/` reuses the saved responses via `trajectory_recipe` and re-extracts at **`[2,10,18,26,34]`+36**. These are **exactly the z-lab DFlash Qwen3-8B target layers** (`convert/dflash` fixture `[1,9,17,25,33]` +1 offset). This is the template for the port.
- Provenance: git SHA (falls back to `"unknown"`), configs, manifests with hashes, Hub revisions. Model dirs are **not** checked against pinned SHAs (`engine.py:66-70`), and some recipes use `revision: main`.

## 5. Serving and evaluation (`notes/03`)

- **vLLM 0.31.0** (commit `db9527a`), patched by `scripts/apply_vllm_generator_patch.py`. It rewrites `config/speculative.py`, `v1/spec_decode/eagle.py` (`EagleProposer` hooks) and `gpu_model_runner.py` in place, with SHA guards.
  - Runs with V1 runner forced, BF16, eager, no chunked prefill, no prefix cache, TP=1, ≤8 seqs.
- Conditioning at serve time is the `target_hidden_states` passed into `propose()`, at the last prompt token. This requires an unchunked prefill.
- **Shared merged path**: `W += (α/r)·B·A` into packed weights. It supports all 7 projections but **asserts one draft layer**.
- **Per-request native multi-LoRA**: supports **only `{o_proj, v_proj}`** (`serving/controller.py:57-62`). Verified.
  - It is 15–17 % slower than merged at equal acceptance, because of CPU-side LoRA mapping overhead.
- **Eval harness**: `speculators benchmark suite` → `launch_vllm.py` → GuideLLM 0.7.1.
  - Acceptance from Prometheus counters (`1 + accepted/drafts`, per-position).
  - 50 prompts/category, 256 max tokens, 3 draft tokens, C1/2/4/8, ID Nemotron + OOD (HumanEval, math-reasoning, summarization).
  - **Correctness `not_checked`**. **Generator latency not netted out**; it is ~6.5 ms/request and included in wall time.
- **Silent fallbacks** (14 listed, `notes/03` §8). Examples:
  - provenance write errors swallowed (`launch_vllm.py:478-482`)
  - status checks skipped on 404
  - `output_mismatch: warn`
  - `continue_on_error: true`

## 6. Experiments and actual results (`notes/04`)

**Evidence quality**:
- Every number is one seed and one run, with 50 prompts per category.
- The target-only greedy output was non-reproducible (23/50 matching).
- The same baseline reran **~5 % faster** at identical acceptance, so any gap below ~5 % is noise.
- Most training ablations exist only as W&B screenshots.

| Comparison (C1, tok/s unless noted) | Result | Source |
|---|---|---|
| DeepSeek ttt7 base → +QKVO generator (paired) | math 86.6→92.3 (+6.6 %), code +11.0 %, STEM +16.1 %, chat +15.5 %; OOD +1.3..+5.3 %; τ math 3.22→3.45 | `docs/experiments/results/GEN-012*` |
| RedHat base vs attention gen 24K/2K (cross-suite) | math 64.3→82.8 (+29 %), chat +8.7 %, OOD +3..+5 % | `GEN-011`, `extended_historical_table.md` |
| **Matched** static QKVO LoRA vs QKVO generator (4K/1500) | math 75.7→78.2 (+3.3 %), acc +5.6 pp; chat 56.5→55.4 (−1.9 %), acc −0.1 pp | `GEN-010` |
| Matched static all-module vs all-module gen | gen *slower*: math −1.5 %, chat −3.9 % (math acc +4.1 pp) | `GEN-010` |
| Conditioning donors (step 4000) | correct vs cross-category: math acc +11 pp; vs fixed donor: math +0.1 pp, chat +5.4 pp; **same-category donor never run** | `eagle3_generator_ablations_v4.md` |
| 24K gen vs "trained LoRA" (presentation slide 3, C2) | math 136.1→142.6 (+4.8 %), chat −0.3 %. **Not matched**: that LoRA is the V/O 4K/1,500 run | `benchmarks/2026-10-24-comparison-v2-concurrency`, `notes/07` |
| Attention gen 24K at 2K/4K/6K updates (C1) | math acc 66.9→68.8 % rises, tok/s flat (2K best, 82.8) | `benchmarks/2026-10-31-attention-24k-6k-c1` |
| Training loss CE vs LK (offline probe agreement, 1k steps) | CE ≈ LK, both > static LoRA | `GEN-006`, slide 6 |
| Rank 8/16/32/64 | acc +6.1/6.8/7.4/7.9 pp, tok/s +10.3/11.6/12.5/13.6 % vs base | `GEN-005`, slide 9 |
| Concurrency (shared cohort) | gen-vs-LoRA margin +5.7 % C1 → +1.1 % C8; chat TTFT 50 ms → 1,448 ms at C8 | `BATCH-001` |

**Ablation verdicts**:
- QKVO > VO. All-module wins offline but loses on cost.
- decoder 256 ≈ 512 (−17 % latency, −49 % params). Trunk depth and condition dim don't matter.
- K=1/2/4 ≈ equal; K=8 never reported.
- Scale 2 (training value) best; 8 collapses.
- Replay, prefix-refresh and feedback are negative.
- 4K→16K and 24K→100K pool gave no gain. The 100K/2K run covered only 0.36 epochs (vs 1.5 for 24K/2K).
- The presentation (`notes/07` §A) adds no new matched evidence. Its slide 3 "same amount of data" comparison is not data/step-matched.
- TTT7 / 10k updates: **no results**.

**Guide's per-position numbers** (0.737/0.535/0.362) **are not in the repo** → unverified.

## 7. Upstream DFlash in this repo (`notes/05`)

- `models/dflash/*` is byte-identical to upstream at `e3041dd`.
- `DFlashDraftModel`: N × `Qwen3DFlashDecoderLayer` (CLI default 5), `fc` (n_aux·H→H, no bias) → `hidden_norm`, final `norm`. Embedding, `lm_head` and `verifier_lm_head`/`verifier_norm` are frozen and come from the target.
- **K/V injection** (`models/dflash/model_definitions.py:116-119`): the **same** `k_proj`/`v_proj` modules are called separately on `target_hidden` (context) and on block tokens, then concatenated.
  - `q_proj`, `o_proj` and the MLP see block tokens only.
  - `target_hidden` is computed once and shared by all layers.
  - So a wrapped-Linear LoRA cannot separate the context and block paths. Editing those 4 lines can.
- **Training**:
  - Random sorted anchors (`--max-anchors` 512 default, examples 3072), **re-sampled every forward**.
  - Blocks of `block_size` (CLI default 16), slot 0 = clean anchor (weight 0).
  - Flex BlockMask: prefix-before-anchor, same document, no cross-block attention.
  - Loss: CE vs target argmax, weighted by **D-PACE** (CLI default for dflash) or `exp(−(k−1)/γ)`, γ default 4 (paper: 7 for bs16).
  - Metrics `full_acc`, `position_{i}_acc`, `eal`.
  - Batches are one packed row (`B=1`), many documents.
- **Mask hazard**: the CLI default makes all layers sliding-window (2048) **with causal in-block attention**. The paper, z-lab checkpoints and vLLM parity test are **bidirectional**, full attention. For paper-faithful training, pass `--full-attention-indices 0 1 2 3 4`.
- Checkpoints:
  - `z-lab/Qwen3-8B-DFlash-b16` (5 layers, bs16, aux `[2,10,18,26,34]`, mask 151669, full attention) is auto-converted by `from_pretrained`.
  - `RedHatAI/Qwen3-8B-speculator.dflash`: config **UNVERIFIED**.
- Paper, Qwen3-8B T=0: DFlash(16) avg 4.86× / τ 6.49 vs EAGLE-3 1.76× / 2.96. 5 layers best; KV injection > input fusion.

## 8. Fork changes to shared code that affect DFlash training (`notes/06` §2)

1. `models/attention.py`: uncompiled CUDA callers get a standalone-compiled flex kernel. This is needed for `functional_call`.
2. `train/optimizers.py`: AdamW gets only `requires_grad` params and raises if none.
3. `train/data.py`: raises on cached-token mismatch when `on_missing=raise`.
4. `train/trainer.py`: logs grad norm and clip. `train/cli.py` imports bank/lora at top.
5. **`schema.py` bank validator hard-requires eagle3 + LoRA.** It must be relaxed for dflash.

## 9. Environment facts

- Pins: torch `>=2.9,<=2.13`, transformers `>=5,<5.17`, eval extra `vllm==0.31.0`, `guidellm==0.7.1`; extras `generator`, `lora` (peft ≥ 0.18).
- **Local Windows box cannot import `speculators`**:
  - the package is not installed;
  - `hs_connectors` imports `fcntl`;
  - torch 2.5.1 lacks the FSDP2 API.
  - The fused loss needs Triton.
  - Local smoke tests therefore need WSL2/Linux + torch ≥ 2.9, or must run on the remote box.
- Hard-coded: Colab `/content/drive/MyDrive` paths (~8 files), HF user `azeeer` (~60 YAMLs + `bank/config.py:124-128`), Modal names.

## 10. Open questions (need network, remote GPU, or user decision)

1. Does vLLM 0.31.0 ship a DFlash proposer? If not, which version does, and does the patch machinery port?
2. RedHat vs z-lab DFlash checkpoint configs: layers, aux ids, window/causality, block size, vocab.
3. Are the Hub trajectories `azeeer/eagle3-qwen3-8b-trajectories-v1` and the DeepSeek activation recipe accessible to us?
4. What changed upstream for DFlash after `e3041dd`? This needs the upstream remote.
5. Is the DeepSeek ttt7 generator result robust (needs repeats + static LoRA control)? That is our best EAGLE-3 reference point.
