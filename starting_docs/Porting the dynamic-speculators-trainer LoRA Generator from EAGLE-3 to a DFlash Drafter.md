# Guide to AhmedZeer/dynamic-speculators-trainer and How to Port It to a DFlash Drafter

You can port this to DFlash, but you'll need to write the code yourself. Upstream vllm-project/speculators already ships a full DFlash drafter (`DFlashDraftModel` in `src/speculators/models/dflash/core.py`).\[1\] The fork's generator code is EAGLE-3-specific: it targets a 1-layer Llama-style drafter and supports only `o_proj`/`v_proj` per request at serving time.\[2\] So the port means building a DFlash-aware LoRA injection layer, a new conditioning/target path and a new serving hook. You don't need to redesign the method.

## TL;DR
- **What the fork is:** a speculators fork with 825 commits. Its README is still the unchanged upstream README.\[3\] The new work appears to be a `speculators generator` package and CLI that trains a hypernetwork.
  - The hypernetwork reads 1–8 prompt-hidden-state "examples" from Qwen3-8B (3 layers concatenated, 12288-dim).
  - It outputs rank-32 LoRA factors for the frozen `RedHatAI/Qwen3-8B-speculator.eagle3` drafter.\[2\]
  - It was compared against a matched static-LoRA baseline, with ablations on target modules, generator width/depth, condition size and context size.\[4\]
- **Where the evidence comes from:** I couldn't read the GitHub source tree, commit list or diff. Everything specific about the generator comes from the author's Hugging Face artifacts (account `azeeer`): bundle configs, training commands and result JSONs.\[4\] Treat file and class names below as inferred unless marked as upstream.
- **How to port:**
  - Keep the generator. Swap the drafter for `DFlashDraftModel`: 5 Qwen3-style layers, block size 16, KV-injected target features through `fc` and `hidden_norm`.\[5\]
  - Add LoRA hooks to the draft layers' q/k/v/o and MLP projections.
  - Replace the EAGLE-3 TTT loss with DFlash's anchored-block cross-entropy using decay or D-PACE position weighting.\[6\]
  - Then run a staged ablation plan that mirrors the EAGLE-3 one, plus DFlash-specific axes: which layer, K/V on context vs. block tokens, per-position gains, and block size.

---

## 1. What I could and couldn't verify

| Item | Status |
|---|---|
| Repo is a fork of vllm-project/speculators, 825 commits, top-level dirs `.buildkite`, `.claude/skills/pr-review`, `docs`, `examples`, `hs_connectors`, `reporting`, `scripts`, `src/speculators`, `tests` | Verified (GitHub landing page) |
| README describes the generator | **No.** The README is the upstream text (DFlash, DSpark, P-EAGLE, hs_connectors news)\[3\] |
| Commit list / per-commit messages / diff vs upstream | **Not accessible** (commits, compare and tree pages were blocked; nothing else was indexed) |
| Generator design, CLI, configs, ablation names, results | From author's HF artifacts (`azeeer`). These record the repo path `/content/dynamic-speculators-trainer` and git SHAs `6cbbb93…` (2026-10-19) and `3ca2a6b…` (2026-11-04)\[2\] |

**First thing to do yourself:** clone the repo and run:
- `git log --author=AhmedZeer --oneline`
- `git diff upstream/main...main --stat` after adding vllm-project/speculators as `upstream`

That gives you the authoritative commit and diff list I couldn't retrieve. Also read `reporting/` and `examples/eagle3-lora/`.

## 2. What the fork adds (reconstructed)

**New package and CLI.** The serving loader is referenced as `speculators.generator.serving.bundle.bundle_state`, which implies `src/speculators/generator/` with a `serving/bundle.py`.\[7\] Two recorded commands:
- `speculators generator campaign --config examples/eagle3-lora/generator_campaign_online.yaml --data-path ./data --trajectories-path ./trajectories --stage all`\[2\]
- `python src/speculators/__main__.py generator train-full-data --config …/launcher/train.yaml --epochs 10 --max-updates 10000 --hub-upload-interval 1000 --hub-repo azeeer/eagle3-generator-attention-100k-10k-v1-ttt7`\[2\]

Recorded environment: speculators 0.8.0.dev0, vllm 0.31.0, transformers 5.16.1, torch 2.13.0.\[2\]

**Setup**
- Target: `Qwen/Qwen3-8B`.
- Base drafter: `RedHatAI/Qwen3-8B-speculator.eagle3` (1 layer, hidden 4096, draft vocab 32000).\[8\]
- Target hidden-state layers [2, 18, 33] concatenated into a 12288-dim generator input.\[2\]

**Generator architecture** (`bundle.json` / `configuration.json`, `"kind": "generator"`)
- `encoder_hidden 512`, `example_dim 128`, `condition_dim 64`, `descriptor_dim 32`, `decoder_width 512`, `residual_blocks 2`.\[2\]
- `condition: "last"`, which is probably the last-token state.
- `min_examples 1`, `max_examples 8`, `conditioning.states: prompt`.\[2\]

My reading of this: each conditioning example (prompt hidden states) is encoded to 128-d and pooled into a 64-d condition. A 32-d per-module/layer descriptor lets one shared decoder output the A/B factors for each targeted module.

**LoRA settings and targets**
- rank 32, alpha 64, dropout 0.05.\[2\]
- Default runs: `o_proj` and `v_proj` only. The `v_proj` shape [8192, 1024] reflects EAGLE-3's concatenated embedding plus hidden input.\[2\]\[9\]
- "attention" runs: q/k/v/o.
- "modules-all" runs: add `gate_proj`, `up_proj`, `down_proj`.\[2\]
- No `fc` or `lm_head` LoRA appears in any artifact. The base drafter appears frozen; only the generator, or the static LoRA in the baseline, is trained.
- Serving caveat (verbatim from a model card): *"Shared merged serving supports attention and MLP projections; the per-request native LoRA hook currently supports o_proj/v_proj."*\[2\]

**Objective and hyperparameters**
- `loss_fn: "lk_hybrid"`, with `-lk-` vs `-ce-` (cross-entropy) variants in repo names.\[2\]\[4\]
- EAGLE-3 TTT unroll: `ttt_steps 7`, `ttt_step_loss_decay 1.0`.\[10\]
- lr 1e-4, weight decay 0.01, grad clip 0.85, 100 warmup updates, 4 episodes per update, token budget 8192, noise_std 0.05.
- 1,500 or 10,000 pretraining updates; 5 conditioning epochs; 1 adaptation epoch; `draft_attn_impl: simple_flex_attention`.\[2\]

**Pipeline stages**
- Stages: pretraining, conditioning, heatmap (seed groups × strides 10/20/30), adaptation.\[2\]
- Two experimental arms: `direct` (the generator) and `fresh_lora` (a matched static LoRA, `matched_generator: "scale-1-direct"`).\[2\]

**Data**
- Source: `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated`, split into math, code, stem and chat subsets.
- Responses regenerated with vLLM (temperature 0.7, top-p 0.8, top-k 20, up to 4096 tokens, thinking off).\[2\]
- Published datasets: `azeeer/eagle3-qwen3-8b-{trajectories,activations,lora-bank-r32,lora-bank-r32-packed}-v1`, with per-domain `bank_{math,code,stem,chat}.json`.\[2\]\[4\] This suggests per-domain LoRAs serve as references or targets.

**Reported numbers** (`azeeer/eagle3-generator-direct-4k-v3`: 1,500 updates, 4k examples, context size 4, 100 validation examples per subset)
- Overall position-0 accuracy 0.737, position-1 0.535, position-2 0.362.
- Math is highest (0.793 at position 0); chat is lowest (0.680).\[2\]
- I didn't find the matched static-LoRA numbers or any end-to-end vLLM speedup. **So no gain from the generator is established yet.**

**Ablations found (repo names only)**
- Target modules: `modules-{mlp,attention,all}`.
- Generator size:
  - `architecture-encoder-hidden-{256,1024}`
  - `residual-blocks-{1,4}`
  - `condition-dim-{32,128}`
  - `decoder-width-{256,1024}`
- Static LoRA baselines: `static-lora-structure-v1-{all,attention}`.
- Context size: `generator-all24k-k{1,2,4,8}`.
- Conditioning source: `replay-{prompt,prefix,prefix-feedback}-lk`.
- Scale: `attention-100k-{2k,10k}-v1(-ttt7)`.
- A `deepseek-ttt7-attention` variant.\[4\]

Each step folder holds `metrics.json`, `result.json` and `timing.json`; pull these to build the outcome tables.\[2\]

## 3. DFlash: what you're porting to

**How it works**
- DFlash, from Z Lab at UC San Diego (ICML 2026), is a small block-diffusion drafter. It predicts a whole block in one forward pass.\[11\]
- Hidden states from several target layers are concatenated and projected by `fc`, then RMSNorm (`hidden_norm`).\[11\]
- The result is injected as extra K/V entries into **every** draft layer.\[12\] The target features only serve as additional KV entries for the masked-block draft tokens; they "bypass the draft model's Q projection, output projection, self-attention update, and FFN."
- Draft tokens attend bidirectionally within the block.\[13\] Embeddings and the LM head are shared with the target and frozen.\[5\]

**Training**
- Random anchor tokens sampled from the response start each block; the remaining positions are masked.\[14\]
- Blocks are trained jointly with a sparse flex-attention mask that blocks cross-block attention.
- CE loss is weighted by w_k = exp(−(k−1)/γ). The paper uses γ = 7 for block size 16 and 4 for block size 8.
- Paper defaults: 5 layers, block size 16, 5 target layers, AdamW lr 6e-4, 512 anchors per sequence.\[5\]\[15\]

**Results that matter for your design**
- Qwen3-8B: about 6.5 average acceptance length vs about 3.0 for EAGLE-3 at temperature 0.
- KV injection beats input fusion.
- 5 layers gives the best speedup; 8 layers gives longer acceptance length but is slower.
- Train with a block size ≥ the inference block size: a model trained at 16 generalises down to 8, but not the reverse.\[5\]

**Upstream speculators support**
- `--speculator-type dflash`, model `DFlashDraftModel(DraftVocabMixin, SpeculatorModel)`.\[13\]\[16\]
- Layers are `nn.ModuleList` of `Qwen3DFlashDecoderLayer`, plus `fc` (len(target_layer_ids)·H → H, no bias), `hidden_norm`, `norm`, and a frozen `verifier_norm`.\[1\]
- Sliding window attention on all draft layers by default.\[3\]
- `from_training_args` sets `speculative_tokens = block_size − 1`.\[1\]
- CLI flags: `--block-size` (default 8), `--max-anchors`, `--dflash-decay-gamma` (default 4.0).\[1\]\[16\]
- RFC #979 recommends block 16, 5 layers, and D-PACE per-position weighting (`--per-position-loss-weight dpace --dpace-alpha 0.5`) with as many anchors as fit in memory.\[6\]
- Serving: speculators-format DFlash models run in vLLM as of PR #38300.\[3\] The z-lab repo also supports SGLang, Transformers and MLX.\[3\]\[11\]\[17\]

## 4. Step-by-step port: generator + DFlash

1. **Start from a trained DFlash drafter.**
   - Use `RedHatAI/Qwen3-8B-speculator.dflash`, or train one with upstream `scripts/train.py --speculator-type dflash --block-size 16 --num-layers 5`.\[3\]\[13\]\[18\]
   - Freeze it, as the EAGLE-3 work did. This keeps the static-LoRA vs generator comparison clean.
2. **Generalise LoRA injection.**
   - Find where the fork wraps EAGLE-3 projections; look in `src/speculators/generator/`.
   - Make the target-module resolver walk `model.layers[i].self_attn.{q,k,v,o}_proj` and `.mlp.{gate,up,down}_proj` for all 5 layers.
   - Extend the 32-d descriptor so it encodes (layer index, module type). EAGLE-3 had a single layer; DFlash has 5 per module type.
   - Shapes become square-ish Qwen3 shapes (H=4096 for Qwen3-8B). The 8192 input width of EAGLE-3 no longer applies.
3. **Decide what K/V LoRA applies to.** In DFlash, `k_proj`/`v_proj` see both the injected context features and the block tokens.
   - LoRA on K/V of the **context** path changes how every layer reads target features. This is the closest DFlash analogue of EAGLE-3's v_proj adaptation.
   - LoRA on the **block** path only changes how draft tokens interact with each other.
   - Implement this as a switch. Also consider LoRA on `fc`, which is shared across layers and so a cheap global lever.
4. **Conditioning input.** Reuse the prompt hidden states. DFlash already extracts multiple target layers, so feed the generator the same `aux_hidden_state_layer_ids`. Upstream example: "2 18 33";\[19\] paper default: 5 layers. This avoids a second extraction. Keep `condition: last`, with mean-pooling as an ablation.
5. **Loss.**
   - Replace the TTT-7 unroll (`ttt_steps`, `lk_hybrid` over steps) with DFlash's anchored-block loss: sample anchors, build blocks, apply the LoRA-adapted draft forward pass, then weighted CE over positions 1..B−1.
   - Port `lk_hybrid` as a per-position term if it is acceptance-oriented.
   - Gradients flow only into the generator. Keep the generated-delta norm regulariser and noise if the fork uses them.
6. **Data.**
   - Reuse the regenerated Nemotron trajectories. They contain no DFlash-specific data, only different hidden-state layers.
   - Make sure the generator's per-episode "examples" and the anchor sampling come from the same query distribution.
   - Raise `--max-anchors`: RFC #979 found that more anchors consistently improved acceptance length.\[6\]
7. **Evaluation.**
   - Offline: per-position accuracy (DFlash metrics) and expected acceptance length τ per domain.
   - Arms: (a) base DFlash, (b) static LoRA, (c) generator.
8. **Serving.**
   - Easiest path: generate the LoRA per query, merge it into a copy of the drafter weights (the "shared merged serving" path), and run vLLM/HF with `method: dflash`.
   - True per-request LoRA in vLLM's DFlash proposer will need a custom hook. The fork's native hook only covers EAGLE-3 `o_proj`/`v_proj`; check whether it patches vLLM.\[2\]
   - Measure end-to-end speedup at concurrency 1, then 8 or more. Account for generator latency: it runs once per request at prefill.

**Pitfalls**
- Sliding window attention is on by default.\[3\]
- Mismatched mask token id or block size between training and serving.
- DFlash checkpoints omit verifier weights; `load_verifier_weights` reconstructs them.\[1\]
- Merged per-query weights break batching across requests.
- Gains measured as offline accuracy may not survive into end-to-end speedup.

## 5. Recommended ablation plan (DFlash)

| Phase | Axis | Settings |
|---|---|---|
| 0 | Baselines | base DFlash; static LoRA (attention, all) at matched rank |
| 1 | Target modules | {o,v}; {q,k,v,o}; MLP; all; +`fc` |
| 1 | Layer scope | layer 0 only; last layer only; all 5; per-layer LoRA scale (learned or fixed: early > late vs. uniform) |
| 1 | K/V path | context-only vs block-only vs both |
| 2 | Rank / alpha | 8, 16, 32, 64; alpha/rank 1–2 |
| 2 | Generator size | encoder 256/512/1024; residual blocks 1/2/4; condition 32/64/128; decoder 256/512/1024 (mirrors the fork) |
| 2 | Conditioning | k = 1/2/4/8 examples; prompt vs prefix vs prefix-feedback; last vs mean pool; target layers 3 vs 5 |
| 3 | Training | lk vs CE vs decay vs D-PACE weighting; anchors 512→3072; 1.5k vs 10k updates; frozen vs co-trained drafter (low lr) |
| 3 | DFlash-specific | block size 8/16 (train ≥ inference); per-position τ gain (does the LoRA help later positions?); full vs sliding window attention |

Report τ and per-position acceptance per domain (math, code, stem, chat), plus end-to-end speedup net of generator cost.

## Caveats
- The fork's internals are inferred from HF artifacts and haven't been verified in source. Exact class names, `lk_hybrid` semantics and the role of the "LoRA bank" need confirming against the code.
- I found no end-to-end speedup numbers or generator-vs-static comparisons, so whether the approach works at all is still unproven.
- Paper speedups (for example, about 6× on Math500) are measured under specific settings (Transformers backend, greedy decoding) and shrink at higher concurrency.\[20\]

## Kaynaklar

1. [dflash - Speculators Docs](https://docs.vllm.ai/projects/speculators/en/latest/reference/speculators/models/dflash/)
2. mcp\_\_Hugging\_Face\_\_hf\_fs
3. [GitHub - AhmedZeer/dynamic-speculators-trainer: A unified library for building, evaluating, and storing speculative decoding algorithms for LLM inference in vLLM](https://github.com/AhmedZeer/dynamic-speculators-trainer)
4. mcp\_\_Hugging\_Face\_\_hub\_repo\_search
5. <https://arxiv.org/pdf/2602.06036>
6. [\[RFC\]: DFlash Training Best Practices — recommended configuration and reasoning · Issue #979 · vllm-project/speculators · GitHub](https://github.com/vllm-project/speculators/issues/979)
7. [azeeer/eagle3-generator-direct-4k-v3 · Hugging Face](https://huggingface.co/azeeer/eagle3-generator-direct-4k-v3)
8. [Speculators Docs](https://docs.vllm.ai/projects/speculators/en/latest/)
9. [Eagle3 - Speculators Docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)
10. [vllm-project/speculators](https://deepwiki.com/vllm-project/speculators)
11. [DFlash: Block Diffusion for Lightning-Fast LLM Speculative Decoding](https://pyshine.com/DFlash-Block-Diffusion-Speculative-Decoding/)
12. [DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding](https://arxiv.org/pdf/2606.02091)
13. [Train and run DFlash speculative decoding with vLLM](https://regolo.ai/train-run-dflash-speculative-decoding-vllm/)
14. [nemo\_automodel.components.speculative.dflash.core](https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/speculative/dflash/core)
15. [(PDF) DFlash: Block Diffusion for Flash Speculative Decoding](https://www.researchgate.net/publication/400505763_DFlash_Block_Diffusion_for_Flash_Speculative_Decoding)
16. [train.py - Speculators Docs](https://docs.vllm.ai/projects/speculators/en/latest/cli/train/)
17. [GitHub - z-lab/dflash: DFlash: Block Diffusion for Flash Speculative Decoding · GitHub](https://github.com/z-lab/dflash)
18. [Train Dflash Model Online - Speculators Docs](https://docs.vllm.ai/projects/speculators/en/v0.5.0_a/user_guide/tutorials/train_dflash_online/)
19. [Speculators v0.5.0: DFlash Support and Online Training](https://vllm.ai/blog/2026-05-28-speculators-v050)
20. [DFlash: Block Diffusion for Flash Speculative Decoding](https://www.alphaxiv.org/abs/2602.06036)
