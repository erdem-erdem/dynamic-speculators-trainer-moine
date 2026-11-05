# 05 - Upstream DFlash drafter: architecture, training, serving, paper facts

Scope: the DFlash drafter as it exists in this repo (fork of vllm-project/speculators, fork point `e3041dd`).
All paths relative to repo root `dynamic-speculators-trainer/`. `core.py` = `src/speculators/models/dflash/core.py`,
`attn.py` = `src/speculators/models/dflash/attention.py` (mask), `md.py` = `.../dflash/model_definitions.py` (layers),
`cli.py` = `src/speculators/train/cli.py`, `schema.py` = `src/speculators/train/config/schema.py`.
Marked UNVERIFIED where I could not run/confirm.

## 0. Did the fork touch DFlash?

`git diff --stat e3041dd HEAD -- src/speculators/models/dflash src/speculators/models/dflash2 src/speculators/models/dspark src/speculators/model.py src/speculators/models/utils.py src/speculators/losses/utils.py scripts/train.py`
returned nothing for those paths: **the DFlash model files, dflash2/dspark, model.py, models/utils.py, losses/utils.py are byte-identical to upstream.**
Fork changes that DO touch the DFlash path (shared infra):
- `src/speculators/models/attention.py` (+58): `flex_attention_forward` gained `speculators_streaming_sdpa/_is_causal` kwargs and `_standalone_flex_attention()` = `torch.compile(flex_attention, fullgraph=True, dynamic=True)` used when `query.is_cuda and not torch.compiler.is_compiling()` (comment: "An uncompiled CUDA caller (e.g. functional LoRA substitution) still needs a fused kernel"). Registered as `"simple_flex_attention"`, which DFlash uses by default.
- `train/trainer.py` (grad_norm logging, `after_optimizer_step` hook, `base_speculator_model(self.model)` for PEFT-wrapped models), `train/data.py` / `dataloader.py` (row_indices, max_train_samples, seeded sampler), `train/cli.py` (+203), `train/config/schema.py` (+103), `train/lora.py` (PEFT LoRA wrap, `target_modules` regex-able; works on `DFlashDraftModel` as a `SpeculatorModel`), `train/bank.py`, `losses/fused.py` (+17), `checkpointer.py`, `logger.py`, `optimizers.py`.
- Fork bank-training validator (schema.py ~L826-): "Bank training requires EAGLE3 with LoRA enabled" -> bank path is hard-wired to eagle3; needs relaxing for dflash.

## 1. DFlashDraftModel architecture (`core.py`)

Class `DFlashDraftModel(DraftVocabMixin, SpeculatorModel)`, registered `"dflash"` (core.py:38-39), config `DFlashSpeculatorConfig` (`dflash/config.py:116-182`).

| Item | Detail | Ref |
|---|---|---|
| Decoder layers | `self.layers = ModuleList([Qwen3DFlashDecoderLayer(tl_config, i) ...])`, count = `transformer_layer_config.num_hidden_layers`. CLI default **5** for dflash/dspark/dflash2, 1 otherwise | core.py:92-101, `_make_decoder_layer` core.py:62-70, schema.py:93-97 & 787-788 |
| Layer class | `Qwen3DFlashDecoderLayer(GradientCheckpointingLayer)`: `self_attn=Qwen3DFlashAttention`, `mlp=Qwen3MLP` (HF; gate/up/down_proj, bias False), `input_layernorm`, `post_attention_layernorm` (Qwen3RMSNorm) | md.py:158-168 |
| Attention | `Qwen3DFlashAttention`: `q_proj,k_proj,v_proj,o_proj` = `nn.Linear(bias=config.attention_bias)`, `q_norm,k_norm` = Qwen3RMSNorm(head_dim), `is_causal=False`, GQA via `num_key_value_groups` | md.py:47-95 |
| `fc` | `nn.Linear(len(target_layer_ids)*hidden, hidden, bias=False)`. Default 3 target layers -> in 12288 (Qwen3-8B) out 4096 | core.py:122-126 |
| `hidden_norm` | `Qwen3RMSNorm(hidden)` applied after `fc` (once, shared by all layers) | core.py:127-130, 418-419 |
| `norm` | final `Qwen3RMSNorm(hidden)` before `lm_head` | core.py:115-118, 468 |
| `verifier_norm` | frozen RMSNorm (class chosen by `resolve_verifier_norm_class`: Gemma-style `1+w` for gemma/qwen3_5 else Qwen3RMSNorm); loaded from verifier `model.norm.weight`; used only to build targets; excluded from save | core.py:131-135, models/utils.py:51-79 |
| `embed_tokens` | `nn.Embedding(verifier_vocab, hidden)`, **frozen, shared-from-target** (copied from verifier at load, NaN-initialised until loaded; `overwrite_embed_tokens=True`; never saved: `_keys_to_ignore_on_save` appends `embed_tokens.weight`) | model.py `DraftVocabMixin._init_vocab` L37-79; core.py:157, 171-176 |
| `lm_head` | `nn.Linear(hidden, draft_vocab_size, bias=False)`, **frozen**. Full vocab: copy of verifier head, not saved. Reduced vocab: rows `t2d`-selected from the verifier head; saved with checkpoint | core.py:158-160; model.py L127-190 |
| `verifier_lm_head` | `nn.Linear(hidden, draft_vocab_size)`, frozen, rebuilt from verifier; produces training targets (logits restricted to draft vocab) | core.py:443-449 |
| Vocab mapping | `t2d` (bool [verifier_vocab]) / `d2t` (long [draft_vocab]) buffers, only if `draft_vocab_size != verifier_vocab_size`. CLI examples use `--draft-vocab-size 32000` (needs `token_freq.pt` built in prepare-data). `load_vocab_mappings` silently ignores mappings when vocab is full (model.py L97-102) | model.py L45-121 |
| Rotary | `Qwen3RotaryEmbedding(flatten_rope_parameters(tl_config))` | core.py:119-120 |
| `mask_token_id` | `config.mask_token_id`; property raises ValueError if None (core.py:309-317). CLI resolves via `resolve_mask_token_id`: explicit > tokenizer.mask_token_id > add `<|MASK|>` if unused embedding slots > pad/eos/unk fallback w/ warning (train/utils.py:16-67). z-lab Qwen3-8B uses 151669 | cli.py:588-593 |
| Block size | `config.block_size` default 8 in config (config.py:144-149); **CLI default 16 for `dflash`** (8 otherwise) (schema.py:515-519, 792-793); `from_training_args` falls back to 8 only if kwargs lack it (core.py:241) | |
| `sample_from_anchor` | default False for dflash: slot 0 = anchor (clean token, not predicted), `block_size-1` speculative tokens; True for dspark | config.py:173-182, core.py:243-259 |
| target_layer_ids | `config.aux_hidden_state_layer_ids`. Default `[2, n//2, n-3]` (Qwen3-8B n=36 -> `[2, 18, 33]`), with a warning; validated distinct and in `[0, n]`; id `n` = final hidden (valid). The final-layer hidden state is extracted by vLLM `--include-last-layer` and delivered as `verifier_last_hidden_states`, NOT as part of `hidden_states` | models/utils.py:85-123; examples use `--target-layer-ids 2 18 33` |
| Sliding window | Layer config from `create_transformer_layer_config` (cli.py:121-262): **all draft layers are `sliding_attention` by default** (`sliding_window=2048`, `--sliding-window`), unless `--full-attention-indices` lists layers; `use_sliding_window=True` when any sliding. `sliding_window_non_causal` default False for dflash (True for dflash2) | cli.py:219-239, schema.py:172-189, 783-784 |
| Attn impl | `simple_flex_attention` (default, forced if `_attn_implementation is None`), or `sdpa`/`eager` | core.py:76-88, schema.py:190-194 |
| Misc | `supports_gradient_checkpointing=True`, `_no_split_modules=["Qwen3DFlashDecoderLayer"]`; `tie_word_embeddings=False` | core.py:41-42 |

**Hazard (mask semantics) - important for the port.** `create_anchor_block_mask_mod` (attn.py:42-43) sets `non_causal = sliding_window is None or sliding_window_non_causal`. With the CLI defaults (all layers sliding, `sliding_window_non_causal=False`) the intra-block attention is **causal** (attn.py:95-98: `same & (kv_idx <= q_idx + total_seq_len)`), while the paper (p5) and docs (`docs/user_guide/algorithms/dflash.md`) say bidirectional within block, and z-lab converted checkpoints have `layer_types` all `full_attention` (transformers 5.5.4 `Qwen3Config` default, verified) = bidirectional. For paper-faithful training pass `--full-attention-indices 0 1 2 3 4` (or `--sliding-window-non-causal`). Whether vLLM's DFlash path honours sliding window / causal-in-block for spec-trained checkpoints: UNVERIFIED. Also sliding window default 2048 vs `--total-seq-len 8192`: context older than 2048 tokens before the anchor is invisible.

Contrast with EAGLE-3 (`models/eagle3/core.py`): 1 decoder layer by default, `fc` over 3 concatenated aux states then concatenated with token embedding in the first layer (q/k/v take 2*hidden input, eagle3/model_definitions.py:14-50), TTT steps (`ttt_steps=3`), `embed_requires_grad` option, KL loss default. DFlash has no embedding concat and no TTT.

## 2. How target features enter K/V - exact code (answer to the key question)

`md.py:97-155` (Qwen3DFlashAttention.forward), verbatim:

```python
bsz, q_len = hidden_states.shape[:-1]
ctx_len = target_hidden.shape[1]
q = self.q_proj(hidden_states)                       # block tokens ONLY
q = q.view(bsz, q_len, -1, self.head_dim)
q = self.q_norm(q).transpose(1, 2)
# This is the main difference from the usual attention mechanism.
k_ctx = self.k_proj(target_hidden)                   # SAME k_proj, context path
k_noise = self.k_proj(hidden_states)                 # SAME k_proj, block path
v_ctx = self.v_proj(target_hidden)                   # SAME v_proj, context path
v_noise = self.v_proj(hidden_states)                 # SAME v_proj, block path
k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
k = self.k_norm(k).transpose(1, 2)                   # k_norm on BOTH ctx and block keys
v = v.transpose(1, 2)
cos, sin = position_embeddings
q, k = apply_rotary_pos_emb(q, k, cos, sin)          # q uses last q_len of cos/sin; k uses all
...
attn_output, attn_weights = attn_fn(self, q, k, v, attention_mask,
        dropout=..., scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
attn_output = attn_output.reshape(bsz, q_len, -1)
attn_output = self.o_proj(attn_output)               # block tokens ONLY
```

Layer (`md.py:170-207`):
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)            # block tokens only
hidden_states = self.self_attn(hidden_states=hidden_states, target_hidden=target_hidden, ...)[0]
hidden_states = residual + hidden_states
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)                        # block tokens only
return residual + hidden_states                                # target_hidden NOT modified
```

Facts:
- **Yes: the same `k_proj` / `v_proj` modules are applied to both context (`target_hidden`) and block tokens**, via two separate `nn.Linear` calls (not one call on a concatenation), then concatenated along the sequence dim: KV layout = `[context (T) | block tokens (A*bs)]`. Because the calls are separate call-sites, a LoRA on `k_proj`/`v_proj` implemented by wrapping the Linear (PEFT) applies identically to both; but a **custom wrapper/patched forward can trivially treat them differently** (edit the 4 lines above; the attention kernel only sees finished q/k/v + BlockMask).
- `q_proj`, `o_proj`, `input_layernorm`, `post_attention_layernorm`, MLP (`gate/up/down_proj`) see **block tokens only**. Context features never pass through q/o/MLP (paper A.3, p12: "They bypass the draft model's Q projection, output projection, self-attention update, and FFN").
- `target_hidden` = `hidden_norm(fc(hidden_states))` computed ONCE (core.py:418-419) and shared (identical tensor) by all layers; it is not passed through each layer's `input_layernorm` (block tokens are). Per layer, context K/V = `k_proj(fc_output)` recomputed per layer (training recomputes for all T positions; each block only sees the prefix before its anchor via mask).
- `k_norm` (per-head RMSNorm) is applied after k_proj to both paths, so an additive LoRA delta on k is partially re-normalised; `v` has no norm.
- Positions: `position_ids = cat([position_ids(T), anchor_pos + arange(bs) per block])` (core.py:422-425); rotary cos/sin length `T + A*bs`.
- `fc` output (T tokens) and block noise embeddings (A*bs tokens) are batch dim 1 always (packed).

Block-token input: `mask_token_ids = full(mask_token_id)`, slot 0 of each block overwritten with the real anchor token `input_ids[:, anchor_positions]` (core.py:408-414), embedded by the frozen `embed_tokens` -> `noise_embedding` `[1, A*bs, H]`.

## 3. Training

Entry: `python -m speculators.train --speculator-type dflash ...` (`scripts/train.py` / `src/speculators/train/__main__.py`; registry lookup cli.py:708-714). Per step `Trainer` calls `self.model(**gpu_batch, **train_call_kwargs)` under bf16 autocast (trainer.py ~L542); returns `(None, loss, metrics)` (core.py:525).

**Data format** (train/data.py:~110-125): one packed row of width `--total-seq-len` (default 8192) per batch, `B=1`:
`hidden_states [1,T,n_aux*H]` (concat of aux layers from vLLM hidden-state extraction, `scripts/launch_vllm.py --target-layer-ids ... [--include-last-layer auto]`, ExampleHiddenStatesConnector), `verifier_last_hidden_states [1,T,H]`, `input_ids [1,T]`, `loss_mask [1,T]`, `position_ids [1,T]`, `document_ids [1,T]` (doc index, -1 pad), multipack sampler `MultipackDistributedBatchSamplerV2` (dataloader.py). No `shift_batch` preprocess for dflash (cli.py:786-792). Data comes online (live vLLM `--vllm-endpoint`, `--on-missing generate --on-generate delete`) or offline cached `hs_*.safetensors` (`--on-missing`, file backend). **Target model is never resident in the trainer**: only its frozen embedding/lm_head/final-norm weights (loaded via `load_model_layers` from safetensors, model.py L144-190). `noise_std=0.05` uniform noise on hidden states (`AddUniformNoise`, dataloader.py:146-150).

**Anchor sampling** (`dflash/utils.py:339-385`, `select_anchors`): `valid = loss_mask.bool()` with the last `block_size` positions disabled; `k=min(max_anchors, #valid)`; random permutation, pick k, **sorted ascending** (contiguous anchors help flex-attn density); padded to `max_anchors` with `anchor_valid=False`. Anchors drawn across the whole packed row (all docs). `--max-anchors` default **512** (schema.py:235-239; paper also 512 per sequence, p12), examples use 3072. Re-sampled every forward.

**Block construction**: block j = positions `anchor_j .. anchor_j+bs-1` (`get_base_indices_for_anchored_blocks`, utils.py:324-336); Q length `A*bs`, KV length `T + A*bs` (attn.py:53-55).

**Mask** (`dflash/attention.py:7-100`, built via `torch.compile(create_block_mask)` core.py:35, 337-344):
- `base_prefix_mod`: query in block j sees base kv iff `kv_idx<T`, same document as the anchor (`document_ids[q_anchor]==document_ids[kv_pos]`, not -1), `kv_pos < q_anchor` (strictly before anchor; anchor token itself arrives via the block's slot 0), and (if sliding) `kv_pos >= q_anchor - sliding_window`.
- `same_block_mod`: query sees synthetic kv of own block only (`q_block==kv_block`); bidirectional if `non_causal` else causal.
- No visibility of other blocks (cross-block isolation), no future base tokens. Per-layer: `full_attn_mask` for non-sliding layers, `sliding_window_attn_mask` for sliding layers (core.py:347-375, 455-466).

**Targets** (core.py:436-453): `verifier_lm_head(verifier_norm(verifier_last_hidden_states[target_idx]))`, `target_idx = anchored_block_indices - 1` (mod T) when `sample_from_anchor=False`, i.e. the verifier's next-token logits at the previous position. Restricted to draft vocab via the lm_head rows. Frozen, `no_grad`.

**Loss** (`models/dflash/metrics.py`, `losses/utils.py`):
- Defaults in schema for `dflash`: `--loss-fn ce`, `--per-position-loss-weight dpace`, `--dpace-alpha 0.5`, `--dflash-decay-gamma 4.0`, `--num-layers 5`, `--block-size 16` (schema.py:748-793, recipe from upstream issue #979). `get_trainer_kwargs` fallbacks (core.py:294-299) are `fixed-exp-decay`/gamma 4.0/max_anchors 512 only if the key is absent.
- `ce` = cross-entropy with **hard labels = argmax of target logits** (`losses/eager.py:92-115`; fused Triton version `fused_ce_loss`, requires Triton; `--loss-implementation eager` for CPU/Windows).
- Fixed decay (paper Eq.4): 
```python
# losses/utils.py:17-45
decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)   # slot0 (anchor) weight 0
```
 i.e. w_k = exp(-(k-1)/gamma), k=1.. block_size-1.
- D-PACE (`dpace_loss_decay`, losses/utils.py:~75-110): q = exp(-CE) per position; smooth = (1-alpha)*q + alpha (masked->1); prefix = cumprod(smooth) within block; weight_k = sum_{j>=k} prefix_j * mask_j (reverse cumsum) -> dynamic, detached (no_grad), requires `ce`.
- Reduction (`loss_function`, utils.py:269-300): `sum(elementwise*mask*decay)/ (sum(mask)+1e-5)` over the whole packed row (mean over all valid trained positions, not per-block).
- Loss mask per slot: `loss_mask[anchored_block_indices] * anchor_valid.repeat_interleave(bs)`, slot 0 zeroed if not sample_from_anchor (core.py:472-484).

**Metrics**: `loss_sum/_total`, `full_acc`, `position_{i}_acc` for i in 1..bs-1, and `eal` = mean over valid blocks of (leading run of correct slots + 1 bonus token) (`models/metrics.py:7-30`, dflash/metrics.py:312-317). Accuracy is vs target argmax (teacher-forced-free: all slots predicted in parallel), so `eal` ~ acceptance-length estimate under greedy verification.

**Trained vs frozen** (`requires_grad`): trainable = `layers.*` (all attn/mlp/norms), `fc`, `hidden_norm`, `norm`. Frozen: `embed_tokens`, `lm_head` (draft head; same as verifier rows), `verifier_lm_head`, `verifier_norm` (core.py:135; model.py L71, 183-190; mixin L63-71). Optimizer built over `requires_grad` params only (optimizers.py:56, 81). Master weights fp32, autocast bf16 (`--hidden-states-dtype bfloat16`).

**Key CLI flags** (schema.py): `--verifier-name-or-path`, `--data-path`, `--save-path`, `--speculator-type dflash`, `--block-size` (16), `--num-layers` (5), `--max-anchors` (512), `--total-seq-len` (8192), `--target-layer-ids`, `--draft-vocab-size`, `--d2t-path/--t2d-path/--token-freq-path`, `--mask-token-id`, `--sliding-window` (2048), `--full-attention-indices`, `--sliding-window-non-causal`, `--sample-from-anchor`, `--draft-attn-impl {simple_flex_attention,sdpa,eager}`, `--draft-config`, `--from-pretrained`, `--draft-hidden-act silu`, `--loss-fn`, `--loss-implementation {fused,eager}`, `--per-position-loss-weight`, `--dpace-alpha`, `--dflash-decay-gamma`, `--lr`, `--epochs`, `--gradient-checkpointing`, `--fsdp-shard`, `--lora-r/--lora-alpha/--lora-target-modules` (PEFT on the whole drafter), `--on-missing`, `--on-generate`, `--dry-run` (saves initialised ckpt, cli.py:630-660). Example recipes: `examples/train/dflash_qwen3_8b_ultrachat_online_5k{,_bestpractices}.sh` (LR 3e-4, 5 epochs, 5k samples, `MAX_ANCHORS=3072`, `TARGET_LAYER_IDS="2 18 33"`, 2 GPUs vLLM + 2 GPUs train).

## 4. Inference / proposal / serving

- **This repo has no DFlash generation loop.** Drafting is done by vLLM only (`vllm serve <ckpt>`, spec method auto-read from the speculators config, or `--speculative-config '{"method":"dflash",...}'`; `examples/demo/demo_side_by_side.sh --method dflash`; `examples/evaluate/example_qwen3_8b_dflash_humaneval.sh` uses `nm-testing/dflash-qwen3-8b-speculators`). The only in-repo "proposal" artifact is `GreedyTokenProposalConfig(speculative_tokens = block_size-1)` (or `block_size` if `sample_from_anchor`), core.py:256-274.
- vLLM minimum version: no vLLM pin in `pyproject.toml` (vllm only a keyword). Docs: `docs/user_guide/tutorials/train.md:93` installs `vllm>=0.27.1`; `response_regeneration.md` says `>=0.19.1`. Fork serving patch pins vLLM 0.31/0.32 (see note 03). Exact DFlash-support minimum version: UNVERIFIED.
- Pinned deps (pyproject `[tool.speculators.dependencies]`): `torch>=2.9.0,<=2.13.0`, `transformers>=5.0.0,<5.17.0`, `datasets>=4.0.0,<=5.0.1`, `peft>=0.18` (extra `lora`).
- vLLM DFlash semantics (from `tests/integration/models/test_dflash_vllm_parity.py:122-160`): context KV is prefix-only (`kv < anchor`), draft KV attends within same block only (non-causal); reference mask in that test uses full-attention config (no sliding window).
- Consistency hazards: (a) `block_size` train vs serve - paper Table 8 (p8): train16/test8 degrades slightly (4.64 -> 3.87x on Math500), train8/test16 degrades; matching is best. (b) `mask_token_id` must equal the ckpt's (serialised in config). (c) `sample_from_anchor` must match engine setting (docs/algorithms/dflash.md). (d) sliding window + causal-in-block default vs full/bidirectional in vLLM and z-lab (section 1 hazard). (e) `target_layer_ids` must match vLLM `eagle_aux_hidden_state_layer_ids`; z-lab ids are offset +1 (converter.py:112-124). (f) fc `hidden_norm` and per-layer unnormalised context path must match the vLLM implementation. (g) any per-request LoRA on k/v/q/o/MLP needs a matching vLLM implementation (fork's vLLM patch targets EAGLE-3; DFlash path unverified).

## 5. Pretrained checkpoints / loading

- Listed in README/docs: `RedHatAI/Qwen3-8B-speculator.dflash`, `RedHatAI/Qwen3-30B-A3B(-Instruct-2507)-speculator.dflash`, `RedHatAI/gemma-4-31B-it-speculator.dflash`, `RedHatAI/Mistral-Small-4-119B-2603.dflash`, Nemotron-3 Super/Ultra `.dflash`; `nm-testing/dflash-qwen3-8b-speculators` (eval example); external `z-lab/Qwen3-8B-DFlash-b16` (README demo, converter docs). I have no network; actual configs UNVERIFIED except the z-lab shape in the converter test fixture (`tests/unit/convert/test_dflash_converter.py:35-52`): `block_size=16`, `num_target_layers=36`, `dflash_config={mask_token_id:151669, target_layer_ids:[1,9,17,25,33]}`, `hidden 4096, intermediate 12288, 5 layers, 32 heads, 8 kv, head_dim 128, vocab 151936`. z-lab ids `+1` -> speculators `aux_hidden_state_layer_ids=[2,10,18,26,34]` (5 layers, matches paper "5 layers between 2nd and third-to-last", p6). No `layer_types` -> HF default all `full_attention`.
- Loading: `SpeculatorModel.from_pretrained(path_or_hub, verifier="Qwen/Qwen3-8B")` (or `DFlashDraftModel.from_pretrained`) - if config lacks `speculators_model_type` it auto-runs `DFlashConverter` (model.py L315-337, `convert/entrypoints.py`), copies layers/fc/hidden_norm/norm as-is, fills embed/lm_head/verifier_norm from the verifier, saves, validates. After HF load it calls `load_vocab_mappings(t2d,d2t)` then `load_verifier_weights()` (model.py L383-387; DFlash override core.py:171-176 forces overwrite of embed and, for full vocab, lm_head). CLI: `speculators convert z-lab/Qwen3-8B-DFlash-b16 --algorithm dflash --base-model Qwen/Qwen3-8B`; training warm start: `--from-pretrained <path>` (cli.py:496-545; re-applies `draft_attn_impl` since `_attn_implementation` is never serialised) or config-only dir to re-init fresh weights.
- Caveat: `load_verifier_weights()` returns silently (leaving NaN embed/lm_head) if `speculators_config` or `verifier.name_or_path` is None (model.py L146-153) - tiny tests fill NaNs manually (`_fill_nan_weights`, tests/integration/conftest.py:85-93).

## 6. Memory profile (Qwen3-8B target, hidden 4096, inter 12288, 32 heads / 8 kv, head_dim 128, vocab 151936)

Per decoder layer: q 16.78M + k 4.19M + v 4.19M + o 16.78M + MLP 3*4096*12288 = 150.99M + norms 8.4K = **192.95M**.
- 5 layers = 964.7M; `fc` (3 aux) 50.3M (5 aux: 83.9M); + 8K norms.
- **Trainable: ~1.015B (3 aux ids) / ~1.049B (5 aux ids).**
- Frozen resident: `embed_tokens` 151936*4096 = 622.3M; `lm_head` + `verifier_lm_head` each 32000*4096 = 131.1M with draft vocab 32k (each 622.3M if full vocab - 2 extra full-vocab heads!), `verifier_norm`. => ~884M frozen (draft vocab 32k) -> total ~1.9B.
- Training footprint (fp32 master weights, Adam): trainable 4.1 GB weights + 4.1 GB grads + 8.1 GB Adam states ~ 16.2 GB + frozen ~3.5 GB (fp32) ~ 20 GB before activations. Activations: logits and targets `[1, A*bs, 32000]` (A=3072, bs=16 -> 49152 rows ~ 3.1 GB each in bf16), KV length T+A*bs ~ 57k; flex-attn block-sparse mask; use `--gradient-checkpointing`; fused CE avoids materialising extra softmax. Fits 80 GB H100 comfortably at A<=3072; start lower (UNVERIFIED peak numbers - not measured).
- Target model is NOT loaded in the trainer. Online mode needs a separate vLLM server on other GPUs; offline mode needs cached hidden states on disk: per token (n_aux+1)*4096*2 bytes (bf16) = 32 KB (3 aux + last), 48 KB (5 aux + last).
- **Local 8GB RTX 4060 smoke test.** A tiny config fits trivially (conftest `TINY_QWEN3_CONFIG`: `vocab 128, hidden 64, inter 256, layers 2, heads 4, kv_heads 4, head_dim 16, max_pos 256`; `make_dflash_model` tests/integration/conftest.py:133-165: `draft_vocab_size=64, block_size=4, aux_hidden_state_layer_ids=[0,1,2], mask_token_id=0, VerifierConfig(name_or_path=None)`, then fill NaNs with N(0,0.02)). Minimal forward inputs: `hidden_states [1,T,3*H]`, `input_ids [1,T]`, `loss_mask [1,T]`, `verifier_last_hidden_states [1,T,H]`, `document_ids [1,T]` (+ optional `position_ids`), call kwargs `loss_config=resolve_loss_config("ce","eager")`, `max_anchors` (small), `block_size` from config. With `draft_vocab_size == vocab_size` no t2d/d2t needed.
- **Local env blockers (verified by running imports, no repo changes):** local Python 3.11, torch 2.5.1+cu121, transformers 5.5.4. `import speculators` fails natively: (1) package metadata `speculators` not installed (`version("speculators")`, config.py:278); (2) `hs_connectors` not installed and imports `fcntl` (POSIX only; hs_connectors/src/hs_connectors/transfer.py:6) -> not importable on native Windows; (3) `from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard` (train/distributed.py:16) fails on torch 2.5.1 (needs the FSDP2 API; repo pins torch>=2.9). So smoke tests must run in WSL/Linux with torch>=2.9 (or import only the dflash module files with stubs). Also flex-attention on torch 2.5.1 lacks `enable_gqa`-era features (UNVERIFIED); use `--draft-attn-impl eager` (eager uses `create_float_mask`, bf16 mask, core.py:82-88) or `sdpa` for the smoke test. The fused loss needs Triton -> use `loss_implementation="eager"` locally. Did not get a successful forward run locally (UNVERIFIED end-to-end).

## 7. Integration seams for a per-sample LoRA generator

Shapes (Qwen3-8B draft; heads 32, kv 8, head_dim 128 => q/o dim 4096, kv dim 1024):

| Module path (per layer i in 0..4) | in -> out | Notes |
|---|---|---|
| `layers[i].self_attn.q_proj` | 4096 -> 4096 | block tokens only |
| `layers[i].self_attn.k_proj` | 4096 -> 1024 | ctx AND block (two calls) |
| `layers[i].self_attn.v_proj` | 4096 -> 1024 | ctx AND block (two calls) |
| `layers[i].self_attn.o_proj` | 4096 -> 4096 | block only |
| `layers[i].mlp.gate_proj / up_proj` | 4096 -> 12288 | block only |
| `layers[i].mlp.down_proj` | 12288 -> 4096 | block only |
| `fc` | 12288 (3 aux) / 20480 (5 aux) -> 4096 | context-feature fusion; applied once to all T positions |

Tiny (conftest): hidden 64; q/o 64 (4x16), k/v 64 with kv_heads=4 (32 if kv_heads=2); MLP 64<->256; `fc` 192->64.

Seams:
1. **Where deltas can be applied without touching flex attention.** `flex_attention_forward` (models/attention.py) receives already projected `q,k,v` `[B,H,L,D]` plus a `BlockMask`; LoRA only changes the Linear outputs, so batched `y = base(x) + scale * ((x @ A) @ B)` at the projection call-sites is fully compatible with flex attention (the fork's own `_standalone_flex_attention` exists precisely for this "functional LoRA substitution" case; note 01 for the eagle3 version). Either wrap the Linears (PEFT `train/lora.py` pattern; `target_modules` names above, `fc` also targetable) or `torch.func.functional_call` with per-sample weights.
2. **Context vs block path.** Edit `md.py:116-119` so `k_ctx/v_ctx` use one delta (or none) and `k_noise/v_noise` another. This is the only place where DFlash lets you separate "target-feature -> KV" from "draft-token -> KV". Plain PEFT wrapping cannot distinguish them (same module object, two calls) - would need a subclass/context flag (e.g. a module attribute toggled between the two calls) or a patched forward. Remember `k_norm` after k.
3. **"Sample" granularity is the hard part.** A DFlash batch is ONE packed row (`B=1`) of many documents; blocks are sampled across docs. Per-sample LoRA therefore needs a token->doc->LoRA-index map: context positions: `document_ids[0, :T]` (-1 = pad); block tokens: `document_ids[0, anchor_positions].repeat_interleave(block_size)` (both available inside `_backbone_forward`/`_build_attention_mask`; core.py:402-414). Options: (a) gather per-token A/B (bmm over doc index, heavy for 12288-wide MLP), (b) segment the packed row per doc and run grouped matmuls, (c) restrict packing to one document per row for generator training (changes throughput; dataloader multipack `batch_max_length=total_seq_len`, data.py/dataloader.py), (d) compute low-rank path as `(x @ A_doc) @ B_doc` via `torch.bmm` after scattering rows by doc id. The generator's conditioning (what text/features feed the hypernetwork) must be computable per doc before the forward.
4. Hook locations: layer-level `_make_decoder_layer` factory (core.py:62-70; DFlash2/DSpark already subclass via this factory - cleanest place to inject a LoRA-aware layer class); `DFlashDraftModel._backbone_forward` (core.py:377-486) is the shared trunk reused by DSpark (loops `self.layers` core.py:455-466) so extra per-sample args need threading through `**kwargs` (kwargs are forwarded to each layer, core.py:465, then to attention) - a `lora_state`/`doc_index` kwarg can ride that path; `forward` (core.py:488-525) and `get_trainer_kwargs` (core.py:281-307) are where new train-call kwargs would be added.
5. **Frozen-base discipline.** Only layer params/fc/norms are trainable by default; freeze all (`requires_grad_(False)`) and train only the generator; `embed_tokens/lm_head/verifier_*` already frozen. PEFT `base_speculator_model()` unwraps for `verify_training_compatible` (trainer.py setup_model).
6. `torch.compile`: `forward` is wrapped by `conditional_torch_compile` (core.py:488; compiled only if CUDA available) and masks are built under `@torch.compiler.disable`; data-dependent shapes (variable #valid anchors padded to `max_anchors`, so shapes are static per `max_anchors`) are OK; per-sample LoRA index tensors of varying length/num docs may cause recompiles (UNVERIFIED) - pad doc count.
7. Gradient checkpointing is implemented via `GradientCheckpointingLayer`; extra tensor kwargs to layers must be checkpoint-safe.

## 8. Paper facts (dflash.pdf, "DFlash: Block Diffusion for Flash Speculative Decoding"; page = PDF page)

- Method (p4-5): fused target context features = `RMSNorm(W_c [H(l1);...;H(l5)])` (A.3, p12), injected into **K and V of every draft layer** (KV cache), bypassing Q/O/FFN; draft tokens produce Q; `K_i=[W^K H_t ; W^K H_d]`, `V_i` likewise (p12) - matches the repo code exactly. Shared embedding + LM head with target, frozen; only draft layers trained (p5-6).
- Training (p5): sample random anchor tokens from the response; block = anchor + (bs-1) masks; bidirectional within block, attends to injected target context, no cross-block attention (Flex Attention); fixed number of masked blocks per sequence, re-sampled each epoch; loss weight `w_k = exp(-(k-1)/gamma)` (Eq.4).
- A.1 (p12): 6 epochs, AdamW lr 6e-4, grad clip 1.0, cosine, warmup ratio 0.04, max seq len 3072 (4096 Coder), **512 anchors/sequence**, **gamma = 7 (bs16), 5 (bs10), 4 (bs8)**; online or offline training; data ~800K samples (Nemotron Post-Training V2 + CodeAlpaca) with target-generated responses (p6). Repo defaults gamma 4.0 regardless of block size (or D-PACE).
- Implementation (p6): **5 layers (8 for Coder), block size 16 (10 for LLaMA-3.1), target features from 5 layers uniformly between 2nd and third-to-last layer** (repo default is 3 layers `[2,n//2,n-3]` - differs).
- Results (Table 1, p6; Qwen3-8B, T=0, thinking off, max 2048 new tokens; speedup / tau): DFlash(16): GSM8K 5.15x/6.54, MATH-500 6.08x/7.87, AIME25 5.62x/7.08, HumanEval 5.14x/6.50, MBPP 4.65x/5.95, LCB 5.51x/7.27, MT-Bench 2.75x/4.24, **avg 4.86x / tau 6.49**. EAGLE-3(16) avg 1.76x/2.96; EAGLE-3(60) avg 2.02x/3.40. T=1 DFlash avg ~4.03x/5.48 (p6 text: 4.9x at T=0 = 2.4x over EAGLE-3(16); ~4.1x at T=1). (Row/model attribution recovered from interleaved pdftotext columns; Q3-4B rows are the adjacent block - minor UNVERIFIED.) Thinking mode (Table 2, p7) Q3-8B T=0: GPQA 4.17x/5.17, MATH-500 4.64x/5.82, AIME25 4.51x/5.74. SGLang B200 Qwen3-8B (Table 3, p7): Math500 conc.1 5.1x, tau 8.01; HumanEval 4.2x, tau 6.50. vLLM Qwen3.5-9B (Table 12, p12): 4.0x at conc.1 on Math500.
- Ablations: 
  - Layers (Table 6, p8, bs16, 5 feat): 3L 4.69x/tau 5.64 (Math500); 5L 4.71x/5.99; 8L 4.64x/6.33 -> **5 layers best speedup**, tau grows with depth.
  - Target features (Table 7, p8, 3L): 3 features 4.49x/5.38 vs 5 features 4.69x/5.64 (more features better, but offline storage scales linearly).
  - Train/test block size (Table 8, p8): b16->b16 4.64x/6.33; b16->b8 3.87x/5.09; b8->b16 3.78x/5.02; b8->b8 3.97x/5.21 (larger-train generalises down, not up).
  - **KV injection vs input fusion** (Table 9, p9, Qwen3-4B, 5L, bs8, tau/speedup): block-diffusion DFlash Input 3.5/2.9x (GSM8K) vs **KV 4.2/3.3x**; autoregressive EAGLE-3-5L Input 4.2/2.1x vs DFlash-AR KV 4.8/2.4x.
  - Loss decay (A.5.1, Fig 5, p13): with decay converges faster/better than uniform. Random anchor sampling (Table 13, p13): standard blocks 4.13x/4.94 vs sampled 4.69x/5.64.
  - Draft w/o target features (Table 10, p12): only 2.8-3.7x / tau 3.4-4.6 (so conditioning matters).
  - Long context (Table 4, p8): 4K-trained drafter degrades beyond 4K, fine-tuning with 1.6K LongAlign samples x3 epochs recovers.
- Latency note (p3-4): drafting cost nearly flat in block size; 5-layer DFlash drafting 16 tokens is cheaper than 1-layer EAGLE-3 drafting 8 (Fig 3).

## 9. Silent fallbacks / default substitutions (file:line)

- `core.py:77-80` `_attn_implementation is None` -> forced to `"simple_flex_attention"`; `core.py:238-240` `kwargs.get("draft_attn_impl","simple_flex_attention")`.
- `md.py:134-141` (attention): `_attn_implementation is None` or `"eager"` -> `eager_attention_forward` silently.
- `core.py:103-112` `getattr(tl_config,"sliding_window",None)`, `layer_types or []` -> no sliding-window layers if absent.
- `core.py:241` `block_size` default 8 (but schema fills 16 for dflash); `core.py:294-299` `kwargs.get` defaults `dflash_decay_gamma=4.0`, `max_anchors=512`, `per_position_loss_weight="fixed-exp-decay"` (schema default is `dpace` for dflash -> inconsistent if `get_trainer_kwargs` called bare), `dpace_alpha=0.5`; `core.py:288-` `loss_implementation` default `"fused"` (needs Triton, else RuntimeError - not silent). `core.py:395` `kwargs.pop("max_anchors",512)`.
- `core.py:437-445` wraparound `(anchored_block_indices-1) % total_seq_len`: an anchor at position 0 silently takes the target at the last position (only safe because loss_mask is 0 on prompt tokens; also packed docs: anchor at a doc start would take the previous doc's last logits - masked by loss_mask on prompts).
- `dflash/utils.py:373-383` `select_anchors`: fewer valid anchors than `max_anchors` -> zero-padded anchors (`anchor_valid=False`); zero valid anchors -> no error, loss ~0 (denominator +1e-5, losses/utils.py:`_LOSS_REDUCTION_EPS`).
- `model.py:146-153` `_load_verifier_weights` returns silently if `speculators_config` is None or `verifier.name_or_path` is None (weights stay NaN); `model.py:~193-199` missing `model.norm.weight` -> `warnings.warn` + weight stays 1.0; `model.py:97-102` t2d/d2t silently ignored when draft vocab == verifier vocab.
- `models/utils.py:36-46` `_verifier_model_type` bare-ish `except (KeyError, OSError, ValueError)` -> warning + assume plain RMSNorm convention (wrong targets for Gemma-style verifiers); `utils.py:56-57` returns False if no verifier path.
- `models/utils.py:86-123` `resolve_target_layer_ids`: unset -> default `[2,n//2,n-3]` + `warnings.warn` (must equal what vLLM extracted); `resolve_draft_intermediate_size` (utils.py:~136) falls back to `3*hidden_size` for MoE verifiers with a warning.
- `train/utils.py:54-62` `resolve_mask_token_id` falls back to pad/eos/unk (warning) and may mutate tokenizer by adding `<|MASK|>`.
- `cli.py:341-348` `--draft-config` vocab_size silently overridden by verifier vocab (logger.warning).
- `cli.py:520-524` `_build_from_config_only`: CLI verifier path used only if saved config has none.
- `models/utils.py:~72` `conditional_torch_compile`: compile only when CUDA available, else eager function (silent perf/behaviour difference).
- `train/data.py` `on_missing` in {generate, skip, warn, raise}: `skip/warn` yield `SampleUnavailable` batches (dropped, counted in `error_records`; empty batch replaced by a placeholder sample, dataloader `__call__`). Fork added `raise` on token mismatch (data.py ~L358).
- `dflash/config.py:189-200` `validate_transformer_config`: dict without `model_type` -> `Qwen3Config` silently.
- `core.py:139-144` warning only for `sample_from_anchor=True` on plain DFlash.
- No bare `except:` in the dflash package itself.

## 10. dflash2 / dspark differences (skim)

- `dflash2/core.py`: subclass of `DFlashDraftModel`; requires full verifier vocab (no draft vocab); overrides `_make_decoder_layer` with `Qwen3DFlash2DecoderLayer` (grouped dynamic causal conv, `conv_kernel_size`, `conv_group_size`) and adds `CandidateSelector` (low-rank bilinear rerank, `selector_top_k`, `selector_rank`); defaults 5 layers, bs 8, KL + fixed decay, `sliding_window_non_causal=True`. Experimental objective.
- `dspark/core.py`: reuses `_backbone_forward` (docstring core.py:387-391) and adds `MarkovHead` (low-rank `W1[prev token] @ W2` logit bias; head types vanilla/gated/rnn) and a `ConfidenceHead`; defaults `sample_from_anchor=True` (block_size speculative tokens), block 8; serves via vLLM method `dspark`.
- Both rely on the `_make_decoder_layer` factory and `_backbone_forward` - the same seams a LoRA-aware DFlash would use.
