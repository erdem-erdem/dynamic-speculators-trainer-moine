# 01 - Generator core: model, LoRA injection, loss, training loop

Repo root: `C:\Users\108573\Desktop\WSL\dynamic-speculators-trainer` (paths below relative to `src/speculators/` unless noted).
Read at HEAD `b17276c`. Everything below is from static reading; nothing was executed (local env: torch 2.5.1+cu121, transformers 5.5.4, no peft, no triton). Items I could not confirm from code are marked UNVERIFIED.

Two training code paths exist in `generator/engine.py`:
- LEGACY "bank regression" path (`_run_job`, jobs `heatmap`/`pretraining` with snapshots): generator is trained with L1 regression onto pre-collected static-LoRA factors (`factor_loss`, model.py:243-251; engine.py:563-623).
- CURRENT "direct / behavioural" path (`generator/campaign_train.py`, objective `"direct"`): generator is trained end-to-end through the frozen drafter's TTT loss. `train_update` raises for any other objective (campaign_train.py:199-200). All production YAMLs (`examples/eagle3-lora/attention_100k_2k_v1/train*.yaml`, `structure_ablations_v1/*`) use this path. Everything below describes the direct path unless stated.

Entry chain for production: `scripts/train_attention_100k.py` -> `generator/structure_study.py:train_structure` / `epoch_training.train_full_data` -> `campaign.run_one` (campaign.py:~118-290) -> `engine.Runtime` + `campaign_train.run_training`.

---------------------------------------------------------------------------------------------------

## 1. Generator architecture (`generator/model.py`, `generator/config.py:275-281`)

`LoRAGenerator(input_dim, shapes, rank, architecture, max_examples=8, feedback_dim=0)` (model.py:28-151). Defaults of `Architecture` (config.py:275-281): `encoder_hidden=512, example_dim=128, condition_dim=64, descriptor_dim=32, decoder_width=512, residual_blocks=2`. All production YAMLs restate exactly these values. (Hypothesis "512/128/64/32/512/2": Verified.)

Modules, with the EAGLE-3 / Qwen3-8B numbers (input_dim = 12288 = 3 aux layers x 4096):
```
encoder   = Linear(input_dim+1 -> 512) GELU Linear(512 -> 128) LayerNorm(128)       model.py:44-49
condition = Linear(2*128+1 -> 128)  GELU Linear(128 -> 64)                          model.py:50-54
module_embedding = Embedding(n_modules, 32)                                          model.py:55
layer_embedding  = Embedding(1, 32)          # single learned vector, "layer 0"     model.py:56
decoder   = Linear(64+32+32 -> 512) GELU  + 2x ResidualMLP(512)                      model.py:63-67
ResidualMLP: x + Linear(GELU(Linear(LayerNorm(x))))                                  model.py:14-25
heads[name]["A"] = Linear(512 -> rank*in_features)                                   model.py:70-75
heads[name]["B"] = Linear(512 -> out_features*rank)
optional feedback_encoder = Linear(feedback_dim -> 64, bias=False), zero-init        model.py:57-62 (only feedback studies)
```
- The `+1` on the encoder input is `log1p(prompt_length)` appended to each example's hidden-state vector (model.py:94-99).
- The decoder and encoder are shared across all target modules; only the output heads are per-module (`self.shapes` dict name -> (in, out)).
- Descriptor is NOT per-layer: it is `cat[condition(64), module_embedding[idx](32), layer_embedding[0](32)]` = 128-d (model.py:138-145). `module_embedding` is a learned embedding indexed by position of the module in `shapes` (dict order = `install_adapters` output, sorted by name). `layer_embedding` has exactly one row -> effectively a learned bias; the generator has no concept of layer index.
- The decoder runs once per module (python loop, model.py:135-150), i.e. n_modules decoder passes, each batched over `leading` dims.

How examples are encoded and pooled (model.py:88-124):
1. `encode_examples(summaries[K, D], lengths[K])` -> `[K,128]`; K must be in `1..max_examples` (=8), else ValueError (model.py:90-93). `lengths` must be >0.
2. `pool_examples`: `cat[mean(0) (128), std(0, correction=0) (128), log1p(K) (1)]` = 257-d (model.py:102-110). Permutation invariant (tested: test_experiments.py:133-147). K=1 gives std=0.
3. `condition` MLP -> 64-d.
4. `forward_cohorts(contexts)` (model.py:112-124) pools each cohort separately then stacks and runs the condition/decoder/heads once with a leading cohort dim; no pooling across cohorts (tested serial==batched in tests/unit/generator/test_generator_batching.py:35-63).

`condition: last` (config.py:262, CONDITIONS = last/last_mean/projected). In the CURRENT campaign path only `"last"` is legal (campaign_config.py:128, campaign_data.py:258-259, online_activations.py:361-362). Meaning: per example, the raw (pre-fc, pre-norm) target-model hidden states of the LAST PROMPT TOKEN, 3 auxiliary layers concatenated, float32:
`states[prompt_length-1, :-1].flatten().float()` (online_activations.py:~148; data.py:36-67 for the legacy `prompt_summary` which also builds `last_mean` = [last, mean over prompt] and `projected` = fc-projected variants). `last_mean`/`projected` only exist in the legacy `Corpus.prepare` path.

A/B factor production (model.py:146-150): `decoded = decoder(descriptor)`; `A = heads.A(decoded).view(*leading, rank, in)`, `B = heads.B(decoded).view(*leading, out, rank)`. Full-factor heads: each output element has its own weight row, i.e. param count scales as `(decoder_width+1) * rank * (in + out)` per module.

Initialisation (model.py:76-81): for every head `weight=0, bias=0`; then the A-head BIAS is overwritten with `kaiming_uniform_(bias.view(rank, in), a=sqrt(5))` (= PEFT default LoRA-A init, U(+-1/sqrt(in))). So at init: `A` = constant kaiming matrix independent of input; `B == 0` exactly -> delta W = 0 exactly (zero-LoRA equivalence at step 0). A-head weights receive zero gradient until B moves (standard LoRA dynamics). Because A lives in a bias, AdamW weight decay (0.01) also decays the A "prior".

Scale/coefficient knobs: none in the generator output (no output scaling, no tanh/norm). The only scale is the fixed `AdapterLinear.scale = alpha / rank` (model.py:160) = 64/32 = 2.0. A serving-time coefficient override exists elsewhere (`examples/eagle3-lora/coefficient_sweep_4000/README.md`: effective coefficient replaces alpha/r at inference; 0.5/4/8 tested, 8 collapses) - not a training knob. "scale" in `campaign.jobs()` (campaign.py:~55-80) and `lora_baseline.py` means DATA scale (1/2/4 subsets per domain), unrelated to LoRA scale.

Parameter counts (my arithmetic, rank 32, 3x4096 input): heads dominate.
- o_proj+v_proj: 293.3M total (285.8M in heads).
- q,k,v,o: 646.3M total (638.8M in heads).
`generator_dtype` is "float32" by default (config.py:396) but all production YAMLs set `bfloat16` (train.yaml:9) -> params, grads and AdamW states all bf16 with NO fp32 master weights (explicitly asserted in test_generator_batching.py:66-87). Risk: lr 1e-4 updates near bf16 resolution for weights ~1e-2..1e-1 (UNVERIFIED impact).

---------------------------------------------------------------------------------------------------

## 2. Conditioning input

- Source: prompt-only target hidden states at layers `bank.hidden_states.layer_ids = [2, 18, 33, 36]` (default config.py bank/config.py:60; recipes `training_recipe*.yaml`). The first three are the EAGLE-3 aux layers (concatenated, 3x4096 = 12288 for Qwen3-8B); the last (36) is the verifier-final layer, dropped by `[:, :-1]` for conditioning. Width of generator input is read from `drafter.fc.weight.shape[1]` (campaign_data.py:221-224; online_activations.py:~348-352) so it is not hard-coded 12288; for the DeepSeek TTT7 drafter recipe layer_ids are `[2,10,18,26,34,36]` -> 20480 (deepseek_ttt7_generator_v1/training_recipe.yaml).
- Position: the last token of the prompt, `prompt_length-1`, where `prompt_length` comes from the prepared Arrow row. Prompt-only: the online path asks vLLM for hidden states of `input_ids[:prompt_length]` only (online_activations.py:111-117) -> no response leakage.
- No noise on the conditioning vector. Noise (`bank.training.noise_std=0.05`) is added in `Corpus.microbatches` only when `training=True`, only to the DRAFTER INPUT `hidden_states` (the 3-layer concat), as `x + 2*(rand_like(x)-0.5)*noise_std` (data.py:315-319) -> uniform in [-0.05, 0.05], actual std = 0.05/sqrt(3) = 0.029 (naming is misleading). Not applied to `verifier_last_hidden_states` (targets) or to validation. (Hypothesis noise 0.05: Verified, with the uniform caveat.)
- Cohort/example counts: `Context.min_examples=1`, `max_examples=8` (hard cap `le=8`, config.py:300-303); campaign `evaluation_sizes=[4]` (campaign_config.py:84, forced at :128).
- Per-episode sampling, campaign path (`campaign_data.py:170-203`, `EpisodeSchedule.update`): for each of `episodes_per_update` episodes, `rng=Random(seed + step*100003 + episode*1009)`; pick domain uniformly, `size=randint(min,max)` (uniform 1..8), subset uniformly within the domain, `cohort=rng.sample(train_indices, size)`. Epoch variant (`epoch_training.EpochSchedule`, used when job has `epochs`): per-epoch shuffle of every row of each subset, chopped into cohorts of random size 1-8 (or fixed `training_cohort_size`), episodes shuffled and batched `episodes_per_update` at a time.
- Optional `training_conditioning_examples` (campaign_config.py:63, campaign_train.py:78-82): condition on a random subset of the cohort while the loss still covers the whole cohort. Default None -> the conditioning set == the supervised cohort (the same K prompts whose responses form the loss). Validation: cohorts of exactly 4 from validation indices, seed `cfg.seed+9000` (campaign_train.py:93), conditioning on the same 4.
- Semantics: one LoRA per cohort; loss on all response tokens of all examples in that cohort.

---------------------------------------------------------------------------------------------------

## 3. LoRA injection mechanism

Mechanism: module replacement + `torch.func.functional_call` parameter substitution; NOT hooks, NOT merged weights during training.
- `install_adapters(model, rank, alpha, dropout, target_modules=None)` (model.py:182-211): selects modules by LEAF NAME (`path.rsplit(".",1)[-1]`), default `("o_proj","v_proj")` (model.py:184-186). Requires EXACTLY ONE `nn.Linear` match per name in the whole model, else `ValueError("Expected exactly one linear layer per selected projection")` (model.py:193-197) - the docstring says "single-layer drafter". Replaces each with `AdapterLinear` and returns `{leaf_name: dotted_path}` sorted (e.g. `{"o_proj": "layers.0.self_attn.o_proj"}`).
- `AdapterLinear` (model.py:154-179): `y = base(x) + scale * linear(linear(dropout(x), a), b)`, `scale = alpha/rank`, `dropout = nn.Dropout(p)` applied on the adapter-branch INPUT only (not on base path), `a: [rank, in] fp32 Parameter`, `b: [out, rank] fp32 Parameter` (kaiming/zeros init), `base.requires_grad_(False)`. No bias on the adapter. Same math as PEFT (alpha/r, dropout on x), so PEFT static LoRAs and generated LoRAs are interchangeable.
- Generated factors enter via `Runtime.forward` (engine.py:176-186): `functional_call(self.drafter, factor_parameters(self.paths, factors), (), kwargs)`, where `factor_parameters` maps `{f"{path}.a": A, f"{path}.b": B}` (model.py:214-219). With `factors=None` the module's own a/b are used (fresh_lora arm / zero-LoRA baseline).
- Shapes (Qwen3-8B EAGLE-3 first layer, `Eagle3FirstLayerMixin`, models/eagle3/model_definitions.py:29-52): q_proj in 8192 -> out 4096; k_proj 8192 -> 1024; v_proj 8192 -> 1024; o_proj 4096 -> 4096; gate/up 4096 -> intermediate, down intermediate -> 4096. `shapes[name] = (in, out)` read from `adapter.base.in_features/out_features` (engine.py:~101-108). v_proj: `shapes=(8192,1024)`, `A:[32,8192]`, `B:[1024,32]`, Linear weight `[1024,8192]`. (Derived from Qwen3-8B config + code; I did not load the checkpoint -> UNVERIFIED numerically.) q/k/v take the 2x-width `cat([embeds, hidden])` input.
- Supported modules: `ADAPTER_MODULES = up_proj, down_proj, gate_proj, q_proj, k_proj, v_proj, o_proj` (config.py:264-272), validated in `ExperimentConfig.valid_target_modules` (config.py:403-411). No `fc`, no `lm_head`, no norms, no embeddings. (The separate static PEFT trainer's `LoraArgs.lora_target_modules` default DOES include `fc` and `lora_alpha` default is 16 - train/config/schema.py `LoraArgs`; that is the bank trainer, not the generator.)
- Per-sample vs per-batch: one set of factors per EPISODE (cohort). The 4 episodes of an update each get their own factors from the generator and are run as separate drafter forwards (campaign_train.py:225-286). Within an episode, all microbatches share that episode's factors; inside one packed microbatch (several examples concatenated into a `[1,T]` sequence) all tokens use the same LoRA. There is no per-sequence batched-LoRA in a single drafter call.
- Module target set config: `ExperimentConfig.target_modules` (YAML `target_modules:`), `None` -> o_proj+v_proj. Production: attention = `[q_proj,k_proj,v_proj,o_proj]` (attention_100k_2k_v1/train.yaml, deepseek_ttt7.../train.yaml); ablations `modules/all.yaml` = all 7; `modules/mlp.yaml` = gate/up/down; `common.yaml` default o_proj+v_proj. (Hypotheses "attention = q/k/v/o", "modules-all adds gate/up/down": Verified. "default targets o_proj+v_proj": Verified for the code default and `bank/config.py:69`, but the latest 100k/10k production runs use q/k/v/o.) `BankConfig.compatible` forces `lora.target_modules == {o_proj, v_proj}` for the static-LoRA bank v1 only (bank/config.py:214).
- Compute precision: drafter loaded in `dtype` (bf16), adapters a/b fp32, the whole forward under `torch.autocast(dtype)` (engine.py:176-186). Generated factors arrive in `generator_dtype`.
- Drafter forward is un-compiled on purpose: `self.drafter.forward = torch.compiler.disable(type(drafter).forward.__get__(drafter), recursive=False)` (engine.py:~123-130) so functional_call substitution is outside the graph; flex-attention kernel is compiled standalone (`models/attention.py` fork diff: `_standalone_flex_attention` = `torch.compile(flex_attention, fullgraph=True, dynamic=True)`).
- Gradient path: per episode, `leaves = detached_factors(factors)` (model.py:222-227); each microbatch does `(loss*weight/len(episodes)).backward()` into the leaves (graph never retained); then `backward_factors(factors, leaves)` pushes accumulated `leaf.grad` into the generator once via `torch.autograd.backward(values, grads)` (model.py:230-240). In batched mode (`generator_batching: true`, used in prod) a single `forward_cohorts` result is split with `value[episode]` and backpropagated once at the end (campaign_train.py:225-264, 287-289). Test: `test_factor_gradient_accumulation_matches_combined_loss` (test_experiments.py:185).

### EAGLE-3-specific assumptions found (exhaustive for my files)
1. `install_adapters` single-match-per-leaf-name rule = single decoder layer (model.py:193-197); error for DFlash's 5 layers.
2. `layer_embedding = Embedding(1, ...)` (model.py:56) - no layer index.
3. `Runtime.__init__` (engine.py:56-132): `drafter.target_layer_ids == bank.hidden_states.layer_ids[:-1]` check (engine.py:96); `SpeculatorModel.config_class.from_pretrained` + `maybe_convert_external_checkpoint(drafter, verifier=target)` (engine.py:82-89; converts external RedHat/DeepSeek EAGLE-3 checkpoints); `config.speculators_config.verifier.name_or_path = target path`.
4. `call_kwargs = {ttt_steps, ttt_step_loss_decay, loss_config}` (engine.py:115-121) = Eagle3 forward kwargs; `resolve_loss_config(spec)` called with default implementation `"fused"` (Triton, CUDA only) - no knob for eager (tests monkeypatch it).
5. `evaluate` / `evaluate_subset` require metric key `full_acc_0` (engine.py:221-226; campaign_train.py:130) - Eagle3 `compute_metrics` emits `full_acc_{ttt_step}`; DFlash emits `full_acc`/`eal`.
6. `Corpus.microbatches` (data.py:271-321): `shift_batch` from `models/eagle3/data.py`; `hidden_states = states[:, :-1].flatten(1)` (3 aux layers concatenated), `verifier_last_hidden_states = states[:, -1]`; row layout `[seq, n_layers, hidden]`; padding of packed length to multiple of `FLEX_ATTENTION_BLOCK_SIZE=128` because `extend_mask_for_draft_tokens` concatenates draft KV segments each TTT step (comment data.py:305-307); `token_budget % 128 == 0` enforced (data.py:261-269); `CollateFn(total, width, dtype=dtype)` leaves `num_target_layers=3` default (only used for the empty-sample fallback, train/data.py CollateFn).
7. Batch keys expected by `Eagle3DraftModel.forward`: `hidden_states, input_ids, document_ids, loss_mask, position_ids, verifier_last_hidden_states` (models/eagle3/core.py:193-206), position ids starting at 1 after shift.
8. Loss mask accounting `sum(loss_mask[1:])` (campaign_train.py:265; engine.py:225) matches the Eagle3 shift.
9. Widths from `drafter.fc.weight.shape[1]` (generic for any drafter with `fc`).
10. `generator/streaming.py` `DraftStep` re-implements the EAGLE-3 forward by hand (`input_norm`, `fc_norm`, `fc`, `embed_tokens`, `cat([embeds, states])`, `rotary_emb(hidden, positions)`, `layers` with `speculators_streaming_sdpa` kwargs, `norm_output`, `lm_head`, `d2t`) - inference/replay only, 100% EAGLE-3 (autoregressive single-token proposals with explicit KV cache); not used in training.
11. Config defaults / naming: `models.drafter = RedHatAI/Qwen3-8B-speculator.eagle3`, `CACHED_LAYER_COUNT=4` (3 aux + final) for producer banks (bank/config.py:10, 220-224), W&B project names `eagle3-generator-*` (config.py:339-352).
12. `ttt_steps` in `Training` defaults to 3 with `loss_fn: Literal["kl_div","ce"]` (bank/config.py:82-84); the generator overrides via `Optimization.loss_fn` (config.py:285).

---------------------------------------------------------------------------------------------------

## 4. Loss, schedule, freezing

Drafter forward (Eagle3, models/eagle3/core.py:193-365): `hidden_states = fc(norms(hidden_states))`; verifier targets `= verifier_lm_head(verifier_norm(verifier_last_hidden_states))` under `no_grad` (draft-vocab logits); for `ttt_step in range(ttt_steps)`: embed current input ids, `cat([embeds, hidden])`, run layers (KV cache grows by one segment per step, mask extended with a diagonal), `logits = lm_head(norm(h))`, per-step loss; next-step input ids = TEACHER-FORCED ground-truth tokens shifted by `1+step` (not the drafter's argmax, core.py:326-339), hidden state fed forward is the drafter's own output.

Per-step loss (models/eagle3/metrics.py:51-119): `align_for_step` slices logits `[:, :-step]`, targets/mask `[:, step:]`; `compound_loss` -> `loss_function` (losses/utils.py:304-339): `elementwise = loss_fn(logits, targets) * loss_mask`, then `* decay`, `decay = exp_loss_decay(pos_idx=step, gamma=ttt_step_loss_decay) = gamma**step`, reduced as `sum / (mask.sum + 1e-5)` (mean over masked tokens per step). Total loss = SUM over steps (not average) (core.py:319). `ttt_step_loss_decay` = 1.0 in all recipes -> equal weights; `ttt_steps` = 3 in `training_recipe.yaml` / `generator_training_recipe.yaml`, 7 in `training_recipe_ttt7.yaml` (used by `train_10k.yaml`; README says "10k variant uses seven TTT steps").

`lk_hybrid` (losses/eager.py:174-213; fused path identical, losses/fused.py:392-399), per position, p=target softmax over draft vocab, q=draft softmax:
```
alpha  = sum_v min(p_v, q_v)              (overlap = acceptance rate)
TV     = 1 - alpha
KL     = KL(p || q)
lambda = exp(-eta * stopgrad(alpha)),  eta = 3.0 (default, not configurable via resolve_loss_config)
loss   = lambda * KL + (1 - lambda) * TV
```
vs `ce` (eager.py:92-115): cross-entropy of draft logits against `argmax(target logits)` (hard labels). `kl_div` is forward KL(p||q). Campaign requires `lk_hybrid` (campaign_config.py:130); `Optimization.loss_fn` default None inherits `bank.training.loss_fn` (config.py:285; bank default `kl_div`). CampaignConfig default optimisation = `loss_fn=lk_hybrid, pretraining_updates=10000, warmup_updates=100, checkpoint_interval=500` (campaign_config.py:85-92). Metric `full_acc_0` = step-0 argmax agreement (validation score).

Aggregation in the generator trainer (campaign_train.py:189-322): within an episode, microbatch loss is weighted by `batch["loss_mask"].sum()/total_response_tokens` (campaign_train.py:279; approximation: per-step denominators differ for steps>0), then `/len(episodes)` -> equal weight per episode regardless of token count; episodes accumulate into one optimizer step. Non-finite loss raises (campaign_train.py:277-278).

Regulariser on delta norm: NONE (no penalty on `||B@A||`, A, B or generator outputs; only AdamW weight decay on generator params). Legacy regression objective `factor_loss` = mean over factors of L1 (model.py:243-251), no gauge alignment.

Optimizer / schedule (engine.py:34-53, campaign_train.py:403-405): `torch.optim.AdamW(generator.parameters(), lr=1e-4, weight_decay=0.01)` (Optimization defaults config.py:286-288; same in YAMLs) - weight decay hits every param incl. biases/LayerNorm/the A-bias. LR: linear warmup `lr*min(step/warmup,1)` with `step=update+1`, then CONSTANT (no decay) (engine.py:49-53, campaign_train.py:206). `warmup_updates`: Optimization default 0 but 100 in CampaignConfig and every production YAML. Grad clip: global-norm 0.85 via `clip_grad_norm_(..., error_if_nonfinite=True)`, logs pre-clip norm (engine.py:34-46; test_experiments.py:112). All of lr 1e-4 / wd 0.01 / clip 0.85 / warmup 100: Verified.
- "episodes per update" = `Optimization.episodes_per_update=4` = number of cohorts (each with its own generated LoRA) accumulated per optimizer step (config.py:293; EpisodeSchedule). Effective examples per update: 4 x U{1..8} (mean 18).
- "token budget" = `Optimization.token_budget=8192` = max packed tokens per DRAFTER MICROBATCH (greedy packing of cohort rows in order; each row `seq_len-1` must be <= budget; data.py:271-286). It is not a per-update cap. Must be multiple of 128 for flex, and `>= hidden_states.sequence_length (8192)` per `BankConfig.compatible` (bank/config.py:236).
- Updates: `pretraining_updates` 2000 (train.yaml) / 10000 (train_10k.yaml); validation every `pretraining_validation_interval=250`; checkpoint every `checkpoint_interval` (500/1000).
- Frozen drafter: `drafter.to(device).requires_grad_(False).eval()` at load (engine.py:98); `install_adapters` calls `model.requires_grad_(False)` BEFORE creating AdapterLinear (new a/b params default requires_grad=True), so `run_training` re-freezes with `runtime.drafter.requires_grad_(False)` when a generator exists (campaign_train.py:400); optimizer only receives `generator.parameters()`; a/b are replaced by functional_call anyway. No explicit assert that no drafter param has requires_grad; enforcement is by call order. `train_update` calls `drafter.train()` (dropout 0.05 active in AdapterLinear), `evaluate` uses `.eval()` (campaign_train.py:89, 202). Verifier head/norm/embeddings are loaded frozen from the target (`DraftVocabMixin`, `verifier_lm_head.requires_grad=False`).
- Determinism: `torch.manual_seed(seed + update*100003 + episode*1009)` per episode (campaign_train.py:233) fixes noise and dropout.

---------------------------------------------------------------------------------------------------

## 5. Static LoRA baseline

Two different "static LoRA" things:
(a) `fresh_lora` arm (the matched baseline), `generator/lora_baseline.py` + `campaign.jobs` + `run_training`: `matched_lora_job(cfg, scale)` copies the generator job at the same `scale` (same `subsets`, `validation_subsets`, `updates`, `training_examples`, `objective`) and sets `arm="fresh_lora"`, `label=scale-{s}-lora` (lora_baseline.py:8-17; test_lora_baseline.py checks equality). In `run_training`: `generator=None` -> `runtime.initialize_direct()` (engine.py:160-174: kaiming A, zero B, only a/b `requires_grad`) and the optimiser is the SAME AdamW(lr 1e-4, wd 0.01) on the drafter's a/b (campaign_train.py:380-405). Everything else is shared: same `EpisodeSchedule` (episodes still sampled as 4 cohorts/update, but all share ONE LoRA), same `lk_hybrid`, TTT, noise, dropout 0.05, token budget, clip 0.85, warmup 100, validation on 4-example cohorts with `generator=None`. Same rank/alpha/target modules (`runtime.paths` from the same cfg). It must use `activation_source == "vllm"` (lora_baseline.py:21-22).
   Mismatches to be aware of: (i) `structure_study.training_job` sets `adapter_dtype = cfg.lora_dtype` default `"bfloat16"` for `arm: static_lora` and `run_training` casts a/b to bf16 (structure_study.py:149,215-216; campaign_train.py:383-397), whereas `train_matched_lora` has no `adapter_dtype` -> fp32 adapters; (ii) the static LoRA sees no conditioning at all (one adapter for the whole domain mix); (iii) weight decay applies to a/b.
(b) PEFT "bank" LoRAs (`train/lora.py`, `train/bank.py`, `train/cli.py` `--lora-r`): upstream-trainer based static LoRA (PEFT `get_peft_model`, `LoraConfig(r, lora_alpha, lora_dropout, target_modules, bias="none")`, train/lora.py:49-91) trained with `BankTrainer`: 1 warmup epoch lr 1e-4 then 4 collection epochs lr 1e-6, 10 snapshots/epoch (train/bank.py:25-55, 168-453; `BankSchedule` defaults), AdamW, grad clip hard-coded 1.0 (train/trainer.py fork diff), 3 seeds [42,43,44]. These snapshots are the regression targets of the legacy path (`FactorCache.get` expects keys `*.{name}.lora_{A,B}.weight`, data.py:416-464). PEFT LoRA init/scaling matches `AdapterLinear` (A kaiming, B zero, alpha/r, dropout on x). PEFT dep is optional (`train/lora.py:19-29`). Static-trainer schema defaults `lora_alpha=16`, targets include `fc` (schema.py LoraArgs); the bank recipes override to r32/a64/p0.05/o_proj+v_proj (Lora in bank/config.py:65-69 - Verified).

---------------------------------------------------------------------------------------------------

## 6. Existing correctness checks / tests (tests/unit/generator)

Exist:
- Zero-LoRA equivalence at init: `test_functional_adapters_preserve_base_and_backpropagate` (test_experiments.py:150-183): output with generated (zero-B) factors == original base module output; generator grads non-zero; all drafter params `requires_grad False` and `.grad is None` after training steps. Also step-0 validation with the generator at init = base-drafter score in the real loop (`validate()` at progress 0).
- Frozen weights: `test_drafter_loss_updates_generator_and_preserves_frozen_weights` (test_experiments.py:932-943) - full drafter `state_dict` unchanged after `train_epoch` (legacy engine path, tiny CPU Eagle3).
- Gradient accumulation == combined loss, including a compiled variant: test_experiments.py:185-225; serial vs batched generator outputs/grads/cohort independence: test_generator_batching.py:35-63.
- Fused-flex vs dense-eager drafter with generated LoRA, forward loss and factor gradients (CUDA only): test_experiments.py:877-930. (Both arms use the same `install_adapters`+`functional_call`, so it checks attention kernels, not the LoRA math itself.)
- Set-encoder permutation invariance; K bounds (test_experiments.py:133-147).
- bf16 drafter keeps fp32 adapter updates (test_experiments.py:246-257); bf16 generator has no fp32 master weights (test_generator_batching.py:66-87).
- Gradient clipping semantics incl. non-finite error (test_experiments.py:112-130).
- Prompt-only summaries ignore response tokens and use actual frozen fc/norm (test_experiments.py:228-244).
- Backward graph never retained across microbatches (test_experiments.py:946-1012); resume equality (test_campaign.py:427-452); merged-weights vs explicit LoRA for the SERVING projection merge (test_projection_merging.py:63-112); matched LoRA data/budget (test_lora_baseline.py); streaming last-only prefill == full model logits and adapter gradients (test_streaming_execution.py:19).
- Tiny CPU harness: `tiny_drafter()` + `synthetic_experiment` fixture build a hidden-8, 1-layer Eagle3 + synthetic hidden-state files and monkeypatch `Runtime` (eager loss, rank 2) - good template for a DFlash smoke test.
Missing / not found:
- No explicit assertion at runtime that `requires_grad` is False for all drafter params in the production `run_training` (only by construction).
- No independent check that `functional_call` factor substitution equals the algebraic merge `W + (alpha/r) B A` in the TRAINING forward (the merge test is in serving). No finite-difference gradcheck of generator -> loss.
- No test of non-zero-delta effect on loss; no test for multi-layer drafters (install_adapters multi-match error path); no test of dropout RNG determinism; no check that generated factor dtype/shape matches `shapes` beyond view().
- No test that A/B head init gives exactly `B==0` for every module in the real-dim generator (small dims only).

---------------------------------------------------------------------------------------------------

## 7. Porting to DFlash (5 Qwen3 decoder layers, fc + hidden_norm KV injection, block-parallel)

DFlash model facts (models/dflash/core.py, model_definitions.py): `layers = ModuleList(Qwen3DFlashDecoderLayer x num_hidden_layers)` each with `self_attn.{q,k,v,o}_proj` (plain hidden-size inputs, no 2x concat) and `mlp.{gate,up,down}_proj`; `fc: Linear(len(target_layer_ids)*hidden -> hidden)` then `hidden_norm`; `k_proj`/`v_proj` are applied to BOTH `target_hidden` (fc_output, ctx) and the noise/mask-token hidden states, so a LoRA on k/v hits both (model_definitions.py:111-118). Forward signature `forward(hidden_states, input_ids, loss_mask, verifier_last_hidden_states, document_ids, position_ids, loss_config, gamma, max_anchors, per_position_loss_weight, dpace_alpha)` returns `(None, loss, metrics)` (core.py:489-524); trainer kwargs from `DFlashDraftModel.get_trainer_kwargs` (core.py:281-307); no TTT loop; random anchor selection with `torch.randperm` inside forward (utils.select_anchors) -> loss is stochastic per call; DFlash needs `mask_token_id`; no `shift_batch` (train/cli.py:788 only maps eagle3/peagle/mtp); metric keys `full_acc`, `eal`, `position_{i}_acc` (dflash/metrics.py:90-112).

Abstraction seams and minimal-change path (ordered):
1. `model.install_adapters` (model.py:182-211): relax uniqueness; select by qualified path (regex/glob over `named_modules()`, e.g. `layers.*.self_attn.q_proj`) and return `{unique_key: path}`. `LoRAGenerator.shapes` keys become `ModuleDict` keys -> must not contain "." (use `layers_0__q_proj` style names). `factor_parameters`/`detached_factors`/`backward_factors` are already key-agnostic.
2. `LoRAGenerator.layer_embedding` -> `Embedding(num_layers, descriptor_dim)` indexed per (layer, module); `module_embedding` currently indexed by dict position (model.py:55,141) - change to module-type index + layer index. Descriptor concat stays 2x32.
3. Head size blow-up: heads are `(512+1)*rank*(in+out)` per module. If DFlash hidden is 4096 (UNVERIFIED for the target DFlash checkpoint), attention-only (q/k/v/o) over 5 layers at rank 32 is ~2.2B head params (5 x 437M, my arithmetic) -> ~17 GB for bf16 weights+grads+AdamW, ~35 GB fp32; all-7-modules larger still. Plan for a smaller rank, smaller `decoder_width`, shared heads across layers with a layer-conditioned decoder, or a low-rank head factorisation. o_proj+v_proj only on 5 layers is ~1.46B (5 x 293M).
4. `Runtime.call_kwargs` (engine.py:115-121): replace the hard-coded TTT kwargs by `model_class.get_trainer_kwargs(...)` output (gamma, max_anchors, per_position_loss_weight, dpace_alpha, loss_config); `Optimization`/`Training` configs need these fields (`ttt_steps` is meaningless). `resolve_loss_config(..., "fused")` default stays.
5. Metrics: `evaluate`/`evaluate_subset` score key `full_acc_0` -> `full_acc` (or accepted length `eal`) (engine.py:221-226; campaign_train.py:130, 146-155).
6. `Corpus.microbatches` (data.py:271-321): drop `shift_batch` (DFlash targets use `anchored_block_indices-1` shifting internally), keep `[1,T]` packing + `document_ids`; padding to 128 is only needed for EAGLE's draft-KV mask extension - DFlash builds its anchor block mask over `Q=max_anchors*block_size`, `KV=T+Q` (core.py:329-344), so check block alignment of `T` for flex; budget/memory differs (512 anchors x block_size queries). Noise semantics: confirm DFlash's training noise (not examined).
7. `Runtime.__init__` (engine.py:75-110): drafter loading via `SpeculatorModel.from_pretrained` is generic (registry), `maybe_convert_external_checkpoint` is EAGLE-3-specific (skip/guard for native DFlash), `target_layer_ids` property exists on DFlash (config.aux_hidden_state_layer_ids), `bank.hidden_states.layer_ids[:-1]` check stays valid; `fc` exists so `summary_width = fc.weight.shape[1]` (5 x 4096 = 20480 if 5 aux layers) works; `config.transformer_layer_config._attn_implementation` assignment generic.
8. `Runtime.forward` -> `drafter.forward` wrapping with `torch.compiler.disable` (engine.py:123-130): DFlash forward also uses `conditional_torch_compile` and a separately `torch.compile`d `create_block_mask` (core.py:35) - re-verify the unwrap trick via `_torchdynamo_orig_callable` and that functional_call works with the compiled mask builder.
9. Conditioning vector: "last token of the prompt, aux layers concat" is drafter-agnostic given a target-hidden extraction with the DFlash layer ids; nothing in model.py depends on 3 layers.
10. Bank/config coupling: `BankConfig.Lora.target_modules` forced {o_proj,v_proj} only for bank producers (bank/config.py:214); consumer recipes + `ExperimentConfig.target_modules` carry the real set. `ADAPTER_MODULES` list is name-based and already covers DFlash module names.
11. Dropout in `AdapterLinear` applied separately at each call; for DFlash k/v called twice per layer -> independent dropout masks for ctx vs noise branches (fine, but differs from merged inference).
12. Zero-LoRA/frozen/gradient tests in section 6 are reusable on a tiny DFlash (tiny Qwen3 config, hidden 8, 2 layers) via the `synthetic_experiment` pattern.
Minimal path: (1)+(2)+(4)+(5)+(6) with `shapes` keyed by sanitized path; reuse `AdapterLinear`, `functional_call`, `detached_factors/backward_factors`, optimizer/clip/LR code, `EpisodeSchedule`, campaign loop unchanged. Streaming/serving modules (`streaming.py`, `generator/serving/*`) are EAGLE-3-only and out of scope for a training port.

---------------------------------------------------------------------------------------------------

## 8. Memory / compute profile

- Resident on the training GPU (direct path): the frozen drafter only (EAGLE-3: 1 decoder layer + fc 12288x4096 + `embed_tokens` 151936x4096 + draft lm_head/`verifier_lm_head` 32000x4096 each; ~2-2.5 GB bf16, my estimate, UNVERIFIED) + the generator (293M/646M params; bf16 weights+grads+2 Adam states ~ 2.3 / 5.2 GB for o+v / qkvo; fp32 2x that) + activations of one microbatch (<= 8192 tokens, TTT steps retained until backward: logits `[1,8192,32000]` per step; fused Triton loss avoids [T,V] float intermediates). README warns TTT7 raises compute/memory (attention_100k_2k_v1/README.md); no measured peak in repo that I read (peak counters are logged: `train/gpu_peak_allocated_bytes`, campaign_train.py:625-637).
- NOT resident: the target Qwen3-8B. Target hidden states come from a vLLM server (separate process; `online.endpoint`, `shared_dir` files in /tmp, `generate_hidden_states` over HTTP; online_activations.py:111-175) - launcher puts vLLM on GPU 0 and the trainer on GPU 1 (README two-GPU section; `--gpu-memory-utilization 0.70` suggestion for 96 GB cards). Only embed/lm_head/norm of the verifier are loaded in the trainer (via `load_model_layers`).
- Cached activations live in CPU RAM / disk, not GPU: `online.ram_cache_gib=120` (train.yaml), `activation_prefetch_workers=2`, lookahead 2 updates; prompt summaries cached on disk + RAM (online_activations.py:332-354).
- Compute per update: 4 episodes x (up to 8 examples, <= 8192-token microbatches) drafter forward+backward with TTT (3 or 7) + one generator pass (batched) + one backward. Generator forward is cheap relative to drafter; its cost is the big head matmuls (decoder_width x rank*in).
- 8 GB RTX 4060 smoke tests: possible: (a) the existing CPU unit tests (tiny hidden-8 Eagle3, eager attention, eager loss); (b) a tiny-config GPU smoke with the real LoRAGenerator but small `shapes`. NOT possible locally: Qwen3-8B target extraction via vLLM (16 GB bf16), and `fused` Triton losses on native Windows (torch 2.5.1, no triton installed; use WSL or an eager loss override since `Runtime` hard-codes "fused"). A real RedHat EAGLE-3 drafter + o/v generator at reduced token_budget (multiple of 128, and `BankConfig` requires `token_budget >= sequence_length`, so shrink both) plausibly fits ~8 GB but is UNVERIFIED; needs pre-extracted activations or a hub source.

---------------------------------------------------------------------------------------------------

## 9. Surprises, risks, hypothesis scorecard

Hypothesis scorecard (secondary guide):
| Claim | Verdict | Evidence |
|---|---|---|
| rank 32, alpha 64, dropout 0.05 | Verified | bank/config.py:65-69; all recipes |
| default targets o_proj+v_proj | Verified (code default); production attention runs override to q/k/v/o | model.py:184-186; train.yaml `target_modules` |
| v_proj shape [8192,1024] | Verified as (in,out) tuple; Linear weight is [1024,8192]; A [32,8192], B [1024,32] | engine.py:~101-108; eagle3 model_definitions.py:48-52 (numbers derived, UNVERIFIED on real ckpt) |
| "attention" = q/k/v/o | Verified | structure_ablations_v1/modules/attention.yaml |
| "modules-all" adds gate/up/down | Verified (all 7) | modules/all.yaml |
| no fc / lm_head LoRA in generator | Verified | config.py:264-272 (static PEFT trainer default does include `fc`, unrelated) |
| lr 1e-4, wd 0.01, clip 0.85, warmup 100 | Verified (warmup default 0 in `Optimization`; 100 in CampaignConfig and YAMLs) | config.py:286-290; campaign_config.py:85-92 |
| 4 episodes/update, token budget 8192 | Verified | config.py:293,297 |
| noise 0.05 | Verified, but it is UNIFORM half-width (std 0.029), drafter inputs only | data.py:315-319 |
| draft_attn_impl simple_flex_attention | Verified (default + recipes) | bank/config.py:113-115 |
| generator dims 512/128/64/32/512/2 | Verified | config.py:275-281 |
| ttt_steps 7 | Partly: 7 only in `training_recipe_ttt7.yaml` (10k variant); 2k variant and generator recipe use 3; `ttt_step_loss_decay=1.0` | recipes |
| lk_hybrid | Verified (eta=3.0 fixed, formula in section 4) | losses/eager.py:174-213 |

Surprising / risky items:
1. Pure-bf16 generator with AdamW and no master weights (generator_dtype bfloat16 in all prod YAMLs) - updates of size lr=1e-4 on weights/biases ~1e-2 are near bf16 spacing; asserted as intended by a test (test_generator_batching.py:66). Potential silent stagnation; consider fp32 master weights for the port.
2. `fresh_lora` baseline precision differs between harnesses (bf16 adapters in structure_study static_lora default vs fp32 in `train_matched_lora`).
3. Weight decay applies to the A-head bias that stores the kaiming A prior, and to all biases/LayerNorm.
4. `install_adapters` ordering: freezes before creating adapters, so a/b start `requires_grad=True`; freezing relies on later `requires_grad_(False)` in `run_training` (campaign_train.py:400); `initialize_direct` re-enables only a/b for the static arm.
5. `Runtime` hard-wires `resolve_loss_config(spec)` to the fused Triton implementation (no config for eager); `evaluate` raises if `full_acc_0` missing; both EAGLE-3-only.
6. Per-microbatch loss weights use step-0 loss-mask counts although later TTT steps have fewer valid tokens (small bias); total loss is a SUM over TTT steps (so TTT7 multiplies loss/grad scale vs TTT3 under the same clip 0.85).
7. Conditioning cohort == loss cohort by default (the generator sees prompts, loss is on those prompts' responses): intended per-request/group specialisation, but validation uses the same coupling (4 prompts -> LoRA -> their own responses), so reported gains are in-cohort adaptation.
8. `norm_before_residual`/`input_norm`/`fc_norm` of the drafter are applied inside drafter forward; conditioning bypasses the drafter's fc/norms entirely (raw hidden states, no normalisation, magnitudes can be large; encoder has a LayerNorm only at its output) - the first Linear sees raw 12289-d inputs.
9. Silent fallbacks: no bare `except` in the training path (model/engine/campaign_train/data). Notable defaults: `CollateFn(..., num_target_layers=3)` implicit default; `Optimization.loss_fn=None` inherits bank loss; `generator_dtype` default float32 vs YAML bf16; `ExperimentConfig.condition` default None in the legacy config vs "last" forced in campaign. `epoch_training.backup_checkpoint` falls back from hardlink to copy on `OSError` (benign).
10. `layer_embedding` is a single constant row - a vestigial layer descriptor; not a bug for 1-layer EAGLE-3 but a trap for DFlash.
11. The optimizer receives only generator params, so even if drafter grads existed they would not be updated; `drafter.train()` only toggles adapter dropout (EAGLE-3 attention_dropout 0 assumed).
12. Eval-time coupling: validation `evaluate_subset` hard-codes cohort size 4 and result key "4" (campaign_train.py:93, 133, 146).
13. Mixed-bank/legacy code (bank regression, heatmap, FactorCache, `last_mean`/`projected` conditions, feedback generators, `Corpus.prepare` summary caches) is dead weight for a DFlash port of the direct objective; keep `campaign_train.py`, `model.py`, `engine.Runtime`, `data.Corpus.microbatches`, `campaign_data.EpisodeSchedule`.
