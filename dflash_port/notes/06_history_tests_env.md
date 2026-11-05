# 06 - History, upstream modifications, tests, dev environment

Scope: `git log/diff e3041dd..HEAD` (no upstream remote; "upstream" = `e3041dd`). 213 commits, almost all `AhmetZeer`. Read-only exploration.

## 1. Commit history by phase

Dates 2026 (all dates below are Sep-Nov). Counts are approximate.

| # | Phase | Dates | ~Commits | Representative SHAs | Key files / notes |
|---|---|---|---|---|---|
| 1 | Initial EAGLE-3 LoRA fine-tune (`--lora-r`) + Colab-style shell recipes | 09-29..09-30 | 17 | 910dbb2 "eagle3 lora", 01e61bd, 675cdc7 (W&B), f4ea259 (5k cfg), 641d3f8 (llama), 239fdcf (qwen) | `train/lora.py`, `LoraArgs` in schema, `examples/eagle3-lora/eagle3_*_nemotronchat_{1k,5k}_lora.sh`; many one-word "fixconfig" commits. Hard-coded `/content/drive/...` (Colab). |
| 2 | LoRA bank generation (many per-subset LoRAs as generator training targets) | 10-03..10-06 | ~20 | eef2b74 "bank generation", 7496e12 (stage responses locally), e794d9c (prefetch), cc550fe (atomic hidden-state publish), c341f78 (bank stages), 5ef9525 (parallel workers), 03013e7 (multiple seeds) | `src/speculators/bank/*` (16 new files), `train/bank.py` (`BankTrainer`), `BankArgs`, resumable hidden-state extraction in `cli/generate_offline_data.py`. 188cedd RAM prefetch cache -> **reverted** 275acbd; a59f26d replaces with buffered reads. |
| 3 | Generator v1: offline hypernetwork on bank | 10-06..10-07 | ~15 | 6a9137d "offline EAGLE3 LoRA generator experiments", 4b194d7 (FlexAttention block alignment), c72a7b7 (grad accumulation w/o retaining compiled graph), 93890d6 (clip 0.85), ad18c90 (LK hybrid loss), 744bb74 (heatmap pretrain validate/250) | `src/speculators/generator/*` (43 new files), `tests/unit/generator/*`, `models/attention.py` edits. |
| 4 | vLLM 0.31.0 serial generator adapter integration | 10-07 | 5 | 6dd9a4a, 8fc7f9d, da0af8c, 669e238 (pin V1 runner), 89b99ad | `generator/serving/*` (19 files), `serving/patches/vllm-0.31.0-generator.patch`, `scripts/apply_vllm_generator_patch.py`, `scripts/launch_vllm.py serve`. |
| 5 | Cloud/Modal + public HF Hub artifacts, sharded activations, append-only scale-up | 10-10..10-16 | ~35 | 91cb7d9 "modal pipeline", d49e2fb, a796da0, 9848a30 (atomic hub commits), 7065731, 31c408b (train from staged raw activations), dd3db62, 25ef567, 6883459 | `scripts/modal/app.py`, `bank/hub.py`/`publish`, `scripts/repack_lora_bank.py`. Hub repos hard-coded to `azeeer/...`. 5ca8cf9 "revert" (config). |
| 6 | Online activation RAM cache + campaigns; direct pretraining & adaptation replaces bank campaign | 10-16..10-19 | ~15 | fde90cf, 7e97f81, 80e3c67, 43a77ef "Replace bank campaign with direct pretraining and adaptation", fa0ca7a "Retire adaptation arms", 52e6d2d (HF checkpoint publishing) | `generator/campaign*.py`, `campaign_resume.py`. Abandoned: bank-campaign schedule, adaptation arms. |
| 7 | HTTP serving + GuideLLM benchmark suites + Nemotron benchmarks (first results) | 10-19..10-22 | ~25 | cc4c490, 6976d0b, 8dde300 (matched-data LoRA baseline), 7241bc9 (full-data generator, exact epochs), 66f6a7b (resumable vLLM+GuideLLM suites), c0ee0e4 | `src/speculators/benchmark/*` (10 files), `cli/benchmark.py`, `scripts/compare_http.py`, `scripts/evaluate/evaluate.py --validation-manifest`. Results under `docs/experiments/results`. |
| 8 | LaTeX report + shared-adapter concurrency / TTFT analysis | 10-22..10-26 | ~25 | af348e1, db35e96, ecff93a "native continuous batching", dc872c5/fb16171 (profiling), 3d7b2c6 | `reporting/` (LaTeX). Fixed-batch shared adapter limits documented (ab8ffc1). |
| 9 | Dynamism experiments, 1000-update matching, causal SDPA replay | 10-26 | ~10 | d52d66f, 1dd23ef (match all at 1000 updates), f93093c "causal SDPA for replay to avoid FlexAttention recompile exhaustion" | `models/attention.py` `speculators_streaming_sdpa` kwarg. |
| 10 | Online feedback continuation (v1/v2/v3) | 10-27..10-28 | ~12 | ce52491, 059744a, d0e5f28 -> **reverted** acd9e8e, 325d2f5, 2ea51de "Restore original saved-response continuation", 52600c0 "negative result" | `generator/feedback_training.py`, `examples/eagle3-lora/feedback_continuation_v{1,2,3}`. Feedback arm = negative result; reverted to original continuation. |
| 11 | Coefficient sweep @ step 4000 | 10-28 | 2 | 5d4501d, 6612aaf | `examples/eagle3-lora/coefficient_sweep_4000`. |
| 12 | Structure/architecture ablations (module, attention-only, decoder256, compact) | 10-30..10-31 | ~25 | cbd323f, b47bae1 (batched cohorts, BF16), 751aa01 (attention-only targets), 7e2ada8, decc6da (24K attention generator), 8e2e853 | `examples/eagle3-lora/structure_ablations_v1`, `attention_24k_6k_v1`. Conclusion: attention-only generator is the champion architecture carried forward. |
| 13 | DeepSeek EAGLE3 ttt7 benchmark + 4K DeepSeek attention generator | 10-31 | ~15 | 80b2884, bd01b0b, e8974c2, 5e53f2b, c2eb467, 948b113 | `benchmark/deepspec.py`, `scripts/prepare_deepspec.py`, `examples/eagle3-lora/deepseek_ttt7_generator_v1`. OOD suites. |
| 14 | 100K Nemotron scale-up | 11-01 | ~20 | 34ed20d, fa0a5ce (multi-node responses by chunk), 545a39d "managed single-command 100K attention training", 6c8036c (restore proven 24K runtime), 6e85f5c, d0f3877, 807a6ae | `scripts/train_attention_100k.py`, `examples/eagle3-lora/attention_100k_2k_v1`. Burst of knob commits ("util", "maxtokesn", "maxbatched"). |
| 15 | Audit 24K vs 100K + latest WIP (see section 3) | 11-01..11-05 | 5 | 0aa0272, 650b091, e1aab8c, 3ca2a6b, b17276c | See section 3. |

Reverts/abandoned: 275acbd (RAM prefetch of bank activations), 5ca8cf9 "revert" (config), acd9e8e (streaming vocab/activation scan optimisation), 2ea51de (online feedback -> original continuation), fa0ca7a (adaptation arms retired), 43a77ef (bank campaign replaced), 1163499 (drop repeated payload validation, "trust saved artifacts"). Last 5 commits are on one branch tip; no merge commits noted.

## 2. Modified pre-existing files (39) - `git diff --diff-filter=M e3041dd HEAD`

DFlash-risk legend: **S** = shared code path also used by DFlash training, behaviour may change; **N** = neutral / EAGLE-only / opt-in.

| File | +/- | What changed | Risk |
|---|---|---|---|
| `src/speculators/models/attention.py` | +47/-11 | (a) `flex_attention_forward` gained kwargs `speculators_streaming_sdpa`, `speculators_streaming_is_causal` -> routes to `F.scaled_dot_product_attention` with dense mask (raises if BlockMask). (b) Default path: if `query.is_cuda and not torch.compiler.is_compiling()` uses `torch.compile(flex_attention, fullgraph=True, dynamic=True)` (lru_cached) instead of bare eager `flex_attention`. (c) `FLEX_ATTENTION_BLOCK_SIZE=128`; ruff noqas. | **S**: DFlash also uses this shared forward. Compiled model: unchanged. Uncompiled CUDA caller (e.g. functional LoRA substitution, eager debug runs) now silently gets a compiled kernel (first-call latency, dynamic-shape recompiles; numerics ~same). Needs torch>=2.9 flex. Test: `tests/unit/models/test_standalone_flex_attention.py`. |
| `src/speculators/losses/fused.py` | +10/-7 | `_autotune_tile` decorator object -> factory `_autotune_tile()` so each kernel owns its own configs list (TorchDynamo mutation tracking). | **S** but behaviour-neutral (same configs/key). Triton only (Linux/CUDA). |
| `train/trainer.py` | +34/-8 | Extracted `create_checkpointer()`; hooks `after_optimizer_step`, `on_train_epoch_start` (no-ops); `verify_training_compatible(base_speculator_model(model))` (unwraps peft); `train_epoch` calls `transform.set_epoch` if present; grad clip now captures norm and adds `grad_norm` + `gradient_clipped` to metrics (clip value stays 1.0). | **S**: extra metrics in DFlash logs; otherwise equal. Unwrap helper is identity for non-LoRA. Hooks used by `train/bank.py`. |
| `train/cli.py` | +~160 | `main()` wraps stages with bank progress; applies `apply_lora` when `lora_r>0`; saves via `save_lora_checkpoint`; bank branch (BankTrainer, BankFileTransfer, validate_cache); `create_train_val_loaders` called with new kwargs; `startup_ready` callback. Imports `speculators.bank.progress`, `train.lora`, `train.checkpointer.patch_config_dtype`. | **S** structurally (same `main`), but non-bank/non-LoRA path should be same. Note imports `train.lora` (needs `peft` at import time? verify - `lora` extra is optional) and `bank.*` at module top. If `train/lora.py` imports peft eagerly, DFlash training would need peft installed. |
| `train/config/schema.py` | +103 | New `data.max_train_samples`; groups `LoraArgs` (r=0 default off), `BankArgs`; `TrainConfig.lora`, `.bank`; validators `_validate_bank` (EAGLE3+LoRA+AdamW+scheduler none...), `_validate_lora` (eagle3 only, needs from_pretrained). | N (defaults off). `flatten()` output gains `lora_*`/`bank_*` keys -> config hashes/provenance differ. LoRA validator explicitly rejects DFlash (`speculator_type != "eagle3"`) - must be relaxed for the port. |
| `train/data.py` | +~40 | `ArrowDataset`: `max_train_samples`, `row_indices` (explicit train/val membership), `file_indices` replaces `start_file_idx` offset; `__getitem__` uses `transform.apply_indexed` if available; `input_ids.tolist()` guarded for non-tensors; **new `on_missing=="raise"` ValueError when cached token ids mismatch**. | **S**: `get_file_index` now `file_indices[index]` (same values for default split). Token-mismatch `raise` is a new strictness (previously mismatch handled differently) - could abort DFlash runs with `on_missing=raise`. |
| `train/dataloader.py` | +~70 | `_bank_worker_init_fn`, `sampler_seed`, `IndexedUniformNoise` (only if seed set AND workers>0), explicit `train_indices/val_indices` (must be paired, disjoint), `include_validation` flag (val_loader may be None), seeded DataLoader generator only when seed. Return type `tuple[DataLoader, DataLoader | None]`. | N for default path; type change to Optional val loader can trip mypy. |
| `train/noise_transforms.py` | +23 | Adds `IndexedUniformNoise` (deterministic per (seed, epoch, index), shared-memory epoch). | N (opt-in). |
| `train/optimizers.py` | +8 | AdamW now receives only `requires_grad` params (not `named_parameters()` tuples of all) and raises if none. | **S**: DFlash with frozen embeddings/lm_head: previously frozen params were in optimizer (no effect); now excluded. Param-group naming changes -> optimizer state_dict layout differs from upstream checkpoints (resume across versions would mismatch). Muon branch untouched. |
| `train/checkpointer.py` | +12/-2 | `SingleGPUCheckpointer`: `load_adapter_checkpoint` early-return on resume; `save_lora_checkpoint` else fallthrough to original `save_pretrained`. | N for non-LoRA (helpers return False). Imports `train.lora`. |
| `train/logger.py` | +16/-7 | `setup_root_logger(level, *, use_rich=True)`; plain handler + `force=True` for bank. | N. Test added. |
| `cli/generate_offline_data.py` | +~300 | Resumable, chunked, prefetching hidden-state extraction: heartbeat/progress logger, `--resume/--skip-existing`, `--row-indices`, chunk size, atomic `publish_hidden_states`, `check_hidden_state_file_header`, sync-every-N; `_finish_startup`. | **S** for DFlash data generation (same command). Default behaviour mostly unchanged but file layout/atomic publish and validation changed; new Typer options. |
| `cli/regenerate_responses.py` | +~170 | Uses new `data_generation.response_output.ResponseOutput` (resumable append, partial-line trim, snapshot copy), chunked multi-node responses, prompt prefetch. | **S** for response regeneration. Tests added. |
| `data_generation/offline.py` | +34 | Hidden-state publish/header-check helpers. | S (offline data format helper). |
| `data_generation/preprocessing.py` | +4/-4 | Small (list-token extraction fix, 7d2648c). | S, minor. |
| `data_generation/vllm_client.py` | +~12 | Minor (timeouts/context len handling; 98a146a "reserve one output token"). | S minor. |
| `cli/__init__.py` | +7 | Registers typers `lora-bank`, `generator`, `benchmark`. | N. |
| `cli/prepare_data.py` | +12 | Added `shuffle` option (and related). | S minor; check default preserves upstream. |
| `provenance.py` / `scripts/_provenance.py` | +1 each | Adds `"peft"` to tracked packages (kept in sync via `sync_provenance.py`). | N. |
| `scripts/launch_vllm.py` | +~85 | New subcommands `responses` (plain vllm serve, no extraction) and `serve` (target/EAGLE3/generator/LoRA serving via `generator.serving.server.build_server`); writes `server_environment.json`, bundles `vllm-0.31.0-generator.patch` into provenance; frontend CPU bound for `train` and `responses`; `os.execvp`-style replace of process (no pipes). | **S** for `train`/`eval` subcommands: DFlash hidden-state serving unchanged except the logging message and execv behaviour. |
| `scripts/evaluate/evaluate.py` | +~55 | `--validation-manifest` frozen-workload mode -> `generator.serving.guidellm.benchmark_endpoint`; `--correctness-report`, `--allow-output-mismatch`, `--server-provenance`, `--warmups`, `--tokenizer-path`, `--validation-grouping`, `--seed`, `--capture-profile`; no longer `sys.exit(1)` when no acceptance counters (logs). | S: acceptance-less runs no longer fail (silent-ish change). |
| `scripts/evaluate/perf_utils.py` | +30 | GuideLLM options: model, request_format, deterministic seed, tokenizer, command_path. | N. |
| `scripts/evaluate/requirements.txt` | | pinned `vllm==0.31.0`, `guidellm==0.7.1`; removed `speculators>0.7.1`. | Env: eval pins exact vLLM. |
| `pyproject.toml` | +13 | `generator` extra (filelock, matplotlib, wandb), `lora` extra (peft>=0.18), package-data for serving patches, dev adds matplotlib+wandb. | See sec. 5. |
| `AGENTS.md` | +11/-2 | Documents `launch_vllm.py responses/serve` (three subcommands), new "Reporting" rule (LaTeX report only on explicit request). | - |
| `.gitignore` | +4 | `tmp/`, `*.zip`, `*.tar.gz` | - |
| docs (`.nav.yml`, `cli/{index,launch_vllm,response_regeneration,train}.md`, `developer/design_docs.md`) | small | Doc updates for new subcommands/flags. | - |
| tests modified | | `tests/unit/models/test_fused_losses.py` (+6), `train/test_dataloader.py` (+35), `train/test_logger.py` (+33), `scripts/test_eval_provenance.py` (+24), `test_launch_vllm.py` (+74), `test_response_regeneration.py` (+73) | |

No file under `src/speculators/models/dflash/` or `models/eagle3/` was modified (only `models/attention.py`; `models/` diff is that single file). DFlash model code is untouched upstream code.

### Shared-path changes that could alter DFlash training (ranked)
1. `models/attention.py` standalone compiled flex-attention for uncompiled CUDA callers (+ new SDPA branch gated by kwarg, default off).
2. `train/optimizers.py` frozen params excluded and optimizer layout changed.
3. `train/data.py` `on_missing="raise"` on token mismatch; index mapping refactor.
4. `train/trainer.py` extra `grad_norm`/`gradient_clipped` metrics (logging only), hooks.
5. `train/cli.py` imports of `speculators.bank.*` and `train.lora` at module import; schema validators only trigger when enabled; `TrainConfig.flatten()` keys changed.
6. `cli/generate_offline_data.py`, `regenerate_responses.py` data-pipeline rewrites (file publication, resume).
7. `losses/fused.py` autotune factory (neutral).

## 3. Latest 5 commits

| SHA | Date | Files | What |
|---|---|---|---|
| b17276c "new ttt" | 11-05 | `examples/.../attention_100k_2k_v1/{README.md, benchmarks/TTT_COMPARISON.md, benchmarks/ttt_comparison_c1.yaml}`, `benchmark/suite.py` (+8), `cli/benchmark.py` (+6), `tests/unit/scripts/test_benchmark_suite.py` (+31) | Benchmark config + comparison table of 2k-step generator vs 10k-step TTT7-trained generator (`azeeer/eagle3-generator-attention-100k-10k-v1-ttt7`) plus suite code change (supports the new comparison) with a test. Comparison doc is a template/table: results may be pending. |
| 3ca2a6b "ttt7" | 11-04 | `training_recipe_ttt7.yaml` (new), `train_10k.yaml` (-/+), `scripts/train_attention_100k.py`, `test_train_attention_100k.py`, README | Adds TTT-step-7 (7 drafter unroll steps) recipe for the 10k-update run; script variants/suffix handling (`--variant`, hub repo `-ttt7` suffix). |
| e1aab8c "parallel training" | 11-03 | `train_attention_100k.py` (+74), `generator/campaign_resume.py`, `test_epoch_training.py` (+25), `test_train_attention_100k.py` (+78), `train_10k.yaml`, README | Parallel training launch (multi-GPU/process) in the managed 100K script; resume compat changes in `campaign_resume`. |
| 650b091 "new scale" | 11-03 | `train_10k.yaml` (new, 181 lines), script (+23), test (+27), `docs/experiments/results/README.md`, README | New 10K-update training config for the attention generator (scale step after 2k). |
| 0aa0272 audit | 11-01 | results `GEN-011`, `GEN-013`, `training_audit/{24k,100k}/*` (bundle/config/result JSON, patches, vllm provenance), `provenance.json` | Audit of actual training exposure 24K vs 100K plus runtime provenance snapshots. Data only. |

State: consistent in that each commit pairs code + test + README/config, and tests were added for script changes. Likely incomplete / WIP: (a) no 10k/TTT7 results recorded yet (TTT_COMPARISON.md is the harness, check numbers before citing); (b) commit messages are uninformative ("new ttt", "ttt7"); (c) `train_10k.yaml` and `training_recipe_ttt7.yaml` hard-code Hub repo names and absolute roots (see sec. 6). I did not read these files line by line; verify status in notes of other agents (results area).

## 4. Tests

Fork-added test files (47). `tests/unit/generator/` (28 files, ~8.8k lines incl. `__init__.py`):

| File | Covers |
|---|---|
| test_activation_cache, test_online_campaign, test_campaign | Activation RAM cache/prefetch; online campaign lookahead; campaign scheduling/resume |
| test_deepspec_training, test_external_validation, test_validation_sampling | DeepSeek EAGLE3 ttt7 native training; Nemotron validation exports; held-out sampling |
| test_epoch_training, test_training_timing, test_tracking_layout, test_studies, test_structure_study, test_experiments | Exact-epoch training, per-component latency, W&B layout, architecture/module ablation studies, experiment identity |
| test_feedback_continuation, test_original_continuation, test_streaming_execution | Online feedback arm; original saved-response continuation; replay/streaming SDPA |
| test_generator_batching, test_fixed_batch_serving, test_per_request_serving, test_serving, test_http_serving, test_guidellm_endpoint | Generator batching, fixed shared-adapter vs per-request adapters, vLLM adapter controller, HTTP serving, GuideLLM endpoint |
| test_lora_baseline, test_projection_merging, test_publish | Matched static LoRA baseline, merged attention/MLP projection, HF publishing |

`tests/unit/scripts/`: test_bank_progress, test_benchmark_suite, test_compare_http, test_deepspec_prepare, test_hidden_state_saving, test_modal_prepare, test_repack_lora_bank, test_runtime_profiling, test_train_attention_100k (new); test_eval_provenance, test_launch_vllm, test_response_regeneration (extended).
`tests/unit/train/`: test_bank_file_reads, test_bank_hub, test_bank_parallel, test_bank_prefetch, test_bank_publication, test_lora_bank, test_raw_activation_staging (new); test_dataloader, test_logger (extended).
`tests/unit/data_generation/`: test_pipeline, test_prompt_buffer, test_response_output (new). `tests/unit/models/`: test_standalone_flex_attention (new), test_fused_losses (extended).
No DFlash-specific fork tests; no end-to-end test of the generator on DFlash.

### Local run attempt (Windows, py3.11.0, torch 2.5.1+cu121, transformers 5.5.4, pytest 9.0.1)
- `python -c "import speculators"`: **ModuleNotFoundError** (not installed). `pip show`: speculators, vllm, peft, wandb **not installed**; loguru, filelock/matplotlib status: loguru missing, peft missing.
- `PYTHONPATH=src python -m pytest tests/unit/generator -x -q`: fails at conftest: `tests/conftest.py` -> `tests.e2e.utils` -> `from loguru import logger` -> `ModuleNotFoundError: loguru`.
- With `--noconftest`: collection error in `test_activation_cache.py`: `speculators/config.py:278 default=version("speculators")` -> `importlib.metadata.PackageNotFoundError: No package metadata was found for speculators` (needs an installed dist; `pip install -e` required, not done). 
- Other blockers expected even if installed: `torch>=2.9.0` required by pyproject (local 2.5.1 too old; flex_attention API, `torch.accelerator` used in trainer needs 2.6+/2.9), `transformers<5.17` OK (5.5.4 is in range), `peft>=0.18`, `wandb`, `loguru`, `datasets>=4`, `numpy>=2`. Triton fused-loss kernels and vLLM are Linux-only. Verdict: generator unit tests cannot run in this local env without installing deps (not allowed). Running them needs the Linux box (or a fresh venv with torch>=2.9). pytest `addopts = -s -vvv --cache-clear`; `.pytest_cache` write gets "Access denied" on this path (harmless warning).

## 5. Dev environment

| Item | Value |
|---|---|
| Python | `requires-python >=3.10`; Modal image uses 3.12 (`scripts/modal/app.py`); local 3.11.0 |
| torch | `>=2.9.0,<=2.13.0` (+torchaudio, torchvision) ; local 2.5.1 is below the minimum |
| transformers | `>=5.0.0,<5.17.0` |
| vLLM | not a base dependency; extras only for eval: `scripts/evaluate/requirements.txt` pins `vllm==0.31.0`, `guidellm==0.7.1`; generator serving patch is `vllm-0.31.0-generator.patch`, V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`) |
| Other pins | datasets `>=4.0.0,<=5.0.1`, numpy `>=2.0,<=2.4.6`, loguru `>=0.7.2,<=0.7.3`, openai>=2, pydantic>=2, typer>=0.12 |
| Fork extras | `generator` = filelock, matplotlib, wandb; `lora` = peft>=0.18,<1; both listed under `dev` partially (matplotlib, wandb). `peft` not in `dev`. package-data for `speculators.generator.serving` patches (`patches/*.json, *.patch`) |
| Build | `setup.py` + setuptools-git-versioning; `hs_connectors` workspace package resolved at build time (dev/nightly/release) - install via `pip install -e .[dev,generator,lora]` (hs-connectors from workspace) |
| Makefile | `quality` (ruff check, ruff format --check, mdformat --check, `mypy --check-untyped-defs`, `scripts/sync_provenance.py --check`), `style` (fixes), plus further targets (build/docs/tests) not listed here |
| pre-commit | ruff v0.15.20 (check --fix + format), mdformat 1.0.0 with gfm/footnote/frontmatter/admon/gfm-alerts, local mypy, `sync-provenance` hook (regenerates `scripts/_provenance.py` from `src/speculators/provenance.py`) |
| `.claude/skills` | only `pr-review/SKILL.md` (upstream-style; design-first PR review) |
| AGENTS.md | Provenance rules: training writes `train_command.txt` + `speculators.patch`; eval writes `eval_command.txt`; `launch_vllm.py` must always get `--provenance-dir` (writes vllm_command.txt, vllm.patch, checkpoint_sha256.txt); include them when publishing. Fork adds: `responses`/`serve` subcommands, and "Reporting" rule: Markdown experiment records are the working docs; LaTeX under `reporting/` only on explicit request, no automatic reports. |
| `.gitignore` adds | `tmp/`, `*.zip`, `*.tar.gz` |
| Local (Windows) | Only smoke tests feasible: RTX 4060 8GB, torch 2.5.1. Needs fresh venv with torch>=2.9 (cu12x) + `pip install -e .[dev,generator,lora]`; Triton/vLLM unavailable on Windows -> fused loss falls back? (check `losses/fused.py` guards) / use WSL |
| Remote Linux H100 | Python 3.10-3.12, CUDA 12.x/13 torch>=2.9, `pip install -e '.[dev,generator,lora]'`, `pip install vllm==0.31.0 guidellm==0.7.1`, apply patch: `python scripts/apply_vllm_generator_patch.py` (generator serving only; not needed for DFlash training), HF_TOKEN + WANDB_API_KEY, large NVMe scratch for activations (`bank_activation_local_dir`), Hub access to `azeeer/*` repos (public) or replace with own |

## 6. Hard-coded environment assumptions (file:line)

| Kind | Location |
|---|---|
| Colab `/content/drive/MyDrive/...` | `examples/eagle3-lora/attention_24k_6k_v1/train.yaml:5`; `examples/eagle3-lora/structure_ablations_v1/common.yaml:4`; `examples/eagle3-lora/generator_math.yaml:3`; `examples/eagle3-lora/vllm_math.yaml:4,8-11`; `examples/eagle3-lora/eagle3_llama3_8b_nemotronchat_5k_lora.sh:35`; `eagle3_qwen3_8b_nemotronchat_1k_lora.sh:35`; `eagle3_qwen3_8b_nemotronchat_5k_lora.sh:35`; `examples/train/eagle3_qwen3_8b_nemotron_lora_offline_smoke.sh:35` |
| HF user `azeeer` in code | `src/speculators/bank/config.py:124,127,128` (banks/trajectories/activations repo defaults); `scripts/repack_lora_bank.py:27,28`; `scripts/train_attention_100k.py:517` (`azeeer/eagle3-generator-attention-100k-{variant}-v1`) |
| HF user `azeeer` in configs | ~60 YAML hits under `examples/eagle3-lora/**` (bank_*.yaml: trajectories/activations/banks repos; benchmark_*.yaml `model:`; feedback_continuation_v{1,2,3}, generator_ablations_v4, attention_24k_6k_v1, attention_100k_2k_v1 incl. `training_recipe*.yaml:43`, `ttt_comparison_c1.yaml`, deepseek_ttt7_generator_v1/train.yaml:77) |
| W&B project defaults | `examples/eagle3-lora/eagle3_*_lora.sh:61-64` (`dynamic-speculators-v2`) |
| Modal infra | `scripts/modal/app.py:24-30` (app name `speculators-prepare`, volumes `speculators-models`/`speculators-data`, CUDA 13.0.2 / Ubuntu 24.04 / Python 3.12 image) |
| Relative local models/output | `scripts/prepare_deepspec.py` defaults `output/deepseek-eagle3-ttt7-vllm`, `--verifier models/qwen3-8b` |
| Upstream example (not fork) | `examples/train/kimi_k3_dspark/launch_train_node.sh:12` `/mnt/shared/weights/kimi-k3` |
| Tokens | No literal tokens (`hf_...`) found in src/scripts/examples/tests; token use is via env / `token=` parameters (`config.py:209`, `model.py:343-389`, upstream). |
Searches covered *.py, *.yaml, *.sh, *.toml in src, scripts, examples, tests, hs_connectors; docs/experiments/results not scanned (data/records). Also `AhmetZeer` in code: none besides git metadata. Also dflash_port/PROGRESS.md exists (our own notes); `/workspace`, `/scratch` not found.

## Caveats
- Fork-addition inventory: src additions are `bank/` (16), `benchmark/` (10), `generator/` (43), `generator/serving/` (19), 3 new CLI modules, 3 data_generation modules, 2 train modules (`lora.py`, `bank.py`).
- Did not verify whether `train/lora.py` imports peft at module level (affects whether DFlash training needs peft); check in generator-internals note.
- Local-run verdict is based on one failing pytest invocation each for with/without conftest; later failures (torch version) are predicted, not observed.
