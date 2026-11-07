# Implementation plan: drafter-family generator (EAGLE-3 → DFlash, later DSpark)

Status: **approved, in execution** (drafted 2026-11-05, approved 2026-11-06). This file covers P1 of `PLAN.md`. Track progress in the checklist at the bottom.
Evidence:
- the generator training path map and the upstream family conventions, both 2026-11-05 agent reports, summarized here;
- `notes/01` §3/§7 and `notes/05` §7/§10.

## 1. How the repo is structured, and what we align with

- **Upstream families**:
  - One package per algorithm: `models/<algo>/{config,core,model_definitions,metrics,…}.py`.
  - Each registers twice under the same key: `@SpeculatorModelConfig.register("dflash")` and `@SpeculatorModel.register("dflash")`. The config field `speculators_model_type` is the discriminator.
  - Family behavior lives on the model class: `get_trainer_kwargs`, `from_training_args`, `forward → (draft, loss, metrics)` and `target_layer_ids`.
  - The upstream trainer's only per-family table outside the model is `preprocess_fns` (`train/cli.py:787-792`: eagle3/peagle → `shift_batch`, mtp → `shift_batch_mtp`, block drafters → none).
  - DSpark and DFlash2 subclass DFlash (model and config) and reuse `_backbone_forward` and `_make_decoder_layer`.
- **The fork's generator** (`src/speculators/generator/`, flat, ~13K lines):
  - It has no family abstraction; EAGLE-3 is assumed everywhere on the training path.
  - The DeepSeek drafter was added by converting it to native EAGLE-3 format (`benchmark/deepspec.py`) and changing YAML only. DFlash is the first real branch point.
  - Training path: CLI `campaign` / `train-full-data` → `CampaignConfig` (+ `training_recipe`, `trajectory_recipe`) → `load_local_inputs` → `OnlineActivations` + `OnlineCorpus` → `Runtime` (`engine.py`) → `campaign_train.run_training` → `persist_checkpoint` / `publish.push_run` → `serving.bundle.export_run`.
- **Alignment decision**:
  - Add a small **drafter-family layer inside the generator**: `src/speculators/generator/drafters/`, keyed by the same `speculators_model_type` strings and using the repo's `ClassRegistryMixin`.
  - EAGLE-3 behavior moves into `drafters/eagle3.py` **unchanged**. DFlash gets `drafters/dflash.py`.
  - A future `drafters/dspark.py` subclasses the DFlash family, just as `DSparkDraftModel` subclasses `DFlashDraftModel`.
  - Generic code (`engine.py`, `data.py`, `campaign_train.py`, …) calls the family instead of hard-coding EAGLE-3.

## 2. The family interface (`drafters/base.py`)

Each method has one job. The EAGLE-3 implementation returns exactly what the code does today.

| Hook | EAGLE-3 (today's code) | DFlash |
|---|---|---|
| `adapter_targets(drafter, target_modules, layers)` → `{key: Target(path, module_type, layer)}` | leaf name, single layer; key = `o_proj` etc. (checkpoint keys unchanged) | `layers.{i}.self_attn.{q,k,v,o}_proj`, optional `mlp.*` and `fc`; key = `l{i}_{type}` |
| `check_drafter(drafter, bank)` | `target_layer_ids == layer_ids[:-1]` | the same, plus mask token id set, block size and attention type equal the checkpoint's, `sample_from_anchor` logged |
| `call_kwargs(training, loss_implementation)` | `ttt_steps`, `ttt_step_loss_decay`, `loss_config` | `type(drafter).get_trainer_kwargs(...)`: `loss_config`, `gamma`, `max_anchors`, `per_position_loss_weight`, `dpace_alpha` |
| `preprocess(sample)` / `trained_length(seq_len)` | `shift_batch` / `seq_len − 1` | none / `seq_len` |
| `supervised_tokens(row)` | `sum(loss_mask[1:])` | `sum(loss_mask)` (anchor positions sampled from it) |
| `microbatch_kwargs(batch, validation)` | `{}` | per-microbatch `max_anchors` from a fixed **anchor density** (default 512 / 8192); validation runs under a fixed per-row seed (`torch.random.fork_rng`) |
| `training_noise(states)` | uniform `noise_std` | none (DFlash trains without it) |
| `score_key` / `metric_label` | `full_acc_0` / "offline agreement (full_acc_0)" | `full_acc`; `eal` and `position_i_acc` are logged as well |
| `identity()` (feeds `run_identity`) | the **same dict as today**, so old EAGLE-3 run identities stay valid | DFlash training params |
| `prepare_drafter(drafter, adapter_cfg)` | no-op | install the `kv_lora_path` attention variant (§4) |

- The family is resolved from `drafter.config.speculators_model_type`. An unregistered family raises.
- No defaults are substituted: every DFlash field is explicit in the recipe, or a documented constant.

## 3. Generator changes (`generator/model.py`)

- `install_adapters(model, rank, alpha, dropout, targets)` takes the family's `targets` and replaces each listed path by `AdapterLinear`. It validates all targets before mutating anything. It returns `{key: path}`. Keys contain no dots, so they are valid `ModuleDict` keys.
- `LoRAGenerator(input_dim, shapes, rank, architecture, …, layout=None, head_sharing="per_module")`:
  - `layout[key] = (module_type, layer)`. When it is None it is `{key: (key, 0)}`, which is exactly today's EAGLE-3 behavior.
  - `head_sharing="per_module"` (today): one A/B head per key.
  - `head_sharing="per_type"` (DFlash default): one head per module type, shared across layers. The layer enters through `layer_embedding = Embedding(num_layers, 32)`. QKVO heads are ≈437M parameters, versus ≈2.2B per layer.
  - `module_embedding` is indexed by module type. For EAGLE-3 the type order equals today's key order, so state-dict keys and outputs are identical.
- **Legacy-equivalence test**: a frozen copy of today's `LoRAGenerator`/`install_adapters` lives in `tests/unit/generator/_legacy_model.py`. With the same seed, the new code must produce identical parameters (state-dict keys and values) and factors. This needs no golden files and works in any environment.

## 4. DFlash-specific pieces

- **K/V path switch (`kv_lora_path ∈ {both, context, block}`)**:
  - A minimal **behavior-identical refactor** of upstream `Qwen3DFlashAttention.forward`: lines 116-119 move into one method, `_project_kv(target_hidden, hidden_states) -> (k_ctx, k_noise, v_ctx, v_noise)`.
  - The generator side gets a subclass, `drafters/dflash_attention.py::KVPathDFlashAttention`. It overrides `_project_kv` and uses `adapter.base(x)` (no delta) on the excluded path.
  - It is installed by `prepare_drafter` only when `kv_lora_path != both`, by switching the instance class (the subclass adds one attribute and no parameters).
  - Tests:
    - (a) the upstream refactor gives bit-identical outputs, via a test against an inline copy of the old forward;
    - (b) each path changes exactly the expected K/V tensors.
  - `both` is the only variant mergeable into vLLM weights (`GUIDE_CORRECTED` §4.3).
  - *Alternative if you prefer zero upstream edits*: the subclass copies the full 60-line `forward`. Duplicating code is worse, so I recommend the seam.
- **`fc` target**: an optional adapter on `fc`, with layer slot = its own embedding row. Its A head is r × 20,480, so it may use a smaller rank. This is an ablation only.
- **Loss**: CE + D-PACE (primary); CE + exp-decay γ; `lk_hybrid` (ablation; verify that the DFlash forward accepts it).
  - Relax `campaign_config.py:130` (which forces `lk_hybrid`) to a **per-family allowed set**. EAGLE-3 keeps requiring `lk_hybrid`.
- **Microbatches**: cohort-only packing, as today. Every token in a microbatch shares the cohort LoRA. There is no shift. Pad to 128 only for flex attention. `CollateFn` is reused unchanged.
- **Config** (`bank/config.py` `Training` and `generator/config.py`), all with `extra=forbid`:
  - `training.block_drafter: {loss_fn, per_position_loss_weight, dflash_decay_gamma, dpace_alpha, anchor_density, validation_anchor_seed}`. It is required when the family is dflash, and must be absent for EAGLE-3.
  - `adapter: {layers: all | [ids], kv_lora_path, head_sharing}`, with the EAGLE-3 defaults equal to today's behavior.
- **Bundle and serving**: `export_run` records `drafter_family`. The serving loaders **raise** for non-EAGLE-3 bundles until P5 ports DFlash serving. Tracking: `full_acc` is mapped to `agreement`, so it doesn't land in `diagnostics/`.

## 5. Paired EAGLE-3 ↔ DFlash comparison

> **Superseded 2026-11-06 (user D7–D10, `PROGRESS.md`):** no EAGLE-3 runs in this project; DFlash mirrors the existing EAGLE-3 experiments (`EXPERIMENT_MATRIX.md`), one seed, compared to existing records. The recipe-swap design below is kept only as the template for the DFlash recipes.

The goal is the same generator type on both methods with the same data. The design makes this a **recipe swap**:
- `examples/dflash-lora/<run>/training_recipe.yaml` has drafter z-lab DFlash-b16 and layers `[2,10,18,26,34,36]`.
- The paired EAGLE-3 recipe is the DeepSeek ttt7 recipe. It is already at `[2,10,18,26,34,36]`.
- Both point `trajectory_recipe` at the same r0001 trajectories.
- The `train.yaml` files are identical in: generator architecture, target modules (QKVO), rank/α, cohort schedule, seed, updates, pool and validation split.
- **Differences, made explicit and logged**:
  - the number of adapted layers (1 vs 5);
  - the native loss (TTT-3 `lk_hybrid` vs CE + D-PACE);
  - LoRA parameter count.
- Extra pairing arms (for P4) to separate "family" from "objective": DFlash with `lk_hybrid`, and DFlash with layer-0 only.
- A test asserts that the two recipes resolve to the same data identity: the trajectory manifest hashes and cohort sequence for a fixed seed.

## 6. Where things go

| Path | Content |
|---|---|
| `src/speculators/generator/drafters/{__init__,base,eagle3,dflash,dflash_attention}.py` | family layer (new) |
| `src/speculators/generator/{model,engine,data,campaign_train,campaign_config,config,tracking_layout,serving/bundle}.py` | call the family instead of EAGLE-3 constants (edits) |
| `src/speculators/bank/config.py` | `Training.block_drafter`; family-aware validators (edit) |
| `src/speculators/models/dflash/model_definitions.py` | `_project_kv` seam (≈10-line behavior-identical refactor; the only upstream edit) |
| `examples/dflash-lora/` | `README.md`, `smoke/`, `generator_4k_1500/`, `static_lora_4k_1500/`, mirroring `examples/eagle3-lora/deepseek_ttt7_generator_v1/` |
| `tests/unit/generator/test_drafter_families.py`, `test_dflash_generator.py`, `test_dflash_rigor.py`, `_legacy_model.py`, `dflash_fixtures.py` | tests (new) |
| `scripts/dflash_port/{remote_setup.sh,run_arm.py,run_matrix.py}` | run layer: remote setup, one-arm runner, matrix/ablation runner (new, §6a) |
| `examples/dflash-lora/<arm>/run.sh`, `examples/dflash-lora/matrices/{smoke,main_4k,ablations}.yaml` | per-experiment entry points and experiment matrices (new, §6a) |
| `dflash_port/{PROGRESS,RUNLOG,IMPLEMENTATION_PLAN,SESSION_PROMPT}.md` | living docs + session-start prompt |

## 6a. How experiments are run (user-facing)

The end state is that the user, or a remote agent, runs shell scripts. Every training, eval and ablation goes through the same three entry points. They mirror the fork's existing pattern, where `examples/eagle3-lora/attention_100k_2k_v1/run.sh` is a thin wrapper around a `scripts/*.py` driver.

1. **`scripts/dflash_port/remote_setup.sh`**: run once per box. It is idempotent and fails loudly.
   - Creates the venv(s) and installs `.[dev,generator,lora]` plus vLLM.
   - Downloads Qwen3-8B@`b968826d…` and z-lab DFlash-b16 at pinned revisions, converts z-lab, and checks the checkpoint SHA256s.
   - Checks the trajectories path against `data.json`.
   - Writes `env.json` (pip freeze, `nvidia-smi`, driver, vLLM version and commit) and appends a `RUNLOG.md` entry.
2. **`examples/dflash-lora/<arm>/run.sh --seed N --out DIR [--dry-run]`**: one experiment arm, for example `generator_4k_1500`, `static_lora_4k_1500` or `base_eval`, plus the paired EAGLE-3 arms under the same names.
   - It calls `scripts/dflash_port/run_arm.py <arm_dir>`, which:
     - (a) writes provenance before any compute: git SHA, `git status`/patch, the resolved config and its hash, seed, env, and trajectory-manifest hashes. A dirty tree is refused unless `--allow-dirty`, and that flag is recorded;
     - (b) starts the co-located vLLM extractor (`launch_vllm.py train … --provenance-dir DIR`);
     - (c) trains the generator or static LoRA with the `speculators generator` CLI;
     - (d) runs eval: offline per-position acceptance and τ per domain, then a vLLM benchmark at C1 and higher concurrency;
     - (e) writes `DONE.json` with the metrics and artifact hashes.
   - A failure at any stage exits non-zero and leaves no `DONE.json`.
3. **`scripts/dflash_port/run_matrix.py --matrix examples/dflash-lora/matrices/<name>.yaml [--dry-run] [--only ARM] [--summarize]`**: experiments and ablations.
   - A matrix lists arms × families × seeds. It expands them, prints the plan, and runs each arm's `run.sh` in turn.
   - An arm is skipped **only** if its `DONE.json` config hash matches; every skip is printed. A hash mismatch is an error, never a silent rerun or reuse.
   - `--summarize` aggregates the results into `summary.{md,csv}`:
     - mean ± std over seeds;
     - per-domain τ and per-position acceptance;
     - paired deltas (generator − static, DFlash − EAGLE-3);
     - net tok/s at C1 and Cn.
   - Matrices:
     - `smoke.yaml` (P2/T3);
     - `main_4k.yaml` (P3: base / static / generator × {DFlash, EAGLE-3-DeepSeek} × 3 seeds);
     - `ablations.yaml` (P4: `kv_lora_path`, `fc`, head sharing, rank, `lk_hybrid`-on-DFlash, layer-0-only, scale-up 24K/2K).
     - **Superseded (D7–D11):** DFlash arms only, one seed; the matrix contents come from `EXPERIMENT_MATRIX.md` after the D11 discussion with the user.
   - Each P3/P4 matrix is launched only after the user approves the budget, because the STOP gates in `PLAN.md` still apply.

`--dry-run` everywhere prints the exact commands and resolved configs without running them. Deliverable 4 (exact configs and commands) is these matrices plus their dry-run output.

## 7. Test tiers: what runs where and when

| Tier | Where | What | When |
|---|---|---|---|
| **T0 local CPU** | This Windows box, a new `uv` venv in `.venv-cpu/` (git-excluded) with torch ≥ 2.9 CPU. Only modules that don't import `fcntl`: `vllm_client` and `hs_connectors` are not imported by `model.py`/`drafters/` | new unit tests + the legacy-equivalence test + the tiny DFlash rigor tests (fp32, `draft_attn_impl="eager"`) | after every step S1–S6 |
| **T1 remote unit** | remote Linux, full env | the whole `tests/unit/generator tests/unit/models tests/unit/convert` (EAGLE-3 regression) + CUDA tests: flex vs eager DFlash with LoRA; bf16 | **baseline before S1** (record the pre-existing failures), then after S6 |
| **T2 remote real-model checks** (= P0) | 1 GPU | download/convert z-lab; finite weights; config assert (5 layers, `[2,10,18,26,34]`, mask 151669, bs16, full attention); layer-index check vs HF `output_hidden_states`; zero-LoRA equals base on the real model; frozen hash; vLLM DFlash version | after S7 |
| **T3 remote smoke** (= P2) | 1–2 GPUs | 50-update tiny overfit; 100 updates × {base eval, static LoRA, generator} on DFlash (D8: no EAGLE-3 runs) on the 4K pool; throughput, memory, step-0 = base, provenance | after T2 → **STOP before P3** |

If the local venv cannot be built (wheel or Python 3.11 issues), T0 moves to the remote box. I'll record that in `RUNLOG.md`.

**Rigor checks, mapped to tests** (`GUIDE_CORRECTED` §4.7). All except item 9 are tiny DFlash, CPU, T0:
1. Zero LoRA gives bit-identical loss and metrics to the base drafter with the same anchors.
2. The drafter `state_dict` hash is unchanged after N updates. All drafter params have `requires_grad=False`, enforced by a runtime assert in `run_training`.
3. Only generator params get gradients, and the optimizer set equals the generator params.
4. `functional_call` LoRA equals an explicit `W + (α/r)·B·A` merge for every target and layer. Perturbing one layer's factors changes only that layer.
5. `kv_lora_path` touches exactly the expected tensors. The upstream `_project_kv` refactor is bit-identical.
6. Block size, mask id, anchors and mask type are consistent between training and validation and are logged. Validation anchors are deterministic across calls and arms.
7. The tiny generator overfits one cohort (loss falls by ≥ 90 %, `full_acc` ≥ 0.95).
8. The same seed gives identical anchors, losses and factors.
9. (T1 / T2) The same as 1–4 on CUDA flex and on the real z-lab checkpoint.

## 8. Steps (each step = code + tests + commit + PROGRESS/RUNLOG update)

- **S0 Env and baseline.**
  - Build the T0 venv and confirm the tiny DFlash forward runs on CPU.
  - Write the T1 baseline command for the remote box. *Done when* the local tiny DFlash forward runs, or the fallback is recorded.
- **S1 Family layer + EAGLE-3 extraction (pure refactor).**
  - `drafters/base.py`, `drafters/eagle3.py`. `engine/data/campaign_train` call the family.
  - *Tests*: the legacy-equivalence test, plus an identity-digest equality test (`run_identity` dict unchanged for an EAGLE-3 config). Existing EAGLE-3 tests are unchanged; run them in T0 where importable, otherwise in T1.
- **S2 Multi-layer adapters + generator layout/head sharing.**
  - *Tests*: install on a tiny DFlash (5 layers × QKVO = 20 targets, keys and paths); shapes; shared vs per-module parameter counts; EAGLE-3 equivalence still holds.
- **S3 DFlash family: kwargs, microbatches, anchors, scoring, config, validators.**
  - *Tests*: a batch-contents test (no shift; document ids); anchor-density math; validation determinism; the score key; config forbid/required rules.
- **S4 `kv_lora_path` seam + `fc` target.**
  - *Tests*: rigor item 5.
- **S5 Rigor suite** (items 1–4 and 6–8) on a tiny DFlash, plus a tiny end-to-end `run_training` (≈20 updates, CPU) for both families with the same fake data (a code test: the EAGLE-3 path must keep working; no EAGLE-3 experiment runs, D8).
- **S6 Recipes + run layer (§6a).**
  - **First (D11, user 2026-11-07):** list the experiments from `EXPERIMENT_MATRIX.md` and how each will run (arms, configs, controls, budget, order, metrics) and discuss with the user. Write recipes/matrices only after agreement. The generic run layer (`run_arm.py`, `run_matrix.py`, `remote_setup.sh`, the `smoke` matrix) can be built before that.
  - `examples/dflash-lora/*` (DFlash arms only; no EAGLE-3 runs, D8).
  - Per-arm `run.sh`, `run_arm.py`, `run_matrix.py`, `remote_setup.sh`, and the matrices `smoke`/`main_4k`/`ablations`.
  - *Tests*:
    - config resolution: both families give the same data identity and cohort sequence;
    - matrix expansion, and skip-only-on-matching-hash: a mismatch raises;
    - `--dry-run` output is stable;
    - `run_arm` refuses a dirty tree without `--allow-dirty`;
    - provenance fields are present.
  - The matrices' dry-run output is the draft of deliverable 4.
- **S7 Bundle/tracking/serving guards and silent-fallback fixes** on the paths we use: git SHA `"unknown"`, unchecked model dirs vs pinned SHAs, NaN verifier weights, the provenance swallow.
- **S8 Review.** `/code-review` over the diff, fix findings, update the docs and commit. Hand-off: a `RUNLOG`-ready runbook for T1–T3 on the remote → **STOP and report.**

## 9. Risks to watch

- `functional_call` interacting with the DFlash compiled mask builder. The forward is already un-compiled by `Runtime`; test on CUDA in T1.
- Microbatch loss weighting with anchors. The family returns the weight from the forward metrics (valid block-token count), so chunked gradients equal unchunked ones in expectation. A test uses fixed anchors.
- Memory: 5 layers × 512 anchors × 16 queries per row. P2 measures it. The `anchor_density` knob is the lever.
- An `extra=forbid` YAML schema means old EAGLE-3 YAMLs must still load. The T1 test loads every `examples/eagle3-lora/**.yaml` (generator configs).

## Checklist (living)

- [x] Plan approved by the user (2026-11-06, "start S0"; the `_project_kv` seam is the recommended option and is revisited at S4)
- [x] S0 env + baseline (2026-11-06 remote: T1 588 passed/10 skipped; tiny DFlash fwd+bwd OK CPU+GPU) · [x] S1 family layer · [x] S2 adapters/generator · [x] S3 DFlash family
- [x] S4 kv path + fc (18da5a4) · [ ] S5 rigor suite · [ ] S6 recipes + run layer (starts with the D11 experiment discussion) · [ ] S7 guards/fallbacks · [ ] S8 review → STOP
