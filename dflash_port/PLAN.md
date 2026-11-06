# DFlash generator port — development / experiment / research plan

Design rationale: `GUIDE_CORRECTED.md` §4–5. Evidence: `FINDINGS.md`. Status: `PROGRESS.md`.
This file is the **execution plan**: phases, gates, where each step runs, and done criteria.
Exact configs and commands are filled in as deliverable 4, after the port compiles. Every run goes through the run layer in `IMPLEMENTATION_PLAN.md` §6a: `remote_setup.sh` once per box, `examples/dflash-lora/<arm>/run.sh` per arm, and `run_matrix.py` per matrix (smoke / main_4k / ablations, with `--dry-run` and `--summarize`).

## Research questions (in priority order)

- **Q1.** Does a per-query generated LoRA raise DFlash acceptance τ and C1 throughput over (a) the base DFlash drafter and (b) a **matched static LoRA**? This is measured net of generator + merge latency.
- **Q2.** Is the gain prompt-specific, or just domain routing? Test with same-category / cross-category / fixed donors.
- **Q3.** Where in DFlash should the LoRA go? Candidates: modules, the K/V context path vs the block path, which layers, `fc`.
- **Q4.** How does the answer compare with EAGLE-3? There is a paired reference: the fork's DeepSeek ttt7 generator uses the *same* conditioning input (`[2,10,18,26,34]`).

## Decisions (resolved 2026-11-05 unless marked open)

| ID | Decision | Outcome |
|---|---|---|
| D1 | Base DFlash checkpoint | **`z-lab/Qwen3-8B-DFlash-b16`** (user). RedHat `.dflash` is a fallback only |
| D2 | Primary loss | **CE + D-PACE**. `lk_hybrid` is a low-priority ablation (EAGLE-3 prior: CE ≈ LK, `notes/07` §A) |
| D3 | Smoke environment | **Remote.** All work there must be resumable by another agent, through `PROGRESS.md` + `RUNLOG.md` (see "Remote protocol") |
| D4 | Upstream sync | Remote added. **No merge**: no DFlash changes upstream; cherry-pick only as needed (`notes/07` §C) |
| D5 | Data | Reuse the **r0001 trajectories** (local copy). Extract activations online at `[2,10,18,26,34]`+36. Do not download the EAGLE-3 activations. r0002-100k math/code/stem only if a 100K arm is approved |
| D6 (open) | 100K arm | It needs ≥1.5 epochs (≈8.3K+ updates) and a trajectory download. Decide at G3 |

## Remote protocol (any agent, any machine)

1. Start by reading `PROGRESS.md`, then the tail of `RUNLOG.md`.
2. Before a run, append a `RUNLOG.md` entry:
   - host, GPU, driver, git SHA, `git status --short`;
   - the exact command, config path and seed, and the output dir.
3. After the run, add the outcome to the same entry: exit status, key metrics, artifact paths, and anomalies. Record failures too.
4. Update the `PROGRESS.md` checkboxes and commit locally. Use the identity `erdem-erdem <yusuf.erd.erdem@gmail.com>` and never push. The user moves code and commits between machines.
5. Large outputs (checkpoints, caches, trajectories) stay out of git. Record their paths and SHA256 in `RUNLOG.md`.

## Phases

### P0 — Environment and baseline reproduction (remote, ~1 GPU-day)

1. Remote Linux box: venv, `pip install -e '.[dev,generator,lora]'` with torch ≥ 2.9, then vLLM in a **separate** env if needed. Run `pytest tests/unit/generator tests/unit/models -q` → green.
2. vLLM DFlash (support confirmed by the user): record the version and SHA. Check that it reads `dflash_config.target_layer_ids`, block size and mask id from the converted z-lab checkpoint.
   - Trajectories: copy the local `datasets/eagle3-qwen3-8b-trajectories-v1` (r0001) to the box and record its `data.json` SHA256s.
   - Layer-index check: features extracted at `[2,10,18,26,34]` must equal the HF `output_hidden_states[id]` on one example (bf16 tolerance) and match z-lab's `[1,9,17,25,33]` convention.
3. Load the D1 checkpoint and assert all weights are finite, plus the config (layers, aux ids, mask id, block size, attention type).
4. Offline base eval with the upstream DFlash forward on our val split (fixed anchors): `eal` and per-position acc per domain.
5. vLLM base eval of target-only and base DFlash on the fork's suite (ID 4 domains + OOD 3, 50 prompts, 256 tokens, C1), with **3 interleaved repeats** to measure our noise floor.
6. Gate G0 passes when the base τ is plausible vs the paper, the noise floor is known, and the env is recorded (pip freeze, GPU, driver).

### P1 — Implementation (local tiny-model tests, then remote CPU/GPU tests)

Order follows the seams in `GUIDE_CORRECTED.md` §4. Each step lands with tests.
1. Multi-layer `install_adapters` and sanitized keys. Regression test: EAGLE-3 single-layer behavior is unchanged.
2. `LoRAGenerator`: layer embedding (N rows), module-type embedding, heads shared per module type (switch: shared | per_layer).
3. DFlash LoRA-aware decoder layer via `_make_decoder_layer`, with `kv_lora_path` {both, context, block}; optional `fc` adapter.
4. `Runtime` drafter-family dispatch: DFlash trainer kwargs, eager/fused loss switch, metric keys `full_acc`/`eal`.
5. DFlash microbatch builder (cohort-only packing, document ids, fixed-seed validation anchors, mask/config asserts).
6. Config/validator relaxations: `campaign_config` loss requirement, `schema.py` bank validator, `evaluation_sizes`. New `examples/dflash-lora/` recipes copied from `deepseek_ttt7_generator_v1/`.
7. Rigor test suite on a tiny DFlash (`tests/unit/generator/test_dflash_*.py`): the 8 checks in `GUIDE_CORRECTED.md` §4.7.
8. Fix silent fallbacks on the paths we use: `git_sha "unknown"`, unchecked model SHAs, NaN verifier weights, provenance swallow, `output_mismatch: warn`.
- Gate G1: all tests green, and a code review (`/code-review`) of the diff.

### P2 — Smoke runs (local tiny, then remote real at small scale) → **STOP and report before P3**

1. Local or remote tiny: 50 updates, generator overfits a single cohort (loss ↓, acc → ~1).
2. Remote real: Qwen3-8B extractor + z-lab DFlash, 4K pool, **100 updates**, all three arms (base eval, static LoRA, generator). Check:
   - throughput (updates/h), peak memory, RAM cache size;
   - step-0 validation equals the base drafter exactly;
   - the drafter hash is unchanged;
   - the provenance files exist.
3. Extrapolate GPU-hours for P3/P4 and write them into this plan.
- Gate G2: the user approves the budget.

### P3 — Main matched comparison (remote) → answers Q1, Q2

- Arms: base / matched static LoRA / generator. Base config 4K pool, 1,500 updates, **3 seeds** each.
- Offline eval (fixed anchors) + vLLM C1, ID + OOD, 3 interleaved serving repeats. Then donor controls (inference-only) on the best generator checkpoint.
- If the result is promising, run the scale-up: 24K/2,000 updates (3 seeds). EAGLE-3 prior: 2K had the best tok/s; acceptance kept rising to 6K.
- 100K (D6) only with ≥1.5 epochs and after downloading the r0002-100k math/code/stem trajectories.
- Gate G3, a decision point:
  - Generator > static LoRA beyond the noise floor → continue to P4.
  - Otherwise → diagnostics first: conditioning controls, train-K=1, loss variants. Then report.

### P4 — Ablations (remote) → answers Q3

Phase-ordered table in `GUIDE_CORRECTED.md` §5, one variable per arm, logged to the experiment ledger (`EXPERIMENTS.md`, created at P2).

### P5 — Serving and efficiency (remote) → completes Q1, Q4

- vLLM DFlash generator patch: merged per request, 5 layers. Measure generator + merge latency separately and report net speedup at C1, C4 and C8.
- Compare with the fork's DeepSeek EAGLE-3 generator under the same suite and session.

## Rigor rules (apply to every run)

- One variable per arm. The same data, splits, seeds, token budget, updates and eval set across matched arms.
- Log per run: git SHA (no `"unknown"` fallback), full resolved config, seed, pip freeze, GPU/driver, data manifest hashes, checkpoint SHA256. Use `--provenance-dir` on every `launch_vllm.py` run.
- Never compare runs that differ in an unlogged way. Interleave serving repeats (A B C A B C …).
- Report the mean ± std over seeds per domain, and τ plus per-position acceptance. Throughput is reported net of generator latency.
- No bare `try/except`, no silent defaults; fail loudly.

## Rough resource notes (to be replaced by P2 measurements)

- Trainer GPU holds:
  - the frozen DFlash drafter (~1.0 B trainable-equivalent + ~0.6–1.3 B frozen embed/heads, bf16);
  - the generator (~0.44 B, QKVO shared heads; fp32 master + Adam ≈ 7 GB);
  - activations for 512 anchors × 16 queries per row.

  This fits easily on an H100 80 GB.
- Extractor GPU: vLLM Qwen3-8B. Host RAM is ~120 GiB for the activation cache: 5 aux + final layer ≈ 48 KB/token, ≈ 320 MiB per 8K-token example.
- Local RTX 4060 (8 GB): only tiny-config tests.
