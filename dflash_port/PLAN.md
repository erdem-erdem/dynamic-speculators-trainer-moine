# DFlash generator port — development / experiment / research plan

Design rationale: `GUIDE_CORRECTED.md` §4–5. Evidence: `FINDINGS.md`. Status: `PROGRESS.md`.
This file is the **execution plan**: phases, gates, where each step runs, and done criteria.
Exact configs and commands are filled in as deliverable 4, after the port compiles.

## Research questions (in priority order)

- **Q1.** Does a per-query generated LoRA raise DFlash acceptance τ and C1 throughput over (a) the base DFlash drafter and (b) a **matched static LoRA**? This is measured net of generator + merge latency.
- **Q2.** Is the gain prompt-specific, or just domain routing? Test with same-category / cross-category / fixed donors.
- **Q3.** Where in DFlash should the LoRA go? Candidates: modules, the K/V context path vs the block path, which layers, `fc`.
- **Q4.** How does the answer compare with EAGLE-3? There is a paired reference: the fork's DeepSeek ttt7 generator uses the *same* conditioning input (`[2,10,18,26,34]`).

## Decisions needed from the user (blocking marked ⛔)

| ID | Decision | Recommendation |
|---|---|---|
| D1 ⛔ | Base DFlash checkpoint | `z-lab/Qwen3-8B-DFlash-b16` (converted). Fallback: RedHat `.dflash` after config check |
| D2 | Primary loss | CE + D-PACE (DFlash default); `lk_hybrid` and γ7-decay as ablations |
| D3 ⛔ | Local smoke environment | Install WSL2 Ubuntu + venv with torch ≥ 2.9 locally, **or** run all smoke tests on the remote box |
| D4 | Sync with upstream speculators | User adds the upstream remote; we check DFlash/vLLM fixes after `e3041dd` before coding |
| D5 | Hub access | Can we read `azeeer/*` trajectories/datasets, or must we regenerate responses (≈ a target-generation job)? |

## Phases

### P0 — Environment and baseline reproduction (remote, ~1 GPU-day)

1. Remote Linux box: venv, `pip install -e '.[dev,generator,lora]'` with torch ≥ 2.9, then vLLM in a **separate** env if needed. Run `pytest tests/unit/generator tests/unit/models -q` → green.
2. vLLM DFlash check: which version has the DFlash proposer, and does it read `dflash_config.target_layer_ids`? Record the version and SHA.
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
- If the result is promising, run the scale-up: 24K/2,000 updates (3 seeds).
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
