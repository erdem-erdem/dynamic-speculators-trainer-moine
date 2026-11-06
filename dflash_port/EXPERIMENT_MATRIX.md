# Experiment matrix: EAGLE-3 ↔ DFlash parallel study + DFlash-specific ablations

Status: **draft, 2026-11-06, awaiting user confirmation** (user asked: parallel EAGLE-3/DFlash coverage of every axis explored for EAGLE-3, plus a few DFlash-only ablations; no norm-gain adapters (A7); match the EAGLE-3 recipe).
Sources: `notes/04` (GEN-001..013 catalogue), `notes/07` §A (presentation slides 2–19), `examples/eagle3-lora/**`.

## 0. Shared protocol (every arm, both families)

- Frozen bases: EAGLE-3 = `deepseek-ai/eagle3_qwen3_8b_ttt7` (same 5 aux layers `[2,10,18,26,34]`+36 as DFlash); DFlash = `z-lab/Qwen3-8B-DFlash-b16`. Target Qwen3-8B@`b968826d`.
- Recipe = the DeepSeek ttt7 generator recipe (GEN-012): rank 32, α 64 (scale 2), dropout 0.05, AdamW 1e-4, wd 0.01, clip 0.85, warmup 100, token budget 8192, cohorts 1–8, seed list, conditioning `last`, generator 512/128/64/32/512/2. Only family-native differences: loss (EAGLE-3 `lk_hybrid` TTT; DFlash CE+D-PACE), number of adapted layers (1 vs 5), param count. All logged.
- Data: r0001 trajectories (4K pool = 1K/category, 1,500 updates) unless the axis is data scale; same cohort schedule per seed for both families (asserted by test).
- Metrics: offline per-position acceptance + τ per domain (math/code/stem/chat) every 250 updates; serving at C1 (+C2/4/8 for headline) on ID + OOD categories; generator latency; params.
- Seeds: **headline (M0) 3 seeds**; sweep arms 1 seed, plus a 2nd seed for any arm whose delta vs reference is < 2× the M0 seed std or which changes a decision.
- Every generator arm that changes capacity/targets has a **matched static LoRA** (same targets, rank, data, updates).

## 1. Parallel matrix (both families, identical settings)

| ID | Axis | Values (★ = reference) | EAGLE-3 precedent | Static control | Eval |
|---|---|---|---|---|---|
| M0 | Headline | base / static LoRA / generator ★ | GEN-010/012 | — (is one) | offline + C1/2/4/8, 3 seeds |
| M1 | Target modules | v/o · q/k/v/o ★ · MLP · all | GEN-009/010 | yes, per set | offline; serving for qkvo & all |
| M2 | LoRA rank | 8 · 16 · 32 ★ · 64 | GEN-005 (slides 4, 9) | yes, at 8 and 64 | offline + C1 |
| M3 | Generator architecture | decoder_width 256/512★/1024 · encoder_hidden 256/512★/1024 · residual_blocks 1/2★/4 · condition_dim 32/64★/128 | GEN-009 (slides 10–13) | n/a | offline; C1 for decoder 256 (latency arm) |
| M4 | LoRA coefficient at inference | s = 0.5 · 2 ★ · 4 · 8 | GEN-008 (slide 8) | n/a | inference-only on the M0 checkpoint |
| M5 | Training loss | EAGLE-3: lk_hybrid ★ / CE / KL · DFlash: CE+D-PACE ★ / CE+exp-decay γ7 / KL / lk_hybrid | GEN-006 (slide 6) | n/a | offline |
| M6 | Conditioning size K | 1 · 2 · 4 ★ · 8 | BATCH-002 | yes (one) | offline + C1 |
| M7 | Conditioning controls | correct ★ · fixed donor · cross-category donor · **same-category random donor** (never run before) | ablations_v4 (slide 5) | n/a | inference-only on M0 checkpoint |
| M8 | Conditioning representation | `last` ★ · `last_mean` | GEN early math study | n/a | offline |
| M9 | Data scale @1,500 upd | pool 4K ★ · 8K · 16K | GEN-001 | yes (4K) | offline |
| M10 | Updates @24K | 2K · 4K · 6K checkpoints of one run | GEN-011 (slides 16–17) | yes @2K | offline + C1 |
| M11 | 100K pool | 100K with ≥ 1.5 epochs (≈ 8.3K updates) vs 24K/2K | GEN-013 (slide 19) | — | offline + C1 (data present on remote) |
| M12 | Concurrency | C1/2/4/8 for M0 arms | BATCH-001 | yes | serving |

Not repeated (negative or retired for EAGLE-3, listed so the omission is deliberate): bank warmup (GEN-004), feedback continuation (GEN-007), prefix refresh / replay (GEN-006). Re-add on request.

## 2. DFlash-only ablations (extra, 1 seed, offline first)

| ID | Axis | Values (★ = default) | Why |
|---|---|---|---|
| D1 | K/V LoRA path | both ★ · context-only · block-only | where the adaptation acts: reading the target vs denoising the block; only `both` is vLLM-mergeable |
| D2 | Adapted layers | all 5 ★ · layer 0 only · last only | separates "more adapted layers" from "family" in the M0 pairing (EAGLE-3 adapts 1 layer) |
| D3 | Generator head sharing | per-type (shared across layers) ★ · per-layer | ≈ 0.44B vs ≈ 2.2B generator params |
| D4 | `fc` context projection adapter | off ★ · on | adapt how the 5 target layers are fused |
| D5 | Anchor density / D-PACE α | 512/8192 ★ vs 256, 1024 · α 0.5 ★ vs 0.25, 1.0 | DFlash training-signal knobs |
| D6 | Draft length at serving | block 16 ★ vs fewer speculative tokens (e.g. 8) | throughput/acceptance trade-off, inference-only |

## 3. Dependencies and order

1. Code (S2–S7) must make every axis above a **config value**: generic target sets (any linear names × layers, `fc`), rank, architecture, loss set per family, K, conditioning representation, inference-time scale and donor overrides in offline eval, head sharing, kv path, anchor density.
2. **DFlash serving with a generator needs the vLLM serving port (P5).** Until then DFlash arms have offline metrics only; all "C1 / tok/s" cells for DFlash wait for P5. EAGLE-3 serving works today.
3. Order: P2 smoke → M0 (3 seeds) → M1, M2, M5 → M3, M6–M8 → M9–M11 (largest cost) → D1–D6. Each block is a separate budget approval (STOP gates).
