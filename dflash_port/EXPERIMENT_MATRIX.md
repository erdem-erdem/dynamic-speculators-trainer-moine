# Experiment matrix: DFlash mirror of the EAGLE-3 generator experiments

Status: **agreed scope, 2026-11-06** (user decisions D7–D11 in `PROGRESS.md`). Not yet budgeted; every block is launched only after a STOP/approval.

> **Not a run plan yet (D11, 2026-11-07).** When we reach experiment preparation (S6, before P2), list every experiment below and how it will run (arms, configs, static controls, data, budget/GPU-hours, order, metrics, success criteria) and **discuss it with the user first**. Only then write the recipes/matrices. Rows may be added, merged or dropped in that discussion.
Sources: `notes/04` (GEN-001..013 catalogue), `notes/07` §A (presentation slides 2–19), `examples/eagle3-lora/**`.

## Rules (user, 2026-11-06)

- **Mirror, don't extend.** DFlash repeats the method and every experiment already run for EAGLE-3; nothing new is added on the EAGLE-3 side.
- **No EAGLE-3 runs in this project.** EAGLE-3 numbers come from the existing records (DeepSeek ttt7 base / static / generator = GEN-010/012; Red Hat drafter sweeps = GEN-001..013). Comparisons to them are descriptive (different runs, env drift ≈ 5 %).
- **One seed (42), as EAGLE-3.**
- **Offline evaluation first** (per-position acceptance + τ per domain, `full_acc`, `eal`). Serving (P5, tok/s) is ported later, after the method and training are established.
- DFlash-only ablations are extras.

## 0. Shared protocol

- Frozen base: `z-lab/Qwen3-8B-DFlash-b16`; target Qwen3-8B@`b968826d`; aux layers `[2,10,18,26,34]`+36.
- Recipe mirrors the DeepSeek ttt7 generator recipe (GEN-012): rank 32, α 64 (scale 2), dropout 0.05, AdamW 1e-4, wd 0.01, clip 0.85, warmup 100, token budget 8192, cohorts 1–8, seed 42, conditioning `last`, generator 512/128/64/32/512/2, targets q/k/v/o.
- Family-native differences (logged): loss CE + D-PACE (EAGLE-3: `lk_hybrid` TTT-3), 5 adapted layers (EAGLE-3: 1), per-type generator heads.
- Data: r0001 trajectories, 4K pool, 1,500 updates unless the axis is data scale.
- Arms changing capacity/targets get a matched DFlash static LoRA, as in the EAGLE-3 records.

## 1. Mirrored experiments (DFlash only)

| ID | Axis | Values (★ = reference) | EAGLE-3 record | Static control |
|---|---|---|---|---|
| M0 | Headline | base / static LoRA / generator ★ | GEN-010/012 | — (is one) |
| M1 | Target modules | v/o · q/k/v/o ★ · MLP · all | GEN-009/010 | yes, per set |
| M2 | LoRA rank | 8 · 16 · 32 ★ · 64 | GEN-005 | as in record (none) |
| M3 | Generator architecture | decoder_width 256/512★/1024 · encoder_hidden 256/512★/1024 · residual_blocks 1/2★/4 · condition_dim 32/64★/128 | GEN-009/010 | n/a |
| M4 | LoRA coefficient at inference | s = 0.5 · 2 ★ · 4 · 8 | GEN-008 | n/a (M0 checkpoint) |
| M5 | Training loss | CE+D-PACE ★ · CE+exp-decay γ7 · KL · lk_hybrid (EAGLE-3 record: LK / CE / KL) | GEN-006 | n/a |
| M6 | Conditioning size K | 1 · 2 · 4 ★ · 8 | BATCH-002 | yes (one) |
| M7 | Conditioning controls | correct ★ · fixed donor · cross-category donor | ablations_v4 | n/a (M0 checkpoint) |
| M8 | Conditioning representation | `last` ★ · `last_mean` | early math study | n/a |
| M9 | Data scale @1,500 upd | pool 4K ★ · 8K · 16K | GEN-001 | yes (4K) |
| M10 | Updates @24K | 2K · 4K · 6K checkpoints of one run | GEN-011 | yes @2K |
| M11 | 100K pool | 100K / 2K updates (record) + ≥ 1.5 epochs (≈ 8.3K) | GEN-013 | — |

**Flagged for possible later addition (not in scope now):**
- bank warmup (GEN-004), feedback continuation (GEN-007), prefix refresh / replay (GEN-006): negative or retired for EAGLE-3;
- same-category random donor control (configured for EAGLE-3 in ablations_v4, never run);
- learning-rate / dropout sweep (never run for EAGLE-3);
- concurrency C1/2/4/8 and all tok/s cells (need P5 serving).

## 2. DFlash-only ablations (extra)

| ID | Axis | Values (★ = default) | Why |
|---|---|---|---|
| D1 | K/V LoRA path | both ★ · context-only · block-only | adapt reading of the target vs denoising of the block; only `both` is vLLM-mergeable |
| D2 | Adapted layers | all 5 ★ · layer 0 only · last only | separates "more adapted layers" from "family" (EAGLE-3 adapts 1 layer) |
| D3 | Generator head sharing | per-type ★ · per-layer | ≈ 0.44B vs ≈ 2.2B generator params |
| D4 | `fc` context-projection adapter | off ★ · on | adapt how the 5 target layers are fused |
| D5 | Anchor density / D-PACE α | 512/8192 ★ vs 256, 1024 · α 0.5 ★ vs 0.25, 1.0 | DFlash training-signal knobs |
| D6 | Draft length | block 16 ★ vs fewer tokens (e.g. 8) | inference-only; offline per-position acceptance shows it, tok/s needs P5 |

**Flagged — important later addition (user 2026-11-06):** D7 *training block size* — train/evaluate the generator with a block size other than the checkpoint's 16 (e.g. 8, 32). Needs an explicit, identity-recorded `block_drafter.block_size` override (no silent default; raises if it disagrees with the checkpoint unless set). Different checkpoints (e.g. a b8 DFlash) already work with no code change, since block size is read from the drafter config. D6 (fewer tokens at inference) is separate.

## 3. Code requirements (S2–S7) and order

1. Every axis is a config value: generic target sets (linear names × layers, `fc`), rank, architecture, per-family loss set, K, conditioning representation, inference-time scale and donor overrides in offline eval, head sharing, kv path, anchor density.
2. Proposed order (to be confirmed in the D11 discussion): P2 smoke → M0 → M1, M2, M5 → M3, M4, M6–M8 → M9–M11 (largest cost) → D1–D6.

## 4. D11 discussion log

### 2026-11-07, round 1 (user)
- **Drafter-agnostic (D12).** The generator must work for any released DFlash drafter, not only `z-lab/Qwen3-8B-DFlash-b16`. Block size, layers, mask id and aux layers already come from the drafter config, checked against the bank. To do:
  - drop z-lab-specific wording from recipes and docs;
  - a compatibility smoke on a second DFlash checkpoint at P0;
  - activations are per drafter, keyed by its aux layer ids.
- **LoRA similarity / dynamism analysis (D13).** Required, mirroring GEN-005 (rank 8/16/32/64: cos within vs between category, dyn_ratio, effective rank, category accuracy). The EAGLE-3 analysis script and metric formulas are not in the repo, so we define our own:
  - computed on ΔW = s·BA per module, exactly via r×r Grams;
  - run on every generator checkpoint, plus static LoRA as the zero-dynamism reference.
- **LoRA coefficient (M4).** Offline sweep: the scale is applied in offline evaluation (no training); tok/s at P5. EAGLE-3's sweep was serving-only (s = 0.5/2/4/8).
- **Rank cost (M2).** Report generator latency (ms per call), parameters and LoRA merge cost next to offline acceptance. tok/s at P5. Undecided (user).
- **Architecture.** Mirror the important EAGLE-3 ablations; not every axis is needed. Priority: target sets (attn ★ / FFN / all, each with a static), then decoder width 256/512/1024. Encoder/condition/residual only if needed.
- **Order (user).** Main runs → small ablations → architecture → scale (24K, 100K; both runnable) → evals.
- **Offline activations.** Requested as an option alongside online extraction, so many runs share one activation store (see round 2).

### 2026-11-07, round 2 (user)
- **Architecture axes are tested on DFlash ourselves.** EAGLE-3 conclusions only set priority and defaults. Decoder width ★ 512 (the presentation slide, "512 > 256", is more recent than GEN-009/010).
- **D2 gets matched static LoRAs** (layer 0 only, last only).
- **Scale strategy.** 4K and 24K are for searching (feasibility + performance) with **offline activation stores** shared by all runs. **100K is the target:** the best configuration is scaled to it afterwards, online or offline; there is disk for a 100K store. The 100K data is local and complete: `datasets/eagle3-qwen3-8b-trajectories-v1/releases/r0002-100k/{chat,code,math,stem}`, 25 subsets each, 3.8 GB. This corrects notes/07, which said "chat only".
- Still open: rank cost (point 4).

## 5. Execution tiers (agreed order; each tier launched only after a STOP/approval)

| Tier | Content | Training runs (≈) |
|---|---|---|
| T-setup | P0: convert/check the drafter (+ second DFlash checkpoint, D12); extract the 4K store; online ≡ offline check; P2 smoke | smoke only |
| 1 Main | M0 base / static / generator @4K-1500 + offline extras on that checkpoint: M4 coefficient sweep, M7 donor controls, D13 similarity | 2 |
| 2 Main ablations | M1 target sets attn★/FFN/all (+ statics) · M2 rank 8/16/32★/64 (+ D13 similarity + generator cost) · M5 loss | ≈ 12 |
| 3 Small ablations | M6 K · M8 last_mean · D1 kv path · D4 fc · D3 head sharing · D5 anchors/α · D2 layers (+ statics) | ≈ 16 |
| 4 Architecture | decoder width 256/512★/1024, then encoder/condition/residual only if width matters | 2–8 |
| 5 Scale | M9 pool 4K/8K/16K · 24K store, M10 (2K/4K/6K) · then the best configuration on 100K (M11) | ≈ 5 + 100K |
| 6 Evals | final eval suite, D6 draft length; P5 serving tok/s (coefficients, rank cost, concurrency) | 0 |
