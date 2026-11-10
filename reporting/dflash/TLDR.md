# DFlash per-prompt LoRA generator: TL;DR of the 4K results

*2026-11-10. Drafter `z-lab/Qwen3-8B-DFlash-b16`, target Qwen3-8B. Every run trained on the same 4K pool (≈ 1,100 rows per category: math, code, stem, chat) for 1,500 updates, ≈ 6.75 epochs, one seed. Per-experiment details are in this folder's other reports.*

## Update: 10K-math comparison (2026-11-10, [MATH10K.md](MATH10K.md))

Static LoRA vs a small generator (1.1B, decoder width 256), both trained on 10,000 math rows for 3 epochs. Evaluated on new leak-checked sets: 500-row test (200 math + 100 code / stem / chat) and 240-row OOD.

| Test (500 rows) | Base | Static LoRA | Generator |
|---|---|---|---|
| τ (category mean) | 7.25 | 7.39 | 7.39 |
| acceptance rate (category mean) | 41.7% | 42.6% | 42.6% |
| accepted draft tokens per 100 blocks (pooled) | 721 | 737 | 738 |
| math τ | 10.02 | 10.22 | 10.24 |
| OOD τ (category mean) | 5.94 | **6.14** | 6.09 |

- **Both arms beat base in every category** (CIs exclude zero).
- **Generator vs static:** tied in distribution (Δτ −0.002, CI [−0.017, +0.012]) and worse OOD (−0.051, CI [−0.092, −0.010]).
- **No overfitting at 3 epochs, but no dynamism either.** The generated LoRAs are almost identical across rows (cross-row cosine 0.965, effective rank 1.04), so the generator behaves like a 1.1B-parameter static LoRA.

## Bottom line (4K ablations)

- **On the 4K pool no generator beats a plain static LoRA.** The static LoRA (4.3M parameters) is the best arm on validation, the held-out test split and the OOD set.
- **Every generator overfits.** It peaks around update 250, about as high as the static, then declines to roughly base level by update 1,500. Across ≈ 25 generator variants the differences are small, about ±0.005 full_acc on 200 rows, and come from single seeds, so most are within noise.
- **Generated LoRAs are row-specific but that does not help.** In every generator arm, a fixed LoRA (one row's, used for all) or another category's LoRA scores at least as well as the row's own (M7 donor control).
- **The generator can be much smaller at no measured cost.** Shared heads give 0.45B and decoder width 256 gives 1.1B, vs 2.2B. Large variants (MLP or all modules, rank 64, decoder 1024) end lower.

## Metrics

All measured offline: the drafter reads stored target hidden states with the true preceding tokens (teacher forcing). Matching is greedy (argmax). Each anchor drafts a block of 16 tokens, 15 of them draft slots.

- **full_acc:** share of draft slots whose top token equals the target's top token. Each slot is judged on its own.
- **τ (accepted length):** leading run of correct slots per block, plus the verifier's bonus token, averaged over blocks. This is the speed-up proxy.
- **Acceptance rate:** (τ − 1) / 15, the average fraction of the 15 drafted tokens per block that would be accepted. It is an offline greedy estimate, not a vLLM serving measurement; serving (P5) has not been run.

## Baselines (M0): base drafter vs static LoRA vs generator

Final checkpoints, LoRA coefficient 2 (= α/r, the training scale). Validation is the 200 fixed rows (50 per category). The test split is held out and was evaluated once. OOD is 150 rows: HumanEval, math_reasoning, summarization.

| Split | Metric | Base drafter | Static LoRA | Generator |
|---|---|---|---|---|
| Validation | full_acc | 0.5246 | **0.5304** | 0.5220 |
| | τ | 7.108 | **7.255** | 7.129 |
| | acceptance rate | 40.7% | **41.7%** | 40.9% |
| Test | full_acc | 0.5258 | **0.5317** | 0.5259 |
| | τ | 7.221 | **7.382** | 7.267 |
| | acceptance rate | 41.5% | **42.5%** | 41.8% |
| OOD | full_acc | 0.4494 | **0.4579** | 0.4481 |
| | τ | 5.971 | **6.153** | 6.012 |
| | acceptance rate | 33.1% | **34.4%** | 33.4% |

Test split by category (τ): math 10.23 / **10.41** / 10.20 · code 7.56 / **7.66** / 7.57 · stem 6.54 / **6.77** / 6.68 · chat 4.55 / **4.68** / 4.62 (base / static / generator). OOD (τ): HumanEval 7.82 / **7.98** / 7.79 · math_reasoning 7.38 / **7.65** / 7.42 · summarization 2.72 / 2.83 / 2.83.

Static vs base is worth about +0.15 τ, i.e. one more accepted token every ≈ 7 blocks.

## All ablations (validation, 200 rows)

The **final** columns are the final checkpoint at the training coefficient (2). **@ s=1** is the same checkpoint with the LoRA scaled to half at inference (M4); the generator is usually best there. † density arms are validated on a different anchor sample (unpaired with the rest). ‡ M6 was trained with K rows per LoRA and is shown in the one-row-per-LoRA view.

| Arm | params | full_acc | τ | accept rate | peak full_acc (update) | full_acc @ s=1 | τ @ s=1 | accept rate @ s=1 |
|---|---|---|---|---|---|---|---|---|
| Base drafter (no LoRA) | — | 0.5246 | 7.108 | 40.7% | — | — | — | — |
| M0 static LoRA (q/k/v/o, r32) | 4.3M | 0.5304 | 7.255 | 41.7% | 0.5307 (1250) | 0.5300 | 7.235 | 41.6% |
| M0 generator (q/k/v/o, r32) | 2.20B | 0.5220 | 7.129 | 40.9% | 0.5289 (250) | 0.5289 | 7.224 | 41.5% |
| M1 static MLP | 7.9M | 0.5294 | 7.251 | 41.7% | 0.5303 (250) | 0.5300 | 7.230 | 41.5% |
| M1 static all | 12.1M | 0.5297 | 7.244 | 41.6% | 0.5305 (250) | 0.5306 | 7.236 | 41.6% |
| M1 generator MLP | 4.05B | 0.5170 | 7.046 | 40.3% | 0.5277 (250) | 0.5271 | 7.195 | 41.3% |
| M1 generator all | 6.23B | 0.5179 | 7.038 | 40.3% | 0.5285 (250) | 0.5286 | 7.209 | 41.4% |
| M2 generator r8 | 0.56B | 0.5239 | 7.176 | 41.2% | 0.5291 (250) | 0.5291 | 7.234 | 41.6% |
| M2 generator r16 | 1.10B | 0.5225 | 7.125 | 40.8% | 0.5286 (250) | 0.5293 | 7.216 | 41.4% |
| M2 generator r64 | 4.38B | 0.5198 | 7.080 | 40.5% | 0.5281 (250) | 0.5288 | 7.212 | 41.4% |
| M5 generator KL | 2.20B | 0.5247 | 7.167 | 41.1% | 0.5288 (250) | 0.5293 | 7.223 | 41.5% |
| M5 generator lk_hybrid | 2.20B | 0.5247 | 7.167 | 41.1% | 0.5292 (250) | 0.5296 | 7.219 | 41.5% |
| M5 generator CE exp-decay γ7 | 2.20B | 0.5165 | 6.971 | 39.8% | 0.5262 (250) | 0.5267 | 7.139 | 40.9% |
| D1 generator K/V context only | 2.20B | 0.5221 | 7.105 | 40.7% | 0.5282 (250) | 0.5293 | 7.227 | 41.5% |
| D1 generator K/V block only | 2.20B | 0.5233 | 7.156 | 41.0% | 0.5290 (250) | 0.5294 | 7.236 | 41.6% |
| D2 static layer 0 | 0.9M | 0.5269 | 7.188 | 41.3% | 0.5272 (1000) | 0.5284 | 7.186 | 41.2% |
| D2 static layer 4 | 0.9M | 0.5258 | 7.157 | 41.0% | 0.5260 (500) | 0.5261 | 7.156 | 41.0% |
| D2 generator layer 0 | 0.45B | 0.5233 | 7.157 | 41.0% | 0.5260 (1000) | 0.5289 | 7.229 | 41.5% |
| D2 generator layer 4 | 0.45B | 0.5239 | 7.115 | 40.8% | 0.5261 (250) | 0.5271 | 7.179 | 41.2% |
| D3 generator shared heads | 0.45B | 0.5227 | 7.139 | 40.9% | 0.5288 (250) | 0.5294 | 7.234 | 41.6% |
| D4 generator + fc | 2.60B | 0.5210 | 7.100 | 40.7% | 0.5286 (250) | 0.5285 | 7.218 | 41.5% |
| D5 generator α 0.25 | 2.20B | 0.5220 | 7.177 | 41.2% | 0.5262 (500) | 0.5291 | 7.257 | 41.7% |
| D5 generator α 1 | 2.20B | 0.5131 | 6.898 | 39.3% | 0.5255 (250) | 0.5258 | 7.148 | 41.0% |
| D5 generator 256 anchors † | 2.20B | 0.5226 | 7.151 | 41.0% | 0.5274 (250) | 0.5283 | 7.215 | 41.4% |
| D5 generator 1024 anchors † | 2.20B | 0.5172 | 7.045 | 40.3% | 0.5257 (250) | 0.5270 | 7.196 | 41.3% |
| M3 generator decoder 256 | 1.11B | 0.5223 | 7.134 | 40.9% | 0.5285 (250) | 0.5296 | 7.252 | 41.7% |
| M3 generator decoder 1024 | 4.38B | 0.5193 | 7.094 | 40.6% | 0.5274 (500) | 0.5290 | 7.228 | 41.5% |
| M6 generator K=2 ‡ | 2.20B | 0.5222 | 7.114 | 40.8% | — | 0.5299 | 7.239 | 41.6% |
| M6 generator K=4 ‡ | 2.20B | 0.5248 | 7.165 | 41.1% | — | 0.5298 | 7.239 | 41.6% |
| M6 generator K=8 ‡ | 2.20B | 0.5239 | 7.155 | 41.0% | — | 0.5301 | 7.243 | 41.6% |

## What each ablation says

| Experiment | Question | Answer on 4K |
|---|---|---|
| M1 targets | attention vs MLP vs all modules | Static: all three ≈ equal (0.529–0.530). Generator: more modules end lower and cost 2× latency. |
| M2 rank | 8 / 16 / 32 / 64 | Higher rank ends lower (0.5239 → 0.5198); all peak ≈ 0.529. |
| M4 coefficient | LoRA scale at inference | Generator best at half scale (s = 1); static best at its training scale (s = 2). Large scales collapse both. |
| M5 loss | CE+D-PACE / KL / lk_hybrid / CE exp-decay γ7 | KL and lk_hybrid overfit least (end at base level); γ7 worst. |
| M6 conditioning K | 2 / 4 / 8 rows per LoRA | Slightly better than K = 1 in the one-row view (0.530 at s = 1); still ≤ static. |
| M7 donors | own vs fixed vs other-category LoRA | In every generator arm, the better of fixed / other-category ≥ own LoRA (closest: M6 K = 4, own 0.5150 vs fixed 0.5138 / other 0.5152). |
| M3 width | decoder 256 / 512 / 1024 | Doesn't matter; 256 matches 512 at half the parameters. |
| D1 K/V path | both / context / block | No difference; keep the mergeable "both". |
| D2 layers | all 5 / layer 0 / layer 4 | Static: all 5 layers best. One-layer generators overfit less but stay below their statics. |
| D3 heads | per layer vs shared across layers | Same results at 1/5 of the parameters (0.45B). |
| D4 fc adapter | off / on | No gain, +0.4B parameters. |
| D5 α, anchors | D-PACE α 0.25 / 0.5 / 1; 256 / 512 / 1024 anchors | Keep α 0.5 and 512 anchors; α 1 is clearly worse. |
| D13 similarity | are generated LoRAs dynamic? | Yes: row-specific and category-separable (centroid accuracy ≈ 0.96). |
| D14 cost | generator overhead | 2.2B parameters, ≈ 4.5 ms per LoRA, merge ≈ 0.7 ms (attention, rank 32). |

## Caveats

- Single seed, 200 validation rows, 4K training rows. Differences under ≈ 0.005 full_acc / 0.05 τ should not be read as real without seeds or confidence intervals.
- Offline, greedy, teacher-forced metrics. The acceptance rate is derived from τ and is not a serving measurement.
- M8 (conditioning representation) was not run because it needs code.

## Next

A 10K-math comparison of static LoRA vs a small generator: rank 32, coefficient 2, 3 epochs. It evaluates on 200 math + 100 each code / stem / chat rows plus a doubled OOD set, with confidence intervals. More data per epoch is the direct test of the overfitting seen here.

## Where things are

Models: `erdem-erdem/dyn-spec-dflash-<run name>` on Hugging Face (public). Store: `erdem-erdem/dyn-spec-dflash-store-qwen3-8b-4k`. OOD set: `erdem-erdem/dyn-spec-dflash-ood-v2`. Training curves are in W&B project `dyn-spec-dflash-v1`. Per-experiment reports are in this folder; the run log is `dflash_port/RUNLOG.md`.
