# 07 — Presentation, local trajectory dataset, upstream sync (2026-11-05)

Sources:
- `../../dynamic-speculators-experiments.pptx`: outside the repo, 19 slides by Ahmed Zeer. Its numbers are images; the matching repo records are cited below.
- `datasets/eagle3-qwen3-8b-trajectories-v1/`: a local HF dataset snapshot. It is **git-excluded** via `.git/info/exclude`; never commit it.
- `git log e3041dd..upstream/main`.

## A. Presentation: what it adds or changes

| Slide | Claim | Repo record / check | Effect on the DFlash port |
|---|---|---|---|
| 2 | The direct generator beats the "oracle" LoRA; bank pretraining is ineffective | Legacy workflow (retired) | None: we use direct only |
| 3 | 24K generator vs "LoRA trained on the same amount of data", 4K steps ≈ 3 epochs; ~1.5× speedup | Numbers are the **C2** rows of `benchmarks/2026-10-24-comparison-v2-concurrency/summary.md`. That `trained-lora` is the **V/O LoRA 4K / 1,500 steps** (labelled so in slide 15 and in `extended_historical_table.md`). **Not data/step-matched** [V] | The only matched evidence remains GEN-010 (4K/1,500). Matched arms at every scale stay mandatory |
| 4 | Dynamism (LoRA spread within vs between categories) rises with rank: dyn_ratio 0.34 → 0.52 for r8 → r64 | `GEN-005` | Log the same dynamism metric for DFlash |
| 5 | Donors at step 4000. Math: correct 79.3 / 65.1 % acc; fixed donor 77.2 / 64.9 %; cross-category 70.1 / 54.0 %. Chat: 60.0 / 40.8 %; 54.0 / 35.4 %; 55.2 / 35.6 % | `eagle3_generator_ablations_v4.md` (already in FINDINGS) | Math looks like domain routing; chat looks prompt-specific. Same-category random donor is still never run, so keep it in Phase 0 |
| 6 | Training with CE instead of LK: "no difference". Prompt-probe agreement is CE ≈ LK, and both are > static LoRA. 1 seed, 1k steps, offline only | `GEN-006`, `generator_ablations_v4/training/prompt_ce.yaml` | Supports **D2 = CE + D-PACE** as primary. The `lk_hybrid` arm drops to low priority |
| 7 | Feedback continuation gives no extra benefit | `GEN-006` | Skip |
| 8 | LoRA scale at test time: 2 (the training value) is best; 8 collapses | `GEN-008` | Keep α/r = 2; no scale sweep |
| 9 | Rank 8 / 16 / 32 / 64: acceptance +6.1 / 6.8 / 7.4 / 7.9 pp, tok/s +10.3 / 11.6 / 12.5 / 13.6 % vs base, so throughput is monotone despite generator cost | `GEN-005` (base 36.1 tok/s; the domain/setup is in GEN-005) | Rank is a real lever. For DFlash (20 module instances) the latency side is heavier, so keep the rank arm |
| 10–13 | Decoder width, condition dim, residual blocks: "no change" | `GEN-009` (screenshots) | Keep 512/64/2; decoder 256 is a latency arm |
| 14–15 | Offline: all > attn > MLP. Serving: attn generator > all generator (faster); all-module static LoRA > all generator; attn generator > all-module LoRA on math, but **not on chat** (55.41 vs 57.26) | `GEN-009/010` | Primary targets = attention (QKVO). On chat a static LoRA is as good |
| 16–17 | Attention generator 24K at 2K / 4K / 6K updates: acceptance keeps rising (math 66.9 → 68.8 %), tok/s does not (2K is best: math 82.80). "3 epochs is enough" | `benchmarks/2026-10-31-attention-24k-6k-c1/` | Scale-up arm = 24K / 2,000 updates (≈1.5 epochs); longer only if acceptance gains carry over to tok/s |
| 18 | DeepSeek ttt7 + attention generator 4K/1,500: +6.6 / +11.0 / +16.1 / +15.5 % (math/code/stem/chat) | `GEN-012` (in FINDINGS) | This is the Q4 reference: same 5-layer conditioning |
| 19 | 100K / 2K updates < 24K / 2K, because of only **0.36 epochs** (100K ≈ 5,552 updates/epoch vs 24K ≈ 1,338) | `attention_100k_2k_v1` | A 100K arm only makes sense with ≥ ~1.5 epochs (≈ 8.3K+ updates) |

Name: the presenter is Ahmed Zeer. The GitHub repo is `AhmedZeer/dynamic-speculators-trainer`, and the git author name is `AhmetZeer`. The old claim that "AhmedZeer is wrong" was itself wrong and has been fixed.

## B. Local trajectory dataset `datasets/eagle3-qwen3-8b-trajectories-v1/` (1.0 GB)

It mirrors the Hub dataset `azeeer/eagle3-qwen3-8b-trajectories-v1`. It contains **token trajectories only, no activations**.

| Release | Content locally | Rows |
|---|---|---|
| `r0001` (the "24K") | math, code, stem, chat × subsets 00000–00005 | 6,000 train + 600 val per category = 26,400. Roles: 00000–00004 `bank`, 00005 `adaptation`. Train tokens: chat 7.4 M, code 10.0 M, math 10.8 M, stem 5.3 M = **33.5 M** |
| `r0002-100k` (the "100K") | **chat only** (subsets 00000–00024: 25,000 train + 2,500 val, 30.2 M tokens) | **correction 2026-11-07:** all four categories are local (`releases/r0002-100k/{chat,code,math,stem}`, 25 subsets each, 3.8 GB); the earlier "chat only" observation was wrong |
| `r0001-training` | stem manifests only (no parquet) | — |

- **Parquet columns**: `input_ids` (int32), `loss_mask` (0 on prompt tokens, 1 on response tokens), `seq_len`, `prompt_length`, `primary_id`, `source_ids`, `category`, `subset_id`, `role`, `membership`, `metadata` (sampling params, finish reason), `source_aliases` (licenses), `source_repo`/`revision`.
- **`r0002-100k` is a superset of `r0001`**: same seed-0 partition; chat-00000 row 0 is identical.
- **How the responses were made**: target `Qwen/Qwen3-8B@b968826d…` via plain `vllm serve` (`launch_vllm.py responses`), on prompts from `openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated@789a044d…`. Settings: T 0.7, top-p 0.8, top-k 20, seed 0, **thinking off**, max 4096 new tokens, context 8192.
  - The drafter name in the manifest (`RedHatAI/Qwen3-8B-speculator.eagle3`) is metadata only. Responses do not depend on any drafter.
- **Activations (`azeeer/eagle3-qwen3-8b-activations-v1`, ≈1.24 TB, never published in full)**:
  - Only `hidden.json` pointers exist, and only for `r0001`. They were produced by a vLLM prefill-only replay of the saved token ids with the hidden-state connector (`--target-layer-ids 2 18 33 --include-last-layer`).
  - Layers **`[2, 18, 33, 36]`**, bf16: 4 × 4096 × 2 B = 32 KiB/token × ~37 M tokens (train + val) ≈ 1.2 TB. This matches the 1.24 TB figure.

**Answers for the DFlash port**
1. **Reuse the trajectories: yes.** DFlash trains on target-generated responses with a response loss mask, which is exactly what this is. The DeepSeek recipe already reuses them via `trajectory_recipe`. Conditions: the same target revision `b968826d…`, and serving/eval in non-thinking mode to match. Whether z-lab trained with thinking on is [U]; it does not block us.
2. **Reuse the activations: no.**
   - z-lab DFlash-b16 needs **`[2,10,18,26,34]` + 36**. The EAGLE-3 dump has 2 and 18 (and 36) but lacks 10, 26 and 34, so neither the drafter's `fc` (20,480-d input) nor the 5-layer generator conditioning can be built from it.
   - Even a complete DFlash dump would be 6 × 8 KiB = 48 KiB/token: ≈ 1.7 TB for 24K and ≈ 6.5 TB for 100K.
3. **Don't download or pre-generate activations.**
   - The fork's current pipeline (`activation_source: vllm`) never reads dumps. A co-located vLLM extractor replays the saved token ids (prefill only, no decoding) at the needed layers, and the trainer caches them in CPU RAM.
   - We run it at `--target-layer-ids 2 10 18 26 34 --include-last-layer`.
   - The extractor cost and the RAM-cache hit rate are measured in P2.
4. **Download to the remote box**:
   - `r0001` (needed for 4K and 24K);
   - the `r0002-100k` math/code/stem trajectories, **only if** a 100K arm is approved. They are not needed for P0–P3 base.

   - The local snapshot already has the layout the loader expects: `<trajectories_path>/releases/<release>/<cat>/data.json`, plus the parquet files listed there. See `local_campaign.py:50-103`, which checks the file size against `data.json`.
   - The loader also checks that the manifest's `hidden_states` and model IDs equal the bank recipe. That check is why the DeepSeek recipe points `trajectory_recipe` at the original EAGLE-3 recipe.
   - So either copy `datasets/eagle3-qwen3-8b-trajectories-v1/` to the remote and set `trajectories_path` to it, or run, after `hf auth login`:
     `hf download azeeer/eagle3-qwen3-8b-trajectories-v1 --repo-type dataset --include "releases/r0001/*" --local-dir trajectories`
     Pin `--revision` to the one in the recipe's `hub.trajectories_revision`.
5. **Layer-index convention (verify in P0)**: the fork documents extraction ids `[2,10,18,26,34]` = outputs of zero-based decoder layers `[1,9,17,25,33]` (the HF `hidden_states` index, where 0 = embeddings). z-lab's config lists `target_layer_ids [1,9,17,25,33]` [R].
   - Test: on one example, compare the extracted features with an HF forward (`output_hidden_states=True`, index `id`), max-abs diff at bf16 tolerance.

## C. Upstream sync (`upstream` remote added by the user)

- `e3041dd..upstream/main` (`9e5508e`): 82 commits.
  - `models/dflash/` changed only via the config refactor `8995690f`. **No DFlash model, loss or mask changes.**
  - 31 files modified by the fork are also changed upstream, so **do not merge**; it risks conflicts with no DFlash benefit.
- Cherry-pick candidates, only if we hit the issue:
  - `e8f88793`: pass the training seed to the batch sampler. Affects upstream DFlash base-training reproducibility, not the generator sampler.
  - `d3c3306c`: final layer in target-layer-ids.
  - `c0df87b7`: provenance records a failed git diff instead of a clean tree. This is one of our silent-fallback items.
  - `bc954015`: frozen weights kept in autocast dtype (perf).
- Not relevant: XPress (`261a82dd`), DFlash2 selector (`48692623`), async dataloader refactor (`7ed657f8`, large).
- vLLM DFlash support: confirmed by the user (2026-11-05). Record the exact version and commit in P0.
