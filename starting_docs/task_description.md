# Task: Port the dynamic LoRA generator from EAGLE-3 to DFlash

## Project
This repo (`dynamic-speculators-trainer`) is a fork of `vllm-project/speculators`. It adds a
**generator** (a hypernetwork) that reads a query's target-model hidden states and outputs LoRA
weights for a **frozen speculative-decoding drafter**. The goal is to raise acceptance length and
speedup per query, dynamically. The developer trained this against an EAGLE-3 drafter
(Qwen3-8B target, `RedHatAI/Qwen3-8B-speculator.eagle3`).

**Our goal:** build the same generator setup on a **DFlash** drafter (block-diffusion, parallel
drafting; upstream already ships `DFlashDraftModel` in `src/speculators/models/dflash/`). Then
test whether the generator helps DFlash as it did EAGLE-3, and run a matched ablation suite.
Priority 1 is understanding the work and getting DFlash training and inference running.
Priority 2 is ablations.

## Sources of truth
1. **The repo is your primary source.** Explore it in full before concluding anything:
   - Commit history and the diff vs upstream: add `vllm-project/speculators` as a remote, then run
     `git log --author=AhmedZeer`, `git diff upstream/main...HEAD --stat`, and read the key diffs.
   - `src/speculators/generator/` (model, LoRA injection, losses, campaign stages, `serving/`).
   - `examples/eagle3-lora/` (configs such as `generator_campaign_online.yaml`), `reporting/`,
     `scripts/`, `hs_connectors/`, tests, and any results, logs or metrics files.
   - Upstream DFlash code: `src/speculators/models/dflash/`, the training scripts and the DFlash
     loss/anchor logic.
2. **`<GUIDE.md>` is secondary.** It was written from public Hugging Face artifacts without
   access to the source. Treat its claims (module targets, `lk_hybrid` loss, generator
   dimensions, pipeline stages, ablation list, port plan) as hypotheses to verify. Where the code
   disagrees, the code wins.

## Deliverables (in order)
1. **Findings report.** What the fork actually adds and changes, grouped by commit or feature,
   with file and function references. Cover:
   - How the generator is conditioned, what it outputs, and where the LoRA is injected.
   - The loss and what is frozen.
   - The data pipeline and the campaign stages.
   - How serving and evaluation work.
   - Which experiments and ablations exist, and their actual results.
2. **Corrected `<GUIDE.md>`.** Fix errors in place, mark each claim as verified or still
   unverified, and **rewrite the port plan and ablation plan** based on what you found.
3. **DFlash port**, implemented as an extension, reusing the fork's generator code with minimal
   changes:
   - DFlash-aware LoRA targets, covering the layers, q/k/v/o, MLP and `fc`, and the
     context-vs-block K/V path.
   - DFlash anchored-block loss.
   - Conditioning that reuses the DFlash target layers.
   - Evaluation and serving.
   - A small smoke run before anything large.
4. **Experiment plan**, with exact configs and commands.

Stop and report after deliverables 1–2, and again before any expensive training run.

## Experimental rigor (non-negotiable)
These comparisons will drive decisions, so a silent bug makes them meaningless.
- **Matched arms:** base DFlash vs a static LoRA (same rank, modules and budget) vs the generator.
  Use identical data, splits, seeds, token budgets, update counts and evaluation sets. Change
  exactly one variable per ablation.
- **Sanity checks before trusting any number:**
  - A zero or disabled LoRA reproduces the base drafter's outputs and metrics exactly.
  - Drafter weights are bit-identical after training (truly frozen).
  - Gradients reach only the intended parameters.
  - LoRA shapes, layer and module mapping, and the scaling factor `alpha/r` are correct.
  - Block size, mask token, anchor sampling and attention mask (including sliding window)
    match between training and evaluation.
  - The generator overfits a tiny batch.
- **No leakage:** conditioning examples and eval queries are disjoint from training data;
  per-domain splits are fixed and recorded.
- **Metrics:** report per-position acceptance and expected acceptance length τ per domain
  (math, code, stem, chat). Use multiple seeds and report variance. Report end-to-end speedup
  measured in vLLM/HF, **net of generator latency**, at concurrency 1 and higher.
- **Reproducibility:** log git SHA, config, seed, environment and data hashes for every run.
  Never compare runs that differ in an unlogged way.
- **No silent fallbacks:** no bare `try/except`, no default substitutions that hide failures.
  Fail loudly. If something is ambiguous or unverifiable, say so rather than guess.

In plans, experiment-development plans-processes, use living progress and todo (brief), update them as you go.

Commit to local repo after checkpoints as a principal. Keep everything in local. Never commit remote or push. When time comes, user will ask for command and you'll only give the command to user.
commit identity: erdem-erdem, yusuf.erd.erdem@gmail.com