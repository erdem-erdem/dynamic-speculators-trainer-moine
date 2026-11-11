"""Step 3: does the drafter use the prompt part of the injected context?

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/analysis/prompt_usage.py \
        --part attention|masking|all

(a) `attention`: attention mass of draft-block queries (slots 1..15 of valid
    blocks) on prompt tokens, response tokens of the injected context, the
    block's anchor token and its mask tokens, per layer and head, for base /
    static / generator, on 200 validation rows (50 per category, rows of at
    most 4,096 tokens so the dense eager weights fit). FlexAttention returns no
    weights, so this pass switches the drafter to the eager attention path
    (`eager_attention_forward` + the float mask of the same mask mod); the
    first rows are also run with flex and compared (metrics and logits).
(b) `masking`: τ / full_acc on all validation rows with the prompt removed
    from the injected context (an extra flex mask term: block queries never
    attend to prompt positions), and two controls removing as many response
    tokens (n = min(prompt length, response length / 2)): random response
    positions, and the earliest response positions. For each arm.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention.flex_attention import and_masks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KEEP = {"mask": None}
MAX_ATTENTION_LEN = 4096
ATTN_ROWS_PER_CATEGORY = 50
EQUIV_ROWS = 5


def patch_mask_mod():
    import speculators.models.dflash.core as core  # noqa: PLC0415

    original = core.create_anchor_block_mask_mod

    def patched(document_ids, total_seq_len, anchor_positions, block_size, **kw):
        mod, q_len, kv_len = original(
            document_ids, total_seq_len, anchor_positions, block_size, **kw)
        keep = KEEP["mask"]
        if keep is None:
            return mod, q_len, kv_len
        if keep.numel() != total_seq_len:
            raise ValueError(f"keep mask {keep.numel()} != {total_seq_len}")
        keep = keep.to(document_ids.device)

        def keep_mod(_b, _h, _q, kv_idx):
            return (kv_idx >= total_seq_len) | keep[torch.remainder(kv_idx, total_seq_len)]

        return and_masks(mod, keep_mod), q_len, kv_len

    core.create_anchor_block_mask_mod = patched


def set_attention(drafter, impl):
    from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401,PLC0415

    import speculators.models.dflash.core as core  # noqa: PLC0415
    from speculators.models.attention import create_float_mask  # noqa: PLC0415

    drafter.config.transformer_layer_config._attn_implementation = impl  # noqa: SLF001
    drafter._attn_impl = impl  # noqa: SLF001
    drafter._create_mask_fn = (  # noqa: SLF001
        core._compiled_create_block_mask  # noqa: SLF001
        if impl == "simple_flex_attention" else create_float_mask
    )
    for layer in drafter.layers:
        if layer.self_attn.config._attn_implementation != impl:  # noqa: SLF001
            raise RuntimeError("Attention config is not shared with the drafter")


class AttentionTap:
    """Per-head attention mass on [prompt, response ctx, anchor, block masks]
    for valid draft-slot queries, plus mass on the 32 most recent context
    tokens, and the uniform-attention expectation of the prompt mass."""

    def __init__(self, drafter):
        self.layout = None
        self.sums = None
        self.handles = [
            layer.self_attn.register_forward_hook(self._hook(i))
            for i, layer in enumerate(drafter.layers)
        ]
        self.n_layers = len(drafter.layers)

    def set_row(self, total, prompt_len, indices, mask):
        """indices: anchored block indices [Q]; mask: aligned loss mask [Q]."""
        q = indices.numel()
        anchors = indices[:: C.BLOCK].repeat_interleave(C.BLOCK)
        valid = mask.bool()
        positions = torch.arange(total, device=indices.device)
        self.layout = {
            "total": total, "prompt_len": prompt_len, "valid": valid,
            "anchors": anchors, "positions": positions, "q": q,
        }

    def _hook(self, layer):
        def hook(_module, _args, output):
            if self.layout is None:
                return
            weights = output[1]  # [1, H, Q, KV]
            if weights is None:
                raise RuntimeError("Attention returned no weights (not eager?)")
            lay = self.layout
            total = lay["total"]
            w = weights[0][:, lay["valid"]].float()  # H, Qv, KV
            base = w[..., :total]
            block = w[..., total:]
            pos = lay["positions"]
            prompt_cols = pos < lay["prompt_len"]
            anchors = lay["anchors"][lay["valid"]]  # Qv
            q_index = torch.nonzero(lay["valid"])[:, 0]
            block_id = q_index // C.BLOCK
            # own-block columns: block j occupies [j*16, (j+1)*16) of the block part
            own = block.view(block.shape[0], block.shape[1], -1, C.BLOCK)
            own = own[:, torch.arange(own.shape[1]), block_id]  # H, Qv, 16
            recent = (pos[None, :] < anchors[:, None]) & (
                pos[None, :] >= anchors[:, None] - 32) & (~prompt_cols)[None, :]
            stats = torch.stack([
                base[..., prompt_cols].sum(-1),
                base[..., ~prompt_cols].sum(-1),
                own[..., 0],
                own[..., 1:].sum(-1),
                (base * recent[None]).sum(-1),
                base[..., :4].sum(-1),  # first 4 prompt tokens (sink candidates)
            ], -1).sum(1)  # H, 6
            visible = torch.minimum(anchors, torch.tensor(total, device=anchors.device))
            uniform_prompt = (
                torch.clamp(torch.full_like(visible, lay["prompt_len"]), max=visible)
                .float() / (visible.float() + C.BLOCK)
            ).sum()
            self.sums[layer] += stats.double()
            self.uniform[layer] += float(uniform_prompt)
            self.queries[layer] += int(q_index.numel())
        return hook

    def reset(self, heads, device):
        self.sums = torch.zeros(self.n_layers, heads, 6, dtype=torch.float64,
                                device=device)
        self.uniform = [0.0] * self.n_layers
        self.queries = [0] * self.n_layers

    def result(self):
        return {
            "sums": self.sums.cpu().numpy().tolist(),
            "uniform_prompt_sum": self.uniform,
            "queries": self.queries,
            "fields": ["prompt", "response_ctx", "anchor", "block_masks", "recent32",
                       "prompt_first4"],
        }


def attention_part(runtime, job, arms, out_dir, limit=None):
    drafter = runtime.drafter
    rows = C.split_rows(runtime, job, "validation")
    chosen = []
    for cat in C.CATEGORIES:
        members = [
            (s, r) for s, r in rows
            if C.category(s) == cat
            and len(runtime.corpus.rows[r]["input_ids"]) <= MAX_ATTENTION_LEN
        ]
        chosen += sorted(random.Random(f"attn-{cat}").sample(
            members, min(limit or ATTN_ROWS_PER_CATEGORY, len(members))))
    heads = drafter.config.transformer_layer_config.num_attention_heads
    tap = AttentionTap(drafter)
    equivalence = []
    results = {}
    started = time.time()
    for arm in ("base", "static", "generator"):
        per_cat = {}
        for n, (subset, row) in enumerate(chosen):
            batch = C.load_batch(runtime, row)
            factors = arms.factors(arm, row)
            if n < EQUIV_ROWS:
                set_attention(drafter, "simple_flex_attention")
                flex = C.run(runtime, batch, factors)
                flex_logits = C.CAPTURE["logits"].float().clone()
            set_attention(drafter, "eager")
            cat = C.category(subset)
            if cat not in per_cat:
                tap.reset(heads, runtime.device)
                per_cat[cat] = None
            else:
                tap.sums, tap.uniform, tap.queries = per_cat[cat]
            start = C.prompt_boundary(batch["loss_mask"][0])
            total = batch["input_ids"].shape[1] + C.BLOCK  # family pad
            tap.layout = None
            # The layout needs the anchors, known only inside the forward:
            # run once untapped (cheap) to get them, then tapped.
            eager = C.run(runtime, batch, factors)
            tap.set_row(total, start, C.CAPTURE["indices"], C.CAPTURE["mask"][0])
            C.run(runtime, batch, factors)
            tap.layout = None
            per_cat[cat] = (tap.sums, tap.uniform, tap.queries)
            if n < EQUIV_ROWS:
                diff = (C.CAPTURE["logits"].float() - flex_logits).abs()
                equivalence.append({
                    "arm": arm, "row": row,
                    "flex_eal": flex["eal_sum"] / flex["eal_total"],
                    "eager_eal": eager["eal_sum"] / eager["eal_total"],
                    "flex_acc": flex["full_acc_sum"] / flex["full_acc_total"],
                    "eager_acc": eager["full_acc_sum"] / eager["full_acc_total"],
                    "max_abs_logit_diff": float(diff.max()),
                    "mean_abs_logit_diff": float(diff.mean()),
                    "argmax_agreement": float(
                        (C.CAPTURE["logits"].argmax(-1) == flex_logits.argmax(-1))
                        .float().mean()),
                })
                print(equivalence[-1], flush=True)
        results[arm] = {}
        for cat, (sums, uniform, queries) in per_cat.items():
            tap.sums, tap.uniform, tap.queries = sums, uniform, queries
            results[arm][cat] = tap.result()
        print(f"attention {arm} done {time.time() - started:.0f}s", flush=True)
    set_attention(drafter, "simple_flex_attention")
    C.write_json(out_dir / "attention.json", {
        "rows": chosen, "equivalence": equivalence, "results": results,
        "max_len": MAX_ATTENTION_LEN,
    })


def masking_part(runtime, job, arms, out_dir, limit=None,
                 conditions=("none", "prompt", "resp_random", "resp_early"),
                 name="masking.json"):
    rows = C.split_rows(runtime, job, "validation")[:limit]
    names = ("base", "static", "generator")
    per_row = []
    started = time.time()
    for n, (subset, row) in enumerate(rows):
        batch = C.load_batch(runtime, row)
        length = batch["input_ids"].shape[1]
        total = length + C.BLOCK
        start = C.prompt_boundary(batch["loss_mask"][0])
        response = length - start
        n_mask = min(start, response // 2)
        keeps = {"none": None}
        keep = torch.ones(total, dtype=torch.bool)
        keep[:start] = False
        keeps["prompt"] = keep
        keep = torch.ones(total, dtype=torch.bool)
        picks = random.Random(row).sample(range(start, length), n_mask)
        keep[picks] = False
        keeps["resp_random"] = keep
        keep = torch.ones(total, dtype=torch.bool)
        keep[start : start + n_mask] = False
        keeps["resp_early"] = keep
        # Prompt removed except its first 4 tokens (attention-sink control).
        keep = torch.ones(total, dtype=torch.bool)
        keep[min(4, start):start] = False
        keeps["prompt_keep4"] = keep
        record = {"subset": subset, "row": row, "prompt_len": start,
                  "response_len": response, "n_mask": n_mask}
        for arm in names:
            factors = arms.factors(arm, row)
            for condition in conditions:
                KEEP["mask"] = keeps[condition]
                try:
                    metrics = C.run(runtime, batch, factors)
                finally:
                    KEEP["mask"] = None
                record[f"{arm}|{condition}"] = [
                    metrics["eal_sum"], metrics["eal_total"],
                    metrics["full_acc_sum"], metrics["full_acc_total"],
                ]
        per_row.append(record)
        if n % 25 == 0:
            print(f"masking row {n + 1}/{len(rows)} {time.time() - started:.0f}s",
                  flush=True)
    C.write_json(out_dir / name, {"rows": per_row})
    for arm in names:
        for condition in conditions:
            values = np.array([r[f"{arm}|{condition}"] for r in per_row])
            print(arm, condition, values[:, 0].sum() / values[:, 1].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("attention", "masking", "sink", "all"),
                        default="all")
    parser.add_argument("--limit", type=int, help="smoke test: few rows")
    args = parser.parse_args()
    out_dir = C.OUT / ("prompt" if args.limit is None else "prompt_smoke")
    patch_mask_mod()

    def body(runtime, job, _output):
        C.install_analysis_forward(runtime)
        arms = C.Arms(runtime)
        if args.part in ("masking", "all"):
            masking_part(runtime, job, arms, out_dir, args.limit)
        if args.part == "sink":
            masking_part(runtime, job, arms, out_dir, args.limit,
                         conditions=("none", "prompt_keep4"), name="masking_sink.json")
        if args.part in ("attention", "all"):
            attention_part(runtime, job, arms, out_dir, args.limit)

    C.with_runtime(body)


if __name__ == "__main__":
    main()
