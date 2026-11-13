"""T0: loss gradients and Fisher traces per weight site, both starting points.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dflash_port/perturbation/t0_grads.py \
        --point base|static [--rows-per-category 1000] [--slot-rows-per-category 50]

Loss: the 24K training loss (KL to the verifier + D-PACE position weights,
`block_drafter` settings of the recipe), one row at a time, validation-style
fixed anchors, eval mode. Gradients w.r.t. an additive zero perturbation at
every site (= the gradient w.r.t. the site's effective weight); K and V get
one leaf per path (context / block), the full K/V gradient is their sum.

Rows: a seeded sample of 24K *training* rows per category (never eval rows).

Writes perturb24k/t0/<point>/:
- `mean_grads_bf16.pt`: {category: {key: mean gradient}} (keys: fc,
  l{i}_{q,o,gate,up,down}, l{i}_{k,v}@{ctx,blk});
- `fisher.pt`: per-row squared gradient norms per key ([rows, keys]) and the
  row categories (empirical Fisher trace = mean over rows; a group's trace is
  the sum of its members', K/V total = |g_ctx + g_blk|^2 kept separately);
- `slot_fisher.pt`: on a smaller sample, the same for the plain KL loss of each
  draft slot 1..15 separately ([rows, 15, keys]);
- `summary.json`: row counts and mean losses.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch._functorch.config

# Per-slot gradients back-propagate one graph 15 times (retain_graph); the
# compiled FlexAttention backward otherwise donates its buffers.
torch._functorch.config.donated_buffer = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

SEED = 20261111


def leaf_keys():
    keys = ["fc"]
    for i in range(H.LAYERS):
        keys += [f"l{i}_{m}" for m in ("q", "o", "gate", "up", "down")]
        keys += [f"l{i}_{m}@{p}" for m in ("k", "v") for p in ("ctx", "blk")]
    return keys


def fisher_keys():
    """Squared-norm keys: every leaf plus the K/V totals."""
    return leaf_keys() + [f"l{i}_{m}" for i in range(H.LAYERS) for m in ("k", "v")]


def make_leaves(runtime):
    leaves = {}
    for key in leaf_keys():
        site = key.split("@")[0]
        w = H.site_weight(runtime, site)
        leaves[key] = torch.zeros(w.shape, dtype=torch.float32, device=runtime.device,
                                  requires_grad=True)
    return leaves


def grads_of(leaves):
    return {k: v.grad for k, v in leaves.items()}


def sq_norms(grads, fkeys):
    out = []
    for key in fkeys:
        if key in grads:
            out.append(grads[key].float().pow(2).sum())
        else:
            g = grads[f"{key}@ctx"].float() + grads[f"{key}@blk"].float()
            out.append(g.pow(2).sum())
    return torch.stack(out)


def slot_losses(logits, targets, mask):
    """Plain KL(pT || p) per draft slot (mean over the row's valid positions)."""
    blocks = logits.shape[1] // H.BLOCK
    lp = torch.log_softmax(logits.float(), -1).reshape(blocks, H.BLOCK, -1)[:, 1:]
    pt = torch.softmax(targets.float(), -1).reshape(blocks, H.BLOCK, -1)[:, 1:]
    lpt = torch.log(pt.clamp_min(1e-30))
    kl = (pt * (lpt - lp)).sum(-1)  # [blocks, 15]
    valid = mask.reshape(blocks, H.BLOCK)[:, 1:].float()
    return (kl * valid).sum(0) / valid.sum(0).clamp_min(1.0), valid.sum(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", choices=("base", "static"), required=True)
    parser.add_argument("--rows-per-category", type=int, default=1000)
    parser.add_argument("--slot-rows-per-category", type=int, default=50)
    args = parser.parse_args()
    out_dir = H.OUT / "t0" / args.point
    out_dir.mkdir(parents=True, exist_ok=True)

    def body(runtime, job, _output):
        H.install_capture(runtime)
        runtime.drafter.requires_grad_(False)
        hooks = H.DeltaHooks(runtime.drafter)
        factors = H.zero_factors(runtime) if args.point == "base" else H.static_factors(runtime)
        leaves = make_leaves(runtime)
        hooks.set({k: H.Delta(w=v) for k, v in leaves.items()})
        rows = H.training_rows(runtime, job, args.rows_per_category, SEED)
        fkeys = fisher_keys()
        acc = {c: {k: torch.zeros_like(v) for k, v in leaves.items()} for c in H.CATEGORIES}
        counts = {c: 0 for c in H.CATEGORIES}
        losses = {c: [] for c in H.CATEGORIES}
        fisher = torch.zeros(len(rows), len(fkeys), dtype=torch.float64)
        started = time.time()
        for n, (cat, row) in enumerate(rows):
            batch = H.load_batch(runtime, row)
            _, loss, _ = runtime.forward(batch, factors)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss on row {row}")
            loss.backward()
            grads = grads_of(leaves)
            fisher[n] = sq_norms(grads, fkeys).double().cpu()
            for k, g in grads.items():
                acc[cat][k].add_(g)
            for v in leaves.values():
                v.grad = None
            counts[cat] += 1
            losses[cat].append(float(loss))
            if n % 200 == 0:
                print(f"[{args.point}] grads {n + 1}/{len(rows)} "
                      f"{time.time() - started:.0f}s {H.memory_report()}", flush=True)
        torch.save(
            {
                c: {k: (v / counts[c]).bfloat16().cpu() for k, v in acc[c].items()}
                for c in H.CATEGORIES
            },
            out_dir / "mean_grads_bf16.pt",
        )
        # Exact float32 norms of the mean gradients (bf16 copies round).
        mean_norms = {
            c: {k: float((v / counts[c]).norm()) for k, v in acc[c].items()}
            for c in H.CATEGORIES
        }
        torch.save(
            {"keys": fkeys, "categories": [c for c, _ in rows], "rows": [r for _, r in rows],
             "sq_norms": fisher},
            out_dir / "fisher.pt",
        )
        del acc
        torch.cuda.empty_cache()

        slot_rows = H.training_rows(runtime, job, args.slot_rows_per_category, SEED + 1)
        slot_fisher = torch.zeros(len(slot_rows), H.SLOTS, len(fkeys), dtype=torch.float64)
        slot_valid = torch.zeros(len(slot_rows), H.SLOTS)
        for n, (cat, row) in enumerate(slot_rows):
            batch = H.load_batch(runtime, row)
            runtime.forward(batch, factors)
            per_slot, valid = slot_losses(
                H.CAPTURE["logits"], H.CAPTURE["targets"], H.CAPTURE["mask"]
            )
            slot_valid[n] = valid.cpu()
            for s in range(H.SLOTS):
                if valid[s] == 0:
                    continue
                per_slot[s].backward(retain_graph=s < H.SLOTS - 1)
                slot_fisher[n, s] = sq_norms(grads_of(leaves), fkeys).double().cpu()
                for v in leaves.values():
                    v.grad = None
            H.CAPTURE.clear()
            if n % 50 == 0:
                print(f"[{args.point}] slot grads {n + 1}/{len(slot_rows)} "
                      f"{time.time() - started:.0f}s {H.memory_report()}", flush=True)
        torch.save(
            {"keys": fkeys, "categories": [c for c, _ in slot_rows],
             "sq_norms": slot_fisher, "valid": slot_valid},
            out_dir / "slot_fisher.pt",
        )
        H.write_json(out_dir / "summary.json", {
            "point": args.point,
            "rows_per_category": args.rows_per_category,
            "slot_rows_per_category": args.slot_rows_per_category,
            "counts": counts,
            "mean_loss": {c: sum(v) / len(v) for c, v in losses.items()},
            "mean_grad_norms": mean_norms,
            "seconds": time.time() - started,
        })
        print("done", time.time() - started, flush=True)

    H.with_runtime(body)


if __name__ == "__main__":
    main()
