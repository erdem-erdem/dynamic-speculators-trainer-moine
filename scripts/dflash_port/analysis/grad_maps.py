"""Step 2: per-category loss-gradient maps over every drafter linear weight.

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/analysis/grad_maps.py \
        --point base|static [--rows-per-category 1000]

The training loss (KL to the verifier + D-PACE weights, the 24K recipe's
`block_drafter` settings) is back-propagated row by row (validation anchors,
eval mode) at the base drafter (zero LoRA) or at the static-adapted point
(the trained static LoRA installed). Gradients are taken w.r.t. fc, q/k/v/o
and gate/up/down of each layer. K and V are split by path: the context path
(target features) and the block path (draft tokens) get separate copies of
the weight, so their gradients are measured separately (their sum is the
ordinary K/V gradient).

Rows: training rows of the 24K pool (r0002-100k subsets 00000..00005), a
seeded sample of `--rows-per-category` per category, alternately assigned to
halves A/B (split-half reliability). Never evaluation rows.

Writes analysis24k/grads/<point>/summary.json (per module x category: mean
gradient Frobenius norm, top-32 singular energy and share, cross-category and
split-half cosines, the share of each category's rank-32 first-order gain a
shared rank-32 update captures) and the per-category accumulated gradients
(bf16) for later re-analysis.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RANK = 32


def module_weights(runtime):
    """{name: weight tensor (leaf, requires grad)}; K/V split by path."""
    from speculators.generator.model import AdapterLinear  # noqa: PLC0415

    drafter = runtime.drafter
    drafter.requires_grad_(False)
    leaves = {}

    def base_linear(module):
        return module.base if isinstance(module, AdapterLinear) else module

    fc = base_linear(drafter.fc)
    fc.weight.requires_grad_(True)
    leaves["fc"] = fc.weight
    for i, layer in enumerate(drafter.layers):
        attn = layer.self_attn
        for name in ("q_proj", "o_proj"):
            linear = base_linear(getattr(attn, name))
            linear.weight.requires_grad_(True)
            leaves[f"l{i}_{name[0]}"] = linear.weight
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(layer.mlp, name)
            linear.weight.requires_grad_(True)
            leaves[f"l{i}_{name.split('_')[0]}"] = linear.weight
        copies = {}
        for name in ("k_proj", "v_proj"):
            linear = base_linear(getattr(attn, name))
            if linear.bias is not None:
                raise ValueError("K/V path split assumes bias-free projections")
            for path in ("ctx", "blk"):
                weight = linear.weight.detach().clone().requires_grad_(True)
                copies[(name, path)] = weight
                leaves[f"l{i}_{name[0]}_{path}"] = weight
        attn._project_kv = make_split_projection(attn, copies)
    return leaves


def make_split_projection(attn, copies):
    from speculators.generator.model import AdapterLinear  # noqa: PLC0415

    def linear(module, weight, x):
        y = F.linear(x, weight)
        if isinstance(module, AdapterLinear):
            y = y + module.scale * F.linear(F.linear(x, module.a), module.b)
        return y

    def project(target_hidden, hidden_states):
        return (
            linear(attn.k_proj, copies[("k_proj", "ctx")], target_hidden),
            linear(attn.k_proj, copies[("k_proj", "blk")], hidden_states),
            linear(attn.v_proj, copies[("v_proj", "ctx")], target_hidden),
            linear(attn.v_proj, copies[("v_proj", "blk")], hidden_states),
        )

    return project


def set_point(runtime, point):
    """Install zero (base) or the trained static factors into the adapters."""
    factors = C.zero_factors(runtime) if point == "base" else C.static_factors(runtime)
    for name, path in runtime.paths.items():
        adapter = runtime.drafter.get_submodule(path)
        a, b = factors[name]
        adapter.a = torch.nn.Parameter(a.clone(), requires_grad=False)
        adapter.b = torch.nn.Parameter(b.clone(), requires_grad=False)


def training_rows(runtime, job, per_category, seed=20261111):
    evaluated = {
        i for s in job["validation_subsets"] + job["test_subsets"]
        for i in runtime.corpus.indices(s, validation=True)
    }
    chosen = []
    for cat in C.CATEGORIES:
        pool = [
            i for s in job["subsets"] if C.category(s) == cat
            for i in runtime.corpus.indices(s)
        ]
        if evaluated & set(pool):
            raise ValueError("Training pool overlaps evaluation rows")
        sample = random.Random(f"{seed}-{cat}").sample(pool, per_category)
        chosen += [(cat, i, n % 2) for n, i in enumerate(sample)]
    return chosen


@torch.no_grad()
def summarize(acc, counts, weights, static_delta=None):
    """Per-module metrics from the per-(category, half) gradient sums."""
    result = {}
    for name in sorted(weights):
        cats = {}
        means = {}
        halves = {}
        for cat in C.CATEGORIES:
            a = acc[(cat, 0)][name].float()
            b = acc[(cat, 1)][name].float()
            n = counts[(cat, 0)] + counts[(cat, 1)]
            g = (a + b) / n
            means[cat] = g
            halves[cat] = (a / counts[(cat, 0)], b / counts[(cat, 1)])
            sv = torch.linalg.svdvals(g)
            energy = (sv**2).sum()
            top = (sv[:RANK] ** 2).sum()
            cats[cat] = {
                "fro": float(g.norm()),
                "top32_energy": float(top),
                "top32_share": float(top / energy),
                "top32_gain": float(top.sqrt()),
                "top1_share": float(sv[0] ** 2 / energy),
                "split_half_cos": cos(halves[cat][0], halves[cat][1]),
                "rows": n,
            }
        g_all = sum(means.values()) / len(means)
        u, s, vh = torch.linalg.svd(g_all, full_matrices=False)
        p32 = (u[:, :RANK] * s[:RANK]) @ vh[:RANK]
        direction = p32 / p32.norm()
        sv_all = s
        entry = {
            "shape": list(weights[name].shape),
            "weight_fro": float(weights[name].float().norm()),
            "all": {
                "fro": float(g_all.norm()),
                "top32_share": float((sv_all[:RANK] ** 2).sum() / (sv_all**2).sum()),
                "top32_gain": float((sv_all[:RANK] ** 2).sum().sqrt()),
            },
            "categories": cats,
            "cos": {},
            "cos_halves": {},
        }
        for cat in C.CATEGORIES:
            # Share of the category's best rank-32 first-order gain that the
            # single shared rank-32 direction (top-32 of the mean) captures.
            shared = float((means[cat] * direction).sum())
            cats[cat]["shared_gain"] = shared
            cats[cat]["shared_capture"] = shared / cats[cat]["top32_gain"]
            # Projection of the category gradient off the shared top-32
            # subspace (left and right): what a static rank-32 cannot follow.
            g = means[cat]
            inside = u[:, :RANK].T @ g @ vh[:RANK].T
            cats[cat]["in_shared_subspace_share"] = float(
                inside.norm() ** 2 / g.norm() ** 2
            )
        # Held-out (noise-matched) comparison of a shared vs a category-specific
        # rank-32 direction: both directions are fitted on one half of the rows
        # and evaluated on the other half (A->B and B->A averaged). The ratio is
        # the share of the category-specific first-order gain that the shared
        # update achieves; 1 = no benefit from a per-category adapter.
        directions = {}
        for h in (0, 1):
            shared_h = sum(halves[c][h] for c in C.CATEGORIES) / len(C.CATEGORIES)
            directions[("all", h)] = top_direction(shared_h)
            for cat in C.CATEGORIES:
                directions[(cat, h)] = top_direction(halves[cat][h])
        for cat in C.CATEGORIES:
            shared_gain, own_gain = [], []
            for h in (0, 1):
                test = halves[cat][h]
                shared_gain.append(float((test * directions[("all", 1 - h)]).sum()))
                own_gain.append(float((test * directions[(cat, 1 - h)]).sum()))
            cats[cat]["heldout_shared_gain"] = sum(shared_gain) / 2
            cats[cat]["heldout_own_gain"] = sum(own_gain) / 2
            cats[cat]["heldout_capture"] = (
                sum(shared_gain) / sum(own_gain) if sum(own_gain) > 0 else None)
        for i, c1 in enumerate(C.CATEGORIES):
            for c2 in C.CATEGORIES[i + 1 :]:
                key = f"{c1}|{c2}"
                entry["cos"][key] = cos(means[c1], means[c2])
                # Cross-half cosines (noise-matched to split-half cosines).
                entry["cos_halves"][key] = sum(
                    cos(halves[c1][x], halves[c2][y]) for x in (0, 1) for y in (0, 1)
                ) / 4
        if static_delta is not None and name in static_delta:
            entry["static_delta_cos_neg_grad"] = cos(static_delta[name], -g_all)
            entry["static_delta_fro"] = float(static_delta[name].norm())
        result[name] = entry
    return result


def top_direction(g):
    """Unit-Frobenius rank-32 truncation of g (its best rank-32 step direction)."""
    u, sv, vh = torch.linalg.svd(g, full_matrices=False)
    p = (u[:, :RANK] * sv[:RANK]) @ vh[:RANK]
    return p / p.norm()


def cos(x, y):
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-30))


def static_deltas(runtime):
    """ΔW = scale · B · A of the trained static LoRA, keyed like the leaves
    (K/V deltas apply to both paths)."""
    deltas = {}
    for key, (a, b) in C.static_factors(runtime).items():
        scale = runtime.drafter.get_submodule(runtime.paths[key]).scale
        delta = scale * b.float() @ a.float()
        layer, module = key.split("_", 1)
        short = module[0]
        if short in ("k", "v"):
            deltas[f"{layer}_{short}_ctx"] = delta
            deltas[f"{layer}_{short}_blk"] = delta
        else:
            deltas[f"{layer}_{short}"] = delta
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", choices=("base", "static"), required=True)
    parser.add_argument("--rows-per-category", type=int, default=1000)
    parser.add_argument("--no-save-grads", action="store_true")
    args = parser.parse_args()
    out_dir = C.OUT / "grads" / args.point

    def body(runtime, job, _output):
        runtime.drafter.eval()
        set_point(runtime, args.point)
        weights = module_weights(runtime)
        rows = training_rows(runtime, job, args.rows_per_category)
        acc = {
            (cat, h): {n: torch.zeros(w.shape, dtype=torch.float32,
                                      device=runtime.device)
                       for n, w in weights.items()}
            for cat in C.CATEGORIES for h in (0, 1)
        }
        counts = {key: 0 for key in acc}
        losses = {cat: [] for cat in C.CATEGORIES}
        started = time.time()
        for n, (cat, row, half) in enumerate(rows):
            batch = C.load_batch(runtime, row)
            _, loss, _ = runtime.forward(batch, None)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss on row {row}")
            loss.backward()
            target = acc[(cat, half)]
            for name, weight in weights.items():
                target[name].add_(weight.grad.float())
                weight.grad = None
            counts[(cat, half)] += 1
            losses[cat].append(float(loss))
            if n % 100 == 0:
                print(f"[{args.point}] {n + 1}/{len(rows)} {time.time() - started:.0f}s "
                      f"mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        deltas = static_deltas(runtime) if args.point == "base" else None
        summary = summarize(acc, counts, weights, deltas)
        out_dir.mkdir(parents=True, exist_ok=True)
        C.write_json(out_dir / "summary.json", {
            "point": args.point,
            "rows_per_category": args.rows_per_category,
            "counts": {f"{c}|{h}": v for (c, h), v in counts.items()},
            "mean_loss": {c: sum(v) / len(v) for c, v in losses.items()},
            "modules": summary,
        })
        if not args.no_save_grads:
            torch.save(
                {
                    cat: {
                        name: ((acc[(cat, 0)][name] + acc[(cat, 1)][name])
                               / (counts[(cat, 0)] + counts[(cat, 1)])).bfloat16().cpu()
                        for name in weights
                    }
                    for cat in C.CATEGORIES
                },
                out_dir / "mean_grads_bf16.pt",
            )
        print("done", time.time() - started)

    C.with_runtime(body)


if __name__ == "__main__":
    main()
