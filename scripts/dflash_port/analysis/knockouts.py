"""Step 4: knock-outs and random-LoRA sensitivity on 200 validation rows.

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/analysis/knockouts.py \
        [--part knockout|sensitivity|all]

Rows: 50 per category from the validation split (seeded), cached on the GPU.
Each variant is one pass of the production forward (validation anchors) with
forward hooks; per-row τ / full_acc counts are saved so Δ and paired CIs can
be computed against the unmodified pass of the same arm.

Knock-outs (arms base, static, generator; heads: base and static):
- `head:L:H`  zero head H's attention output in layer L (o_proj input slice);
- `attn:L`    zero layer L's attention output;
- `mlp:L`     zero layer L's MLP output;
- `ctx:L` / `blk:L`   zero K and V of the context path (target features) /
  of the block path (draft tokens) in layer L; `ctx:all` / `blk:all`.

Sensitivity (base arm): a random rank-32 update ΔW = B·A (Gaussian factors)
added to one module (fc or q/k/v/o/gate/up/down of a layer), Frobenius norm
matched to the trained static LoRA's ΔW of that module (MLP / fc: the static's
mean relative norm ‖ΔW‖/‖W‖ times ‖W‖), at 1x and 4x, 3 seeds. Plus
`transplant:<module>`: the trained static ΔW of that single module alone.
"""

import argparse
import random
import zlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROWS_PER_CATEGORY = 50
SEEDS = (0, 1, 2)
SCALES = (1.0, 4.0)


def pick_rows(runtime, job, per_category=ROWS_PER_CATEGORY, seed=20261111):
    rows = C.split_rows(runtime, job, "validation")
    chosen = []
    for cat in C.CATEGORIES:
        members = [(s, r) for s, r in rows if C.category(s) == cat]
        chosen += sorted(random.Random(f"{seed}-{cat}").sample(members, per_category))
    return chosen


class Hooks:
    def __init__(self):
        self.handles = []

    def add(self, handle):
        self.handles.append(handle)

    def clear(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def zero_path(attn, which):
    original = type(attn)._project_kv

    def project(target_hidden, hidden_states):
        k_ctx, k_blk, v_ctx, v_blk = original(attn, target_hidden, hidden_states)
        if which == "ctx":
            k_ctx, v_ctx = torch.zeros_like(k_ctx), torch.zeros_like(v_ctx)
        else:
            k_blk, v_blk = torch.zeros_like(k_blk), torch.zeros_like(v_blk)
        return k_ctx, k_blk, v_ctx, v_blk

    attn._project_kv = project


def restore_paths(drafter):
    for layer in drafter.layers:
        layer.self_attn.__dict__.pop("_project_kv", None)


def apply_knockout(drafter, hooks, variant, head_dim):
    kind, _, rest = variant.partition(":")
    layers = drafter.layers
    if kind == "head":
        layer, head = (int(x) for x in rest.split(":"))
        lo, hi = head * head_dim, (head + 1) * head_dim

        def pre(_module, args):
            x = args[0].clone()
            x[..., lo:hi] = 0
            return (x,)

        hooks.add(layers[layer].self_attn.o_proj.register_forward_pre_hook(pre))
    elif kind in ("attn", "mlp"):
        module = layers[int(rest)].self_attn.o_proj if kind == "attn" else layers[
            int(rest)].mlp
        hooks.add(module.register_forward_hook(
            lambda _m, _a, out: torch.zeros_like(out)))
    elif kind in ("ctx", "blk"):
        targets = range(len(layers)) if rest == "all" else [int(rest)]
        for layer in targets:
            zero_path(layers[layer].self_attn, kind)
    elif kind != "none":
        raise ValueError(variant)


def module_path(name):
    if name == "fc":
        return "fc"
    layer, kind = name.split("_", 1)
    block = "mlp" if kind in ("gate", "up", "down") else "self_attn"
    return f"layers.{layer[1:]}.{block}.{kind}_proj"


def base_weight(module):
    return getattr(module, "base", module).weight


def add_delta(drafter, hooks, name, delta):
    module = drafter.get_submodule(module_path(name))

    def post(_m, args, out):
        x = args[0]
        return out + torch.nn.functional.linear(x, delta.to(x.dtype))

    hooks.add(module.register_forward_hook(post))


def evaluate(runtime, cache, factors_for, label, results, started):
    per_row = []
    for (subset, row), batch in cache:
        metrics = C.run(runtime, batch, factors_for(row))
        per_row.append({
            "subset": subset, "row": row,
            "eal_sum": metrics["eal_sum"], "eal_total": metrics["eal_total"],
            "acc_sum": metrics["full_acc_sum"], "acc_total": metrics["full_acc_total"],
        })
    results[label] = per_row
    tau = sum(r["eal_sum"] for r in per_row) / sum(r["eal_total"] for r in per_row)
    print(f"{label}: tau_pooled={tau:.4f} ({time.time() - started:.0f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("knockout", "sensitivity", "all"),
                        default="all")
    args = parser.parse_args()
    out_dir = C.OUT / "knockouts"

    def body(runtime, job, _output):
        C.install_analysis_forward(runtime)
        arms = C.Arms(runtime)
        drafter = runtime.drafter
        rows = pick_rows(runtime, job)
        cache = [((s, r), C.load_batch(runtime, r)) for s, r in rows]
        gen_cache = {r: arms.factors("generator", r) for _, r in rows}
        providers = {
            "base": lambda r: arms.zero,
            "static": lambda r: arms.static,
            "generator": lambda r: gen_cache[r],
        }
        attn0 = drafter.layers[0].self_attn
        heads = drafter.config.transformer_layer_config.num_attention_heads
        n_layers = len(drafter.layers)
        hooks = Hooks()
        started = time.time()
        if args.part in ("knockout", "all"):
            results = {}
            variants = ["none"]
            variants += [f"attn:{i}" for i in range(n_layers)]
            variants += [f"mlp:{i}" for i in range(n_layers)]
            variants += [f"ctx:{i}" for i in range(n_layers)] + ["ctx:all"]
            variants += [f"blk:{i}" for i in range(n_layers)] + ["blk:all"]
            head_variants = [f"head:{i}:{h}" for i in range(n_layers)
                             for h in range(heads)]
            for arm in ("base", "static", "generator"):
                for variant in variants + (head_variants if arm != "generator" else []):
                    apply_knockout(drafter, hooks, variant, attn0.head_dim)
                    try:
                        evaluate(runtime, cache, providers[arm], f"{arm}|{variant}",
                                 results, started)
                    finally:
                        hooks.clear()
                        restore_paths(drafter)
                C.write_json(out_dir / "knockouts.json",
                             {"rows": rows, "results": results})
        if args.part in ("sensitivity", "all"):
            results = {}
            deltas, rel = {}, []
            for key, (a, b) in arms.static.items():
                adapter = drafter.get_submodule(runtime.paths[key])
                delta = adapter.scale * b.float() @ a.float()
                layer, module = key.split("_", 1)
                name = f"{layer}_{module.split('_')[0]}"
                deltas[name] = delta
                rel.append(float(delta.norm() / base_weight(adapter).float().norm()))
            mean_rel = sum(rel) / len(rel)
            names = ["fc"] + [f"l{i}_{m}" for i in range(n_layers)
                              for m in ("q", "k", "v", "o", "gate", "up", "down")]
            norms = {}
            for name in names:
                weight = base_weight(drafter.get_submodule(module_path(name))).float()
                target = (float(deltas[name].norm()) if name in deltas
                          else mean_rel * float(weight.norm()))
                norms[name] = {"weight_fro": float(weight.norm()), "delta_fro": target}
            evaluate(runtime, cache, providers["base"], "base|none", results, started)
            for name in names:
                if name in deltas:
                    add_delta(drafter, hooks, name, deltas[name])
                    try:
                        evaluate(runtime, cache, providers["base"],
                                 f"base|transplant:{name}", results, started)
                    finally:
                        hooks.clear()
                weight = base_weight(drafter.get_submodule(module_path(name)))
                out_f, in_f = weight.shape
                for scale in SCALES:
                    for seed in SEEDS:
                        gen = torch.Generator(device="cpu").manual_seed(
                            zlib.crc32(f"{name}:{seed}".encode()))
                        a = torch.randn(32, in_f, generator=gen)
                        b = torch.randn(out_f, 32, generator=gen)
                        delta = (b @ a).to(runtime.device)
                        delta *= scale * norms[name]["delta_fro"] / delta.norm()
                        add_delta(drafter, hooks, name, delta)
                        try:
                            evaluate(runtime, cache, providers["base"],
                                     f"base|random:{name}:x{scale:g}:s{seed}",
                                     results, started)
                        finally:
                            hooks.clear()
                C.write_json(out_dir / "sensitivity.json",
                             {"rows": rows, "norms": norms, "mean_rel": mean_rel,
                              "results": results})

    C.with_runtime(body)


if __name__ == "__main__":
    main()
