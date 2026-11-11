"""T2: short trainings from the base drafter (movement-vs-gain paths).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/dflash_port/perturbation/t2_train.py \
        --run full|lora_attn_r512|lora_ffn_r512 [--updates 600] [--no-wandb]

Common (plan D2/D3/D5/D6): start from the base drafter; loss KL to the
verifier + D-PACE weights (the 24K recipe's `block_drafter` settings, through
the production forward); 24K training rows only (a seeded permutation of the
24,000-row pool, the same order for every run, no repeats within 600 x 18);
18 rows per update, one row per episode, loss averaged over the 18 episodes
(as campaign_train); training anchors at 512 per 8,192 tokens (the recipe's
anchor density, random per microbatch); AdamW (torch defaults betas/eps),
linear warmup then constant LR (engine.set_learning_rate), gradient clipping
at 0.85 on all trainable parameters.

Runs:
- full: every linear weight of the drafter (fc; q/k/v/o, gate/up/down of the
  5 layers; ~1.05B) as float32 trainable dW added on the hook path
  (W_eff = W + dW, K/V one weight on both paths); LR 1e-5, wd 0.01 applied
  to the full weight W + dW (decoupled), warmup 50;
- lora_attn_r512 / lora_ffn_r512: LoRA rank 512, alpha 1024 (scale 2) on
  q/k/v/o resp. gate/up/down of all 5 layers; A kaiming-uniform(a=sqrt 5),
  B = 0, dropout 0.05 on the LoRA input in training; LR 1e-4, wd 0.01 on the
  factors, warmup 100 (the repo's static settings).

At updates 0 / 150 / 300 / 600 the current dW is evaluated against the base
drafter on the 100 sweep rows (sweep.run_sweep, saved to
perturb24k/t2/<run>/eval_u<update>.pt); at the end also the category-mean
τ / full_acc on all 500 validation rows. W&B: run name dflash-perturb24k-<run>.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402
import sweep as S  # noqa: E402

RUNS = {
    "full": {"lr": 1e-5, "warmup": 50, "sites": "all_linear"},
    "lora_attn_r512": {"lr": 1e-4, "warmup": 100, "sites": H.ATTN, "rank": 512, "alpha": 1024},
    "lora_ffn_r512": {"lr": 1e-4, "warmup": 100, "sites": H.FFN, "rank": 512, "alpha": 1024},
}
ROWS_PER_UPDATE = 18
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.85
DROPOUT = 0.05
EVAL_AT = (0, 150, 300, 600)
ORDER_SEED = 20261111
ENTITY = "yusuf-erd-erdem-y-ld-z-technical-university"
PROJECT = "dyn-spec-dflash-v1"


class TrainDelta:
    """Trainable additive map: dense float32 dW, or LoRA scale * B (A drop(x))."""

    def __init__(self, w=None, a=None, b=None, scale=1.0, dropout=0.0):
        self.w, self.a, self.b, self.scale, self.dropout = w, a, b, scale, dropout
        self.training = True

    def __call__(self, x):
        if self.w is not None:
            return F.linear(x, self.w)
        if self.training and self.dropout:
            x = F.dropout(x, self.dropout, training=True)
        return self.scale * F.linear(F.linear(x, self.a), self.b)

    def parameters(self):
        return [self.w] if self.w is not None else [self.a, self.b]

    @torch.no_grad()
    def dense(self):
        if self.w is not None:
            return self.w.detach()
        return self.scale * self.b.detach() @ self.a.detach()


def build_deltas(runtime, spec):
    deltas = {}
    if spec["sites"] == "all_linear":
        for site in H.all_sites():
            w = H.site_weight(runtime, site)
            deltas[site] = TrainDelta(
                w=torch.zeros(w.shape, dtype=torch.float32, device=runtime.device,
                              requires_grad=True))
        return deltas
    torch.manual_seed(ORDER_SEED)
    for i in range(H.LAYERS):
        for m in spec["sites"]:
            site = f"l{i}_{m}"
            out_f, in_f = H.site_weight(runtime, site).shape
            a = torch.empty(spec["rank"], in_f, device=runtime.device, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            b = torch.zeros(out_f, spec["rank"], device=runtime.device, dtype=torch.float32)
            deltas[site] = TrainDelta(a=a.requires_grad_(True), b=b.requires_grad_(True),
                                      scale=spec["alpha"] / spec["rank"], dropout=DROPOUT)
    return deltas


def evaluate_checkpoint(runtime, hooks, rows, base_factors, deltas, out_path, name, update):
    for d in deltas.values():
        d.training = False
    runtime.drafter.eval()
    frozen = {site: H.Delta(w=d.dense().to(runtime.dtype)) for site, d in deltas.items()}
    pert = S.Perturbation(name, {"kind": "train", "run": name, "update": update,
                                 "location": name, "point": "base"}, frozen)
    S.run_sweep(runtime, hooks, rows, base_factors, [pert], out_path, save_every=100)
    for d in deltas.values():
        d.training = True
    data = torch.load(out_path, map_location="cpu", weights_only=False)
    slot, block = data["slot"][0], data["block"][0]
    k = {key: i for i, key in enumerate(data["slot_keys"])}
    n = slot[..., k["n"]].sum()
    return {
        "eval/kl_from_base": float(slot[..., k["kl12"]].sum() / n),
        "eval/flip_rate": float(slot[..., k["flip"]].sum() / n),
        "eval/delta_full_acc": float((slot[..., k["c2"]].sum() - slot[..., k["c1"]].sum()) / n),
        "eval/delta_tau_pooled": float((block[:, 2].sum() - block[:, 1].sum()) / block[:, 0].sum()),
        "eval/delta_kl_to_target": float((slot[..., k["klT2"]].sum() - slot[..., k["klT1"]].sum()) / n),
        "eval/fixes": float(slot[..., k["fix"]].sum() / n),
        "eval/breaks": float(slot[..., k["brk"]].sum() / n),
    }


@torch.no_grad()
def full_validation(runtime, hooks, job, base_factors, deltas):
    runtime.drafter.eval()
    hooks.set({site: H.Delta(w=d.dense().to(runtime.dtype)) for site, d in deltas.items()})
    counts = {}
    for subset, row in H.split_rows(runtime, job, "validation"):
        batch = H.load_batch(runtime, row)
        _, _, metrics = runtime.forward(batch, base_factors)
        target = counts.setdefault(subset, {})
        for key, value in metrics.items():
            target[key] = target.get(key, 0.0) + float(value)
    hooks.clear()
    return H.score(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=tuple(RUNS), required=True)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    spec = RUNS[args.run]
    name = args.run
    out_dir = H.OUT / "t2" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_at = [u for u in EVAL_AT if u <= args.updates]

    def body(runtime, job, _output):
        H.install_capture(runtime)
        runtime.drafter.requires_grad_(False)
        hooks = H.DeltaHooks(runtime.drafter)
        base = H.zero_factors(runtime)
        rows = H.sweep_rows(runtime, job)
        deltas = build_deltas(runtime, spec)
        params = [p for d in deltas.values() for p in d.parameters()]
        n_params = sum(p.numel() for p in params)
        full = spec["sites"] == "all_linear"
        optimizer = torch.optim.AdamW(params, lr=spec["lr"],
                                      weight_decay=0.0 if full else WEIGHT_DECAY)
        base_weights = (
            {site: H.site_weight(runtime, site) for site in deltas} if full else None
        )
        wnorm = {site: float(H.site_weight(runtime, site).float().norm()) for site in deltas}
        pool = H.training_pool(runtime, job)
        order = random.Random(ORDER_SEED).sample(pool, len(pool))
        needed = args.updates * ROWS_PER_UPDATE
        if needed > len(order):
            raise ValueError("More rows requested than the pool holds (no repeats)")
        tracker = None
        if not args.no_wandb:
            import wandb  # noqa: PLC0415

            tracker = wandb.init(
                entity=ENTITY, project=PROJECT, name=f"dflash-perturb24k-{name}",
                group="perturb24k", job_type="t2",
                config={"run": name, **{k: v for k, v in spec.items() if k != "sites"},
                        "sites": str(spec["sites"]), "updates": args.updates,
                        "rows_per_update": ROWS_PER_UPDATE, "weight_decay": WEIGHT_DECAY,
                        "max_grad_norm": MAX_GRAD_NORM, "trainable_parameters": n_params,
                        "loss": "kl_div + dpace", "start": "base drafter",
                        "order_seed": ORDER_SEED},
            )
        print(f"[{name}] trainable={n_params:,} {H.memory_report()}", flush=True)
        hooks.set(deltas)
        history = []
        started = time.time()
        for update in range(args.updates + 1):
            if update in eval_at:
                hooks.clear()
                ev = evaluate_checkpoint(runtime, hooks, rows, base, deltas,
                                         out_dir / f"eval_u{update}.pt", name, update)
                hooks.set(deltas)
                rel = (sum(float(d.dense().pow(2).sum()) for d in deltas.values())
                       / sum(v**2 for v in wnorm.values())) ** 0.5
                ev["eval/rel_delta_norm"] = rel
                ev["update"] = update
                history.append(ev)
                print(f"[{name}] eval u{update}: {ev}", flush=True)
                if tracker:
                    tracker.log(ev, step=update)
                H.write_json(out_dir / "history.json", history)
            if update == args.updates:
                break
            runtime.drafter.train()
            lr = spec["lr"] * min((update + 1) / spec["warmup"], 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for episode in range(ROWS_PER_UPDATE):
                row = order[update * ROWS_PER_UPDATE + episode]
                torch.manual_seed(ORDER_SEED + update * 100003 + episode * 1009)
                batch = H.load_batch(runtime, row, training=True)
                _, loss, _ = runtime.forward(batch, base)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at update {update}")
                (loss / ROWS_PER_UPDATE).backward()
                losses.append(float(loss))
            H.CAPTURE.clear()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                params, MAX_GRAD_NORM, error_if_nonfinite=True))
            optimizer.step()
            if full:
                with torch.no_grad():
                    for site, d in deltas.items():
                        # Decoupled weight decay on the full weight W + dW.
                        d.w.sub_(lr * WEIGHT_DECAY * (base_weights[site].float() + d.w))
            log = {"train/loss": sum(losses) / len(losses), "train/grad_norm": grad_norm,
                   "train/lr": lr}
            if tracker:
                tracker.log(log, step=update + 1)
            if update % 10 == 0:
                print(f"[{name}] u{update + 1} loss={log['train/loss']:.4f} "
                      f"gn={grad_norm:.3f} lr={lr:.2e} {time.time() - started:.0f}s "
                      f"{H.memory_report()}", flush=True)
        runtime.drafter.eval()
        val = full_validation(runtime, hooks, job, base, deltas)
        H.write_json(out_dir / "validation500.json", val)
        if tracker:
            tracker.log({"val500/full_acc": val["full_acc"]["all"],
                         "val500/tau": val["eal"]["all"]}, step=args.updates)
        torch.save({site: d.dense().float().cpu() if full else
                    {"a": d.a.detach().cpu(), "b": d.b.detach().cpu(), "scale": d.scale}
                    for site, d in deltas.items()}, out_dir / "final_delta.pt")
        H.write_json(out_dir / "summary.json", {
            "run": name, "spec": {k: str(v) for k, v in spec.items()},
            "updates": args.updates, "trainable_parameters": n_params,
            "history": history, "validation500": val, "seconds": time.time() - started,
            "wandb": None if tracker is None else {"name": tracker.name, "id": tracker.id,
                                                   "url": tracker.url},
        })
        if tracker:
            tracker.finish()
        print("done", time.time() - started, flush=True)

    H.with_runtime(body)



if __name__ == "__main__":
    main()
