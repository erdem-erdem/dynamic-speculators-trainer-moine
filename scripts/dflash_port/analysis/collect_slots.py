"""Step 1 (+ step 5 collection): per-slot records for every arm.

    CUDA_VISIBLE_DEVICES=0 python scripts/dflash_port/analysis/collect_slots.py \
        --split validation [--hidden]

For each row of the split and each arm (base, static, generator; with
--hidden also the generator conditioned on a same-category and on a
cross-category donor row), one production forward (validation anchors) and,
per draft slot: validity, correctness, the target's top-1 probability and
entropy, the drafter's rank / probability of the target's top-1 token, the
drafter's own top-1 probability, and the target's probability / rank of the
drafter's prediction. The per-subset metric counts are kept to check the
recorded evaluation (reproduction).

--hidden also stores, for 6 seeded fully valid blocks per row (the same
blocks for every arm), the drafter's per-layer states at the 15 draft slots
(input embedding + 5 layer outputs) and the target's 6 stored layers at the
position each slot is predicted from (anchor + slot - 1).

Outputs under /workspace/yusuf/runs/dflash/analysis24k/slots/<split>/.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HIDDEN_BLOCKS = 6
CHUNK = 512


@torch.no_grad()
def slot_stats(logits, targets, mask, data_tokens):
    """Per-position statistics (slot 0 included; callers drop it).
    `data_tokens`: the response token at each query position (anchor + slot),
    so `dm` says whether the data (sampled) token equals the target's argmax
    label there (the in-block path is the target's greedy path)."""
    logits, targets = logits[0], targets[0]
    out = {k: [] for k in (
        "correct", "p_top", "ent", "rank", "p_d_tgt", "p_d_top", "t_p_pred",
        "t_rank_pred",
    )}
    for start in range(0, logits.shape[0], CHUNK):
        lg = logits[start : start + CHUNK].float()
        tg = targets[start : start + CHUNK].float()
        t_logp = torch.log_softmax(tg, -1)
        t_p = t_logp.exp()
        top = tg.argmax(-1)
        pred = lg.argmax(-1)
        d_logp = torch.log_softmax(lg, -1)
        lg_top = lg.gather(-1, top[:, None])
        tg_pred = tg.gather(-1, pred[:, None])
        out["correct"].append(pred == top)
        out["p_top"].append(t_p.gather(-1, top[:, None])[:, 0])
        out["ent"].append(-(t_p * t_logp).sum(-1))
        out["rank"].append((lg > lg_top).sum(-1) + 1)
        out["p_d_tgt"].append(d_logp.gather(-1, top[:, None])[:, 0].exp())
        out["p_d_top"].append(d_logp.max(-1).values.exp())
        out["t_p_pred"].append(t_p.gather(-1, pred[:, None])[:, 0])
        out["t_rank_pred"].append((tg > tg_pred).sum(-1) + 1)
    result = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    result["valid"] = mask[0].bool().cpu().numpy()
    labels = targets.argmax(-1)  # targets is [N, V] here
    result["dm"] = (labels == data_tokens).cpu().numpy()
    return result


def compact(stats, row_id, anchors_rel):
    """Draft slots 1..15 of blocks with any valid slot, as flat arrays."""
    n = stats["valid"].shape[0] // C.BLOCK
    valid = stats["valid"].reshape(n, C.BLOCK)
    keep = valid[:, 1:].any(1)
    blocks = np.nonzero(keep)[0]
    out = {}
    for key, value in stats.items():
        value = value.reshape(n, C.BLOCK)[blocks, 1:]
        out[key] = value.reshape(-1)
    count = len(blocks)
    out["row"] = np.full(count * 15, row_id, np.int32)
    out["block"] = np.repeat(blocks.astype(np.int32), 15)
    out["slot"] = np.tile(np.arange(1, 16, dtype=np.int8), count)
    out["anchor_rel"] = np.repeat(anchors_rel[blocks].astype(np.int32), 15)
    dtypes = {
        "p_top": np.float16, "ent": np.float16, "p_d_tgt": np.float16,
        "p_d_top": np.float16, "t_p_pred": np.float16,
        "rank": np.int32, "t_rank_pred": np.int32,
    }
    for key, dtype in dtypes.items():
        out[key] = out[key].astype(dtype)
    return out


def donors(rows):
    """Same-category donor: the next row of the category (cyclic);
    cross-category donor: the row at the same position of the next category."""
    by_cat = {}
    for subset, row in rows:
        by_cat.setdefault(C.category(subset), []).append(row)
    same, cross = {}, {}
    cats = [c for c in C.CATEGORIES if c in by_cat]
    for c_index, cat in enumerate(cats):
        members = by_cat[cat]
        other = by_cat[cats[(c_index + 1) % len(cats)]]
        for i, row in enumerate(members):
            same[row] = members[(i + 1) % len(members)]
            cross[row] = other[i % len(other)]
    return same, cross


class HiddenTap:
    """Drafter layer states at selected query positions (forward hooks)."""

    def __init__(self, drafter):
        self.positions = None
        self.states = []
        self.handles = [
            drafter.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)
        ]
        for layer in drafter.layers:
            self.handles.append(layer.register_forward_hook(self._post))

    def _pre(self, _module, _args, kwargs):
        if self.positions is not None:
            self.states = [kwargs["hidden_states"][0, self.positions].half().cpu()]

    def _post(self, _module, _args, output):
        if self.positions is not None:
            self.states.append(output[0, self.positions].half().cpu())

    def take(self):
        states, self.states = torch.stack(self.states, 0), []
        return states  # [6, n, hidden]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out_dir = C.OUT / "slots" / args.split

    def body(runtime, job, _output):
        C.install_analysis_forward(runtime)
        arms = C.Arms(runtime)
        rows = C.split_rows(runtime, job, args.split)
        if args.limit:
            rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
        same, cross = donors(rows)
        names = ["base", "static", "generator"]
        if args.hidden:
            names += ["generator_same", "generator_cross"]
        records = {name: [] for name in names}
        counts = {name: {} for name in names}
        row_table = []
        hidden = {name: [] for name in names}
        target_states, hidden_labels = [], []
        tap = HiddenTap(runtime.drafter) if args.hidden else None
        started = time.time()
        for row_id, (subset, row) in enumerate(rows):
            batch = C.load_batch(runtime, row)
            info = runtime.corpus.rows[row]
            start = C.prompt_boundary(batch["loss_mask"][0])
            row_table.append({
                "row_id": row_id, "subset": subset, "category": C.category(subset),
                "index": row, "primary_id": info["primary_id"],
                "seq_len": int(batch["input_ids"].shape[1]), "prompt_len": start,
                "same_donor": same[row], "cross_donor": cross[row],
            })
            chosen = None
            for name in names:
                if name == "generator_same":
                    factors = arms.factors("generator", row, donor=same[row])
                elif name == "generator_cross":
                    factors = arms.factors("generator", row, donor=cross[row])
                else:
                    factors = arms.factors(name, row)
                if tap is not None and chosen is not None:
                    tap.positions = chosen
                metrics = C.run(runtime, batch, factors)
                C.accumulate(counts[name], subset, metrics)
                cap = C.CAPTURE
                indices = cap["indices"]
                anchors = indices[:: C.BLOCK]
                padded = torch.nn.functional.pad(batch["input_ids"][0], (0, C.BLOCK))
                stats = slot_stats(cap["logits"], cap["targets"], cap["mask"],
                                   padded[indices])
                rel = (anchors - start).cpu().numpy()
                records[name].append(compact(stats, row_id, rel))
                if tap is not None and chosen is None:
                    # Select the blocks once (base arm); anchors are identical
                    # across arms, then rerun this arm with the tap on.
                    valid = stats["valid"].reshape(-1, C.BLOCK)[:, 1:].all(1)
                    full = np.nonzero(valid)[0].tolist()
                    pick = sorted(random.Random(row).sample(
                        full, min(HIDDEN_BLOCKS, len(full))))
                    query = torch.tensor(
                        [b * C.BLOCK + s for b in pick for s in range(1, C.BLOCK)],
                        device=runtime.device, dtype=torch.long,
                    )
                    chosen = query
                    tap.positions = chosen
                    C.run(runtime, batch, factors)
                    source = indices[query] - 1  # target position of each slot
                    target_states.append(torch.cat([
                        batch["hidden_states"][0, source].view(len(query), 5, batch["verifier_last_hidden_states"].shape[-1]),
                        batch["verifier_last_hidden_states"][0, source][:, None],
                    ], 1).transpose(0, 1).half().cpu())
                    lab_correct = stats["correct"].reshape(-1, C.BLOCK)[pick, 1:]
                    accepted = np.cumprod(lab_correct, 1).astype(bool)
                    hidden_labels.append({
                        "row": np.full(len(query), row_id, np.int32),
                        "block": np.repeat(np.array(pick, np.int32), 15),
                        "slot": np.tile(np.arange(1, 16, dtype=np.int8), len(pick)),
                        "base_correct": lab_correct.reshape(-1),
                        "base_accepted": accepted.reshape(-1),
                        "base_p_d_top": stats["p_d_top"].reshape(-1, C.BLOCK)[
                            pick, 1:].reshape(-1).astype(np.float16),
                    })
                if tap is not None:
                    hidden[name].append(tap.take())
                    tap.positions = None
                if tap is not None and name == names[0]:
                    pass
            if row_id % 25 == 0:
                print(f"[{args.split}] row {row_id + 1}/{len(rows)} "
                      f"{time.time() - started:.0f}s", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        C.write_json(out_dir / "rows.json", row_table)
        for name in names:
            merged = {k: np.concatenate([r[k] for r in records[name]])
                      for k in records[name][0]}
            np.savez_compressed(out_dir / f"slots_{name}.npz", **merged)
        recorded = C.recorded_scores()
        repro = {}
        for name in names:
            mine = C.score(counts[name])
            entry = {"computed": mine, "counts": counts[name]}
            if name in recorded and not args.limit:
                rec = recorded[name][args.split]
                entry["recorded_full_acc"] = rec["score"]
                entry["recorded_categories"] = rec["categories"]
                entry["abs_diff_full_acc"] = abs(mine["full_acc"]["all"] - rec["score"])
            repro[name] = entry
        C.write_json(out_dir / "reproduction.json", repro)
        if tap is not None:
            hdir = C.OUT / "hidden" / args.split
            hdir.mkdir(parents=True, exist_ok=True)
            for name in names:
                np.save(hdir / f"drafter_{name}.npy",
                        torch.cat(hidden[name], 1).numpy())
            np.save(hdir / "target.npy", torch.cat(target_states, 1).numpy())
            labels = {k: np.concatenate([h[k] for h in hidden_labels])
                      for k in hidden_labels[0]}
            np.savez(hdir / "labels.npz", **labels)
        print({n: (repro[n]["computed"]["full_acc"]["all"],
                   repro[n].get("recorded_full_acc")) for n in names})

    C.with_runtime(body)


if __name__ == "__main__":
    main()
