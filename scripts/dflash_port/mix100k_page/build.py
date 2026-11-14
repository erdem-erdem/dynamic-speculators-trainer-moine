"""Build the MIX100K results page (static r512 vs generator r64 on 100K rows, with the 24K runs as context).

    python scripts/dflash_port/mix100k_page/build.py RESULTS.json OUT.html

RESULTS.json is `scripts/dflash_port/abl24k_collect.py` output with the labels in ARMS (missing
runs are skipped). The direct paired comparisons in DIRECT are recomputed here from the stored
per-row counts (same bootstrap as compare_arms.py).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / "perturbation"
sys.path.insert(0, str(HERE.parent))
from compare_arms import bootstrap, load_rows, macro  # noqa: E402

# label, name, group (P1 colour), colour token, data, epochs, updates/epoch, in curves, in heatmap
ARMS = [
    ("s100", "100K static r512 (3 ep)", "s100", "--s1", "100k", 3, 5555.67, True, True),
    ("g100", "100K generator r64, shared heads (3 ep)", "g100", "--s2", "100k", 3, 5555.67, True, True),
    ("s24_512", "24K static r512 (2 ep)", "s24", "--s3", "24k", 2, 1333.5, True, True),
    ("g24_64", "24K generator r64, shared heads (2 ep)", "g24", "--s4", "24k", 2, 1333.5, True, True),
    ("s24_32", "24K static r32 (2 ep)", "s24", "--s7", "24k", 2, 1333.5, False, True),
    ("g24_pm", "24K generator 1.1B per-module r32 (2 ep)", "g24", "--s5", "24k", 2, 1333.5, False, False),
]
SCALE = [
    {"name": "static r512", "color": "--s1", "points": [{"label": "s24_512", "rows": 24000}, {"label": "s100", "rows": 100000}]},
    {"name": "generator r64 (shared heads)", "color": "--s2", "points": [{"label": "g24_64", "rows": 24000}, {"label": "g100", "rows": 100000}]},
]
DIRECT = [
    ("100K generator r64 − 100K static r512", "g100", "s100"),
    ("100K static r512 − 24K static r512", "s100", "s24_512"),
    ("100K generator r64 − 24K generator r64", "g100", "g24_64"),
    ("100K static r512 − 24K static r32", "s100", "s24_32"),
]
SETTINGS = {
    "arms": ["static r512", "generator r64"],
    "rows": [
        ["Run", "mix100k-static-lora-attn-r512-3ep-kl", "mix100k-generator-attn-r64-3ep-kl-pt256"],
        ["Training rows", "100,000: train splits of 25 subsets × math / code / stem / chat (25K each)", "same"],
        ["Schedule", "3 epochs = 16,667 updates × 18 rows; validation every 500 updates", "same"],
        ["Loss", "KL to the target + D-PACE position weights (α 0.5)", "same"],
        ["Optimizer", "AdamW, lr 1e-4, weight decay 0.01, clip 0.85, warmup 100", "same"],
        ["Adapter", "q / k / v / o of all 5 layers, rank 512, α 1024 (coefficient 2), dropout 0.05", "q / k / v / o of all 5 layers, rank 64, α 128 (coefficient 2), one LoRA generated per prompt"],
        ["Trainable", "68.2M (one LoRA)", "≈ 0.45B: encoder 512 → 128, condition 32, decoder 256 with 2 residual blocks, heads shared across the 5 layers"],
        ["Conditioning", "—", "last prompt token's target hidden states (5 layers × 4,096) + log length; one prompt per LoRA"],
        ["Drafter", "z-lab/Qwen3-8B-DFlash-b16 (frozen), block 16, 512 anchors / 8,192 tokens", "same"],
        ["Evaluation", "validation 500 / test 500 rows (200 math + 100 code / stem / chat), OOD v3 240 rows; same rows as every 24K run", "same"],
        ["Seed", "42 (one seed)", "same"],
    ],
}


def r5(v, d=5):
    if isinstance(v, float):
        return float(f"{v:.{d}g}")
    if isinstance(v, list):
        return [r5(x, d) for x in v]
    if isinstance(v, dict):
        return {k: r5(x, d) for k, x in v.items()}
    return v


def paired(run_a, run_b):
    out = {}
    for sp in ("validation", "test", "ood"):
        try:
            a, b = load_rows(run_a, sp, "coef-2_donor-correct"), load_rows(run_b, sp, "coef-2_donor-correct")
        except (IndexError, FileNotFoundError, KeyError):
            continue
        if sorted(a) != sorted(b):
            continue
        cats = sorted({v[0] for v in a.values()})
        groups = [[k for k in sorted(a) if a[k][0] == c] for c in cats]
        ci = bootstrap(a, b, groups, 2000, 0)["tau"]
        out[sp] = {"delta": macro(a, groups)["tau"] - macro(b, groups)["tau"], "ci": list(ci)}
    return out or None


def main():
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    R = json.loads(results.read_text())
    runs = {}
    dirs = {}
    for label, run in R["runs"].items():
        run = dict(run)
        dirs[label] = run.pop("run_dir", None)
        for sp in ("validation", "test", "ood"):
            if sp in run:
                run[sp].pop("pooled", None)
        runs[label] = run
    direct = [{"name": n, "result": paired(dirs[a], dirs[b]) if a in dirs and b in dirs else None} for n, a, b in DIRECT]
    arms = [dict(label=a[0], name=a[1], group=a[2], color=a[3], data=a[4], epochs=a[5], updates_per_epoch=a[6], curve=a[7], heat=a[8])
            for a in ARMS]
    data = {"arms": arms, "runs": runs, "base": R["base"], "scale": SCALE, "direct": direct, "settings": SETTINGS}
    html = (KIT / "artifact_template.html").read_text()
    html = html.replace("<title>Move vs Improve: Results</title>", "<title>MIX100K Static vs Generator</title>")
    html = html.replace("/*__CHARTS__*/", (KIT / "artifact_charts.js").read_text())
    html = html.replace("/*__RENDER__*/", (HERE / "render.js").read_text())
    html = html.replace("<!--__BODY__-->", (HERE / "body.html").read_text())
    html = html.replace("/*__DATA__*/null", json.dumps(r5(data), separators=(",", ":")))
    out.write_text(html)
    print(out, f"{len(html) / 1e6:.2f} MB")
    for d in direct:
        print(d["name"], d["result"])


if __name__ == "__main__":
    main()
