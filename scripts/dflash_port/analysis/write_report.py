"""Render reporting/dflash/ANALYSIS24K.md and the artifact HTML from the
computed results (every number comes from analysis24k/results/*.json or the
raw step outputs).

    python scripts/dflash_port/analysis/write_report.py [--artifact-url URL]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import report_text as T  # noqa: E402

RES = C.OUT / "results"
REPORT = C.REPO / "reporting/dflash/ANALYSIS24K.md"
TEMPLATE = Path(__file__).resolve().parent / "artifact_template.html"
HTML = C.OUT / "artifact" / "where_dflash_moves.html"


def load(name):
    return json.loads((RES / name).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-url", default="(published with the Artifact tool; see the final summary)")
    args = parser.parse_args()
    data = load("artifact_data.json")
    results = {
        "errors_test": load("errors_test.json"),
        "errors_validation": load("errors_validation.json"),
        "knockouts": load("knockouts.json"),
        "sensitivity": load("sensitivity.json"),
        "masking": load("masking.json"),
        "repr": load("repr_validation.json"),
        "reproduction_validation": json.loads(
            (C.OUT / "slots/validation/reproduction.json").read_text()),
        "reproduction_test": json.loads((C.OUT / "slots/test/reproduction.json").read_text()),
        "grads_base": json.loads((C.grads_dir("base") / "summary.json").read_text()),
        "grads_static": json.loads((C.grads_dir("static") / "summary.json").read_text()),
        "attention": json.loads((C.OUT / "prompt/attention.json").read_text()),
    }
    numbers = T.numbers(data, results)
    REPORT.write_text(T.markdown(numbers, args.artifact_url))
    text = T.artifact_text(numbers)
    html = TEMPLATE.read_text()
    html = html.replace("/*DATA*/", json.dumps(slim(data), separators=(",", ":")))
    html = html.replace("/*TEXT*/", json.dumps(text, separators=(",", ":")))
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(html)
    C.write_json(RES / "report_numbers.json", numbers)
    print(REPORT, HTML, f"{len(html) / 1e3:.0f} kB")


def slim(data):
    """Only what the page draws."""
    keep = ("block_outcomes", "transition_contrib", "slot_acc_onpath", "error_summary",
            "grads_base", "grads_static", "head_knockouts", "module_knockouts",
            "sensitivity", "masking", "attention", "repr")
    out = {k: data[k] for k in keep if k in data}
    if "attention" in out:
        out["attention"] = {a: {k: v for k, v in r.items() if k != "per_category_layer"}
                            for a, r in out["attention"].items()}
    for point in ("grads_base", "grads_static"):
        if point in out:
            out[point] = {k: v for k, v in out[point].items()
                          if k in ("modules", "gain", "capture", "heldout_capture",
                                   "split_half", "corrected_cos", "fc")}
    if "repr" in out:
        out["repr"] = {k: out["repr"][k] for k in ("rel_change", "cka", "probes")}
    if "masking" in out:
        out["masking"] = {"arms": out["masking"]["arms"]}
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
