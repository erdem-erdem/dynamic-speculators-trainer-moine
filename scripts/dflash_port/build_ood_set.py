"""Build the offline out-of-distribution set (`speculators.generator.ood_set`)
against a running `launch_vllm.py responses` server; see build_ood_set.sh.

    python scripts/dflash_port/build_ood_set.py --output output/dflash-lora/ood-v1 \\
        --endpoint http://127.0.0.1:8001/v1 --tokenizer models/qwen3-8b \\
        --provenance-dir output/dflash-lora/ood-v1.provenance/vllm
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests
from transformers import AutoTokenizer

from speculators.generator import ood_set
from speculators.generator.activation_store import check_extraction_provenance
from speculators.provenance import find_repo_root, require_git_sha

TARGET = "Qwen/Qwen3-8B"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=ood_set.DEFAULT_BENCHMARKS)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    # Hashed target weights and an identified vLLM, as for the store.
    server = check_extraction_provenance(args.provenance_dir)
    provenance = {
        "git_sha": require_git_sha(find_repo_root(Path(__file__))),
        "server": server,
        "server_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.provenance_dir.iterdir())
            if path.is_file()
        },
        "target": TARGET,
        "tokenizer": str(args.tokenizer.resolve()),
    }
    session = requests.Session()

    def post(payload):
        response = session.post(
            f"{args.endpoint}/completions", json=payload, timeout=3600
        )
        response.raise_for_status()
        return response.json()

    manifest = ood_set.build(
        args.output,
        AutoTokenizer.from_pretrained(args.tokenizer),
        post,
        TARGET,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        seed=args.seed,
        provenance=provenance,
        workers=args.workers,
    )
    print(json.dumps(manifest["subsets"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
