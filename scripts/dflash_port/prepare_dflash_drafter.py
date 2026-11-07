"""Prepare any released DFlash drafter at a pinned commit for generator training.

Example (P0):
    python scripts/dflash_port/prepare_dflash_drafter.py \\
        --repository z-lab/Qwen3-8B-DFlash-b16 --revision <40-char commit SHA> \\
        --verifier models/qwen3-8b --output-root output/dflash-drafters/<name>

Writes `<output-root>/<revision>/` (converted weights, `config.json`,
`original_config.json`, `preparation.json`) and prints the path. The bank must
name the same repository/revision (`models.drafter`, `models.drafter_revision`)
and aux layers (`hidden_states.layer_ids[:-1]`); the runtime verifies this and
the weight SHA256 before training.
"""

import argparse

from speculators.generator.drafters.prepare_dflash import prepare_dflash

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True, help="exact commit SHA")
    parser.add_argument("--verifier", required=True, help="local verifier directory")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(
        prepare_dflash(args.repository, args.revision, args.verifier, args.output_root)
    )
