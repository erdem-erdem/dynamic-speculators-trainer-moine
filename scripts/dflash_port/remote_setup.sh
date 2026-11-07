#!/usr/bin/env bash
# DFlash port: remote box setup check (P0). Idempotent; fails loudly.
#
#   scripts/dflash_port/remote_setup.sh                 # check only (default)
#   scripts/dflash_port/remote_setup.sh --install-vllm  # offline vLLM install
#   scripts/dflash_port/remote_setup.sh --download-target
#   scripts/dflash_port/remote_setup.sh --prepare-drafter <40-char commit SHA>
#
# Check mode reports every item and prints the exact command for anything
# missing; long steps (installs, downloads, drafter preparation) run only with
# their flag. Writes output/dflash-lora/env/env-<UTC time>.json (pip freeze of
# both venvs, nvidia-smi, git SHA). Run from the repository root.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
TARGET_REPO="Qwen/Qwen3-8B"
TARGET_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
TARGET_DIR="models/qwen3-8b"
DRAFTER_REPO="z-lab/Qwen3-8B-DFlash-b16"
DRAFTER_ROOT="output/dflash-drafters/z-lab-qwen3-8b-dflash-b16"
VLLM_VERSION="0.31.0"
TRAJECTORIES="datasets/eagle3-qwen3-8b-trajectories-v1/releases/r0001"

install_vllm=0; download_target=0; drafter_sha=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-vllm) install_vllm=1 ;;
    --download-target) download_target=1 ;;
    --prepare-drafter) drafter_sha="${2:?--prepare-drafter needs a commit SHA}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

missing=0
ok() { echo "  [ok]      $*"; }
todo() { echo "  [missing] $1"; echo "            run: $2"; missing=1; }

echo "== training env (.venv)"
if .venv/bin/python -c "import speculators" 2>/dev/null; then
  ok "speculators $(.venv/bin/python -c 'from importlib.metadata import version; print(version("speculators"))')"
else
  todo ".venv with speculators" "see dflash_port/RUNLOG.md S0 (offline wheels in ./wheels/train)"
fi

echo "== extraction env (.venv-vllm)"
if [[ $install_vllm == 1 ]]; then
  [[ -x .venv-vllm/bin/python ]] || python3.12 -m venv .venv-vllm
  .venv-vllm/bin/pip install --no-index --find-links wheels/vllm "vllm==${VLLM_VERSION}"
fi
if .venv-vllm/bin/python -c "import vllm" 2>/dev/null; then
  found="$(.venv-vllm/bin/python -c 'import vllm; print(vllm.__version__)')"
  [[ "$found" == "$VLLM_VERSION" ]] || { echo "vLLM $found != $VLLM_VERSION" >&2; exit 1; }
  ok "vllm $found"
  if .venv-vllm/bin/python -c "import hs_connectors" 2>/dev/null; then
    ok "hs_connectors (extraction backend)"
  else
    echo "  [note]    hs_connectors absent: launch_vllm.py uses its vendored File backend"
  fi
else
  todo "vllm ${VLLM_VERSION} in .venv-vllm" "$0 --install-vllm"
fi

echo "== target model (${TARGET_REPO}@${TARGET_REVISION:0:8})"
if [[ $download_target == 1 ]]; then
  .venv/bin/hf download "$TARGET_REPO" --revision "$TARGET_REVISION" --local-dir "$TARGET_DIR"
fi
if [[ -f "$TARGET_DIR/config.json" ]]; then
  ok "$TARGET_DIR"
else
  todo "$TARGET_DIR" "$0 --download-target"
fi

echo "== trajectories (r0001, 4K pool)"
for category in chat code math stem; do
  data="$TRAJECTORIES/$category/data.json"
  [[ -f "$data" ]] || { todo "$data" "restore the local trajectory release"; continue; }
  want="$(.venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$data.sha256.json")"
  have="$(sha256sum "$data" | cut -d' ' -f1)"
  [[ "$want" == "$have" ]] || { echo "SHA256 mismatch: $data" >&2; exit 1; }
  ok "$data ($have)"
done

echo "== DFlash drafter (${DRAFTER_REPO})"
if [[ -n "$drafter_sha" ]]; then
  [[ -f "$TARGET_DIR/config.json" ]] || { echo "target model needed first" >&2; exit 1; }
  .venv/bin/python scripts/dflash_port/prepare_dflash_drafter.py \
    --repository "$DRAFTER_REPO" --revision "$drafter_sha" \
    --verifier "$TARGET_DIR" --output-root "$DRAFTER_ROOT"
fi
prepared="$(ls -d "$DRAFTER_ROOT"/*/ 2>/dev/null | head -1 || true)"
if [[ -n "$prepared" && -f "$prepared/preparation.json" ]]; then
  ok "$prepared (set training_recipe drafter_revision and online.drafter_model_path)"
else
  todo "prepared drafter" "$0 --prepare-drafter <commit SHA of ${DRAFTER_REPO}>"
fi

mkdir -p output/dflash-lora/env
env_file="output/dflash-lora/env/env-$(date -u +%Y%m%dT%H%M%SZ).json"
.venv/bin/python - "$env_file" <<'PY'
import json, subprocess, sys
def run(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
json.dump({
    "git_sha": run("git", "rev-parse", "HEAD").strip(),
    "git_status": run("git", "status", "--short"),
    "nvidia_smi": run("nvidia-smi"),
    "venv": run(".venv/bin/pip", "freeze"),
    "venv_vllm": run(".venv-vllm/bin/pip", "freeze"),
}, open(sys.argv[1], "w"), indent=2)
PY
echo "== environment record: $env_file"
if [[ $missing == 1 ]]; then
  echo "Setup incomplete (see [missing] above)." >&2
  exit 1
fi
echo "Setup complete."
