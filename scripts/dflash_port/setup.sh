#!/usr/bin/env bash
# DFlash generator port: set up a GPU box (clone -> environments -> data ->
# models -> prepared drafter -> machine config). Idempotent; fails loudly.
# Run from the repository root. Steps run only with their flag; with no flag
# the script only checks and prints the next command for anything missing.
#
#   scripts/dflash_port/setup.sh --fetch-wheels     # HF wheel repo -> ./wheels
#   scripts/dflash_port/setup.sh --create-envs      # .venv + .venv-vllm (offline)
#   scripts/dflash_port/setup.sh --fetch-data r0001 # 4K/24K pools (r0002-100k: 100K)
#   scripts/dflash_port/setup.sh --target /path/to/Qwen3-8B   # existing copy
#   scripts/dflash_port/setup.sh --download-target  # or download it
#   scripts/dflash_port/setup.sh --prepare-drafter <z-lab commit SHA>
#   scripts/dflash_port/setup.sh --fetch-store <commit SHA>  # 4K activation store
#                                                   # from HF instead of P0 extraction
#   scripts/dflash_port/setup.sh                    # check only
#
# Flags combine; they run in the order above. See examples/dflash-lora/QUICKSTART.md.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

WHEELS_REPO="erdem-erdem/dyn-spec-train-vllm-wheels-fixed-vllm"
WHEELS_REVISION="44f654615311a464b7845a7ee92e7bc7e67198a4"
DATA_REPO="azeeer/eagle3-qwen3-8b-trajectories-v1"
DATA_REVISION="93750ccc4ba1701c2ec6b75ca5e4c1a5d867daa6"
DATA_DIR="datasets/eagle3-qwen3-8b-trajectories-v1"
TARGET_REPO="Qwen/Qwen3-8B"
TARGET_REVISION="b968826d9c46dd6066d109eabc6255188de91218"
TARGET_DIR="models/qwen3-8b"
DRAFTER_REPO="z-lab/Qwen3-8B-DFlash-b16"
DRAFTER_ROOT="output/dflash-drafters/z-lab-qwen3-8b-dflash-b16"
MACHINE="examples/dflash-lora/machine.local.yaml"
STORE_REPO="erdem-erdem/dyn-spec-dflash-store-qwen3-8b-4k"
STORE_DIR="output/dflash-lora/activation-store-4k"
VLLM_VERSION="0.31.0"
GUIDELLM_VERSION="0.7.1"
# SHA256 of each release manifest (data.json) used by the recorded runs.
declare -A DATA_SHA256=(
  [r0001/chat]=da2f80c43b4a4b8b886f684267ac72385e758774cb998c61acff855f0285c980
  [r0001/code]=af3b2d1df46786f8bedcc8a223abec655ecc714ba40210c09ca7f288a13db215
  [r0001/math]=59edf56d93d16c7b71c06975aa57416c19dd3f642ea872c78a82a3ef8e8b4806
  [r0001/stem]=84303548406fafe670eb062ca2cafbdd0604c29ad1c212efc5015760ab5a6f39
  [r0002-100k/chat]=f72a3db6987d1109cf0aed62142574c275aef229499aee66487ea7289cafad17
  [r0002-100k/code]=eaa8450575061466d7a21073ed0b395420158855b624844a54912823da6e23d8
  [r0002-100k/math]=e84ce9d0c67b1bc711a8b4febead3b8bd9472e80d9039b076d88557da4867614
  [r0002-100k/stem]=c96d13528dfe57aba3dd95e1f2716d121f3a9d206f07e1aa666d2808d6c3f0dc
)

fetch_wheels=0; create_envs=0; pools=(); target=""; download_target=0; drafter_sha=""
store_sha=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch-wheels) fetch_wheels=1 ;;
    --create-envs) create_envs=1 ;;
    --fetch-data) pools+=("${2:?--fetch-data needs r0001 or r0002-100k}"); shift ;;
    --target) target="${2:?--target needs a directory}"; shift ;;
    --download-target) download_target=1 ;;
    --prepare-drafter) drafter_sha="${2:?--prepare-drafter needs a commit SHA}"; shift ;;
    --fetch-store) store_sha="${2:?--fetch-store needs the store repo commit SHA}"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "ERROR: $*" >&2; exit 1; }
hf_cli() {
  # huggingface_hub CLI: from .venv once it exists, else from PATH.
  if [[ -x .venv/bin/hf ]]; then .venv/bin/hf "$@";
  elif command -v hf >/dev/null; then hf "$@";
  else die "no 'hf' CLI: 'pip install huggingface_hub' (any Python), or copy ./wheels by hand"; fi
}

# 1. Wheels (offline installs; PyPI may be blocked on GPU boxes).
if [[ $fetch_wheels == 1 ]]; then
  hf_cli download "$WHEELS_REPO" --revision "$WHEELS_REVISION" --local-dir wheels
fi

# 2. Environments. Order matters for .venv-vllm: guidellm FIRST, then vLLM.
if [[ $create_envs == 1 ]]; then
  [[ -d wheels/train && -d wheels/vllm ]] || die "wheels missing: run with --fetch-wheels"
  command -v python3.12 >/dev/null || die "python3.12 is required"
  [[ -x .venv/bin/python ]] || python3.12 -m venv .venv
  .venv/bin/pip install --no-index --find-links wheels/train \
    -e ".[dev,generator,lora]" ./hs_connectors
  [[ -x .venv-vllm/bin/python ]] || python3.12 -m venv .venv-vllm
  .venv-vllm/bin/pip install --no-index --find-links wheels/vllm "guidellm==${GUIDELLM_VERSION}"
  .venv-vllm/bin/pip install --no-index --find-links wheels/vllm "vllm==${VLLM_VERSION}"
fi

# 3. Trajectory pools (r0001 = 4K and 24K pools; r0002-100k = 100K).
for pool in "${pools[@]}"; do
  [[ "$pool" == r0001 || "$pool" == r0002-100k ]] || die "unknown pool $pool"
  hf_cli download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" \
    --include "releases/${pool}/*" --local-dir "$DATA_DIR"
done

# 4. Target model: an existing copy at the pinned revision, or a download.
if [[ -n "$target" ]]; then
  meta="$target/.cache/huggingface/download/config.json.metadata"
  [[ -f "$target/config.json" ]] || die "$target has no config.json"
  if [[ -f "$meta" ]]; then
    [[ "$(head -1 "$meta")" == "$TARGET_REVISION" ]] \
      || die "$target is revision $(head -1 "$meta"), not $TARGET_REVISION"
  else
    die "$target has no HF download metadata to prove revision $TARGET_REVISION; use --download-target"
  fi
  mkdir -p models
  if [[ -L "$TARGET_DIR" || ! -e "$TARGET_DIR" ]]; then
    ln -sfn "$(cd "$target" && pwd)" "$TARGET_DIR"
  else
    die "$TARGET_DIR exists and is not a symlink"
  fi
fi
if [[ $download_target == 1 ]]; then
  hf_cli download "$TARGET_REPO" --revision "$TARGET_REVISION" --local-dir "$TARGET_DIR"
fi

# 5. DFlash drafter at a pinned commit -> machine config for the run scripts.
if [[ -n "$drafter_sha" ]]; then
  [[ -f "$TARGET_DIR/config.json" ]] || die "target model needed first (--target or --download-target)"
  prepared="$DRAFTER_ROOT/$drafter_sha"
  if [[ ! -f "$prepared/preparation.json" ]]; then
    .venv/bin/python scripts/dflash_port/prepare_dflash_drafter.py \
      --repository "$DRAFTER_REPO" --revision "$drafter_sha" \
      --verifier "$TARGET_DIR" --output-root "$DRAFTER_ROOT"
  fi
  .venv/bin/python - "$MACHINE" "$ROOT/$TARGET_DIR" "$drafter_sha" "$ROOT/$prepared" <<'PY'
import sys, yaml
path, target, revision, drafter = sys.argv[1:]
yaml.safe_dump(
    {"target_model_path": target, "drafter_revision": revision,
     "drafter_model_path": drafter},
    open(path, "w"),
)
print(f"wrote {path}")
PY
fi

# 6. Activation store from the Hub (pinned commit), verified file by file.
if [[ -n "$store_sha" ]]; then
  [[ "$store_sha" =~ ^[0-9a-f]{40}$ ]] || die "--fetch-store needs a 40-hex commit SHA"
  hf_cli download "$STORE_REPO" --repo-type dataset --revision "$store_sha" \
    --local-dir "$STORE_DIR"
  .venv/bin/python scripts/dflash_port/push_store.py verify "$STORE_DIR"
fi

# Check (always).
missing=0
ok() { echo "  [ok]      $*"; }
todo() { echo "  [missing] $1"; echo "            run: $2"; missing=1; }
echo "== wheels"
if [[ -d wheels/train && -d wheels/vllm ]]; then ok "wheels/{train,vllm}"; else todo wheels "$0 --fetch-wheels"; fi
echo "== training env (.venv)"
if .venv/bin/python -c "import speculators, hs_connectors" 2>/dev/null; then
  ok "speculators $(.venv/bin/python -c 'from importlib.metadata import version; print(version("speculators"))')"
else
  todo ".venv" "$0 --create-envs"
fi
echo "== extraction env (.venv-vllm)"
if (cd /tmp && "$ROOT/.venv-vllm/bin/python" -c "import vllm") 2>/dev/null; then
  found="$(cd /tmp && "$ROOT/.venv-vllm/bin/python" -c 'import vllm; print(vllm.__version__)')"
  [[ "$found" == "$VLLM_VERSION" ]] || die "vLLM $found != $VLLM_VERSION"
  ok "vllm $found"
  if (cd /tmp && "$ROOT/.venv-vllm/bin/python" -c "from hs_connectors import HiddenStatesBackend") 2>/dev/null; then
    ok "hidden-state backend: hs_connectors"
  else
    echo "  [note]    hidden-state backend: launch_vllm.py's vendored File backend"
  fi
else
  todo ".venv-vllm" "$0 --create-envs   (guidellm ${GUIDELLM_VERSION} first, then vllm ${VLLM_VERSION})"
fi
echo "== trajectories"
for key in "${!DATA_SHA256[@]}"; do
  data="$DATA_DIR/releases/$key/data.json"
  pool="${key%%/*}"
  if [[ ! -f "$data" ]]; then
    [[ "$pool" == r0001 ]] && todo "$data" "$0 --fetch-data r0001"
    [[ "$pool" == r0002-100k ]] && echo "  [later]   $data (100K: $0 --fetch-data r0002-100k)"
    continue
  fi
  have="$(sha256sum "$data" | cut -d' ' -f1)"
  [[ "$have" == "${DATA_SHA256[$key]}" ]] || die "SHA256 mismatch: $data"
  ok "$data"
done
echo "== target model ($TARGET_REPO@${TARGET_REVISION:0:8})"
if [[ -f "$TARGET_DIR/config.json" ]]; then ok "$TARGET_DIR"; else
  todo "$TARGET_DIR" "$0 --target /path/to/existing/Qwen3-8B   or   $0 --download-target"; fi
echo "== DFlash drafter ($DRAFTER_REPO) + machine config"
if [[ -f "$MACHINE" ]]; then ok "$MACHINE"; cat "$MACHINE" | sed 's/^/            /'; else
  todo "$MACHINE" "$0 --prepare-drafter <commit SHA of $DRAFTER_REPO>"; fi

echo "== 4K activation store"
if [[ -f "$STORE_DIR/identity.json" && -f "$STORE_DIR/extractor.json" ]]; then
  ok "$STORE_DIR ($(find "$STORE_DIR" -name '*.safetensors' | wc -l) entries)"
else
  echo "  [later]   $STORE_DIR: P0 fill (scripts/dflash_port/fill_store.sh) or $0 --fetch-store <SHA>"
fi

mkdir -p output/dflash-lora/env
env_file="output/dflash-lora/env/env-$(date -u +%Y%m%dT%H%M%SZ).json"
if [[ -x .venv/bin/python ]]; then
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
fi
if [[ $missing == 1 ]]; then echo "Setup incomplete (see [missing] above)." >&2; exit 1; fi
echo "Setup complete."
