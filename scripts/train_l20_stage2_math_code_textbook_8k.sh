#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/l20_edu_135m_stage2_math_code_textbook_8k.yaml}"
OUTPUT_DIR="$(python - <<'PY' "$CONFIG"
from pathlib import Path
import sys
import yaml

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["output_dir"])
PY
)"

INIT_DIR="$(python - <<'PY' "$CONFIG"
from pathlib import Path
import sys
import yaml

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["init_model_name_or_path"])
PY
)"

if [ ! -d "$INIT_DIR" ]; then
  echo "Stage-1 final checkpoint is missing: $INIT_DIR" >&2
  exit 2
fi

RESUME_DIR=""
if [ -d "$OUTPUT_DIR" ]; then
  RESUME_DIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'step-*' | sort | tail -n 1)"
fi

if [ -n "$RESUME_DIR" ] && [ -f "$RESUME_DIR/trainer_state.pt" ]; then
  echo "Resuming stage2 from $RESUME_DIR"
  python -m l20_pretrain.train "$CONFIG" --resume "$RESUME_DIR"
else
  echo "Starting stage2 from $INIT_DIR"
  python -m l20_pretrain.train "$CONFIG"
fi
