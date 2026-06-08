#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1

CONFIG="${1:-configs/l20_edu_135m_continue_1b.yaml}"
OUTPUT_DIR="$(python - <<'PY' "$CONFIG"
from pathlib import Path
import sys
import yaml

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["output_dir"])
PY
)"

RESUME_DIR=""
if [ -d "$OUTPUT_DIR" ]; then
  RESUME_DIR="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'step-*' | sort | tail -n 1)"
fi

if [ -n "$RESUME_DIR" ] && [ -f "$RESUME_DIR/trainer_state.pt" ]; then
  echo "Resuming from $RESUME_DIR"
  set +e
  python -m l20_pretrain.train "$CONFIG" --resume "$RESUME_DIR"
  STATUS=$?
  set -e
else
  echo "Starting from config init_model_name_or_path"
  set +e
  python -m l20_pretrain.train "$CONFIG"
  STATUS=$?
  set -e
fi

echo "train_exit_code=$STATUS"
exit "$STATUS"
