#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_DIR="${DATA_DIR:-data/l20_stage3_dclm_edu_replay_8k}"
POLISH_FINAL="${POLISH_FINAL:-runs/l20-edu-135m-stage2-replay-polish-8k/final}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/l20_edu_135m_stage3_dclm_edu_replay_8k.yaml}"
PREPARE_LOG="${PREPARE_LOG:-$LOG_DIR/l20_stage3_dclm_edu_replay_prepare_latest.log}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/l20_stage3_dclm_edu_replay_train_latest.log}"

mkdir -p "$LOG_DIR"

active_train_pids() {
  pgrep -af "python -m l20_pretrain.train" || true
}

data_complete() {
  "$PYTHON" - "$DATA_DIR" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
metadata_path = root / "metadata.json"
if not metadata_path.is_file() or not (root / "train.bin").is_file() or not (root / "val.bin").is_file():
    raise SystemExit(1)
metadata = json.loads(metadata_path.read_text())
target = int(metadata.get("target_tokens") or 0)
train_tokens = int(metadata.get("train_tokens") or 0)
val_tokens = int(metadata.get("val_tokens") or 0)
if target <= 0 or train_tokens < target or val_tokens <= 0:
    raise SystemExit(1)
PY
}

echo "stage3_wait_for_polish $(date -Is)"
while [ ! -e "$POLISH_FINAL" ]; do
  echo "waiting for polish final: $POLISH_FINAL"
  sleep "$POLL_SECONDS"
done

while active_train_pids | grep -q .; do
  echo "waiting for active training to finish"
  active_train_pids
  sleep "$POLL_SECONDS"
done

if data_complete; then
  echo "stage3_data_complete $(date -Is)"
else
  echo "stage3_prepare_start $(date -Is)"
  PYTHON="$PYTHON" bash scripts/prepare_l20_stage3_dclm_edu_replay_8k.sh 2>&1 | tee "$PREPARE_LOG"
fi

echo "stage3_train_start $(date -Is)"
PYTHON="$PYTHON" bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh "$TRAIN_CONFIG" 2>&1 | tee "$TRAIN_LOG"
echo "stage3_train_done $(date -Is)"
