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
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LOG_DIR="${LOG_DIR:-logs}"
DATA_DIR="${DATA_DIR:-data/l20_stage3_dclm_edu_replay_8k}"
POLISH_FINAL="${POLISH_FINAL:-runs/l20-edu-135m-stage2-replay-polish-8k/final}"
STAGE3_CONFIG="${STAGE3_CONFIG:-configs/l20_edu_135m_stage3_dclm_edu_replay_8k.yaml}"
STAGE3_SAFE_CONFIG="${STAGE3_SAFE_CONFIG:-configs/l20_edu_135m_stage3_dclm_edu_replay_8k_safe.yaml}"
SFT_CONFIG="${SFT_CONFIG:-configs/l20_edu_135m_sft_tulu3_anti_forgetting.yaml}"

mkdir -p "$LOG_DIR" data/sft eval_results
if [ -x ".venv-eval/bin/lm_eval" ]; then
  export PATH="$PWD/.venv-eval/bin:$PATH"
fi

log() {
  echo "[$(date -Is)] $*"
}

active_train_pids() {
  pgrep -af "python -m l20_pretrain.train( |$)|python -m l20_pretrain.train_sft( |$)" || true
}

wait_for_no_training() {
  while active_train_pids | grep -q .; do
    log "waiting for active training process"
    active_train_pids
    sleep "$POLL_SECONDS"
  done
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
if target < 10_000_000_000 or train_tokens < target or val_tokens <= 0:
    raise SystemExit(1)
PY
}

log "ultra_pipeline_start"
while [ ! -e "$POLISH_FINAL" ]; do
  log "waiting for polish final: $POLISH_FINAL"
  sleep "$POLL_SECONDS"
done

wait_for_no_training

if data_complete; then
  log "stage3_data_complete data_dir=$DATA_DIR"
else
  log "stage3_prepare_start target_tokens=10000000000"
  PYTHON="$PYTHON" bash scripts/prepare_l20_stage3_dclm_edu_replay_8k.sh \
    2>&1 | tee "$LOG_DIR/l20_stage3_ultra_prepare_latest.log"
fi

log "stage3_train_start config=$STAGE3_CONFIG"
set +e
PYTHON="$PYTHON" bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh "$STAGE3_CONFIG" \
  2>&1 | tee "$LOG_DIR/l20_stage3_ultra_train_latest.log"
status="${PIPESTATUS[0]}"
set -e

if [ "$status" -ne 0 ]; then
  if grep -qi "out of memory\\|cuda" "$LOG_DIR/l20_stage3_ultra_train_latest.log"; then
    log "stage3_fast_config_failed_status=$status; retrying safe config=$STAGE3_SAFE_CONFIG"
    PYTHON="$PYTHON" bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh "$STAGE3_SAFE_CONFIG" \
      2>&1 | tee "$LOG_DIR/l20_stage3_ultra_train_safe_latest.log"
  else
    log "stage3_train_failed_status=$status"
    exit "$status"
  fi
fi

log "stage3_train_done"

if command -v lm_eval >/dev/null 2>&1; then
  log "stage3_smollm_eval_start"
  CANDIDATE=ours-stage3 OUTPUT_ROOT=eval_results/stage3_smollm_target_latest \
    bash scripts/eval_smollm_benchmark.sh \
      "ours-stage3=runs/l20-edu-135m-stage3-dclm-edu-replay-8k/final" \
      "smollm-135m=HuggingFaceTB/SmolLM-135M" \
      "smollm2-135m=HuggingFaceTB/SmolLM2-135M" \
      2>&1 | tee "$LOG_DIR/l20_stage3_smollm_eval_latest.log"
else
  log "stage3_smollm_eval_skipped lm_eval_not_on_path"
fi

if [ ! -s data/sft/tulu3_anti_forgetting_30k.jsonl ] || [ ! -s data/sft/tulu3_anti_forgetting_eval_1k.jsonl ]; then
  log "sft_mix_prepare_start"
  "$PYTHON" scripts/prepare_sft_anti_forgetting_mix.py \
    --target-size 30000 \
    --eval-size 1024 \
    --sft-source-limit 300000 \
    --replay-ratio 0.15 \
    --output data/sft/tulu3_anti_forgetting_30k.jsonl \
    --eval-output data/sft/tulu3_anti_forgetting_eval_1k.jsonl \
    --summary-output data/sft/tulu3_anti_forgetting_summary.json \
    2>&1 | tee "$LOG_DIR/sft_anti_forgetting_prepare_latest.log"
fi

log "sft_train_start config=$SFT_CONFIG"
"$PYTHON" -m l20_pretrain.train_sft "$SFT_CONFIG" \
  2>&1 | tee "$LOG_DIR/sft_tulu3_anti_forgetting_latest.log"

log "sft_sanity_eval_start"
"$PYTHON" scripts/eval_sft_sanity.py runs/l20-edu-135m-sft-tulu3-anti-forgetting/final \
  --output eval_results/sft_tulu3_anti_forgetting/results.jsonl \
  --markdown-output eval_results/sft_tulu3_anti_forgetting/report.md \
  2>&1 | tee "$LOG_DIR/sft_tulu3_anti_forgetting_eval_latest.log"

log "ultra_pipeline_done"
