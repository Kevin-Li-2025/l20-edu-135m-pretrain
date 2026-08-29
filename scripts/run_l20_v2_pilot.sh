#!/usr/bin/env bash
set -euo pipefail

cd "${L20_PRETRAIN_DIR:-/home/hhai/l20-pretrain}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PARQUET_RANGE_WORKERS="${PARQUET_RANGE_WORKERS:-2}"
export PARQUET_RANGE_CHUNK_BYTES="${PARQUET_RANGE_CHUNK_BYTES:-33554432}"

STATE_DIR=runs/l20-v2-pilot-state
LOG_DIR=logs/l20-v2-pilot
mkdir -p "$STATE_DIR" "$LOG_DIR" data/benchmark_contamination models

write_state() {
  local stage="$1"
  printf '{"status":"running","stage":"%s","updated_at":"%s"}\n' \
    "$stage" "$(date -Is)" > "$STATE_DIR/status.json"
}

if [ ! -f models/l20-edu-135m/config.json ]; then
  write_state download_checkpoint
  .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AliceYin/l20-edu-135m",
    local_dir="models/l20-edu-135m",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.py"],
)
PY
fi

if [ ! -f data/benchmark_contamination/eval_5tasks.jsonl ]; then
  write_state contamination_index
  PYTHONPATH=src .venv/bin/python scripts/build_benchmark_contamination_index.py \
    --out data/benchmark_contamination/eval_5tasks.jsonl \
    2>&1 | tee "$LOG_DIR/contamination_index.log"
fi

write_state prepare_stage_a
while true; do
  resume_args=()
  if [ -f data/l20_v2_stage_a_targeted_5k_pilot/resume_state.pkl ]; then
    resume_args=(--resume)
  fi
  set +e
  HF_ENDPOINT="$HF_ENDPOINT" PYTHONPATH=src .venv/bin/python -m l20_pretrain.prepare_mixture_shards \
    --recipe configs/mixtures/l20_v2_stage_a_targeted_pilot.yaml \
    --checkpoint-interval 2000 \
    --max-rss-gb 6 \
    "${resume_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/prepare_stage_a.log"
  code=${PIPESTATUS[0]}
  set -e
  if [ "$code" -eq 0 ]; then
    break
  fi
  if [ "$code" -ne 75 ]; then
    exit "$code"
  fi
  write_state prepare_stage_a_restart
done

write_state train_stage_a_pilot
PYTHONPATH=src .venv/bin/python -m l20_pretrain.train \
  configs/l20_v2_stage_a_5k_pilot.yaml \
  2>&1 | tee "$LOG_DIR/train_stage_a_pilot.log"

printf '{"status":"complete","stage":"complete","updated_at":"%s"}\n' \
  "$(date -Is)" > "$STATE_DIR/status.json"
