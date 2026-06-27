#!/usr/bin/env bash
set -euo pipefail

cd "${L20_PRETRAIN_DIR:-/home/hhai/l20-pretrain}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PARQUET_RANGE_WORKERS="${PARQUET_RANGE_WORKERS:-2}"
export PARQUET_RANGE_CHUNK_BYTES="${PARQUET_RANGE_CHUNK_BYTES:-33554432}"
export PATH="$PWD/.venv/bin:$PATH"

STATE_DIR=runs/stage7-smollm2-trial-state
LOG_DIR=logs/stage7-smollm2-trial
DATA_DIR=data/l20_stage7_smollm2_trial_50m
RUN_DIR=runs/l20-stage7-smollm2-trial-50m
EVAL_DIR=eval_results/stage7_smollm2_trial_50m
mkdir -p "$STATE_DIR" "$LOG_DIR" data/benchmark_contamination "$RUN_DIR" "$EVAL_DIR"

LOCK_FILE="$STATE_DIR/run.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Stage7 SmolLM2 trial runner is already active; refusing to start a second writer." >&2
  exit 70
fi

write_state() {
  local stage="$1"
  local status="${2:-running}"
  printf '{"status":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$status" "$stage" "$(date -Is)" > "$STATE_DIR/status.json"
}

if [ ! -e runs/l20-stage6-edu-reasoning-300m/final ]; then
  write_state missing_stage6_final failed
  echo "Missing Stage6 final checkpoint." >&2
  exit 2
fi

if [ ! -f data/benchmark_contamination/eval_5tasks.jsonl ]; then
  write_state contamination_index
  PYTHONPATH=src .venv/bin/python scripts/build_benchmark_contamination_index.py \
    --out data/benchmark_contamination/eval_5tasks.jsonl \
    2>&1 | tee "$LOG_DIR/contamination_index.log"
fi

write_state prepare_stage7
while true; do
  resume_args=()
  if [ -f "$DATA_DIR/resume_state.pkl" ]; then
    resume_args=(--resume)
  fi
  set +e
  PYTHONPATH=src .venv/bin/python -m l20_pretrain.prepare_mixture_shards \
    --recipe configs/mixtures/l20_stage7_smollm2_trial_50m.yaml \
    --checkpoint-interval 1000 \
    --max-rss-gb 6 \
    "${resume_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/prepare.log"
  code=${PIPESTATUS[0]}
  set -e
  if [ "$code" -eq 0 ]; then
    break
  fi
  if [ "$code" -ne 75 ]; then
    write_state prepare_stage7_failed failed
    exit "$code"
  fi
  write_state prepare_stage7_restart
done

write_state train_stage7
PYTHONPATH=src .venv/bin/python -m l20_pretrain.train \
  configs/l20_stage7_smollm2_trial_50m.yaml \
  2>&1 | tee "$LOG_DIR/train.log"

write_state eval_stage7
TASKS="${STAGE7_EVAL_TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande}" \
DEVICE="${STAGE7_EVAL_DEVICE:-cuda:0}" \
DTYPE="${STAGE7_EVAL_DTYPE:-bfloat16}" \
BATCH_SIZE="${STAGE7_EVAL_BATCH_SIZE:-auto}" \
  scripts/eval_lm_harness.sh "$RUN_DIR/final" "$EVAL_DIR" 2>&1 | tee "$LOG_DIR/eval.log"

PYTHONPATH=src .venv/bin/python scripts/summarize_smollm_benchmark.py \
  --result stage7-smollm2-trial="$EVAL_DIR" \
  --result stage6="$PWD/eval_results/stage6_edu_reasoning_300m" \
  --candidate stage7-smollm2-trial \
  --baseline stage6 \
  --out-md results/stage7_smollm2_trial_50m.md \
  --out-json results/stage7_smollm2_trial_50m.json \
  --out-csv results/stage7_smollm2_trial_50m.csv \
  2>&1 | tee "$LOG_DIR/summary.log"

write_state complete complete
