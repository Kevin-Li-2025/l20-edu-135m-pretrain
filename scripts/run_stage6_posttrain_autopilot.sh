#!/usr/bin/env bash
set -euo pipefail

cd "${L20_PRETRAIN_DIR:-/home/hhai/l20-pretrain}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PATH="$PWD/.venv/bin:$PATH"

STATE_DIR="${STAGE6_STATE_DIR:-runs/stage6-edu-reasoning-state}"
LOG_DIR="${STAGE6_LOG_DIR:-logs/stage6-edu-reasoning}"
POST_DIR="$LOG_DIR/posttrain"
PROFILE_ROOT="${STAGE6_PROFILE_ROOT:-$LOG_DIR/profile}"
MODEL_DIR="${STAGE6_MODEL_DIR:-runs/l20-stage6-edu-reasoning-300m}"
MODEL_FINAL="${STAGE6_MODEL_FINAL:-$MODEL_DIR/final}"
EVAL_DIR="${STAGE6_EVAL_DIR:-eval_results/stage6_edu_reasoning_300m}"
SUMMARY_JSON="${STAGE6_SUMMARY_JSON:-results/stage6/posttrain_summary.json}"
SUMMARY_MD="${STAGE6_SUMMARY_MD:-results/stage6/posttrain_summary.md}"
WAIT_SECONDS="${STAGE6_WAIT_SECONDS:-60}"
TRAIN_PATTERN="${STAGE6_TRAIN_PATTERN:-.venv/bin/python -m l20_pretrain.train configs/l20_stage6_edu_reasoning_300m.yaml}"
PROFILE_PATTERN="${STAGE6_PROFILE_PATTERN:-run_stage6_tensor_profile_after_train.sh|ncu.*stage6_tensor_profile|l20_stage6_edu_reasoning_300m_tensor_profile.yaml}"

mkdir -p "$STATE_DIR" "$POST_DIR" "$PROFILE_ROOT" "$(dirname "$SUMMARY_JSON")" "$(dirname "$SUMMARY_MD")"

LOCK_FILE="$STATE_DIR/posttrain.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Stage6 post-train autopilot is already active; refusing to start a second copy." >&2
  exit 70
fi

json_status() {
  local event="$1"
  local detail="${2:-}"
  if [ -n "$detail" ]; then
    printf '{"event":"%s","detail":"%s","updated_at":"%s"}\n' \
      "$event" "$detail" "$(date -Is)" | tee "$POST_DIR/status.json"
  else
    printf '{"event":"%s","updated_at":"%s"}\n' "$event" "$(date -Is)" | tee "$POST_DIR/status.json"
  fi
}

have_lm_eval_result() {
  [ -d "$EVAL_DIR" ] && find "$EVAL_DIR" -type f -name "*.json" -size +0c | grep -q .
}

latest_profile_dir() {
  find "$PROFILE_ROOT" -maxdepth 1 -type d -name "tensor_profile_*" 2>/dev/null | sort | tail -1
}

profile_done() {
  local latest
  latest="$(latest_profile_dir)"
  [ -n "$latest" ] && [ -f "$latest/status.json" ] && grep -q '"ncu_profile_done"' "$latest/status.json"
}

wait_for_stage6_train() {
  while pgrep -af "$TRAIN_PATTERN" >/dev/null; do
    json_status waiting_for_stage6_train
    sleep "$WAIT_SECONDS"
  done
  if [ ! -e "$MODEL_FINAL" ]; then
    json_status missing_final_checkpoint "$MODEL_FINAL"
    echo "Stage6 final checkpoint is missing: $MODEL_FINAL" >&2
    exit 2
  fi
}

run_or_wait_for_profile() {
  if profile_done; then
    json_status ncu_profile_already_done "$(latest_profile_dir)"
    return
  fi

  if pgrep -af "$PROFILE_PATTERN" >/dev/null; then
    while pgrep -af "$PROFILE_PATTERN" >/dev/null; do
      json_status waiting_for_existing_ncu_profile
      sleep "$WAIT_SECONDS"
    done
    if profile_done; then
      json_status ncu_profile_done "$(latest_profile_dir)"
      return
    fi
  fi

  json_status ncu_profile_start
  set +e
  scripts/run_stage6_tensor_profile_after_train.sh 2>&1 | tee "$POST_DIR/tensor_profile.log"
  profile_code=${PIPESTATUS[0]}
  set -e
  if [ "$profile_code" -ne 0 ]; then
    json_status ncu_profile_failed "exit_code=$profile_code"
    if [ "${STAGE6_REQUIRE_NCU:-0}" = "1" ]; then
      exit "$profile_code"
    fi
    return
  fi
  if ! profile_done; then
    json_status ncu_profile_missing_after_run
    echo "Nsight Compute profile did not produce a completed profile status." >&2
    exit 3
  fi
  json_status ncu_profile_done "$(latest_profile_dir)"
}

run_eval() {
  if [ "${STAGE6_FORCE_EVAL:-0}" != "1" ] && have_lm_eval_result; then
    json_status eval_already_done "$EVAL_DIR"
    return
  fi

  mkdir -p "$EVAL_DIR"
  json_status eval_start "$EVAL_DIR"
  TASKS="${STAGE6_EVAL_TASKS:-arc_challenge,arc_easy,hellaswag,lambada_openai,piqa,winogrande}" \
  DEVICE="${STAGE6_EVAL_DEVICE:-cuda:0}" \
  DTYPE="${STAGE6_EVAL_DTYPE:-bfloat16}" \
  BATCH_SIZE="${STAGE6_EVAL_BATCH_SIZE:-auto}" \
    scripts/eval_lm_harness.sh "$MODEL_FINAL" "$EVAL_DIR" 2>&1 | tee "$POST_DIR/eval.log"
  json_status eval_done "$EVAL_DIR"
}

write_summary() {
  json_status summarize_start
  PYTHONPATH=src .venv/bin/python scripts/summarize_stage6_posttrain.py \
    --train-log "$LOG_DIR/train_stage6.log" \
    --profile-dir "$PROFILE_ROOT" \
    --eval-dir "$EVAL_DIR" \
    --out-json "$SUMMARY_JSON" \
    --out-md "$SUMMARY_MD" \
    2>&1 | tee "$POST_DIR/summarize.log"
  json_status complete "$SUMMARY_JSON"
}

wait_for_stage6_train
run_or_wait_for_profile
run_eval
write_summary
