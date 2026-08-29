#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
CONFIG="${1:-${PROJECT_ROOT}/configs/posttrain/skill_curriculum_sft_lr_pilot_v1.yaml}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/posttrain/skill-curriculum-sft-lr-pilot-v1}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/posttrain/skill-curriculum-sft-lr-pilot-v1}"
TRAIN_DATA="${PROJECT_ROOT}/data/posttrain/skill_curriculum_v1/final/train.jsonl"

test -s "${TRAIN_DATA}"
test -s "${PROJECT_ROOT}/data/posttrain/skill_curriculum_v1/final/manifest.json"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

names=(lr5e5 lr1e4 lr2e4 lr3e4 lr6e4)
learning_rates=(0.00005 0.0001 0.0002 0.0003 0.0006)
pids=()

for gpu in 0 1 2 3 4; do
  name="${names[$gpu]}"
  output="${RUN_ROOT}/${name}"
  if [[ -s "${output}/step-000200/model.safetensors" ]]; then
    continue
  fi
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=1 \
      PYTHONPATH="${PROJECT_ROOT}" \
      "${PYTHON_BIN}" -m l20_pretrain.train_sft "${CONFIG}" \
        --device cuda \
        --run-name "skill-curriculum-sft-${name}" \
        --output-dir "${output}" \
        --learning-rate "${learning_rates[$gpu]}"
  ) > "${LOG_ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
