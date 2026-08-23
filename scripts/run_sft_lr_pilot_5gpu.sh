#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
CONFIG="${1:-${PROJECT_ROOT}/configs/posttrain/smol_smoltalk_sft_pilot_v1.yaml}"
LOG_ROOT="${PROJECT_ROOT}/logs/posttrain/sft-lr-pilot-v1"
RUN_ROOT="${PROJECT_ROOT}/runs/posttrain/sft-lr-pilot-v1"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

names=(lr5e5 lr1e4 lr3e4 lr6e4 lr1e3)
learning_rates=(0.00005 0.0001 0.0003 0.0006 0.001)
pids=()

for gpu in 0 1 2 3 4; do
  name="${names[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${PROJECT_ROOT}" \
    "${PYTHON_BIN}" -m l20_pretrain.train_sft "${CONFIG}" \
      --device cuda \
      --run-name "smol-smoltalk-sft-${name}" \
      --output-dir "${RUN_ROOT}/${name}" \
      --learning-rate "${learning_rates[$gpu]}" \
      > "${LOG_ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
