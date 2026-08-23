#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
CONFIG="${1:-${PROJECT_ROOT}/configs/posttrain/smol_smoltalk_sft_pilot_v1.yaml}"
LOG_ROOT="${PROJECT_ROOT}/logs/posttrain/sft-throughput-v1"
RUN_ROOT="${PROJECT_ROOT}/runs/posttrain/sft-throughput-v1"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

names=(micro16_liger micro24_liger micro32_liger micro24_plain micro24_liger_compile)
micro_batches=(16 24 32 24 24)
liger_flags=(--liger-kernel --liger-kernel --liger-kernel --no-liger-kernel --liger-kernel)
compile_flags=(--no-compile --no-compile --no-compile --no-compile --compile)
pids=()

for gpu in 0 1 2 3 4; do
  name="${names[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${PROJECT_ROOT}" \
    "${PYTHON_BIN}" -m l20_pretrain.train_sft "${CONFIG}" \
      --device cuda \
      --run-name "sft-throughput-${name}" \
      --output-dir "${RUN_ROOT}/${name}" \
      --micro-batch-size "${micro_batches[$gpu]}" \
      --gradient-accumulation-steps 1 \
      --max-steps 30 \
      --warmup-steps 3 \
      --eval-interval 0 \
      --save-interval 0 \
      --no-save-final \
      "${liger_flags[$gpu]}" \
      "${compile_flags[$gpu]}" \
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
