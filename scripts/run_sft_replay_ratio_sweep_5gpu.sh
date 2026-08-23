#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
LOG_ROOT="${PROJECT_ROOT}/logs/posttrain"
mkdir -p "${LOG_ROOT}"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}"

for ratio in 20 30; do
  config="${PROJECT_ROOT}/configs/posttrain/smol_smoltalk_sft_replay${ratio}_v1.yaml"
  run_root="${PROJECT_ROOT}/runs/posttrain/smol-smoltalk-sft-replay${ratio}-v1"
  log_path="${LOG_ROOT}/smol-smoltalk-sft-replay${ratio}-v1.log"
  mkdir -p "${run_root}"
  if [[ -f "${run_root}/.complete" ]]; then
    continue
  fi
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=5 \
    -m l20_pretrain.train_sft \
    "${config}" \
    2>&1 | tee -a "${log_path}"
  touch "${run_root}/.complete"
done
