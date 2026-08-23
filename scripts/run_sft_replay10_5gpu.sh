#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/a40-pretrain}"
PYTHON_BIN="${PYTHON_BIN:-/opt/a40-pretrain-venv/bin/python}"
CONFIG="${1:-${PROJECT_ROOT}/configs/posttrain/smol_smoltalk_sft_replay10_v1.yaml}"
RUN_ROOT="${PROJECT_ROOT}/runs/posttrain/smol-smoltalk-sft-replay10-v1"
LOG_ROOT="${PROJECT_ROOT}/logs/posttrain"
LOG_PATH="${LOG_ROOT}/smol-smoltalk-sft-replay10-v1.log"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
if [[ -e "${RUN_ROOT}/.complete" ]]; then
  printf 'run already complete: %s\n' "${RUN_ROOT}" >&2
  exit 0
fi

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=5 \
  -m l20_pretrain.train_sft \
  "${CONFIG}" \
  2>&1 | tee -a "${LOG_PATH}"

touch "${RUN_ROOT}/.complete"
