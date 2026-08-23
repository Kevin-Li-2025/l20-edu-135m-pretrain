#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

nproc="${A40_NPROC_PER_NODE:-6}"
case "$nproc" in
  4)
    default_config="configs/a40_5x_l20_edu_135m_12b.yaml"
    ;;
  5)
    default_config="configs/a40_5x_l20_edu_135m_12b.yaml"
    ;;
  6)
    default_config="configs/a40_6x_l20_edu_135m_12b.yaml"
    ;;
  *)
    echo "A40_NPROC_PER_NODE must be 4, 5, or 6, got: $nproc" >&2
    exit 2
    ;;
esac

config="${1:-$default_config}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2
  exit 2
fi
python_bin="${A40_PYTHON:-}"
if [[ -z "$python_bin" && -x /opt/a40-pretrain-venv/bin/python ]]; then
  python_bin=/opt/a40-pretrain-venv/bin/python
fi
python_bin="${python_bin:-python}"

tokenized_path="$("$python_bin" -c 'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("dataset", {}).get("tokenized_path", ""))' "$config")"
if [[ -n "$tokenized_path" && ! -f "$tokenized_path/train.bin" ]]; then
  echo "Missing pretokenized data: $tokenized_path/train.bin" >&2
  exit 2
fi

visible_gpus="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
if (( visible_gpus < nproc )); then
  echo "Need $nproc visible GPUs, but PyTorch sees $visible_gpus" >&2
  exit 2
fi

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# This host's native mixed P2P/SHM ring hangs between the cross-NUMA GPU0
# and GPUs 1-4. Host-mediated collectives pass the same five-rank smoke test.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

extra_args=("$@")
if [[ -n "${A40_PREFLIGHT_STEPS:-}" ]]; then
  extra_args+=(
    --max-steps "$A40_PREFLIGHT_STEPS"
    --output-dir "${A40_PREFLIGHT_OUTPUT_DIR:-runs/a40-preflight-${nproc}gpu}"
  )
fi

exec "$python_bin" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$nproc" \
  -m l20_pretrain.train \
  "$config" \
  "${extra_args[@]}"
