#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
run_dir=$project_root/runs/l20-edu-135m-gpu-smoke

if [[ ! -x "$runtime_python" ]]; then
  echo "runtime Python is missing: $runtime_python" >&2
  exit 2
fi
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing smoke run: $run_dir" >&2
  exit 3
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$project_root/tmp"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$project_root"

echo "execution_scope=synthetic_smoke_only_not_quality_evidence"
echo "hostname=$(hostname)"
nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader

"$runtime_python" scripts/verify_shard_manifest.py data/v2_smoke_shards
"$runtime_python" -m pytest -q
"$runtime_python" -m l20_pretrain.train configs/l20_135m_gpu_smoke.yaml --device cuda

final_checkpoint=$run_dir/step-000003
test -s "$final_checkpoint/model.safetensors"
test -s "$final_checkpoint/trainer_state.pt"
sha256sum \
  "$final_checkpoint/model.safetensors" \
  "$final_checkpoint/trainer_state.pt"

nvidia-smi \
  --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
echo "smoke_status=pass"
