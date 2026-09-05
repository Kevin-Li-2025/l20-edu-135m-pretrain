#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
source_checkpoint=$project_root/runs/l20-edu-135m-gpu-smoke/step-000002
expected_checkpoint=$project_root/runs/l20-edu-135m-gpu-smoke/step-000003
run_dir=$project_root/runs/l20-edu-135m-gpu-smoke-resume

test -s "$source_checkpoint/model.safetensors"
test -s "$source_checkpoint/trainer_state.pt"
test -s "$expected_checkpoint/model.safetensors"
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing resume-equivalence run: $run_dir" >&2
  exit 2
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$project_root/tmp"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$project_root"
echo "execution_scope=resume_equivalence_not_quality_evidence"
"$runtime_python" -m l20_pretrain.train \
  configs/l20_135m_gpu_smoke_resume.yaml \
  --resume "$source_checkpoint" \
  --device cuda

actual_checkpoint=$run_dir/step-000003
cmp "$expected_checkpoint/model.safetensors" "$actual_checkpoint/model.safetensors"
sha256sum \
  "$expected_checkpoint/model.safetensors" \
  "$actual_checkpoint/model.safetensors"
echo "resume_equivalence=byte_exact"
