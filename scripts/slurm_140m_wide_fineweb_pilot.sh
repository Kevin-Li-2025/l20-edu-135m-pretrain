#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
run_dir=$project_root/runs/l20-edu-140m-wide-fineweb-pilot-50m
telemetry_path=$project_root/remote_logs/wide-fineweb-pilot-telemetry-${SLURM_JOB_ID:-manual}.csv

if [[ ! -x "$runtime_python" ]]; then
  echo "runtime Python is missing: $runtime_python" >&2
  exit 2
fi
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing wide pilot run: $run_dir" >&2
  exit 3
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$project_root/tmp"

cd "$project_root"
echo "execution_scope=matched_50m_wide_baseline_not_final_quality_evidence"
"$runtime_python" scripts/verify_shard_manifest.py data/v2_fineweb_pilot_50m

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop=10 >"$telemetry_path" &
telemetry_pid=$!
cleanup() {
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup EXIT

"$runtime_python" -m l20_pretrain.train \
  configs/l20_140m_wide_fineweb_pilot_50m.yaml \
  --device cuda

final_checkpoint=$run_dir/step-000313
test -s "$final_checkpoint/model.safetensors"
test -s "$final_checkpoint/trainer_state.pt"
sha256sum \
  "$final_checkpoint/model.safetensors" \
  "$final_checkpoint/trainer_state.pt"
echo "telemetry=$telemetry_path"
echo "wide_pilot_status=pass"
