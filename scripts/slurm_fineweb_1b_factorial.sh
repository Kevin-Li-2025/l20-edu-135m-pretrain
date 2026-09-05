#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must select a factorial cell}
configs=(
  configs/l20_135m_fineweb_1b.yaml
  configs/l20_135m_fineweb_wsd_1b.yaml
  configs/l20_140m_wide_fineweb_1b.yaml
  configs/l20_140m_wide_fineweb_wsd_1b.yaml
)

if (( task_id < 0 || task_id >= ${#configs[@]} )); then
  echo "invalid factorial task id: $task_id" >&2
  exit 2
fi
if [[ ! -x "$runtime_python" ]]; then
  echo "runtime Python is missing: $runtime_python" >&2
  exit 2
fi

config_path=${configs[$task_id]}
readarray -t config_values < <(
  "$runtime_python" - "$project_root/$config_path" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(config["run_name"])
print(config["output_dir"])
print(config["trainer"]["max_steps"])
PY
)
run_name=${config_values[0]}
run_dir=$project_root/${config_values[1]}
max_steps=${config_values[2]}
final_checkpoint=$(printf '%s/step-%06d' "$run_dir" "$max_steps")
telemetry_path=$project_root/remote_logs/${run_name}-telemetry-${SLURM_JOB_ID:-manual}.csv

if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing 1B run: $run_dir" >&2
  exit 3
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$project_root/tmp"

cd "$project_root"
echo "execution_scope=matched_1b_factorial_not_final_quality_evidence"
echo "factorial_cell=$task_id config=$config_path"
"$runtime_python" scripts/verify_shard_manifest.py data/v2_fineweb_1b

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

"$runtime_python" -m l20_pretrain.train "$config_path" --device cuda

test -s "$final_checkpoint/model.safetensors"
test -s "$final_checkpoint/trainer_state.pt"
sha256sum "$final_checkpoint/model.safetensors" "$final_checkpoint/trainer_state.pt"
echo "telemetry=$telemetry_path"
echo "fineweb_1b_factorial_status=pass"
