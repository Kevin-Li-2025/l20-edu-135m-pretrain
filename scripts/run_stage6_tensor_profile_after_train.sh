#!/usr/bin/env bash
set -euo pipefail

cd "${L20_PRETRAIN_DIR:-/home/hhai/l20-pretrain}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
else
  export PYTHONPATH="$PWD/src"
fi

STATE_DIR=runs/stage6-edu-reasoning-state
PROFILE_DIR=logs/stage6-edu-reasoning/profile
mkdir -p "$STATE_DIR" "$PROFILE_DIR"

LOCK_FILE="$STATE_DIR/tensor-profile.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Stage6 Tensor Core profiler is already active; refusing to start a second profiler." >&2
  exit 70
fi

NCU_BIN="${NCU_BIN:-}"
if [ -z "$NCU_BIN" ]; then
  for candidate in /usr/local/cuda-13.0/bin/ncu /opt/nvidia/nsight-compute/2025.3.1/ncu /usr/local/cuda/bin/ncu; do
    if [ -x "$candidate" ]; then
      NCU_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$NCU_BIN" ] || [ ! -x "$NCU_BIN" ]; then
  echo "Nsight Compute ncu binary not found." >&2
  exit 127
fi

main_pattern=".venv/bin/python -m l20_pretrain.train configs/l20_stage6_edu_reasoning_300m.yaml"
while pgrep -af "$main_pattern" >/dev/null; do
  echo "{\"event\":\"waiting_for_stage6_train\",\"updated_at\":\"$(date -Is)\"}" | tee "$PROFILE_DIR/wait.log"
  sleep "${PROFILE_WAIT_SECONDS:-60}"
done

if [ ! -e runs/l20-stage6-edu-reasoning-300m/final ]; then
  echo "Stage6 final checkpoint is missing; not profiling." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_dir="$PROFILE_DIR/tensor_profile_${timestamp}"
mkdir -p "$run_dir"

echo "{\"event\":\"ncu_profile_start\",\"updated_at\":\"$(date -Is)\",\"ncu\":\"$NCU_BIN\"}" | tee "$run_dir/status.json"

set +e
"$NCU_BIN" \
  --target-processes all \
  --set roofline \
  --launch-skip "${NCU_LAUNCH_SKIP:-200}" \
  --launch-count "${NCU_LAUNCH_COUNT:-80}" \
  --force-overwrite \
  --export "$run_dir/stage6_tensor_profile" \
  --log-file "$run_dir/ncu_profile.log" \
  .venv/bin/python -m l20_pretrain.train configs/l20_stage6_edu_reasoning_300m_tensor_profile.yaml
ncu_code=$?
set -e

if [ "$ncu_code" -ne 0 ]; then
  printf '{"event":"ncu_profile_failed","updated_at":"%s","exit_code":%s,"run_dir":"%s"}\n' \
    "$(date -Is)" "$ncu_code" "$run_dir" | tee "$run_dir/status.json"
  exit "$ncu_code"
fi

"$NCU_BIN" --import "$run_dir/stage6_tensor_profile.ncu-rep" --page details \
  > "$run_dir/ncu_details.txt" 2>&1 || true
"$NCU_BIN" --import "$run_dir/stage6_tensor_profile.ncu-rep" --page raw --csv \
  > "$run_dir/ncu_raw.csv" 2>&1 || true

echo "{\"event\":\"ncu_profile_done\",\"updated_at\":\"$(date -Is)\",\"run_dir\":\"$run_dir\"}" | tee "$run_dir/status.json"
