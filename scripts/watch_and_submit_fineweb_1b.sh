#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
prep_pid_file=$project_root/remote_logs/prepare-fineweb-1b.pid
data_dir=$project_root/data/v2_fineweb_1b
watch_log=$project_root/remote_logs/watch-and-submit-fineweb-1b.log

exec >>"$watch_log" 2>&1
echo "watch_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ ! -s "$prep_pid_file" ]]; then
  echo "missing preparation PID receipt: $prep_pid_file" >&2
  exit 2
fi
prep_pid=$(<"$prep_pid_file")
if [[ ! "$prep_pid" =~ ^[0-9]+$ ]]; then
  echo "invalid preparation PID: $prep_pid" >&2
  exit 2
fi

while kill -0 "$prep_pid" 2>/dev/null; do
  sleep 60
done
if [[ ! -s "$data_dir/metadata.json" ]]; then
  echo "preparation process exited before metadata was committed" >&2
  exit 3
fi

cd "$project_root"
/ssd/scxi253/pretraining2/runtime/venv/bin/python scripts/verify_shard_manifest.py "$data_dir"
for run_dir in \
  runs/l20-edu-135m-fineweb-1b \
  runs/l20-edu-135m-fineweb-wsd-1b \
  runs/l20-edu-140m-wide-fineweb-1b \
  runs/l20-edu-140m-wide-fineweb-wsd-1b; do
  if [[ -e "$run_dir" ]]; then
    echo "refusing to launch because a target exists: $run_dir" >&2
    exit 4
  fi
done

submission=$(sbatch --parsable \
  --partition=gpu_4090 \
  --gres=gpu:1 \
  --cpus-per-task=6 \
  --time=08:00:00 \
  --array=0-3 \
  --job-name=fineweb-1b-factorial \
  --output="$project_root/remote_logs/fineweb-1b-factorial-%A_%a.out" \
  --error="$project_root/remote_logs/fineweb-1b-factorial-%A_%a.err" \
  scripts/slurm_fineweb_1b_factorial.sh)
echo "submission=$submission"
echo "watch_completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
