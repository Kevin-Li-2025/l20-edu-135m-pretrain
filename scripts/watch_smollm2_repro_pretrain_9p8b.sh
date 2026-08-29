#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

main_log="${SMOLLM2_LONGRUN_LOG:-$repo_root/a40_smollm2_repro_pretrain_9p8b_run.log}"
watchdog_log="${SMOLLM2_WATCHDOG_LOG:-$repo_root/a40_smollm2_repro_pretrain_9p8b_watchdog.log}"
lock_file="${SMOLLM2_WATCHDOG_LOCK:-$repo_root/.smollm2_repro_pretrain_9p8b_watchdog.lock}"
config="configs/a40_5x_smollm2_repro_pretrain_9p8b.yaml"
process_pattern="/opt/a40-pretrain-venv/bin/python -u -m l20_pretrain.train ${config}$"
max_restarts="${SMOLLM2_MAX_RESTARTS:-3}"
poll_seconds="${SMOLLM2_WATCHDOG_POLL_SECONDS:-30}"
restarts=0

exec 9>"$lock_file"
if ! flock -n 9; then
  printf '%s watchdog already active; exiting\n' "$(date -Is)" >> "$watchdog_log"
  exit 0
fi

while true; do
  if [[ -s "$main_log" ]] && grep -q '"event": "done"' "$main_log"; then
    printf '%s training complete; watchdog exiting\n' "$(date -Is)" >> "$watchdog_log"
    exit 0
  fi
  if pgrep -f -- "$process_pattern" >/dev/null; then
    sleep "$poll_seconds"
    continue
  fi
  if (( restarts >= max_restarts )); then
    printf '%s restart limit reached (%s)\n' "$(date -Is)" "$max_restarts" >> "$watchdog_log"
    exit 1
  fi

  restarts=$((restarts + 1))
  printf '%s training absent; restart %s/%s\n' \
    "$(date -Is)" "$restarts" "$max_restarts" >> "$watchdog_log"
  set +e
  bash scripts/run_smollm2_repro_pretrain_9p8b.sh >> "$main_log" 2>&1
  status=$?
  set -e
  printf '%s training command exited with status %s\n' \
    "$(date -Is)" "$status" >> "$watchdog_log"
  sleep 10
done
