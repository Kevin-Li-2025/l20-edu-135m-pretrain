#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/hhai/l20-pretrain"
cd "$ROOT"

STATE_DIR="$ROOT/runs/stage4-full-autopilot-state"
LOG_DIR="$ROOT/logs"
PIPELINE="scripts/run_stage4_full_autopilot.sh"
PYTHON="$ROOT/.venv-continue/bin/python"
WATCH_LOG="$LOG_DIR/stage4_full_autopilot_watchdog.log"
PID_FILE="$STATE_DIR/watchdog.pid"
MAX_RESTARTS="${STAGE4_WATCHDOG_MAX_RESTARTS:-12}"
SLEEP_SECONDS="${STAGE4_WATCHDOG_SLEEP_SECONDS:-300}"
RESTART_BACKOFF_SECONDS="${STAGE4_WATCHDOG_RESTART_BACKOFF_SECONDS:-120}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 8>"$STATE_DIR/watchdog.lock"
if ! flock -n 8; then
  echo "[$(date -Is)] another watchdog already owns $STATE_DIR/watchdog.lock" | tee -a "$WATCH_LOG"
  exit 0
fi
printf "%s\n" "$$" > "$PID_FILE"

log() { printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "$WATCH_LOG"; }

status_field() {
  local field="$1"
  "$PYTHON" - "$STATE_DIR/status.json" "$field" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(0)
d=json.loads(p.read_text())
print(d.get(sys.argv[2], ""))
PY
}

pipeline_running() {
  pgrep -f "bash $PIPELINE" >/dev/null || pgrep -f "$PIPELINE" >/dev/null
}

train_running() {
  pgrep -f "l20_pretrain.train .*l20_edu_135m_stage4_hq_crossdedup_8k.yaml" >/dev/null || \
  pgrep -f "l20_pretrain.train_sft .*l20_edu_135m_stage4_sft_anti_forgetting.yaml" >/dev/null
}

restarts=0
log "watchdog_start max_restarts=$MAX_RESTARTS sleep=${SLEEP_SECONDS}s"
while true; do
  status="$(status_field status)"
  stage="$(status_field stage)"
  if [ "$status" = "complete" ] || [ -s "$STATE_DIR/pipeline.done" ]; then
    log "pipeline_complete stage=$stage"
    exit 0
  fi

  if pipeline_running; then
    log "pipeline_running status=${status:-unknown} stage=${stage:-unknown}"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if train_running; then
    log "train_running_without_parent status=${status:-unknown} stage=${stage:-unknown}; waiting"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "max_restarts_reached restarts=$restarts status=${status:-unknown} stage=${stage:-unknown}"
    exit 2
  fi

  restarts=$((restarts + 1))
  log "restarting_pipeline attempt=$restarts previous_status=${status:-unknown} previous_stage=${stage:-unknown}"
  nohup bash "$PIPELINE" >> "$LOG_DIR/stage4_full_autopilot_restart_${restarts}.log" 2>&1 &
  sleep "$RESTART_BACKOFF_SECONDS"
done
