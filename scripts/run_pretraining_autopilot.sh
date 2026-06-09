#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AUTOPILOT_CONFIG="${AUTOPILOT_CONFIG:-configs/pretraining_autopilot.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

ts() {
  date -Is
}

log() {
  echo "[$(ts)] $*"
}

yaml_get() {
  local expr="$1"
  python - "$AUTOPILOT_CONFIG" "$expr" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = config
for part in sys.argv[2].split("."):
    if part == "":
        continue
    if isinstance(value, list):
        value = value[int(part)]
    else:
        value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print("" if value is None else value)
PY
}

stage_json() {
  local index="$1"
  python - "$AUTOPILOT_CONFIG" "$index" <<'PY'
from pathlib import Path
import json
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
stages = config.get("stages") or []
print(json.dumps(stages[int(sys.argv[2])], sort_keys=True))
PY
}

stage_field() {
  local stage="$1"
  local field="$2"
  python - "$stage" "$field" <<'PY'
import json
import sys

stage = json.loads(sys.argv[1])
value = stage.get(sys.argv[2])
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print("" if value is None else value)
PY
}

active_pretraining_pids() {
  pgrep -af "l20_pretrain\.(prepare_mixture_shards|train)( |$)" || true
}

wait_for_active_pretraining() {
  local poll_seconds
  poll_seconds="$(yaml_get poll_seconds)"
  poll_seconds="${poll_seconds:-600}"
  while active_pretraining_pids | grep -q .; do
    log "another pretraining job is active; waiting ${poll_seconds}s"
    active_pretraining_pids | sed 's/^/[active] /'
    sleep "$poll_seconds"
  done
}

data_complete() {
  local data_dir="$1"
  python - "$data_dir" <<'PY'
from pathlib import Path
import json
import sys

metadata_path = Path(sys.argv[1]) / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(1)
metadata = json.loads(metadata_path.read_text())
target = int(metadata.get("target_tokens") or 0)
val_target = int(metadata.get("val_tokens_target") or metadata.get("val_tokens") or 0)
train_tokens = int(metadata.get("train_tokens") or 0)
val_tokens = int(metadata.get("val_tokens") or 0)
if target <= 0 or train_tokens < target:
    raise SystemExit(1)
if val_target > 0 and val_tokens < val_target:
    raise SystemExit(1)
PY
}

train_complete() {
  local train_config="$1"
  python - "$train_config" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
output_dir = Path(config["output_dir"])
max_steps = int(config["trainer"]["max_steps"])
final_step = output_dir / f"step-{max_steps:06d}" / "trainer_state.pt"
if not final_step.is_file():
    raise SystemExit(1)
PY
}

run_logged() {
  local log_path="$1"
  shift
  log "running: $*"
  set +e
  "$@" 2>&1 | tee "$log_path"
  local status="${PIPESTATUS[0]}"
  set -e
  log "exit_code=${status}: $*"
  return "$status"
}

run_stage() {
  local stage="$1"
  local name data_dir train_config prepare_script train_script log_prefix

  name="$(stage_field "$stage" name)"
  data_dir="$(stage_field "$stage" data_dir)"
  train_config="$(stage_field "$stage" train_config)"
  prepare_script="$(stage_field "$stage" prepare_script)"
  train_script="$(stage_field "$stage" train_script)"
  log_prefix="$(stage_field "$stage" log_prefix)"
  log_prefix="${log_prefix:-$name}"

  log "stage_start name=${name}"

  if data_complete "$data_dir"; then
    log "data_complete name=${name} data_dir=${data_dir}"
  else
    run_logged "$LOG_DIR/${log_prefix}_prepare_autopilot.log" bash "$prepare_script"
  fi

  if train_complete "$train_config"; then
    log "train_complete name=${name} config=${train_config}"
  else
    run_logged "$LOG_DIR/${log_prefix}_train_autopilot.log" bash "$train_script" "$train_config"
  fi

  log "stage_done name=${name}"
}

main() {
  log "autopilot_start config=${AUTOPILOT_CONFIG}"
  wait_for_active_pretraining

  local stage_count
  stage_count="$(python - "$AUTOPILOT_CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(len(config.get("stages") or []))
PY
)"

  local index
  for ((index = 0; index < stage_count; index++)); do
    local stage enabled
    stage="$(stage_json "$index")"
    enabled="$(stage_field "$stage" enabled)"
    if [ "$enabled" = "false" ]; then
      log "stage_skip index=${index}"
      continue
    fi
    run_stage "$stage"
  done

  log "autopilot_done"
}

main "$@"
