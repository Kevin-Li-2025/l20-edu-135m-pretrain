#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/hhai/l20-pretrain"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PARQUET_RANGE_CHUNK_BYTES="${PARQUET_RANGE_CHUNK_BYTES:-16777216}"
export PARQUET_RANGE_WORKERS="${PARQUET_RANGE_WORKERS:-12}"
export PARQUET_CHUNK_MAX_SECONDS="${PARQUET_CHUNK_MAX_SECONDS:-900}"
export PARQUET_MIN_BYTES_PER_SEC="${PARQUET_MIN_BYTES_PER_SEC:-65536}"
export PARQUET_LOW_SPEED_SECONDS="${PARQUET_LOW_SPEED_SECONDS:-30}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="$ROOT/.venv-eval/bin:$PATH"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

PYTHON="$ROOT/.venv-continue/bin/python"
LOG_DIR="$ROOT/logs"
STATE_DIR="$ROOT/runs/stage4-full-autopilot-state"
DATA_DIR="$ROOT/data/l20_stage4_hq_crossdedup_8k"
RUN_DIR="$ROOT/runs/l20-edu-135m-stage4-hq-crossdedup-8k"
TRAIN_CONFIG="$ROOT/configs/l20_edu_135m_stage4_hq_crossdedup_8k.yaml"
STAGE3_CONFIG="$ROOT/configs/l20_edu_135m_stage3_current_shard_8k.yaml"
EVAL_ROOT="$ROOT/eval_results/stage4_release"
SFT_RUN_DIR="$ROOT/runs/l20-edu-135m-stage4-sft-anti-forgetting"
SFT_CONFIG="$ROOT/configs/l20_edu_135m_stage4_sft_anti_forgetting.yaml"
mkdir -p "$LOG_DIR" "$STATE_DIR" "$EVAL_ROOT" "$ROOT/data/sft"

exec 9>"$STATE_DIR/pipeline.lock"
if ! flock -n 9; then
  echo "Another Stage 4 autopilot owns $STATE_DIR/pipeline.lock" >&2
  exit 3
fi

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
mark_done() { printf '%s\n' "$(date -Is)" > "$STATE_DIR/$1.done"; }
is_done() { [ -s "$STATE_DIR/$1.done" ]; }

on_exit() {
  status=$?
  "$PYTHON" - "$STATE_DIR/status.json" "$status" "${CURRENT_STAGE:-startup}" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "complete" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "stage": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2))
PY
  log "pipeline_exit status=$status stage=${CURRENT_STAGE:-startup}"
}
trap on_exit EXIT
trap 'log "signal_received"; exit 130' INT TERM HUP

set_stage() {
  CURRENT_STAGE="$1"
  "$PYTHON" - "$STATE_DIR/status.json" "$CURRENT_STAGE" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "running",
    "stage": sys.argv[2],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}, indent=2))
PY
}

set_stage preflight
"$PYTHON" scripts/stage4_preflight.py \
  --root "$ROOT" --out runs/stage4-full-autopilot-state/preflight.json
mark_done preflight

checkpoint_complete() {
  "$PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys, torch, yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
final = Path(config["output_dir"]) / "final"
if not final.exists():
    raise SystemExit(1)
state_path = final.resolve() / "trainer_state.pt"
if not state_path.is_file():
    raise SystemExit(1)
state = torch.load(state_path, map_location="cpu", weights_only=False)
if int(state.get("step", -1)) != int(config["trainer"]["max_steps"]):
    raise SystemExit(1)
PY
}

summary_has_models() {
  "$PYTHON" - "$1" "${@:2}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
d = json.loads(p.read_text())
means = d.get("means") or {}
if any(name not in means or means[name] is None for name in sys.argv[2:]):
    raise SystemExit(1)
PY
}

jsonl_count_at_least() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p, minimum = Path(sys.argv[1]), int(sys.argv[2])
if not p.is_file():
    raise SystemExit(1)
count = 0
with p.open() as f:
    for line in f:
        json.loads(line)
        count += 1
if count < minimum:
    raise SystemExit(1)
PY
}

archive_partial_dir() {
  path="$1"
  if [ -e "$path" ]; then
    mv "$path" "${path}.partial.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
}

set_stage wait_stage3
log "waiting_for_stage3"
while pgrep -f "l20_pretrain.train configs/l20_edu_135m_stage3_current_shard_8k.yaml" >/dev/null; do
  sleep 300
done
if ! checkpoint_complete "$STAGE3_CONFIG"; then
  log "stage3 did not reach configured max_steps; refusing to continue"
  exit 2
fi
mark_done stage3_verified

set_stage prepare_data
if [ ! -s "$DATA_DIR/metadata.json" ]; then
  log "stage4_prepare_start"
  bash scripts/prepare_l20_stage4_hq_crossdedup_8k.sh \
    2>&1 | tee -a "$LOG_DIR/l20_stage4_hq_crossdedup_prepare_latest.log"
fi

set_stage data_gate
log "stage4_data_gate"
"$PYTHON" scripts/check_stage4_data_gate.py \
  --data-dir "$DATA_DIR" --out "$EVAL_ROOT/data_gate.json" \
  2>&1 | tee "$LOG_DIR/l20_stage4_data_gate.log"
mark_done data_gate

set_stage train_smoke
if ! checkpoint_complete "$ROOT/configs/l20_edu_135m_stage4_smoke.yaml" 2>/dev/null; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import yaml
p=Path("configs/l20_edu_135m_stage4_hq_crossdedup_8k.yaml")
d=yaml.safe_load(p.read_text())
d["run_name"]="l20-edu-135m-stage4-smoke"
d["output_dir"]="runs/l20-edu-135m-stage4-smoke"
d["trainer"]["max_steps"]=3
d["trainer"]["warmup_steps"]=1
d["trainer"]["eval_interval"]=0
d["trainer"]["save_interval"]=3
d["trainer"]["keep_last_checkpoints"]=1
Path("configs/l20_edu_135m_stage4_smoke.yaml").write_text(yaml.safe_dump(d,sort_keys=False))
PY
  rm -rf "$ROOT/runs/l20-edu-135m-stage4-smoke"
  bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh \
    configs/l20_edu_135m_stage4_smoke.yaml \
    2>&1 | tee "$LOG_DIR/l20_stage4_train_smoke.log"
fi
checkpoint_complete "$ROOT/configs/l20_edu_135m_stage4_smoke.yaml"
mark_done train_smoke

set_stage train_stage4
if ! checkpoint_complete "$TRAIN_CONFIG"; then
  log "stage4_train_start_or_resume"
  bash scripts/train_l20_stage2_math_code_textbook_replay_8k.sh "$TRAIN_CONFIG" \
    2>&1 | tee -a "$LOG_DIR/l20_stage4_hq_crossdedup_train_latest.log"
fi
checkpoint_complete "$TRAIN_CONFIG"
mark_done train_stage4

set_stage select_base
"$PYTHON" scripts/select_best_pretrain_checkpoint.py \
  --log "$LOG_DIR/l20_stage4_hq_crossdedup_train_latest.log" \
  --run-dir "$RUN_DIR" --out "$EVAL_ROOT/best_checkpoint.json"
BEST="$("$PYTHON" -c 'import json; print(json.load(open("eval_results/stage4_release/best_checkpoint.json"))["checkpoint"])')"
[ -s "$BEST/model.safetensors" ] && [ -s "$BEST/config.json" ] && [ -s "$BEST/tokenizer.json" ]
mark_done select_base

set_stage eval_base
if ! summary_has_models "$EVAL_ROOT/base_eval/summary.json" stage4-best smollm-135m smollm2-135m; then
  archive_partial_dir "$EVAL_ROOT/base_eval"
  CANDIDATE=stage4-best OUTPUT_ROOT="$EVAL_ROOT/base_eval" \
    bash scripts/eval_smollm_benchmark.sh \
      "stage4-best=$BEST" \
      "smollm-135m=HuggingFaceTB/SmolLM-135M" \
      "smollm2-135m=HuggingFaceTB/SmolLM2-135M" \
    2>&1 | tee "$LOG_DIR/l20_stage4_eval_latest.log"
fi
summary_has_models "$EVAL_ROOT/base_eval/summary.json" stage4-best smollm-135m smollm2-135m
mark_done eval_base

set_stage prepare_sft
SFT_TRAIN="$ROOT/data/sft/stage4_smol_smoltalk_hq.jsonl"
SFT_EVAL="$ROOT/data/sft/stage4_smol_smoltalk_eval_2k.jsonl"
SFT_DATA_SUMMARY="$ROOT/data/sft/stage4_smol_smoltalk_summary.json"
if ! jsonl_count_at_least "$SFT_TRAIN" 100000 || ! jsonl_count_at_least "$SFT_EVAL" 2048; then
  rm -f "$SFT_TRAIN" "$SFT_EVAL" "$SFT_DATA_SUMMARY"
  "$PYTHON" scripts/prepare_sft_anti_forgetting_mix.py \
    --target-size 0 --eval-size 2048 --sft-source-limit 500000 \
    --max-example-tokens 8192 \
    --output "$SFT_TRAIN" --eval-output "$SFT_EVAL" \
    --summary-output "$SFT_DATA_SUMMARY" \
    2>&1 | tee "$LOG_DIR/stage4_sft_prepare_latest.log"
fi
jsonl_count_at_least "$SFT_TRAIN" 100000
jsonl_count_at_least "$SFT_EVAL" 2048

"$PYTHON" - "$BEST" "$SFT_DATA_SUMMARY" <<'PY'
import json, math
from pathlib import Path
import sys, yaml
p=Path("configs/l20_edu_135m_sft_tulu3_anti_forgetting.yaml")
d=yaml.safe_load(p.read_text())
d["run_name"]="l20-edu-135m-stage4-sft-anti-forgetting"
d["base_model"]=sys.argv[1]
d["output_dir"]="runs/l20-edu-135m-stage4-sft-anti-forgetting"
d["block_size"]=8192
d["dataset"]["local_jsonl_path"]="data/sft/stage4_smol_smoltalk_hq.jsonl"
d["dataset"]["eval_local_jsonl_path"]="data/sft/stage4_smol_smoltalk_eval_2k.jsonl"
d["dataset"]["max_chars"]=50000
d["dataset"]["eval_max_examples"]=2048
d["dataset"]["system_prompt"]="You are a helpful, accurate, concise AI assistant."
d["trainer"]["micro_batch_size"]=4
d["trainer"]["gradient_accumulation_steps"]=4
rows=int(json.loads(Path(sys.argv[2]).read_text())["train_rows"])
d["dataset"]["max_examples"]=rows
steps=math.ceil(rows * 2 / (
    d["trainer"]["micro_batch_size"] * d["trainer"]["gradient_accumulation_steps"]
))
d["trainer"]["learning_rate"]=3e-4
d["trainer"]["liger_kernel"]=True
d["trainer"]["max_steps"]=steps
d["trainer"]["warmup_steps"]=max(1, math.ceil(steps * 0.10))
d["trainer"]["eval_interval"]=max(500, steps // 8)
d["trainer"]["save_interval"]=max(500, steps // 8)
d["trainer"]["keep_last_checkpoints"]=4
Path("configs/l20_edu_135m_stage4_sft_anti_forgetting.yaml").write_text(
    yaml.safe_dump(d,sort_keys=False)
)
PY
mark_done prepare_sft

set_stage train_sft
if ! checkpoint_complete "$SFT_CONFIG"; then
  RESUME_DIR="$(find "$SFT_RUN_DIR" -maxdepth 1 -type d -name 'step-*' 2>/dev/null | sort | tail -n 1)"
  if [ -n "$RESUME_DIR" ] && [ -s "$RESUME_DIR/trainer_state.pt" ]; then
    "$PYTHON" -m l20_pretrain.train_sft "$SFT_CONFIG" --resume "$RESUME_DIR" \
      2>&1 | tee -a "$LOG_DIR/stage4_sft_train_latest.log"
  else
    "$PYTHON" -m l20_pretrain.train_sft "$SFT_CONFIG" \
      2>&1 | tee -a "$LOG_DIR/stage4_sft_train_latest.log"
  fi
fi
checkpoint_complete "$SFT_CONFIG"
"$PYTHON" scripts/select_best_pretrain_checkpoint.py \
  --log "$LOG_DIR/stage4_sft_train_latest.log" \
  --run-dir "$SFT_RUN_DIR" --out "$EVAL_ROOT/best_sft_checkpoint.json"
SFT_BEST="$("$PYTHON" -c 'import json; print(json.load(open("eval_results/stage4_release/best_sft_checkpoint.json"))["checkpoint"])')"
[ -s "$SFT_BEST/model.safetensors" ] && [ -s "$SFT_BEST/config.json" ]
mark_done train_sft

set_stage eval_sft_regression
BLEND_ROOT="$SFT_RUN_DIR/interpolated"
for spec in "a0875:0.875" "a0750:0.750" "a0500:0.500" "a0250:0.250"; do
  label="${spec%%:*}"
  alpha="${spec##*:}"
  if [ ! -s "$BLEND_ROOT/$label/model.safetensors" ]; then
    rm -rf "$BLEND_ROOT/$label"
    "$PYTHON" scripts/interpolate_checkpoints.py \
      --base "$BEST" --sft "$SFT_BEST" --alpha "$alpha" \
      --output "$BLEND_ROOT/$label"
  fi
done
if ! summary_has_models "$EVAL_ROOT/sft_eval/summary.json" \
  stage4-sft-full stage4-sft-a0875 stage4-sft-a0750 \
  stage4-sft-a0500 stage4-sft-a0250 stage4-base; then
  archive_partial_dir "$EVAL_ROOT/sft_eval"
  CANDIDATE=stage4-sft-full OUTPUT_ROOT="$EVAL_ROOT/sft_eval" \
    bash scripts/eval_smollm_benchmark.sh \
      "stage4-sft-full=$SFT_BEST" \
      "stage4-sft-a0875=$BLEND_ROOT/a0875" \
      "stage4-sft-a0750=$BLEND_ROOT/a0750" \
      "stage4-sft-a0500=$BLEND_ROOT/a0500" \
      "stage4-sft-a0250=$BLEND_ROOT/a0250" \
      "stage4-base=$BEST" \
    2>&1 | tee "$LOG_DIR/stage4_sft_eval_latest.log"
fi
"$PYTHON" - "$SFT_BEST" "$BLEND_ROOT" <<'PY'
import json
from pathlib import Path
import sys
d=json.loads(Path("eval_results/stage4_release/sft_eval/summary.json").read_text())
base=float(d["means"]["stage4-base"])
candidates=[
    ("stage4-sft-full",1.0,Path(sys.argv[1])),
    ("stage4-sft-a0875",0.875,Path(sys.argv[2])/"a0875"),
    ("stage4-sft-a0750",0.750,Path(sys.argv[2])/"a0750"),
    ("stage4-sft-a0500",0.500,Path(sys.argv[2])/"a0500"),
    ("stage4-sft-a0250",0.250,Path(sys.argv[2])/"a0250"),
]
evaluations=[]
selected=None
for name,alpha,path in candidates:
    mean=float(d["means"][name])
    drops={
        row["task"]: float(row["stage4-base"])-float(row[name])
        for row in d["tasks"]
        if row.get("stage4-base") is not None and row.get(name) is not None
    }
    failures=[]
    if base-mean > 0.015:
        failures.append(f"mean drop {base-mean:.4f} exceeds 0.015")
    for task,drop in drops.items():
        if drop > 0.04:
            failures.append(f"{task} drop {drop:.4f} exceeds 0.04")
    item={"name":name,"alpha":alpha,"path":str(path),"mean":mean,
          "mean_drop":base-mean,"task_drops":drops,"failures":failures}
    evaluations.append(item)
    if selected is None and not failures:
        selected=item
if selected is None:
    payload={"status":"fail","base_mean":base,"candidates":evaluations}
    Path("eval_results/stage4_release/sft_regression_gate.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)
    )
    raise SystemExit("No SFT interpolation candidate passed the regression gate")
payload={"status":"pass","base_mean":base,"selected":selected,"candidates":evaluations}
Path("eval_results/stage4_release/sft_regression_gate.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)
)
selected_summary=dict(d)
selected_summary["means"]=dict(d["means"])
selected_summary["means"]["stage4-sft"]=d["means"][selected["name"]]
selected_summary["tasks"]=[
    {**row,"stage4-sft":row.get(selected["name"])} for row in d["tasks"]
]
Path("eval_results/stage4_release/sft_eval/selected_summary.json").write_text(
    json.dumps(selected_summary,indent=2,sort_keys=True)
)
Path("eval_results/stage4_release/selected_sft_checkpoint.json").write_text(
    json.dumps(selected,indent=2,sort_keys=True)
)
print(json.dumps(payload,indent=2,sort_keys=True))
PY
SFT_SELECTED="$("$PYTHON" -c 'import json; print(json.load(open("eval_results/stage4_release/selected_sft_checkpoint.json"))["path"])')"
[ -s "$SFT_SELECTED/model.safetensors" ] && [ -s "$SFT_SELECTED/config.json" ]
mark_done eval_sft_regression

set_stage eval_sft_sanity
"$PYTHON" scripts/eval_sft_sanity.py "$SFT_SELECTED" \
  --output "$EVAL_ROOT/sft_sanity/results.jsonl" \
  --markdown-output "$EVAL_ROOT/sft_sanity/report.md" \
  2>&1 | tee "$LOG_DIR/stage4_sft_sanity_latest.log"
[ -s "$EVAL_ROOT/sft_sanity/results.jsonl" ]
mark_done eval_sft_sanity

set_stage build_release
"$PYTHON" scripts/build_stage4_release_card.py \
  --base-summary "$EVAL_ROOT/base_eval/summary.json" \
  --sft-summary "$EVAL_ROOT/sft_eval/selected_summary.json" \
  --data-gate "$EVAL_ROOT/data_gate.json" \
  --best-checkpoint "$EVAL_ROOT/best_checkpoint.json" \
  --sft-data-summary "$SFT_DATA_SUMMARY" \
  --out "$EVAL_ROOT/README.md"
[ -s "$EVAL_ROOT/README.md" ]
mark_done build_release

set_stage publish
if ! is_done publish; then
  "$PYTHON" scripts/publish_release_to_hf.py \
    --repo-id AliceYin/l20-edu-135m --model-dir "$BEST" \
    --model-card "$EVAL_ROOT/README.md" --eval-dir "$EVAL_ROOT" \
    --revision-dir stage4-best --skip-eval
  "$PYTHON" scripts/publish_release_to_hf.py \
    --repo-id AliceYin/l20-edu-135m --model-dir "$SFT_SELECTED" \
    --model-card "$EVAL_ROOT/README.md" --eval-dir "$EVAL_ROOT" \
    --revision-dir stage4-sft --root-model
  mark_done publish
fi

set_stage complete
mark_done pipeline
log "stage4_full_autopilot_done"
