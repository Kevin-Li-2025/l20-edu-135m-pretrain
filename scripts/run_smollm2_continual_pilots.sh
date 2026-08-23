#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
baseline_eval="${CONTINUAL_BASELINE_EVAL:-/workspace/a40-pretrain/eval_results/smollm2-repro-pretrain-9p8b-broad}"
result_root="${CONTINUAL_EVAL_ROOT:-/workspace/a40-pretrain/eval_results/smollm2-continual-pilots}"
confidence="${CONTINUAL_PILOT_CONFIDENCE:-0.9833}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export A40_NPROC_PER_NODE=5
export A40_PYTHON="$python_bin"

if [[ ! -f "$baseline_eval/.complete" ]]; then
  bash scripts/eval_pretrain_broad_parallel.sh \
    /workspace/a40-pretrain/runs/a40-5x-smollm2-repro-pretrain-9p8b/step-009969 \
    "$baseline_eval"
fi

configs=(
  configs/a40_5x_smollm2_continual_control_500m_lr3e4.yaml
  configs/a40_5x_smollm2_continual_repair_500m_lr3e4.yaml
  configs/a40_5x_smollm2_continual_repair_500m_lr6e4.yaml
  configs/a40_5x_smollm2_continual_repair_500m_1k_lr3e4.yaml
)

for config in "${configs[@]}"; do
  run_name="$($python_bin -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["run_name"])' "$config")"
  output_dir="$($python_bin -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$config")"
  checkpoint="$output_dir/step-000510"
  eval_dir="$result_root/$run_name/eval"
  mkdir -p "$result_root/$run_name"

  if [[ ! -s "$checkpoint/model.safetensors" ]]; then
    bash scripts/train_a40_ddp.sh "$config" \
      > "$result_root/$run_name/train.log" 2>&1
  fi
  if [[ ! -f "$eval_dir/.complete" ]]; then
    bash scripts/eval_pretrain_broad_parallel.sh "$checkpoint" "$eval_dir"
  fi

  "$python_bin" -m l20_pretrain.paired_eval \
    --baseline "$baseline_eval" \
    --candidate "$eval_dir" \
    --confidence "$confidence" \
    --out "$result_root/$run_name/paired-primary.json"
  "$python_bin" -m l20_pretrain.paired_eval \
    --baseline "$baseline_eval" \
    --candidate "$eval_dir" \
    --task openbookqa \
    --task sciq \
    --task boolq \
    --task swag \
    --confidence "$confidence" \
    --out "$result_root/$run_name/paired-development.json"
  "$python_bin" scripts/check_continual_pilot.py \
    --primary-paired "$result_root/$run_name/paired-primary.json" \
    --development-paired "$result_root/$run_name/paired-development.json" \
    --min-confidence "$confidence" \
    --out "$result_root/$run_name/pilot-gate.json"
done

"$python_bin" - "$result_root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/pilot-gate.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run": path.parent.name,
            "status": payload["status"],
            "primary_delta": payload["checks"]["primary"]["delta"],
            "primary_ci": payload["checks"]["primary"]["paired_bootstrap_ci"],
            "development_delta": payload["checks"]["development"]["delta"],
            "required_task_improvements": payload["checks"]["required_task_improvements"],
        }
    )
summary = {"status": "complete", "pilots": rows}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
