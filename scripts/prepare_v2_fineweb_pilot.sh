#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
output_dir=$project_root/data/v2_fineweb_pilot_50m

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing pilot shards: $output_dir" >&2
  exit 2
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TMPDIR="$project_root/tmp"

cd "$project_root"
"$runtime_python" -m l20_pretrain.prepare_shards \
  --output-dir "$output_dir" \
  --tokenizer assets/tokenizer-smollm2-135m \
  --dataset HuggingFaceFW/fineweb-edu \
  --config-name sample-10BT \
  --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --split train \
  --target-tokens 50000000 \
  --val-tokens 524288 \
  --block-size 2048 \
  --min-chars 300 \
  --max-chars 50000 \
  --min-score 3.0 \
  --min-int-score 3 \
  --report-interval 1000

"$runtime_python" scripts/verify_shard_manifest.py "$output_dir"
