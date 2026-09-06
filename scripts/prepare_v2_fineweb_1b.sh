#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-/ssd/scxi253/l20-edu-135m-v2-dev}
runtime_python=${RUNTIME_PYTHON:-/ssd/scxi253/pretraining2/runtime/venv/bin/python}
overlay=${RUNTIME_OVERLAY:-$project_root/runtime-overlay}
output_dir=$project_root/data/v2_fineweb_1b

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing 1B shards: $output_dir" >&2
  exit 2
fi

export PYTHONPATH="$overlay:$project_root/src"
export PYTHONNOUSERSITE=1
export HF_HOME="$project_root/hf-cache"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export TMPDIR="$project_root/tmp"

cd "$project_root"
"$runtime_python" -m l20_pretrain.prepare_shards \
  --output-dir "$output_dir" \
  --tokenizer assets/tokenizer-smollm2-135m \
  --tokenizer-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
  --dataset HuggingFaceFW/fineweb-edu \
  --config-name sample-10BT \
  --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --split train \
  --target-tokens 1000000000 \
  --val-tokens 4194304 \
  --block-size 2048 \
  --min-chars 300 \
  --max-chars 50000 \
  --min-score 3.0 \
  --min-int-score 3 \
  --report-interval 10000

"$runtime_python" scripts/verify_shard_manifest.py "$output_dir"
