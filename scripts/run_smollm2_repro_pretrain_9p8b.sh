#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
input_dir="${SMOLLM2_10B_INPUT_DIR:-/tmp/smollm2-135m-10b-parquet}"
data_dir="${SMOLLM2_10B_TOKENIZED_DIR:-/tmp/smollm2-135m-10b-tokenized}"
persistent_data_dir="${SMOLLM2_10B_PERSISTENT_DIR:-/workspace/a40-pretrain/data/smollm2-135m-10b-tokenized}"
contamination_index="${SMOLLM2_CONTAMINATION_INDEX:-/tmp/l20_benchmark_contamination/eval_6tasks.jsonl}"
config="${SMOLLM2_PRETRAIN_CONFIG:-configs/a40_5x_smollm2_repro_pretrain_9p8b.yaml}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"
export PARQUET_CACHE_DIR="${PARQUET_CACHE_DIR:-/tmp/l20-parquet-cache}"

if [[ ! -s "$data_dir/metadata.json" && -s "$persistent_data_dir/metadata.json" ]]; then
  echo "Restoring persistent packed data from $persistent_data_dir"
  mkdir -p "$data_dir"
  rsync -a --partial \
    "$persistent_data_dir/train.bin" \
    "$persistent_data_dir/val.bin" \
    "$persistent_data_dir/metadata.json" \
    "$data_dir/"
fi

parquet_count="$(find "$input_dir/data" -maxdepth 1 -type f -name '*.parquet' | wc -l)"
if [[ "$parquet_count" -ne 85 ]]; then
  echo "Expected 85 SmolLM2 parquet shards under $input_dir/data, found $parquet_count" >&2
  exit 2
fi
if [[ ! -s "$contamination_index" ]]; then
  echo "Missing benchmark contamination index: $contamination_index" >&2
  exit 2
fi

"$python_bin" scripts/prepare_parallel_parquet.py \
  --input-dir "$input_dir" \
  --output-dir "$data_dir" \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --workers "${PREP_WORKERS:-16}" \
  --rayon-threads "${PREP_RAYON_THREADS:-4}" \
  --batch-size "${PREP_BATCH_SIZE:-512}" \
  --target-tokens 9800000000 \
  --val-tokens 8388608 \
  --block-size 2048 \
  --contamination-index "$contamination_index"

"$python_bin" - "$data_dir" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
assert metadata["status"] == "complete"
assert metadata["train_tokens"] == 9_800_000_000
assert metadata["val_tokens"] == 8_388_608
assert (root / "train.bin").stat().st_size == metadata["train_tokens"] * 4
assert (root / "val.bin").stat().st_size == metadata["val_tokens"] * 4
data = np.memmap(root / "train.bin", mode="r", dtype=np.uint32)
rng = np.random.default_rng(8)
sample = data[rng.integers(0, len(data), size=1_000_000)]
assert int(sample.min()) >= 0
assert int(sample.max()) < 49_152
print(
    json.dumps(
        {
            "event": "data_integrity_pass",
            "train_tokens": metadata["train_tokens"],
            "val_tokens": metadata["val_tokens"],
            "sample_min": int(sample.min()),
            "sample_max": int(sample.max()),
            "contaminated_docs_removed": metadata["contaminated_docs"],
        }
    ),
    flush=True,
)
PY

if [[ "${PERSIST_TOKENIZED_DATA:-1}" == "1" ]]; then
  (
    mkdir -p "$persistent_data_dir"
    rsync -a --partial "$data_dir/train.bin" "$data_dir/val.bin" "$persistent_data_dir/"
    rsync -a "$data_dir/metadata.json" "$persistent_data_dir/metadata.json"
    echo "Persistent packed-data copy complete: $persistent_data_dir"
  ) > "$repo_root/persist_smollm2_10b_data.log" 2>&1 &
  echo "Persistent packed-data copy queued with PID $!"
fi

export A40_NPROC_PER_NODE=5
export A40_PYTHON="$python_bin"
output_dir="$($python_bin -c 'import sys, yaml; print((yaml.safe_load(open(sys.argv[1])) or {})["output_dir"])' "$config")"
resume_args=()
latest_checkpoint="$(find "$output_dir" -maxdepth 1 -type d -name 'step-*' 2>/dev/null | sort | tail -n 1)"
if [[ -n "$latest_checkpoint" && -s "$latest_checkpoint/trainer_state.pt" ]]; then
  resume_args=(--resume "$latest_checkpoint")
  echo "Resuming long pretraining from $latest_checkpoint"
fi
exec bash scripts/train_a40_ddp.sh "$config" "${resume_args[@]}"
