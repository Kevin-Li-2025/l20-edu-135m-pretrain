#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${A40_PYTHON:-/opt/a40-pretrain-venv/bin/python}"
input_dir="${SMOLLM2_10B_INPUT_DIR:-/tmp/smollm2-135m-10b-parquet}"
output_dir="${SMOLLM2_REPAIR_DIR:-/tmp/smollm2-135m-repair-5b-tokenized}"
persistent_dir="${SMOLLM2_REPAIR_PERSISTENT_DIR:-/workspace/a40-pretrain/data/smollm2-135m-repair-5b-tokenized}"
contamination_index="${SMOLLM2_CONTAMINATION_INDEX:-/tmp/l20_benchmark_contamination/eval_6tasks.jsonl}"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/tmp/l20-hf-cache}"

# The source rates yield approximately this 5B-token composition before the
# deterministic final truncation: 60% FineWeb-Edu, 26% DCLM-Edu, 5% Stack-Edu,
# 4% InfiMM-WebMath, 3% FineMath, and 2% Cosmopedia. It deliberately raises the
# educational share to address ARC without removing broad web language data.
"$python_bin" scripts/prepare_parallel_parquet.py \
  --input-dir "$input_dir" \
  --output-dir "$output_dir" \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --workers "${PREP_WORKERS:-16}" \
  --rayon-threads "${PREP_RAYON_THREADS:-4}" \
  --batch-size "${PREP_BATCH_SIZE:-512}" \
  --target-tokens 5000000000 \
  --val-tokens 4194304 \
  --block-size 2048 \
  --contamination-index "$contamination_index" \
  --source-keep-rate dclm_edu=0.25 \
  --source-keep-rate fineweb_edu=0.85 \
  --source-keep-rate stack_edu=0.40 \
  --source-keep-rate cosmopedia_v2=1.0 \
  --source-keep-rate finemath=0.90 \
  --source-keep-rate infimm_webmath=1.0

"$python_bin" - "$output_dir" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
assert metadata["status"] == "complete"
assert metadata["train_tokens"] == 5_000_000_000
assert metadata["val_tokens"] == 4_194_304
assert (root / "train.bin").stat().st_size == metadata["train_tokens"] * 4
assert (root / "val.bin").stat().st_size == metadata["val_tokens"] * 4
expected = {
    "dclm_edu",
    "fineweb_edu",
    "stack_edu",
    "cosmopedia_v2",
    "finemath",
    "infimm_webmath",
}
assert set(metadata["prepared_source_tokens"]) == expected
print(json.dumps({"event": "repair_data_integrity_pass", **metadata}, sort_keys=True))
PY

if [[ "${PERSIST_TOKENIZED_DATA:-1}" == "1" ]]; then
  mkdir -p "$persistent_dir"
  rsync -a --partial \
    "$output_dir/train.bin" \
    "$output_dir/val.bin" \
    "$output_dir/metadata.json" \
    "$persistent_dir/"
fi
