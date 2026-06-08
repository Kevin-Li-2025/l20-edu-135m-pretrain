#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade "liger-kernel>=0.5"
python - <<'PY'
from liger_kernel.transformers import apply_liger_kernel_to_llama

print("liger_kernel_available=true")
print("apply_liger_kernel_to_llama=", apply_liger_kernel_to_llama)
PY
