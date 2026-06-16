#!/usr/bin/env bash
set -euo pipefail

# L20 is Ada Lovelace (SM 8.9). Building only this architecture avoids
# flash-attn's default multi-architecture build exhausting small-RAM hosts.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-1}"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-FALSE}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

python -m pip install --upgrade "ninja>=1.11" packaging psutil

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
then
  python - <<'PY'
import flash_attn
print("flash_attn_available=true")
print("flash_attn_version=", getattr(flash_attn, "__version__", "unknown"))
PY
  exit 0
fi

if python -m pip install --only-binary=:all: flash-attn; then
  :
else
  python -m pip install --no-build-isolation flash-attn
fi

python - <<'PY'
import flash_attn
print("flash_attn_available=true")
print("flash_attn_version=", getattr(flash_attn, "__version__", "unknown"))
PY
