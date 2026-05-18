#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv-eval}"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
python -m pip install "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git"
python scripts/check_sota_readiness.py
