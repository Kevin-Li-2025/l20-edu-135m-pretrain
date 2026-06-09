#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-hhai@100.111.150.63}"
REMOTE_DIR="${REMOTE_DIR:-/home/hhai/l20-pretrain}"
SESSION="${SESSION:-pretraining_autopilot}"

ssh -o StrictHostKeyChecking=no "$REMOTE" "cd '$REMOTE_DIR' && mkdir -p logs && tmux kill-session -t '$SESSION' 2>/dev/null || true && tmux new-session -d -s '$SESSION' \"bash -lc 'cd $REMOTE_DIR && . .venv-continue/bin/activate && bash scripts/run_pretraining_autopilot.sh 2>&1 | tee logs/pretraining_autopilot_latest.log'\" && tmux ls"
