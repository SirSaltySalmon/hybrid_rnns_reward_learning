#!/usr/bin/env bash
# Run a custom experiment on GPU from WSL.
#
# Usage (Windows PowerShell):
#   wsl -d Ubuntu-24.04 bash custom/wsl_run.sh debug
#   wsl -d Ubuntu-24.04 bash custom/wsl_run.sh paper_replication
#   wsl -d Ubuntu-24.04 bash custom/wsl_run.sh --list

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/custom/.wsl_venv_path" ]]; then
  VENV_DIR="$(cat "$REPO_ROOT/custom/.wsl_venv_path")"
else
  VENV_DIR="${HYBRNN_VENV:-$HOME/venvs/hybrid_rnns_reward_learning}"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "WSL GPU venv not found at $VENV_DIR"
  echo "Run first: wsl -d Ubuntu-24.04 bash custom/setup_wsl_gpu.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

# Avoid JAX preallocating all VRAM on 8GB laptop GPUs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

exec python custom/run.py "$@"
