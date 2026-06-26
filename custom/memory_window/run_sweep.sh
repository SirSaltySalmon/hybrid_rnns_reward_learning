#!/usr/bin/env bash
# Run the memory-window sweep on GPU from WSL.
#
# Usage (Windows PowerShell):
#   wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh smoke
#   wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh full
#   wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh full_0  # N=3, 5 seeds, 1M steps
#   wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh one
#   wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh full --seed 42  # full grid, one seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/custom/.wsl_venv_path" ]]; then
  VENV_DIR="$(cat "$REPO_ROOT/custom/.wsl_venv_path")"
else
  VENV_DIR="${HYBRNN_VENV_WSL:-$HOME/hybrnn_venv_wsl}"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "WSL GPU venv not found at $VENV_DIR"
  echo "Run first: wsl -d Ubuntu-24.04 bash custom/setup_wsl_gpu.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

PRESET="${1:-smoke}"
shift || true

exec python custom/memory_window/sweep.py --preset "$PRESET" "$@"
