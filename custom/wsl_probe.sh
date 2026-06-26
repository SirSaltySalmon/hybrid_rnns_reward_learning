#!/usr/bin/env bash
# Probe WSL JAX GPU (called from train_models.ipynb — avoid inline bash -c; wsl.exe
# strips $VAR in -c strings on Windows).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c "import jax; print(jax.__version__); print(jax.default_backend()); print(jax.devices())"
