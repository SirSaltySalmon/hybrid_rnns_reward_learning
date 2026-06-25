#!/usr/bin/env bash
# One-time WSL2 + GPU JAX setup (no sudo if uv is installed).
#
# From Windows PowerShell:
#   wsl -d Ubuntu-24.04 bash custom/setup_wsl_gpu.sh
#
# Train on GPU from WSL:
#   wsl -d Ubuntu-24.04 bash custom/wsl_run.sh debug
#   wsl -d Ubuntu-24.04 bash custom/wsl_run.sh paper_replication

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${HYBRNN_VENV_WSL:-$HOME/hybrnn_venv_wsl}"

echo "==> Repo: $REPO_ROOT"
echo "==> Venv: $VENV_DIR (Linux filesystem — required for CUDA wheels)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found in WSL."
  echo "Update the NVIDIA Windows driver, then run: wsl --shutdown"
  exit 1
fi

echo "==> GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (user-local, no sudo)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv..."
  uv venv "$VENV_DIR"
fi

echo "==> Installing project + JAX CUDA 12..."
uv pip install --python "$VENV_DIR/bin/python" -e .
uv pip install --python "$VENV_DIR/bin/python" "jax[cuda12]"

echo "==> Verifying JAX GPU..."
"$VENV_DIR/bin/python" - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("JAX is not using GPU.")
x = jax.numpy.ones((1024, 1024))
y = jax.numpy.dot(x, x).block_until_ready()
print("GPU matmul OK")
PY

echo ""
echo "Setup complete."
echo "Run training:"
echo "  wsl -d Ubuntu-24.04 bash custom/wsl_run.sh debug"
