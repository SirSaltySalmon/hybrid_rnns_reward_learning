#!/usr/bin/env bash
# Install WSL venv + GPU JAX. Venv lives on Linux FS (not /mnt/c) for speed & reliability.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HYBRNN_VENV:-$HOME/venvs/hybrid_rnns_reward_learning}"

echo "==> Repo: $REPO_ROOT"
echo "==> Venv: $VENV_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found in WSL. Update NVIDIA driver, then: wsl --shutdown"
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

mkdir -p "$(dirname "$VENV_DIR")"
if [[ -d "$VENV_DIR" ]] && ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "==> Removing broken venv..."
  rm -rf "$VENV_DIR"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "==> Creating venv..."
  if python3 -m venv "$VENV_DIR" 2>/dev/null; then
    :
  else
    python3 -m venv --without-pip "$VENV_DIR"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV_DIR/bin/python" /tmp/get-pip.py
    rm -f /tmp/get-pip.py
  fi
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_ROOT"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
python -m pip install -U "jax[cuda12]"

python - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("JAX is not using GPU")
x = jax.numpy.ones((1024, 1024))
y = jax.numpy.dot(x, x).block_until_ready()
print("GPU matmul OK")
PY

cat > "$REPO_ROOT/custom/.wsl_venv_path" <<EOF
$VENV_DIR
EOF

echo ""
echo "Setup complete."
echo "  venv: $VENV_DIR"
echo "  train: wsl -d Ubuntu-24.04 bash custom/wsl_run.sh debug"
