#!/usr/bin/env bash
# Quick update script for the existing training runtime virtual environment.
# Upgrades PyTorch, TorchAO, and SageAttention without re-downloading model weights.
#
# Usage (run this on your Linux GPU machine):
#   cd /home/d13/lab/x-fti/xFTI-feature/feature_pipeline_hub
#   ./scripts/upgrade_training_runtime.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HUB_ROOT="$(dirname -- "$SCRIPT_DIR")"
RUNTIME_DIR="${FTI_TRAINING_RUNTIME_DIR:-$HUB_ROOT/training_runtime}"
VENV_DIR="$RUNTIME_DIR/venv"

echo "=========================================================="
echo "  Feature Pipeline Hub - Training Runtime Upgrade"
echo "  Target: RTX 4070 Ti (Ada Lovelace) + 64GB RAM"
echo "=========================================================="

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[-] Error: Training venv not found at $VENV_DIR" >&2
    echo "    Run scripts/setup_training_runtime.sh first." >&2
    exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[-] Error: Python executable not found or not executable at $PYTHON_BIN" >&2
    exit 1
fi

echo "[1/4] Upgrading pip and core build tools..."
"$PIP_BIN" install --upgrade pip setuptools wheel --quiet

echo "[2/4] Installing updated PyTorch stack (CUDA 12.6/12.8)..."
CUDA_TAG="${FTI_TORCH_CUDA_TAG:-cu126}"
"$PIP_BIN" install --upgrade \
    "torch>=2.7.0" "torchvision" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" || \
"$PIP_BIN" install --upgrade \
    "torch" "torchvision" \
    --index-url "https://download.pytorch.org/whl/cu124"

echo "[3/4] Installing TorchAO (Memory & Quantization Kernels)..."
"$PIP_BIN" install --upgrade "torchao>=0.8.0"

echo "[4/4] Installing optional attention acceleration (non-fatal)..."
set +e
if ! "$PIP_BIN" install "sageattention" --quiet 2>/dev/null; then
    "$PIP_BIN" install "git+https://github.com/thu-ml/SageAttention.git" --quiet 2>/dev/null || \
    echo "    [i] SageAttention omitido (se utilizará FlashAttention/CuDNN SDPA nativo de PyTorch)."
fi
set -e

echo ""
echo "=========================================================="
echo "  Verification & Installed Versions:"
echo "=========================================================="
"$PYTHON_BIN" -c "
import torch
print(f'  ✓ PyTorch:      {torch.__version__}')
print(f'  ✓ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ Device:       {torch.cuda.get_device_name(0)}')
    print(f'  ✓ CUDA Version: {torch.version.cuda}')
try:
    import torchao
    print(f'  ✓ TorchAO:      {torchao.__version__}')
except Exception as e:
    print(f'  [-] TorchAO:    Not loaded ({e})')
try:
    import sageattention
    print(f'  ✓ SageAttention: Available')
except Exception:
    print(f'  [i] SageAttention: Not installed (using native SDPA)')
"
echo "=========================================================="
echo "✓ Training runtime updated successfully!"
