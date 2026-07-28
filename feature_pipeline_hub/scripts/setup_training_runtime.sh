#!/usr/bin/env bash
# One-time (idempotent) provisioning for the self-contained training runtime.
#
# Copies the Krea-2-NF4 model (~46GB) from an existing LoRAlab checkout and
# creates a dedicated Python venv with the training + curation dependencies.
# Run this yourself when you're ready — it is NOT triggered automatically by
# the app (see the plan: Etapa 1, "Setup manual"). Safe to re-run: rsync only
# copies what changed, and pip install is a no-op once packages match.
#
# Usage:
#   FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2 ./scripts/setup_training_runtime.sh
#
# Optional:
#   FTI_TRAINING_RUNTIME_DIR   defaults to <this repo>/feature_pipeline_hub/training_runtime
#   FTI_TRAINING_VENV_PYTHON   base interpreter used to CREATE the venv (defaults to python3.13,
#                               matching the LoRAlab venv this was pinned against)

set -euo pipefail

if [[ -z "${FTI_LORALAB_ROOT:-}" ]]; then
    echo "Set FTI_LORALAB_ROOT to your AcademiaSD_LoRAlab-Krea2 checkout." >&2
    echo "  e.g. FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2 $0" >&2
    exit 1
fi

if [[ ! -d "$FTI_LORALAB_ROOT/Krea-2-NF4" ]]; then
    echo "No Krea-2-NF4/ found under FTI_LORALAB_ROOT ($FTI_LORALAB_ROOT)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
HUB_ROOT="$(dirname -- "$SCRIPT_DIR")"
RUNTIME_DIR="${FTI_TRAINING_RUNTIME_DIR:-$HUB_ROOT/training_runtime}"
VENV_PYTHON="${FTI_TRAINING_VENV_PYTHON:-python3.13}"

mkdir -p "$RUNTIME_DIR"

echo "==> Copying model weights (this is the slow part, ~46GB, safe to Ctrl+C and re-run)"
mkdir -p "$RUNTIME_DIR/model"
rsync -a --info=progress2 --exclude ".cache" \
    "$FTI_LORALAB_ROOT/Krea-2-NF4/" "$RUNTIME_DIR/model/"

echo "==> Creating training venv at $RUNTIME_DIR/venv"
if [[ ! -x "$RUNTIME_DIR/venv/bin/python" ]]; then
    "$VENV_PYTHON" -m venv "$RUNTIME_DIR/venv"
fi

echo "==> Installing pinned dependencies (versions matched to the LoRAlab venv this runtime replaces)"
"$RUNTIME_DIR/venv/bin/pip" install --upgrade pip --quiet
"$RUNTIME_DIR/venv/bin/pip" install --quiet \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$RUNTIME_DIR/venv/bin/pip" install --quiet \
    "diffusers==0.39.0" \
    "accelerate==1.14.0" \
    "bitsandbytes==0.49.2" \
    "peft==0.19.1" \
    "transformers==5.14.1" \
    "safetensors==0.8.0" \
    "insightface==1.0.1" \
    "opencv-python==5.0.0.93" \
    "onnxruntime==1.28.0" \
    "numpy==2.4.4" \
    "Pillow==12.2.0"

echo "==> Done."
echo "    Model: $RUNTIME_DIR/model"
echo "    Venv:  $RUNTIME_DIR/venv"
echo
echo "Set FTI_TRAINING_RUNTIME_DIR before starting the app if you didn't use the default location."
