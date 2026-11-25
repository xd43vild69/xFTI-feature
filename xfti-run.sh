#!/usr/bin/env bash
set -euo pipefail

# Resolver el directorio raíz del repositorio sin importar desde dónde se invoque el script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB_DIR="${SCRIPT_DIR}/feature_pipeline_hub"

if [[ ! -d "${HUB_DIR}" ]]; then
    echo "Error: No se encontró el directorio '${HUB_DIR}'." >&2
    exit 1
fi

# Asegurar que rutas comunes de uv / binarios locales estén en PATH
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

cd "${HUB_DIR}"

# Asegurar que src/ esté en PYTHONPATH
export PYTHONPATH="${HUB_DIR}/src:${PYTHONPATH:-}"

# Evitar conflictos de VIRTUAL_ENV heredado o entre sistemas operativos (Linux vs macOS por SMB)
unset VIRTUAL_ENV || true
OS_NAME="$(uname -s)"
if [[ "${OS_NAME}" == "Darwin" ]]; then
    export UV_PROJECT_ENVIRONMENT="${HUB_DIR}/.venv-macos"
else
    export UV_PROJECT_ENVIRONMENT="${HUB_DIR}/.venv"
fi

echo "=================================================="
echo " Iniciando xFTI Feature Pipeline Hub (Streamlit)"
echo " Directorio: ${HUB_DIR}"
echo " Entorno:    ${UV_PROJECT_ENVIRONMENT}"
echo "=================================================="

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
PIP_BIN="${UV_PROJECT_ENVIRONMENT}/bin/pip"
STREAMLIT_BIN="${UV_PROJECT_ENVIRONMENT}/bin/streamlit"

# Verificar salud completa del entorno virtual; si falta cualquier dependencia o está corrupto, reconstruir limpio
if [[ ! -x "${PYTHON_BIN}" ]] || ! "${PYTHON_BIN}" -c "import streamlit, urllib3, requests, pandas, pydantic, pydantic_core, PIL, imagehash, numpy, mcp, huggingface_hub; assert int(pandas.__version__.split('.')[0]) < 3" >/dev/null 2>&1; then
    echo "==> Entorno virtual incompleto o corrupto. Reconstruyendo limpiamente ${UV_PROJECT_ENVIRONMENT}..."
    rm -rf "${UV_PROJECT_ENVIRONMENT}"
    if command -v uv >/dev/null 2>&1; then
        uv sync --link-mode=copy
    else
        python3 -m venv "${UV_PROJECT_ENVIRONMENT}"
        "${UV_PROJECT_ENVIRONMENT}/bin/pip" install --upgrade pip --quiet
        "${UV_PROJECT_ENVIRONMENT}/bin/pip" install -e "${HUB_DIR}"
    fi
fi

if command -v uv >/dev/null 2>&1; then
    exec uv run --no-sync streamlit run ui/app.py --browser.gatherUsageStats=false "$@"
elif [[ -x "${STREAMLIT_BIN}" ]]; then
    source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
    exec "${STREAMLIT_BIN}" run ui/app.py --browser.gatherUsageStats=false "$@"
else
    exec python3 -m streamlit run ui/app.py --browser.gatherUsageStats=false "$@"
fi
