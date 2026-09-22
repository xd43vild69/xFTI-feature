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

if command -v uv >/dev/null 2>&1; then
    # Verificar que el entorno virtual tenga streamlit operable; si no, sincronizar
    if [[ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]] || ! "${UV_PROJECT_ENVIRONMENT}/bin/python" -c "import streamlit" >/dev/null 2>&1; then
        echo "Sincronizando entorno virtual con uv..."
        uv sync
    fi
    exec uv run --no-sync streamlit run ui/app.py "$@"
elif [[ -x "${UV_PROJECT_ENVIRONMENT}/bin/streamlit" ]]; then
    source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
    exec "${UV_PROJECT_ENVIRONMENT}/bin/streamlit" run ui/app.py "$@"
else
    exec python3 -m streamlit run ui/app.py "$@"
fi
