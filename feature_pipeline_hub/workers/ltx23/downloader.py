"""LTX-2.3 Model Downloader / Validator."""

import json
import os
from pathlib import Path

HF_BASE_REPO_ID = "diffusers/LTX-2.3-Diffusers"
HF_NF4_REPO_ID = "AcademiaSD/LTX23_NF4"


def get_hf_token(project_root: str | None = None) -> str | None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hub_root = os.path.dirname(os.path.dirname(current_dir))
    candidates = [
        "HF_token.json",
        os.path.join(hub_root, "HF_token.json"),
        os.path.join(hub_root, "training_runtime", "HF_token.json"),
    ]
    if project_root:
        candidates.append(os.path.join(project_root, "HF_token.json"))
    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    token_data = json.load(f)
                token = str(token_data.get("token", "")).strip()
                if token:
                    return token
            except Exception:
                pass
    token = os.environ.get("HF_TOKEN", "").strip()
    return token if token else None


def ensure_ltx23_model_downloaded(local_path: str | Path) -> Path:
    """Verifies that the LTX 2.3 base model and NF4 weights exist locally, downloading if missing."""
    p = Path(local_path).resolve()

    has_base = (p / "model_index.json").is_file()
    has_nf4 = (p / "index.json").is_file()

    if has_base and has_nf4:
        return p

    p.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo base LTX-2.3 y NF4 a {p}...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub es necesario. Instala con: pip install huggingface_hub")

    token = get_hf_token()
    if token:
        print("✓ Usando Hugging Face Token")

    print(f"Descargando base diffusers: {HF_BASE_REPO_ID}")
    snapshot_download(
        repo_id=HF_BASE_REPO_ID,
        local_dir=str(p),
        token=token,
        max_workers=4,
    )

    print(f"Descargando NF4 weights: {HF_NF4_REPO_ID}")
    snapshot_download(
        repo_id=HF_NF4_REPO_ID,
        local_dir=str(p),
        token=token,
        max_workers=4,
    )

    if not (p / "model_index.json").is_file():
        raise RuntimeError(f"Descarga incompleta: falta model_index.json en {p}")

    if not (p / "index.json").is_file():
        raise RuntimeError(f"Descarga incompleta: falta index.json en {p}")

    print(f"[OK] Modelo LTX 2.3 listo en {p}")
    return p
