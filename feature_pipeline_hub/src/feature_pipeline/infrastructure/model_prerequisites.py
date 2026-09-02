"""Model prerequisites validation, token management, and downloading.

Validates that model weights and checkpoints exist inside training_runtime before
pre-cache or training can begin, eliminating external system dependencies.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from feature_pipeline.infrastructure.app_settings import resolve_model_dir
from feature_pipeline.infrastructure.storage import training_runtime_dir

ModelArch = Literal["krea2", "ltx23"]

# Hugging Face Repositories
KREA2_HF_REPO_ID = "AcademiaSD/Krea-2-NF4-for-LoRA-Training"
LTX23_BASE_HF_REPO_ID = "diffusers/LTX-2.3-Diffusers"
LTX23_NF4_HF_REPO_ID = "AcademiaSD/LTX23_NF4"


class ModelPrerequisitesMissingError(RuntimeError):
    """Raised when attempting to train/precache without the required base model downloaded."""


@dataclass(frozen=True)
class ModelStatus:
    """Status of model weights and checkpoints in local storage."""

    target_model: ModelArch
    model_dir: Path
    is_ready: bool
    missing_items: list[str] = field(default_factory=list)
    disk_size_bytes: int = 0
    message: str = ""

    @property
    def disk_size_gb(self) -> float:
        return self.disk_size_bytes / (1024**3)


def hub_root_dir() -> Path:
    """Root directory of the feature_pipeline_hub project."""
    return Path(__file__).resolve().parents[3]


def get_saved_hf_token(project_root: Path | None = None) -> str | None:
    """Retrieve HF_TOKEN from candidate secret files, HF cache, or environment."""
    if project_root is not None:
        candidate_paths = [project_root / "HF_token.json"]
    else:
        candidate_paths = [
            hub_root_dir() / "HF_token.json",
            training_runtime_dir() / "HF_token.json",
            Path.home() / ".cache" / "huggingface" / "token",
        ]
    for path in candidate_paths:
        if path.is_file():
            try:
                raw = path.read_text(encoding="utf-8").strip()
                if raw.startswith("{"):
                    data = json.loads(raw)
                    token = str(data.get("token", "")).strip()
                    if token:
                        return token
                elif raw:
                    return raw
            except Exception:
                pass

    env_token = os.environ.get("HF_TOKEN", "").strip()
    return env_token if env_token else None


def save_hf_token(token: str, project_root: Path | None = None) -> Path:
    """Persist HF_TOKEN to HF_token.json in project root and training_runtime."""
    cleaned = token.strip()
    if project_root is not None:
        target = project_root / "HF_token.json"
        target.write_text(json.dumps({"token": cleaned}, indent=2), encoding="utf-8")
    else:
        root = hub_root_dir()
        target = root / "HF_token.json"
        target.write_text(json.dumps({"token": cleaned}, indent=2), encoding="utf-8")
        try:
            rt_target = training_runtime_dir() / "HF_token.json"
            rt_target.parent.mkdir(parents=True, exist_ok=True)
            rt_target.write_text(json.dumps({"token": cleaned}, indent=2), encoding="utf-8")
        except Exception:
            pass

    os.environ["HF_TOKEN"] = cleaned
    return target


def default_model_dir(target_model: ModelArch = "krea2") -> Path:
    """Self-contained local destination for model weights (consulting settings, env vars, or training_runtime)."""
    return resolve_model_dir(target_model, fallback_runtime_dir=training_runtime_dir())


def _directory_size_bytes(directory: Path) -> int:
    """Calculate total size of directory contents in bytes."""
    if not directory.is_dir():
        return 0
    total = 0
    try:
        for p in directory.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def check_model_status(
    target_model: ModelArch = "krea2",
    custom_dir: Path | None = None,
) -> ModelStatus:
    """Check if all required checkpoints and files exist locally for the specified model."""
    model_dir = custom_dir or default_model_dir(target_model)

    if not model_dir.is_dir():
        return ModelStatus(
            target_model=target_model,
            model_dir=model_dir,
            is_ready=False,
            missing_items=[f"Directory '{model_dir.name}' does not exist"],
            disk_size_bytes=0,
            message=f"Model directory not found at {model_dir}",
        )

    size_bytes = _directory_size_bytes(model_dir)

    if target_model == "ltx23":
        missing: list[str] = []
        has_base = (model_dir / "model_index.json").is_file()
        index_file = model_dir / "index.json"
        has_nf4 = index_file.is_file()

        if not has_base:
            missing.append("model_index.json (Base Diffusers)")
        if not has_nf4:
            missing.append("index.json (NF4 Transformer)")
        else:
            try:
                index_data = json.loads(index_file.read_text(encoding="utf-8"))
                quantized = index_data.get("quantized", {})
                missing_count = 0
                for info in list(quantized.values()):
                    fname = info.get("file", "")
                    candidates = [
                        model_dir / "weights" / fname,
                        model_dir / fname,
                        model_dir / "weights" / Path(fname).name,
                        model_dir / Path(fname).name,
                    ]
                    if not any(c.is_file() for c in candidates):
                        missing_count += 1
                if missing_count > 0:
                    missing.append(f"{missing_count} archivos de pesos NF4 en weights/")
            except Exception:
                missing.append("index.json ilegible")

        is_ready = len(missing) == 0
        message = (
            f"LTX 2.3 model ready ({size_bytes / (1024**3):.1f} GB)"
            if is_ready
            else f"Missing required LTX 2.3 components: {', '.join(missing)}"
        )
        return ModelStatus(
            target_model=target_model,
            model_dir=model_dir,
            is_ready=is_ready,
            missing_items=missing,
            disk_size_bytes=size_bytes,
            message=message,
        )

    # Krea 2 check
    missing_krea: list[str] = []
    required_dirs = ["transformer", "text_encoder", "vae"]
    has_subdirs = all((model_dir / d).is_dir() for d in required_dirs)
    has_index = any((model_dir / f).is_file() for f in ["index.json", "model_index.json", "config.json"])

    if not (has_subdirs or has_index):
        for d in required_dirs:
            if not (model_dir / d).is_dir():
                missing_krea.append(f"{d}/")

    is_ready = bool(has_subdirs or has_index)
    message = (
        f"Krea 2 model ready ({size_bytes / (1024**3):.1f} GB)"
        if is_ready
        else f"Incomplete Krea 2 model at {model_dir}"
    )
    return ModelStatus(
        target_model=target_model,
        model_dir=model_dir,
        is_ready=is_ready,
        missing_items=missing_krea,
        disk_size_bytes=size_bytes,
        message=message,
    )


def download_model_prerequisites(
    target_model: ModelArch = "krea2",
    hf_token: str | None = None,
    destination: Path | None = None,
) -> Path:
    """Download required weights from Hugging Face directly into the training_runtime directory.

    Requires huggingface_hub to be installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required for downloading. Install with: pip install huggingface_hub")

    dest = destination or default_model_dir(target_model)
    dest.mkdir(parents=True, exist_ok=True)

    token = hf_token or get_saved_hf_token()
    if token:
        save_hf_token(token)

    if target_model == "ltx23":
        # 1. Base Diffusers
        snapshot_download(
            repo_id=LTX23_BASE_HF_REPO_ID,
            local_dir=str(dest),
            token=token,
            max_workers=4,
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"],
        )
        # 2. NF4 Transformer weights
        snapshot_download(
            repo_id=LTX23_NF4_HF_REPO_ID,
            local_dir=str(dest),
            token=token,
            max_workers=4,
        )
    else:
        # Krea 2
        snapshot_download(
            repo_id=KREA2_HF_REPO_ID,
            local_dir=str(dest),
            token=token,
            max_workers=4,
        )

    status = check_model_status(target_model, custom_dir=dest)
    if not status.is_ready:
        raise ModelPrerequisitesMissingError(
            f"Download finished but model verification failed: {status.message}"
        )

    return dest
