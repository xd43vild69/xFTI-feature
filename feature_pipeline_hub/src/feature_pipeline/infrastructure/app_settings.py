"""Persistent application settings management for Feature Pipeline Hub.

Allows users to customize model storage paths (Krea 2, LTX 2.3), training runtime
directories, and training interpreter paths via the UI or config files.
Persisted in data/hub_config.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ModelArch = Literal["krea2", "ltx23"]


@dataclass
class AppSettings:
    """Configurable settings for Feature Pipeline Hub."""

    krea2_model_dir: str | None = None
    ltx23_model_dir: str | None = None
    training_runtime_dir: str | None = None
    training_python_path: str | None = None
    hf_token: str | None = None
    custom_options: dict[str, str] = field(default_factory=dict)


def default_config_path() -> Path:
    """Path to the persistent configuration JSON file."""
    override = os.environ.get("FTI_CONFIG_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data" / "hub_config.json"


def load_app_settings() -> AppSettings:
    """Load settings from the persistent JSON file, returning defaults if missing."""
    config_file = default_config_path()
    if not config_file.is_file():
        return AppSettings()

    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return AppSettings(
            krea2_model_dir=data.get("krea2_model_dir"),
            ltx23_model_dir=data.get("ltx23_model_dir"),
            training_runtime_dir=data.get("training_runtime_dir"),
            training_python_path=data.get("training_python_path"),
            hf_token=data.get("hf_token"),
            custom_options=data.get("custom_options", {}),
        )
    except Exception:
        return AppSettings()


def save_app_settings(settings: AppSettings) -> None:
    """Persist settings to data/hub_config.json."""
    config_file = default_config_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    config_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_app_settings() -> AppSettings:
    """Clear custom settings file and return clean defaults."""
    config_file = default_config_path()
    if config_file.is_file():
        try:
            config_file.unlink()
        except OSError:
            pass
    return AppSettings()


def _normalize_existing_path(path_str: str | None) -> Path | None:
    """Normalize a path string and check existence, with transparent fallback for macOS/Linux mount mappings."""
    if not path_str:
        return None

    p = Path(path_str).expanduser()
    if p.exists():
        return p

    # Cross-platform fallback: /mnt/backup <-> /Volumes/backup-1
    if path_str.startswith("/mnt/backup/"):
        alt = Path("/Volumes/backup-1" + path_str[len("/mnt/backup") :])
        if alt.exists():
            return alt
    elif path_str.startswith("/Volumes/backup-1/"):
        alt = Path("/mnt/backup" + path_str[len("/Volumes/backup-1") :])
        if alt.exists():
            return alt

    return None


def resolve_model_dir(target_model: ModelArch = "krea2", fallback_runtime_dir: Path | None = None) -> Path:
    """Resolve the active model directory following priority:
    1. Saved user setting in AppSettings
    2. Environment variables (FTI_LTX23_ROOT, FTI_KREA2_ROOT / FTI_MODEL_ROOT)
    3. Default under training_runtime_dir/
    """
    settings = load_app_settings()

    if target_model == "ltx23":
        # 1. User setting
        user_path = _normalize_existing_path(settings.ltx23_model_dir)
        if user_path and user_path.is_dir():
            return user_path

        # 2. Env var
        env_ltx = _normalize_existing_path(os.environ.get("FTI_LTX23_ROOT"))
        if env_ltx and env_ltx.is_dir():
            return env_ltx

        # 3. Default
        rt_dir = fallback_runtime_dir or resolve_training_runtime_dir()
        return rt_dir / "LTX23-NF4"

    # target_model == "krea2"
    # 1. User setting
    user_path = _normalize_existing_path(settings.krea2_model_dir)
    if user_path and user_path.is_dir():
        return user_path

    # 2. Env var
    env_krea = _normalize_existing_path(os.environ.get("FTI_KREA2_ROOT") or os.environ.get("FTI_MODEL_ROOT"))
    if env_krea and env_krea.is_dir():
        return env_krea

    # 3. Default
    rt_dir = fallback_runtime_dir or resolve_training_runtime_dir()
    return rt_dir / "model"


def resolve_training_runtime_dir() -> Path:
    """Resolve the training runtime directory following priority:
    1. Saved user setting in AppSettings
    2. Environment variable FTI_TRAINING_RUNTIME_DIR
    3. Default under <repo>/feature_pipeline_hub/training_runtime
    """
    settings = load_app_settings()
    if settings.training_runtime_dir and Path(settings.training_runtime_dir).is_dir():
        return Path(settings.training_runtime_dir)

    env_override = os.environ.get("FTI_TRAINING_RUNTIME_DIR")
    if env_override and Path(env_override).is_dir():
        return Path(env_override)

    default_base = Path(__file__).resolve().parents[3] / "training_runtime"
    default_base.mkdir(parents=True, exist_ok=True)
    return default_base


def resolve_training_python_path() -> Path:
    """Resolve the training Python interpreter path following priority:
    1. Saved user setting in AppSettings
    2. Environment variable FTI_TRAINING_PYTHON
    3. Default under <training_runtime_dir>/venv/bin/python
    """
    settings = load_app_settings()
    if settings.training_python_path and Path(settings.training_python_path).is_file():
        return Path(settings.training_python_path)

    env_python = os.environ.get("FTI_TRAINING_PYTHON", "").strip()
    if env_python and Path(env_python).expanduser().is_file():
        return Path(env_python).expanduser()

    rt_dir = resolve_training_runtime_dir()
    return rt_dir / "venv" / "bin" / "python"
