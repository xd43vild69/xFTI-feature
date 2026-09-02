"""Unit tests for AppSettings and dynamic path resolution."""

import json
from pathlib import Path
import pytest

from feature_pipeline.infrastructure.app_settings import (
    AppSettings,
    default_config_path,
    load_app_settings,
    reset_app_settings,
    resolve_model_dir,
    resolve_training_python_path,
    resolve_training_runtime_dir,
    save_app_settings,
)
from feature_pipeline.infrastructure.model_prerequisites import default_model_dir


def test_app_settings_lifecycle(tmp_path, monkeypatch):
    config_file = tmp_path / "test_config.json"
    monkeypatch.setenv("FTI_CONFIG_PATH", str(config_file))

    # Clean default
    settings = load_app_settings()
    assert settings.krea2_model_dir is None
    assert settings.ltx23_model_dir is None

    # Save
    new_settings = AppSettings(
        krea2_model_dir=str(tmp_path / "custom_krea"),
        ltx23_model_dir=str(tmp_path / "custom_ltx"),
        training_runtime_dir=str(tmp_path / "custom_runtime"),
        training_python_path=str(tmp_path / "custom_python"),
        hf_token="hf_secret_token",
    )
    save_app_settings(new_settings)

    loaded = load_app_settings()
    assert loaded.krea2_model_dir == str(tmp_path / "custom_krea")
    assert loaded.ltx23_model_dir == str(tmp_path / "custom_ltx")
    assert loaded.hf_token == "hf_secret_token"

    # Reset
    reset_app_settings()
    reset_loaded = load_app_settings()
    assert reset_loaded.krea2_model_dir is None


def test_resolve_model_dir_priority(tmp_path, monkeypatch):
    config_file = tmp_path / "config.json"
    monkeypatch.setenv("FTI_CONFIG_PATH", str(config_file))

    user_krea = tmp_path / "user_krea"
    user_krea.mkdir()
    user_ltx = tmp_path / "user_ltx"
    user_ltx.mkdir()

    # When user settings exist
    save_app_settings(AppSettings(krea2_model_dir=str(user_krea), ltx23_model_dir=str(user_ltx)))

    assert resolve_model_dir("krea2") == user_krea
    assert resolve_model_dir("ltx23") == user_ltx
    assert default_model_dir("krea2") == user_krea
    assert default_model_dir("ltx23") == user_ltx

    # When env vars are set but user settings override
    env_krea = tmp_path / "env_krea"
    env_krea.mkdir()
    monkeypatch.setenv("FTI_KREA2_ROOT", str(env_krea))
    assert resolve_model_dir("krea2") == user_krea

    # When user settings reset, env var takes precedence
    reset_app_settings()
    assert resolve_model_dir("krea2") == env_krea
