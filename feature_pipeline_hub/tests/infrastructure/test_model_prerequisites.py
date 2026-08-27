"""Unit tests for model prerequisites validation and HF_TOKEN management."""
from pathlib import Path
import json
import pytest

from feature_pipeline.infrastructure.model_prerequisites import (
    ModelPrerequisitesMissingError,
    check_model_status,
    default_model_dir,
    get_saved_hf_token,
    save_hf_token,
)


def test_default_model_dirs(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "custom_runtime"
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime_dir))

    krea_dir = default_model_dir("krea2")
    assert krea_dir == runtime_dir / "model"

    ltx_dir = default_model_dir("ltx23")
    assert ltx_dir == runtime_dir / "LTX23-NF4"


def test_hf_token_save_and_get(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert get_saved_hf_token(tmp_path) is None

    save_hf_token("hf_test_123456", project_root=tmp_path)
    token = get_saved_hf_token(tmp_path)
    assert token == "hf_test_123456"

    # Environment variable fallback
    tmp_empty = tmp_path / "empty_dir"
    tmp_empty.mkdir()
    monkeypatch.setenv("HF_TOKEN", "hf_from_env_999")
    assert get_saved_hf_token(tmp_empty) == "hf_from_env_999"


def test_check_model_status_krea2_missing(tmp_path):
    model_dir = tmp_path / "model"
    status = check_model_status("krea2", custom_dir=model_dir)
    assert not status.is_ready
    assert "not found" in status.message or "does not exist" in status.message


def test_check_model_status_krea2_ready(tmp_path):
    model_dir = tmp_path / "model"
    for subdir in ("transformer", "text_encoder", "vae"):
        (model_dir / subdir).mkdir(parents=True)
        (model_dir / subdir / "weights.safetensors").write_bytes(b"x" * (2 * 1024 * 1024))

    status = check_model_status("krea2", custom_dir=model_dir)
    assert status.is_ready
    assert len(status.missing_items) == 0
    assert status.disk_size_bytes > 0


def test_check_model_status_ltx23_missing_components(tmp_path):
    ltx_dir = tmp_path / "LTX23-NF4"
    ltx_dir.mkdir(parents=True)

    status = check_model_status("ltx23", custom_dir=ltx_dir)
    assert not status.is_ready
    assert any("model_index.json" in item for item in status.missing_items)
    assert any("index.json" in item for item in status.missing_items)

    # Only base
    (ltx_dir / "model_index.json").write_text("{}", encoding="utf-8")
    status = check_model_status("ltx23", custom_dir=ltx_dir)
    assert not status.is_ready
    assert any("index.json" in item for item in status.missing_items)


def test_check_model_status_ltx23_ready(tmp_path):
    ltx_dir = tmp_path / "LTX23-NF4"
    ltx_dir.mkdir(parents=True)
    (ltx_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (ltx_dir / "index.json").write_text("{}", encoding="utf-8")

    status = check_model_status("ltx23", custom_dir=ltx_dir)
    assert status.is_ready
    assert len(status.missing_items) == 0
