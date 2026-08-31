"""Unit tests for ltx23.config resolution and immutability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from ltx23.config import DEFAULTS, LTX23TrainConfig, load_config


def test_default_config_resolution(tmp_path: Path) -> None:
    cfg = load_config(settings_path=str(tmp_path / "nonexistent.json"), project_root=str(tmp_path))
    assert isinstance(cfg, LTX23TrainConfig)
    assert cfg.lora_rank == DEFAULTS["lora_rank"]
    assert cfg.lr == DEFAULTS["lr"]
    assert cfg.total_steps == 3200
    assert cfg.save_every == 200
    assert cfg.timestep_shift == 1.0
    assert cfg.use_loss_weighting is False
    assert cfg.conditioning_mode == "i2v"
    assert cfg.lr_schedule == "constant_with_warmup"
    assert cfg.cond_noise_prob == 0.15


def test_config_overrides(tmp_path: Path) -> None:
    settings = {
        "lora_rank": 64,
        "lr": 2e-4,
        "use_audio_loss": True,
        "project_name": "test_exp",
    }
    settings_file = tmp_path / "train_settings.json"
    settings_file.write_text(json.dumps(settings), encoding="utf-8")

    cfg = load_config(settings_path=str(settings_file), project_root=str(tmp_path))
    assert cfg.lora_rank == 64
    assert cfg.lr == 2e-4
    assert cfg.use_audio_loss is True
    assert cfg.project_name == "test_exp"
    assert "test_exp" in cfg.cache_dir
    assert "test_exp" in cfg.output_dir


def test_config_is_frozen(tmp_path: Path) -> None:
    cfg = load_config(project_root=str(tmp_path))
    with pytest.raises(Exception):
        cfg.lr = 1e-3  # type: ignore[misc]
