"""Unit tests for ltx23.metrics formatting and CSV logging."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from ltx23.metrics import (
    CheckpointLog,
    TrainLog,
    format_eta,
    format_progress,
    progress_bar,
    smooth,
)


def test_format_eta() -> None:
    assert format_eta(0) == "00:00:00"
    assert format_eta(3665) == "01:01:05"


def test_progress_bar() -> None:
    bar_0 = progress_bar(0.0)
    assert "░" in bar_0 and "█" not in bar_0
    bar_1 = progress_bar(1.0)
    assert "█" in bar_1 and "░" not in bar_1


def test_smooth() -> None:
    assert smooth(0.0, 1.0) == 1.0
    assert abs(smooth(1.0, 2.0, weight=0.1) - 1.1) < 1e-6


def test_train_log_and_checkpoint_log(tmp_path: Path) -> None:
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    logger = TrainLog(str(out_dir))
    logger.write_row(
        step=10,
        update=2,
        loss=0.05,
        loss_avg=0.06,
        grad_norm=0.15,
        lr=1e-4,
        secs=0.5,
        vram_peak_gb=12.5,
    )
    logger.close()

    log_path = out_dir / "train_log.csv"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "step,update,loss,loss_avg" in content
    assert "10,2,0.050000,0.060000" in content

    ckpt_logger = CheckpointLog(str(out_dir), launch_id="test_launch")
    ckpt_logger.record(step=10, reason="periodic", num_images=5)
    ckpt_logger.close()

    ckpt_path = out_dir / "checkpoint_log.csv"
    assert ckpt_path.exists()
    ckpt_content = ckpt_path.read_text(encoding="utf-8")
    assert "periodic" in ckpt_content
    assert "test_launch" in ckpt_content
