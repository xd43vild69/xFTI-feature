"""Metrics logging, terminal progress formatting, and CSV tracking for LTX 2.3.

Free of torch: covered by mypy and pytest.
"""
from __future__ import annotations

import csv
import os
import time
import uuid
from typing import Any, Callable, IO, Sequence

TRAIN_COLUMNS = [
    "step",
    "update",
    "loss",
    "loss_avg",
    "grad_norm",
    "lr",
    "secs",
    "vram_peak_gb",
]

CHECKPOINT_COLUMNS = [
    "step",
    "reason",
    "timestamp",
    "elapsed_seconds",
    "steps_delta",
    "num_images",
    "launch_id",
]

CHECKPOINT_REASONS = ("periodic", "interrupt", "final")

BAR_WIDTH = 20
BAR_FULL = "█"
BAR_EMPTY = "░"


def format_eta(seconds: float) -> str:
    """Format seconds as HH:MM:SS, clamped at 0."""
    seconds = max(0.0, seconds)
    return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"


def progress_bar(fraction: float) -> str:
    """Fixed-width progress bar for fraction in [0, 1]."""
    filled = int(max(0.0, min(1.0, fraction)) * BAR_WIDTH)
    return BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled)


def smooth(previous: float, sample: float, weight: float = 0.1) -> float:
    """Exponential moving average of step duration."""
    return sample if previous == 0 else weight * sample + (1 - weight) * previous


def format_progress(
    *,
    step: int,
    total_steps: int,
    avg_loss: float,
    grad_norm: float,
    lr: float,
    seconds_per_step: float,
) -> str:
    """Single line progress readout for terminal and UI telemetry."""
    fraction = step / max(1, total_steps)
    eta = format_eta((total_steps - step) * seconds_per_step)

    return (
        f"Step/Paso {step:4d}/{total_steps} [{progress_bar(fraction)}] {fraction * 100:5.1f}% | "
        f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
        f"lr {lr:.2e} | {seconds_per_step:.2f}s/it | ETA {eta}"
    )


class TrainLog:
    """Appends update metrics to train_log.csv."""

    def __init__(self, output_dir: str) -> None:
        self.path = os.path.join(output_dir, "train_log.csv")
        exists = os.path.exists(self.path)
        os.makedirs(output_dir, exist_ok=True)
        self._file: IO[str] = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not exists:
            self._writer.writerow(TRAIN_COLUMNS)
            self._file.flush()

    def write_row(
        self,
        *,
        step: int,
        update: int,
        loss: float,
        loss_avg: float,
        grad_norm: float,
        lr: float,
        secs: float,
        vram_peak_gb: float = 0.0,
    ) -> None:
        self._writer.writerow([
            step,
            update,
            f"{loss:.6f}",
            f"{loss_avg:.6f}",
            f"{grad_norm:.6f}",
            f"{lr:.8e}",
            f"{secs:.3f}",
            f"{vram_peak_gb:.2f}",
        ])
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class CheckpointLog:
    """Logs each saved checkpoint and measures span durations."""

    def __init__(self, output_dir: str, launch_id: str | None = None) -> None:
        self.path = os.path.join(output_dir, "checkpoint_log.csv")
        self.launch_id = launch_id or os.environ.get("FTI_RUN_ID", str(uuid.uuid4())[:8])
        exists = os.path.exists(self.path)
        os.makedirs(output_dir, exist_ok=True)
        self._file: IO[str] = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not exists:
            self._writer.writerow(CHECKPOINT_COLUMNS)
            self._file.flush()
        self._last_mark = time.time()
        self._last_step = 0

    def record(self, *, step: int, reason: str, num_images: int = 0) -> None:
        now = time.time()
        elapsed = now - self._last_mark
        delta = step - self._last_step if self._last_step > 0 else step
        self._writer.writerow([
            step,
            reason,
            int(now),
            f"{elapsed:.2f}",
            delta,
            num_images,
            self.launch_id,
        ])
        self._file.flush()
        self._last_mark = now
        self._last_step = step

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
