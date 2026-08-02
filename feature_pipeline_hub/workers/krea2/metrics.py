"""What a run reports about itself: the progress line and the CSV logs.

Two audiences. The progress line is for whoever is watching, rewritten in place on one
terminal row and surfaced by the Streamlit dashboard. The CSVs are for afterwards —
appended across restarts so a resumed run keeps its history, and the only record of what
a finished run actually did.

Free of torch: these format numbers and write text, so they are covered by mypy and by
tests/workers/ like the rest of the torch-free layer.
"""
from __future__ import annotations

import csv
import os
from typing import Any, IO, Sequence

TRAIN_COLUMNS = ["step", "update", "epoch", "loss", "loss_avg", "grad_norm",
                 "lr", "sigma", "bucket_h", "bucket_w", "secs", "vram_peak_gb"]
VAL_COLUMNS = ["step", "update", "epoch", "val_loss"]

BAR_WIDTH = 20
BAR_FULL = "█"
BAR_EMPTY = "░"


def format_eta(seconds: float) -> str:
    """Seconds as HH:MM:SS, clamped at zero so an overshoot never shows negative time."""
    seconds = max(0.0, seconds)
    return (f"{int(seconds // 3600):02d}:"
            f"{int((seconds % 3600) // 60):02d}:"
            f"{int(seconds % 60):02d}")


def progress_bar(fraction: float) -> str:
    """A fixed-width bar for a fraction in [0, 1]."""
    filled = int(max(0.0, min(1.0, fraction)) * BAR_WIDTH)
    return BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled)


def smooth(previous: float, sample: float, weight: float = 0.1) -> float:
    """Exponential moving average of step duration; `previous` of 0 means first sample.

    Smoothed because a single step that happened to collide with a checkpoint save would
    otherwise throw the ETA off by minutes.
    """
    return sample if previous == 0 else weight * sample + (1 - weight) * previous


def format_progress(
    *,
    step: int,
    total_steps: int,
    avg_loss: float,
    grad_norm: float,
    lr: float,
    epoch: int,
    seconds_per_step: float,
    multiphase: bool = False,
    phase_index: int = 0,
    phase_count: int = 1,
    phase_label: str = "",
    global_step_offset: int = 0,
    global_total_steps: int = 0,
) -> str:
    """The one-line progress readout.

    Under progressive resolution the bar, percentage and ETA are *global*: the hand-off
    carries the weights forward, so restarting them per phase would suggest the run is
    beginning again when it is two thirds done.
    """
    done = global_step_offset + step
    total = global_total_steps if multiphase else total_steps
    fraction = done / max(1, total)
    eta = format_eta((total - done) * seconds_per_step)

    if multiphase:
        head = (f"[F{phase_index + 1}/{phase_count} {phase_label}²] "
                f"Paso {step:4d}/{total_steps} · global {done:5d}/{global_total_steps}")
    else:
        head = f"Step/Paso {step:4d}/{total_steps}"

    return (f"{head} [{progress_bar(fraction)}] {fraction * 100:5.1f}% | "
            f"Loss {avg_loss:.4f} | gnorm {grad_norm:.3f} | "
            f"lr {lr:.2e} | ep {epoch} | {seconds_per_step:.2f}s/it | ETA {eta}")


def average_loss(mode: str, window: Sequence[float], running_total: float,
                 steps_done: int) -> float:
    """Displayed loss: a sliding window, or the cumulative mean.

    The cumulative mean flattens by construction and hides late movement — after a few
    hundred steps a new value barely shifts it — so the window is what shows whether a
    run is still learning.
    """
    if mode == "window":
        return sum(window) / max(1, len(window))
    return running_total / max(1, steps_done)


class CsvLogs:
    """The train and validation CSVs, appended across restarts.

    Headers are written only for a file that did not already exist, so resuming a run
    continues its history instead of starting a second table inside the same file.
    """

    def __init__(self, output_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._train_file: IO[str] | None = None
        self._val_file: IO[str] | None = None
        self._train: Any = None
        self._val: Any = None
        if not enabled:
            return

        train_path = os.path.join(output_dir, "train_log.csv")
        val_path = os.path.join(output_dir, "val_log.csv")
        new_train = not os.path.exists(train_path)
        new_val = not os.path.exists(val_path)

        self._train_file = open(train_path, "a", newline="", encoding="utf-8")
        self._val_file = open(val_path, "a", newline="", encoding="utf-8")
        self._train = csv.writer(self._train_file)
        self._val = csv.writer(self._val_file)
        if new_train:
            self._train.writerow(TRAIN_COLUMNS)
        if new_val:
            self._val.writerow(VAL_COLUMNS)

    def log_step(self, *, step: int, update: int, epoch: int, loss: float,
                 avg_loss: float, grad_norm: float, lr: float, sigma: float,
                 bucket: tuple[int, int], seconds: float, vram_peak_gb: float) -> None:
        """Append one row. Called only on real optimizer updates, not every micro-step."""
        if self._train is None or self._train_file is None:
            return
        self._train.writerow([
            step, update, epoch, f"{loss:.6f}", f"{avg_loss:.6f}", f"{grad_norm:.4f}",
            f"{lr:.3e}", f"{sigma:.4f}", bucket[0], bucket[1],
            f"{seconds:.3f}", f"{vram_peak_gb:.2f}",
        ])
        # Flushed per row: a run killed by SIGKILL still leaves a complete log.
        self._train_file.flush()

    def log_validation(self, *, step: int, update: int, epoch: int,
                       val_loss: float) -> None:
        if self._val is None or self._val_file is None:
            return
        self._val.writerow([step, update, epoch, f"{val_loss:.6f}"])
        self._val_file.flush()

    def close(self) -> None:
        for handle in (self._train_file, self._val_file):
            if handle is not None:
                handle.close()
        self._train_file = self._val_file = None
        self._train = self._val = None

    def __enter__(self) -> "CsvLogs":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
