"""Progress readout and CSV logs.

Formatting bugs are cosmetic; the two that are not are the multiphase progress accounting
(which tells an operator whether a four-hour run is a third or two-thirds done) and the
CSV append behavior (a resumed run must extend its history, not start a second table
inside the same file).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.metrics import (  # noqa: E402
    BAR_WIDTH, CHECKPOINT_COLUMNS, TRAIN_COLUMNS, VAL_COLUMNS, CheckpointLog, CsvLogs,
    average_loss, format_eta, format_progress, progress_bar, smooth,
)


# ── formatting ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00:00"), (59, "00:00:59"), (60, "00:01:00"),
    (3600, "01:00:00"), (3661, "01:01:01"), (86399, "23:59:59"),
])
def test_eta_formats_as_clock_time(seconds: float, expected: str) -> None:
    assert format_eta(seconds) == expected


def test_negative_eta_clamps_to_zero() -> None:
    """A resumed run can overshoot its total; a negative ETA would look like a bug."""
    assert format_eta(-500) == "00:00:00"


@pytest.mark.parametrize("fraction,filled", [
    (0.0, 0), (0.5, BAR_WIDTH // 2), (1.0, BAR_WIDTH),
    (-1.0, 0), (2.0, BAR_WIDTH),  # clamped
])
def test_bar_fills_proportionally(fraction: float, filled: int) -> None:
    bar = progress_bar(fraction)
    assert len(bar) == BAR_WIDTH
    assert bar.count("█") == filled


def test_smoothing_takes_the_first_sample_whole() -> None:
    assert smooth(0.0, 2.5) == 2.5


def test_smoothing_damps_a_spike() -> None:
    """One step colliding with a checkpoint save must not throw the ETA off by minutes."""
    assert smooth(1.0, 11.0) == pytest.approx(2.0)


# ── average loss ────────────────────────────────────────────────────────────

def test_window_average_uses_only_the_window() -> None:
    assert average_loss("window", [1.0, 2.0, 3.0], 999.0, 999) == pytest.approx(2.0)


def test_cumulative_average_uses_the_running_total() -> None:
    assert average_loss("cumulative", [1.0], 30.0, 3) == pytest.approx(10.0)


def test_averages_survive_an_empty_start() -> None:
    assert average_loss("window", [], 0.0, 0) == 0.0
    assert average_loss("cumulative", [], 0.0, 0) == 0.0


# ── progress line ───────────────────────────────────────────────────────────

def progress(**kwargs: object) -> str:
    base = dict(step=100, total_steps=1200, avg_loss=0.1234, grad_norm=0.5,
                lr=1e-4, epoch=2, seconds_per_step=1.5)
    return format_progress(**{**base, **kwargs})  # type: ignore[arg-type]


def test_single_phase_shows_local_progress() -> None:
    line = progress()
    assert "Step/Paso  100/1200" in line
    assert "8.3%" in line
    assert "Loss 0.1234" in line
    assert "ep 2" in line


def test_multiphase_reports_global_progress() -> None:
    """The hand-off carries weights forward, so per-phase percentages would mislead."""
    line = progress(multiphase=True, phase_index=1, phase_count=3, phase_label="768",
                    global_step_offset=1200, global_total_steps=3600, step=600)
    assert "[F2/3 768²]" in line
    assert "global  1800/3600" in line
    assert "50.0%" in line          # global, not the 50% of phase 2 alone
    assert "Paso  600/1200" in line  # local step still visible


def test_eta_is_global_under_multiphase() -> None:
    line = progress(multiphase=True, phase_count=3, global_step_offset=0,
                    global_total_steps=3600, step=0, seconds_per_step=1.0)
    assert "ETA 01:00:00" in line   # 3600 remaining steps at 1s


# ── CSV logs ────────────────────────────────────────────────────────────────

def read_rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def sample_step(logs: CsvLogs, step: int, loss: float = 0.5) -> None:
    logs.log_step(step=step, update=step // 4, epoch=1, loss=loss, avg_loss=loss,
                  grad_norm=0.25, lr=1e-4, sigma=0.5, bucket=(64, 64),
                  seconds=1.0, vram_peak_gb=8.0)


def test_the_hubs_copy_of_the_header_still_matches_this_one() -> None:
    """The hub reads this CSV but cannot import this module.

    `workers/` is not on the hub's sys.path and the training runtime is a separate
    interpreter, so feature_pipeline.domain.train_log keeps its own copy of the
    column list. This is the only place both sides are importable at once, so it
    is the only place the duplication can be pinned.
    """
    from feature_pipeline.domain.train_log import TRAIN_LOG_COLUMNS

    assert list(TRAIN_LOG_COLUMNS) == TRAIN_COLUMNS


def test_writes_a_header_then_rows(tmp_path: Path) -> None:
    with CsvLogs(str(tmp_path)) as logs:
        sample_step(logs, 4)
        sample_step(logs, 8)
    rows = read_rows(tmp_path / "train_log.csv")
    assert rows[0] == TRAIN_COLUMNS
    assert len(rows) == 3


def test_reopening_appends_without_a_second_header(tmp_path: Path) -> None:
    """A resumed run must extend its history, not restart the table mid-file."""
    with CsvLogs(str(tmp_path)) as logs:
        sample_step(logs, 4)
    with CsvLogs(str(tmp_path)) as logs:
        sample_step(logs, 8)

    rows = read_rows(tmp_path / "train_log.csv")
    assert rows[0] == TRAIN_COLUMNS
    assert len(rows) == 3
    assert [r[0] for r in rows[1:]] == ["4", "8"]


def test_rows_are_flushed_immediately(tmp_path: Path) -> None:
    """A run killed by SIGKILL should still leave a complete log."""
    logs = CsvLogs(str(tmp_path))
    sample_step(logs, 4)
    assert len(read_rows(tmp_path / "train_log.csv")) == 2  # readable before close
    logs.close()


def test_validation_log_is_separate(tmp_path: Path) -> None:
    with CsvLogs(str(tmp_path)) as logs:
        logs.log_validation(step=50, update=12, epoch=1, val_loss=0.321)
    rows = read_rows(tmp_path / "val_log.csv")
    assert rows[0] == VAL_COLUMNS
    assert rows[1] == ["50", "12", "1", "0.321000"]


def test_disabled_logs_write_nothing(tmp_path: Path) -> None:
    with CsvLogs(str(tmp_path), enabled=False) as logs:
        sample_step(logs, 4)
        logs.log_validation(step=1, update=1, epoch=1, val_loss=0.1)
    assert list(tmp_path.iterdir()) == []


def test_closing_twice_is_safe(tmp_path: Path) -> None:
    logs = CsvLogs(str(tmp_path))
    logs.close()
    logs.close()


def test_logging_after_close_is_a_noop(tmp_path: Path) -> None:
    logs = CsvLogs(str(tmp_path))
    logs.close()
    sample_step(logs, 4)  # must not raise on a closed handle
    assert len(read_rows(tmp_path / "train_log.csv")) == 1


# ── checkpoint log ──────────────────────────────────────────────────────────

class FakeClock:
    """A clock the test advances by hand, so spans can be asserted exactly."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def checkpoint_rows(path: Path) -> list[list[str]]:
    return read_rows(path / "checkpoint_log.csv")[1:]


def test_the_hubs_copy_of_the_checkpoint_header_still_matches_this_one() -> None:
    """Same duplication, same only-place-to-pin-it as the train log's columns."""
    from feature_pipeline.domain.checkpoint_log import CHECKPOINT_LOG_COLUMNS

    assert list(CHECKPOINT_LOG_COLUMNS) == CHECKPOINT_COLUMNS


def test_each_span_is_measured_from_the_previous_checkpoint(tmp_path: Path) -> None:
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(120.0)
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
        clock.advance(90.0)
        log.record(step=200, epoch=2, reason="periodic", num_images=40)

    rows = checkpoint_rows(tmp_path)
    assert [row[0] for row in rows] == ["100", "200"]
    assert [row[4] for row in rows] == ["120.000", "90.000"]
    assert [row[5] for row in rows] == ["100", "100"]      # steps_delta
    assert [row[6] for row in rows] == ["40", "40"]


def test_the_first_span_is_measured_from_construction(tmp_path: Path) -> None:
    """Not from the previous launch's last save.

    This is the resume case: the clock starts when the process comes back up, so the
    hours a stopped run spent waiting to be resumed are never billed as training.
    """
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(60.0)
        log.record(step=100, epoch=1, reason="periodic", num_images=40)

    clock.advance(10_000.0)  # the process was dead for this long

    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(75.0)
        log.record(step=200, epoch=3, reason="periodic", num_images=40)

    rows = checkpoint_rows(tmp_path)
    assert [row[4] for row in rows] == ["60.000", "75.000"]


def test_reopening_the_checkpoint_log_appends_without_a_second_header(tmp_path: Path) -> None:
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
    with CheckpointLog(str(tmp_path), now=clock) as log:
        log.record(step=200, epoch=2, reason="periodic", num_images=40)

    rows = read_rows(tmp_path / "checkpoint_log.csv")
    assert rows[0] == CHECKPOINT_COLUMNS
    assert len(rows) == 3


def test_a_stop_records_the_partial_span(tmp_path: Path) -> None:
    """The user's stop lands between two save_every boundaries."""
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(120.0)
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
        clock.advance(30.0)
        log.record(step=137, epoch=1, reason="interrupt", num_images=40)

    partial = checkpoint_rows(tmp_path)[-1]
    assert partial[2] == "interrupt"
    assert partial[4] == "30.000"
    assert partial[5] == "37"


def test_a_resumed_launch_counts_only_the_steps_it_ran(tmp_path: Path) -> None:
    """Not every step since zero — 248, for a launch that resumed at 1852."""
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        log.set_start_step(1852)
        clock.advance(643.0)
        log.record(step=2100, epoch=5, reason="periodic", num_images=21)

    row = checkpoint_rows(tmp_path)[0]
    assert row[5] == "248"


def test_a_resume_seam_records_a_non_positive_delta(tmp_path: Path) -> None:
    """The reader drops these from its medians; the writer just reports them."""
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(10.0)
        log.record(step=300, epoch=2, reason="periodic", num_images=40)
        clock.advance(10.0)
        log.record(step=200, epoch=1, reason="periodic", num_images=40)

    assert checkpoint_rows(tmp_path)[-1][5] == "-100"


def test_a_backwards_clock_never_records_negative_time(tmp_path: Path) -> None:
    clock = FakeClock()
    with CheckpointLog(str(tmp_path), now=clock) as log:
        clock.advance(-50.0)
        log.record(step=100, epoch=1, reason="periodic", num_images=40)

    assert checkpoint_rows(tmp_path)[0][4] == "0.000"


def test_every_row_carries_the_launch_id(tmp_path: Path, monkeypatch) -> None:
    """It comes from FTI_RUN_ID, the same id _telemetry.py stamps on log.txt."""
    monkeypatch.setenv("FTI_RUN_ID", "run-abc")
    with CheckpointLog(str(tmp_path)) as log:
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
        log.record(step=200, epoch=2, reason="periodic", num_images=40)

    assert [row[7] for row in checkpoint_rows(tmp_path)] == ["run-abc", "run-abc"]


def test_two_launches_over_one_file_are_distinguishable(tmp_path: Path, monkeypatch) -> None:
    """The resume case: without this the two processes' spans read as one process's."""
    monkeypatch.delenv("FTI_RUN_ID", raising=False)
    with CheckpointLog(str(tmp_path)) as log:
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
    with CheckpointLog(str(tmp_path)) as log:
        log.record(step=200, epoch=2, reason="periodic", num_images=40)

    first, second = (row[7] for row in checkpoint_rows(tmp_path))
    assert first and second and first != second


def test_a_missing_image_count_writes_an_empty_cell(tmp_path: Path) -> None:
    with CheckpointLog(str(tmp_path)) as log:
        log.record(step=100, epoch=1, reason="periodic", num_images=None)

    assert checkpoint_rows(tmp_path)[0][6] == ""


def test_checkpoint_rows_are_flushed_immediately(tmp_path: Path) -> None:
    log = CheckpointLog(str(tmp_path))
    log.record(step=100, epoch=1, reason="periodic", num_images=40)
    assert len(checkpoint_rows(tmp_path)) == 1  # readable before close
    log.close()


def test_disabled_checkpoint_log_writes_nothing(tmp_path: Path) -> None:
    with CheckpointLog(str(tmp_path), enabled=False) as log:
        log.record(step=100, epoch=1, reason="periodic", num_images=40)
    assert list(tmp_path.iterdir()) == []


def test_recording_after_close_is_a_noop(tmp_path: Path) -> None:
    log = CheckpointLog(str(tmp_path))
    log.close()
    log.record(step=100, epoch=1, reason="periodic", num_images=40)
    assert checkpoint_rows(tmp_path) == []
