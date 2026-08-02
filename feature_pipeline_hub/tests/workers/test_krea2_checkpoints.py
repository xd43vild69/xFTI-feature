"""Checkpoint filesystem discipline: atomic writes, retention, and run ownership.

Every property here is about a run that was interrupted rather than one that finished.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.checkpoints import atomic_write, belongs_to_run, rotate  # noqa: E402


# ── atomic_write ────────────────────────────────────────────────────────────

def test_writes_through_to_the_target(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    atomic_write(str(target), lambda p: Path(p).write_text("payload"))
    assert target.read_text() == "payload"


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path: Path) -> None:
    """A half-written adapter still loads, and then trains to garbage."""
    target = tmp_path / "out.txt"
    target.write_text("original")

    def explode(path: str) -> None:
        Path(path).write_text("partial")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        atomic_write(str(target), explode)

    assert target.read_text() == "original"


def test_a_failed_write_removes_its_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    def explode(path: str) -> None:
        Path(path).write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_write(str(target), explode)

    assert list(tmp_path.iterdir()) == []


def test_no_target_is_left_behind_when_the_writer_never_created_one(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with pytest.raises(FileNotFoundError):
        atomic_write(str(target), lambda _p: None)
    assert not target.exists()


# ── rotate ──────────────────────────────────────────────────────────────────

def populate(directory: Path, steps: list[int]) -> None:
    for step in steps:
        (directory / f"Krea2_LoRA_step_{step}.safetensors").write_text("x")
    (directory / "Krea2_FINAL_LoRA.safetensors").write_text("x")
    (directory / "train_log.csv").write_text("x")


def remaining_steps(directory: Path) -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in directory.iterdir()
                  if p.name.startswith("Krea2_LoRA_step_"))


def test_keeps_the_most_recent_by_step_number(tmp_path: Path) -> None:
    populate(tmp_path, [25, 50, 100, 200, 1000, 75])
    rotate(str(tmp_path), 3, log=lambda _m: None)
    assert remaining_steps(tmp_path) == [100, 200, 1000]


def test_ordering_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    """Sorted as text, step_1000 would rank below step_75 and get pruned first."""
    populate(tmp_path, [75, 1000])
    rotate(str(tmp_path), 1, log=lambda _m: None)
    assert remaining_steps(tmp_path) == [1000]


def test_zero_or_negative_keeps_everything(tmp_path: Path) -> None:
    for keep in (0, -1):
        directory = tmp_path / f"k{keep}"
        directory.mkdir()
        populate(directory, [25, 50, 100])
        rotate(str(directory), keep, log=lambda _m: None)
        assert remaining_steps(directory) == [25, 50, 100]


def test_keeping_more_than_exist_removes_nothing(tmp_path: Path) -> None:
    populate(tmp_path, [25, 50])
    assert rotate(str(tmp_path), 10, log=lambda _m: None) == []


def test_the_final_export_and_unrelated_files_are_never_touched(tmp_path: Path) -> None:
    populate(tmp_path, [25, 50, 100])
    rotate(str(tmp_path), 1, log=lambda _m: None)
    assert (tmp_path / "Krea2_FINAL_LoRA.safetensors").exists()
    assert (tmp_path / "train_log.csv").exists()


def test_returns_what_it_removed(tmp_path: Path) -> None:
    populate(tmp_path, [25, 50, 100])
    removed = rotate(str(tmp_path), 1, log=lambda _m: None)
    assert [Path(p).name for p in removed] == [
        "Krea2_LoRA_step_25.safetensors", "Krea2_LoRA_step_50.safetensors"]


# ── belongs_to_run ──────────────────────────────────────────────────────────

def test_no_run_id_accepts_any_checkpoint(tmp_path: Path) -> None:
    """A standalone trainer has no pipeline identity to check against."""
    assert belongs_to_run(str(tmp_path / "run_id.txt"), "")


def test_matching_run_id_is_accepted(tmp_path: Path) -> None:
    marker = tmp_path / "run_id.txt"
    marker.write_text("run-abc\n")
    assert belongs_to_run(str(marker), "run-abc")


def test_stale_run_id_is_rejected(tmp_path: Path) -> None:
    """Otherwise the phase resumes at start_step == total_steps and runs an empty loop."""
    marker = tmp_path / "run_id.txt"
    marker.write_text("run-old")
    assert not belongs_to_run(str(marker), "run-new")


def test_missing_marker_is_rejected(tmp_path: Path) -> None:
    assert not belongs_to_run(str(tmp_path / "absent.txt"), "run-abc")
