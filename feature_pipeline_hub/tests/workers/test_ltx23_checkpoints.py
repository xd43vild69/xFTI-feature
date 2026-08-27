"""Unit tests for ltx23.checkpoints filesystem utility."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from ltx23.checkpoints import atomic_write, belongs_to_run, rotate


def test_atomic_write(tmp_path: Path) -> None:
    target = tmp_path / "test.txt"
    atomic_write(str(target), lambda tmp: open(tmp, "w").write("hello"))
    assert target.exists()
    assert target.read_text() == "hello"


def test_rotate(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    for step in (10, 20, 30, 40):
        (out_dir / f"LTX23_LoRA_step_{step}.safetensors").write_bytes(b"dummy")

    removed = rotate(str(out_dir), keep=2, prefix="LTX23_LoRA")
    assert len(removed) == 2
    assert (out_dir / "LTX23_LoRA_step_10.safetensors") not in [Path(p) for p in removed] or not (out_dir / "LTX23_LoRA_step_10.safetensors").exists()
    assert (out_dir / "LTX23_LoRA_step_30.safetensors").exists()
    assert (out_dir / "LTX23_LoRA_step_40.safetensors").exists()


def test_belongs_to_run(tmp_path: Path) -> None:
    run_id_file = tmp_path / "run_id.txt"
    run_id_file.write_text("run_12345", encoding="utf-8")

    assert belongs_to_run(str(run_id_file), "run_12345") is True
    assert belongs_to_run(str(run_id_file), "run_other") is False
    assert belongs_to_run(str(tmp_path / "missing.txt"), "run_12345") is False
