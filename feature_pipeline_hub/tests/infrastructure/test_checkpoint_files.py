"""Reading the checkpoints a past run left behind.

The header is parsed by hand rather than through the safetensors package, so the
format itself is what these pin: an 8-byte little-endian length, that many bytes of
JSON, tensors after it. Everything else here is about not raising — the caller's
fallback for "no reconstruction" is the same as for "nothing on disk", so a truncated
or foreign file must land there rather than break the Metrics page.
"""

import json
import struct
from pathlib import Path

from feature_pipeline.infrastructure.checkpoint_files import (
    list_checkpoint_files,
    read_metadata,
)


def write_checkpoint(path: Path, *, step: int | None = None, epoch: int = 0,
                     images: str | None = "21") -> Path:
    """A file shaped like a real safetensors export, without the tensors."""
    metadata: dict[str, str] = {"modelspec.architecture": "krea2/lora"}
    if images is not None:
        metadata["ss_num_train_images"] = images
    if step is not None:
        metadata["training_info"] = json.dumps({"step": step, "epoch": epoch})

    header = json.dumps({"__metadata__": metadata}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return path


# ── the header ──────────────────────────────────────────────────────────────

def test_metadata_is_read_without_loading_the_tensors(tmp_path):
    path = write_checkpoint(tmp_path / "x_step_300.safetensors", step=300)

    assert read_metadata(path)["ss_num_train_images"] == "21"


def test_a_truncated_file_reads_as_no_metadata(tmp_path):
    path = tmp_path / "x_step_300.safetensors"
    path.write_bytes(b"\x04\x00")

    assert read_metadata(path) == {}


def test_a_header_length_larger_than_any_real_one_is_refused(tmp_path):
    """Otherwise a foreign file could ask for a gigabyte-long read."""
    path = tmp_path / "x_step_300.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40) + b"{}")

    assert read_metadata(path) == {}


def test_a_header_that_is_not_json_reads_as_no_metadata(tmp_path):
    path = tmp_path / "x_step_300.safetensors"
    path.write_bytes(struct.pack("<Q", 5) + b"not{}")

    assert read_metadata(path) == {}


def test_a_missing_file_reads_as_no_metadata(tmp_path):
    assert read_metadata(tmp_path / "absent.safetensors") == {}


# ── listing a run's checkpoints ─────────────────────────────────────────────

def test_checkpoints_are_listed_oldest_first(tmp_path):
    later = write_checkpoint(tmp_path / "dh_bd_v1_step_600.safetensors", step=600)
    earlier = write_checkpoint(tmp_path / "dh_bd_v1_step_300.safetensors", step=300)
    import os
    os.utime(earlier, (1_000, 1_000))
    os.utime(later, (1_731, 1_731))

    found = list_checkpoint_files(tmp_path)

    assert [f.step for f in found] == [300, 600]
    assert [f.written_at for f in found] == [1_000.0, 1_731.0]


def test_the_final_export_is_placed_by_its_header(tmp_path):
    """Its filename carries no step, so the header is the only way to know."""
    write_checkpoint(tmp_path / "dh_bd_v1_FINAL.safetensors", step=3000, epoch=4)

    found = list_checkpoint_files(tmp_path)

    assert [(f.step, f.epoch, f.is_final) for f in found] == [(3000, 4, True)]


def test_a_final_with_no_readable_header_is_skipped(tmp_path):
    """Guessing its step would fabricate a span."""
    (tmp_path / "dh_bd_v1_FINAL.safetensors").write_bytes(b"\x00" * 4)

    assert list_checkpoint_files(tmp_path) == []


def test_a_step_file_falls_back_to_its_filename(tmp_path):
    """Older exports may not carry training_info; the name still does."""
    write_checkpoint(tmp_path / "dh_bd_v1_step_1200.safetensors", step=None)

    found = list_checkpoint_files(tmp_path)

    assert [f.step for f in found] == [1200]


def test_the_image_count_is_none_when_the_header_omits_it(tmp_path):
    write_checkpoint(tmp_path / "x_step_300.safetensors", step=300, images=None)

    assert list_checkpoint_files(tmp_path)[0].num_images is None


def test_anything_that_is_not_a_checkpoint_is_ignored(tmp_path):
    write_checkpoint(tmp_path / "dh_bd_v1_step_300.safetensors", step=300)
    (tmp_path / "train_log.csv").write_text("step,update\n")
    (tmp_path / "optimizer.pt").write_bytes(b"\x00")
    (tmp_path / "current_step.txt").write_text("300")
    # resume_checkpoint/ is rewritten in place on every save, so its mtime says
    # nothing about any particular checkpoint.
    resume = tmp_path / "resume_checkpoint"
    resume.mkdir()
    write_checkpoint(resume / "adapter_model.safetensors", step=300)

    assert [f.step for f in list_checkpoint_files(tmp_path)] == [300]


def test_a_missing_directory_lists_nothing(tmp_path):
    assert list_checkpoint_files(tmp_path / "reclaimed") == []
