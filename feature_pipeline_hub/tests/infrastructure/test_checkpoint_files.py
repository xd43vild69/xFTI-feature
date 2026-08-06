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

import pytest

from feature_pipeline.infrastructure.checkpoint_files import (
    copy_resume_state,
    list_checkpoint_files,
    materialize_warm_start,
    read_metadata,
    rewrite_export_to_resume_layout,
    warm_start_step,
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


# ── copy_resume_state ───────────────────────────────────────────────────────

def _write_resume_checkpoint(source_dir: Path, *, step: int = 1500) -> None:
    resume = source_dir / "resume_checkpoint"
    resume.mkdir(parents=True)
    (resume / "adapter_model.safetensors").write_bytes(b"\x00" * 8)
    (resume / "adapter_config.json").write_text('{"r": 32}')
    (source_dir / "optimizer.pt").write_bytes(b"\x01" * 8)
    (source_dir / "current_step.txt").write_text(str(step))
    # Not part of the checkpoint state — must never be copied.
    (source_dir / "train_log.csv").write_text("step,update\n")
    (source_dir / "checkpoint_log.csv").write_text("step,reason\n")
    (source_dir / f"Krea2_LoRA_step_{step}.safetensors").write_bytes(b"\x02" * 8)


def _has_complete_checkpoint(output_dir: Path) -> bool:
    return (
        (output_dir / "optimizer.pt").is_file()
        and (output_dir / "resume_checkpoint" / "adapter_model.safetensors").is_file()
        and (output_dir / "current_step.txt").is_file()
    )


def test_copy_resume_state_copies_all_three_artifacts(tmp_path):
    source = tmp_path / "parent"
    _write_resume_checkpoint(source, step=1500)
    destination = tmp_path / "branch"

    step = copy_resume_state(source, destination)

    assert step == 1500
    assert (destination / "optimizer.pt").read_bytes() == b"\x01" * 8
    assert (destination / "resume_checkpoint" / "adapter_model.safetensors").is_file()
    assert (destination / "resume_checkpoint" / "adapter_config.json").read_text() == '{"r": 32}'
    assert (destination / "current_step.txt").read_text().strip() == "1500"


def test_copy_resume_state_leaves_the_source_untouched(tmp_path):
    source = tmp_path / "parent"
    _write_resume_checkpoint(source, step=1500)
    destination = tmp_path / "branch"

    copy_resume_state(source, destination)

    assert _has_complete_checkpoint(source)
    assert (source / "current_step.txt").read_text().strip() == "1500"


def test_copy_resume_state_does_not_copy_step_exports_or_csvs(tmp_path):
    source = tmp_path / "parent"
    _write_resume_checkpoint(source, step=1500)
    destination = tmp_path / "branch"

    copy_resume_state(source, destination)

    assert not (destination / "train_log.csv").exists()
    assert not (destination / "checkpoint_log.csv").exists()
    assert not (destination / "Krea2_LoRA_step_1500.safetensors").exists()
    assert list(destination.glob("*.safetensors")) == []


def test_copy_resume_state_is_idempotent(tmp_path):
    source = tmp_path / "parent"
    _write_resume_checkpoint(source, step=1500)
    destination = tmp_path / "branch"

    copy_resume_state(source, destination)
    step = copy_resume_state(source, destination)

    assert step == 1500
    assert _has_complete_checkpoint(destination)


def test_a_destination_missing_current_step_txt_reads_as_incomplete(tmp_path):
    source = tmp_path / "parent"
    _write_resume_checkpoint(source, step=1500)
    destination = tmp_path / "branch"
    destination.mkdir()
    # Simulate an interruption after the first two artifacts landed but before the
    # step file was written — the ordering copy_resume_state itself follows.
    import shutil as _shutil
    _shutil.copytree(source / "resume_checkpoint", destination / "resume_checkpoint")
    _shutil.copyfile(source / "optimizer.pt", destination / "optimizer.pt")

    assert not _has_complete_checkpoint(destination)


def test_copy_resume_state_raises_when_the_source_has_no_checkpoint(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    destination = tmp_path / "branch"

    with pytest.raises(OSError):
        copy_resume_state(source, destination)


# ── warm start ──────────────────────────────────────────────────────────────
#
# The layout half of these is pinned against a real 224 MB export from a finished run:
# 528 tensors in, 528 out, key set identical to the resume adapter PEFT wrote for the
# same step, and every tensor bit-identical once loaded through safetensors. What the
# fixtures below cover is the mechanics — that the blob is carried through untouched
# and that the marker is the last thing written.


def write_step_export(path: Path, *, names: list[str], payload: int = 8) -> bytes:
    """A per-step export's shape: `transformer.*` keys over a contiguous blob."""
    header: dict = {"__metadata__": {"ss_num_train_images": "21"}}
    for index, name in enumerate(names):
        header[name] = {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [index * payload, (index + 1) * payload],
        }
    blob = bytes(range(len(names) * payload))
    encoded = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    return blob


def read_header(path: Path) -> tuple[dict, bytes]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), handle.read()


EXPORT_KEYS = [
    "transformer.img_in.lora_A.default.weight",
    "transformer.img_in.lora_B.default.weight",
]
RESUME_KEYS = {
    "base_model.model.img_in.lora_A.weight",
    "base_model.model.img_in.lora_B.weight",
}


def test_rewrite_maps_export_keys_onto_the_layout_peft_loads(tmp_path):
    source = write_step_export(tmp_path / "p_step_900.safetensors", names=EXPORT_KEYS)
    destination = tmp_path / "out" / "adapter_model.safetensors"

    renamed = rewrite_export_to_resume_layout(
        tmp_path / "p_step_900.safetensors", destination
    )

    header, blob = read_header(destination)
    assert renamed == 2
    assert set(header) - {"__metadata__"} == RESUME_KEYS
    assert blob == source


def test_rewrite_carries_the_tensor_blob_through_untouched(tmp_path):
    """Every data_offsets pair is relative to the blob, not the file, so renaming keys
    cannot move a tensor — which is the whole reason this needs no torch."""
    write_step_export(tmp_path / "p_step_900.safetensors", names=EXPORT_KEYS)
    destination = tmp_path / "out" / "adapter_model.safetensors"

    rewrite_export_to_resume_layout(tmp_path / "p_step_900.safetensors", destination)

    original, original_blob = read_header(tmp_path / "p_step_900.safetensors")
    rewritten, rewritten_blob = read_header(destination)
    assert rewritten_blob == original_blob
    assert [rewritten[k]["data_offsets"] for k in sorted(RESUME_KEYS)] == [
        original[k]["data_offsets"] for k in sorted(EXPORT_KEYS)
    ]


def test_rewrite_keeps_the_blob_eight_byte_aligned(tmp_path):
    """safetensors pads the header with spaces so the tensors start aligned; a rewritten
    header is a different length than the one it replaced."""
    write_step_export(tmp_path / "p_step_900.safetensors", names=EXPORT_KEYS)
    destination = tmp_path / "out" / "adapter_model.safetensors"

    rewrite_export_to_resume_layout(tmp_path / "p_step_900.safetensors", destination)

    with destination.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
    assert (8 + length) % 8 == 0


def test_rewrite_preserves_the_metadata_block(tmp_path):
    write_step_export(tmp_path / "p_step_900.safetensors", names=EXPORT_KEYS)
    destination = tmp_path / "out" / "adapter_model.safetensors"

    rewrite_export_to_resume_layout(tmp_path / "p_step_900.safetensors", destination)

    header, _ = read_header(destination)
    assert header["__metadata__"]["ss_num_train_images"] == "21"


def test_rewrite_refuses_an_export_carrying_alpha_tensors(tmp_path):
    """export_alpha_tensors adds keys PEFT reports as unexpected — and dropping them
    would mean rebuilding the blob this function copies verbatim."""
    write_step_export(
        tmp_path / "p_step_900.safetensors", names=[*EXPORT_KEYS, "transformer.img_in.alpha"]
    )

    with pytest.raises(ValueError, match="alpha"):
        rewrite_export_to_resume_layout(
            tmp_path / "p_step_900.safetensors", tmp_path / "out" / "a.safetensors"
        )


def test_rewrite_refuses_a_file_that_is_not_an_export(tmp_path):
    write_step_export(
        tmp_path / "other.safetensors", names=["base_model.model.img_in.lora_A.weight"]
    )

    with pytest.raises(ValueError, match="not an exported LoRA key"):
        rewrite_export_to_resume_layout(
            tmp_path / "other.safetensors", tmp_path / "out" / "a.safetensors"
        )


def test_rewrite_leaves_no_staging_file_behind(tmp_path):
    write_step_export(tmp_path / "p_step_900.safetensors", names=EXPORT_KEYS)
    destination = tmp_path / "out" / "adapter_model.safetensors"

    rewrite_export_to_resume_layout(tmp_path / "p_step_900.safetensors", destination)

    assert [p.name for p in destination.parent.iterdir()] == ["adapter_model.safetensors"]


# ── materialize_warm_start ──────────────────────────────────────────────────


def _staged_export(tmp_path: Path) -> Path:
    write_step_export(tmp_path / "parent" / "p_step_900.safetensors", names=EXPORT_KEYS)
    (tmp_path / "parent" / "resume_checkpoint").mkdir(parents=True, exist_ok=True)
    (tmp_path / "parent" / "resume_checkpoint" / "adapter_config.json").write_text('{"r": 32}')
    return tmp_path / "parent" / "p_step_900.safetensors"


def test_materialize_stages_the_adapter_and_the_marker(tmp_path):
    export = _staged_export(tmp_path)
    destination = tmp_path / "branch"

    step = materialize_warm_start(
        step_export=export,
        adapter_config=export.parent / "resume_checkpoint" / "adapter_config.json",
        destination_dir=destination, step=900, source_label="p_step_900.safetensors",
    )

    assert step == 900
    assert (destination / "resume_checkpoint" / "adapter_model.safetensors").is_file()
    assert (destination / "resume_checkpoint" / "adapter_config.json").read_text() == '{"r": 32}'
    marker = json.loads((destination / "warm_start.json").read_text())
    assert marker == {"format_version": 1, "step": 900, "source": "p_step_900.safetensors"}


def test_materialize_writes_no_optimizer_or_step_file(tmp_path):
    """Their absence is what keeps has_checkpoint()/checkpoint_step() answering
    'nothing restorable here' — which is exactly true of a warm start."""
    export = _staged_export(tmp_path)
    destination = tmp_path / "branch"

    materialize_warm_start(
        step_export=export, adapter_config=None, destination_dir=destination, step=900
    )

    assert not (destination / "optimizer.pt").exists()
    assert not (destination / "current_step.txt").exists()
    assert sorted(p.name for p in destination.iterdir()) == [
        "resume_checkpoint", "warm_start.json"
    ]


def test_materialize_tolerates_a_missing_adapter_config(tmp_path):
    """It is copied for inspectability; the trainer builds the adapter from its own
    config and never reads this back."""
    export = _staged_export(tmp_path)
    destination = tmp_path / "branch"

    materialize_warm_start(
        step_export=export, adapter_config=tmp_path / "absent.json",
        destination_dir=destination, step=900,
    )

    assert warm_start_step(destination) == 900


def test_materialize_refuses_a_non_positive_step(tmp_path):
    export = _staged_export(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        materialize_warm_start(
            step_export=export, adapter_config=None,
            destination_dir=tmp_path / "branch", step=0,
        )


def test_materialize_leaves_no_marker_when_the_export_is_unusable(tmp_path):
    """The marker is written last for the same reason current_step.txt is: it is the
    assertion that the weights it points at are complete."""
    write_step_export(
        tmp_path / "bad_step_900.safetensors", names=["base_model.model.img_in.lora_A.weight"]
    )
    destination = tmp_path / "branch"

    with pytest.raises(ValueError):
        materialize_warm_start(
            step_export=tmp_path / "bad_step_900.safetensors", adapter_config=None,
            destination_dir=destination, step=900,
        )

    assert warm_start_step(destination) is None
    assert not (destination / "warm_start.json").exists()


# ── warm_start_step ─────────────────────────────────────────────────────────


def test_warm_start_step_is_none_without_a_marker(tmp_path):
    tmp_path.joinpath("branch").mkdir()

    assert warm_start_step(tmp_path / "branch") is None


def test_warm_start_step_is_none_when_the_adapter_is_missing(tmp_path):
    """A marker pointing at nothing must not read as a usable start."""
    destination = tmp_path / "branch"
    destination.mkdir()
    (destination / "warm_start.json").write_text('{"format_version": 1, "step": 900}')

    assert warm_start_step(destination) is None


def test_warm_start_step_is_none_for_an_unknown_marker_version(tmp_path):
    export = _staged_export(tmp_path)
    destination = tmp_path / "branch"
    materialize_warm_start(
        step_export=export, adapter_config=None, destination_dir=destination, step=900
    )
    (destination / "warm_start.json").write_text('{"format_version": 99, "step": 900}')

    assert warm_start_step(destination) is None


def test_warm_start_step_is_none_for_an_unreadable_marker(tmp_path):
    export = _staged_export(tmp_path)
    destination = tmp_path / "branch"
    materialize_warm_start(
        step_export=export, adapter_config=None, destination_dir=destination, step=900
    )
    (destination / "warm_start.json").write_text("not json")

    assert warm_start_step(destination) is None
