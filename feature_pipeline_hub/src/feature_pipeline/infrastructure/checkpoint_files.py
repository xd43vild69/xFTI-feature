"""Reading the checkpoints a run left on disk, for runs that predate the timing log.

`checkpoint_log.csv` only exists for runs launched after the trainer started writing
it. Everything before that left the checkpoints themselves and nothing else — but a
`.safetensors` carries its own mtime, which is when it was written, and its header
carries the step it was written at and how many images the run trained on. That is
enough to reconstruct the spans (see `domain.checkpoint_log.reconstruct`).

The header is read by hand rather than through `safetensors`: the format puts an
8-byte little-endian length followed by that many bytes of JSON, and the tensors after
it. So the metadata costs one small read regardless of how large the adapter is, and
this module stays free of both torch and the safetensors package — neither of which is
installed in the hub's environment.

Nothing here raises. A directory that is gone, a file truncated by a crash mid-write,
a header from some future version — all of them mean "no reconstruction for this run",
which is a state the caller already has to handle.
"""

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

# Krea2_LoRA_step_1200.safetensors / dh_bd_v1_FINAL.safetensors. The prefix is
# configurable per run (`checkpoint_prefix`), so it is matched rather than assumed —
# a run whose prefix changed mid-lineage still lists both.
_STEP_FILE = re.compile(r"^(?P<prefix>.+)_step_(?P<step>\d+)\.safetensors$")
_FINAL_FILE = re.compile(r"^(?P<prefix>.+)_FINAL\.safetensors$")

# How much of the file to trust as a header length before deciding it is not one.
_MAX_HEADER_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class CheckpointFileInfo:
    """What one checkpoint file on disk can say about itself."""

    path: Path
    step: int
    written_at: float
    num_images: int | None
    epoch: int
    is_final: bool


def list_checkpoint_files(output_dir: Path) -> list[CheckpointFileInfo]:
    """Every checkpoint in a run's output_dir, oldest first.

    The per-step exports and the FINAL are both included; they overlap at the last
    step, which the domain merges rather than double-counting. `resume_checkpoint/`
    is skipped — it is PEFT layout, rewritten in place on every save, so its mtime
    says when the run last saved and nothing about any particular checkpoint.
    """
    try:
        entries = sorted(output_dir.iterdir())
    except OSError:
        return []

    found: list[CheckpointFileInfo] = []
    for path in entries:
        if not path.is_file() or path.suffix != ".safetensors":
            continue

        final = _FINAL_FILE.match(path.name) is not None
        match = _STEP_FILE.match(path.name)
        if not final and match is None:
            continue

        try:
            written_at = path.stat().st_mtime
        except OSError:
            continue

        metadata = read_metadata(path)
        step = _step_from(metadata)
        if step is None:
            if match is None:
                # A FINAL whose header did not survive: its step is unrecoverable,
                # and guessing one would fabricate a span.
                continue
            step = int(match.group("step"))

        found.append(
            CheckpointFileInfo(
                path=path,
                step=step,
                written_at=written_at,
                num_images=_int_or_none(metadata.get("ss_num_train_images")),
                epoch=_epoch_from(metadata),
                is_final=final,
            )
        )

    return sorted(found, key=lambda info: (info.written_at, info.step))


def read_metadata(path: Path) -> dict:
    """The `__metadata__` block of a safetensors file, or {} if it cannot be read."""
    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) < 8:
                return {}
            length = struct.unpack("<Q", raw_length)[0]
            if length <= 0 or length > _MAX_HEADER_BYTES:
                return {}
            header = json.loads(handle.read(length))
    except (OSError, struct.error, json.JSONDecodeError, UnicodeDecodeError):
        return {}

    if not isinstance(header, dict):
        return {}
    metadata = header.get("__metadata__")
    return metadata if isinstance(metadata, dict) else {}


def _training_info(metadata: dict) -> dict:
    """`training_info` is itself a JSON string inside the metadata dict."""
    raw = metadata.get("training_info")
    if not isinstance(raw, str):
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _step_from(metadata: dict) -> int | None:
    """The step out of the header — the only way to place a FINAL, which has no
    step in its filename."""
    return _int_or_none(_training_info(metadata).get("step"))


def _epoch_from(metadata: dict) -> int:
    return _int_or_none(_training_info(metadata).get("epoch")) or 0


def _int_or_none(value: object) -> int | None:
    """The header is JSON written by another process; every field is str or int here."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None
