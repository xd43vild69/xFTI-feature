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

Nothing in the *reading* half raises. A directory that is gone, a file truncated by a
crash mid-write, a header from some future version — all of them mean "no
reconstruction for this run", which is a state the caller already has to handle. The
writing half (`copy_resume_state`, `rewrite_export_to_resume_layout`,
`materialize_warm_start`) is the exception and raises deliberately; each says so.
"""

import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

# Mirror of the same names in application/training_service.py (STEP_FILE,
# OPTIMIZER_STATE_FILE, RESUME_ADAPTER) — duplicated rather than imported so this
# infrastructure module never depends on the application layer above it.
_RESUME_DIR_NAME = "resume_checkpoint"
_OPTIMIZER_STATE_FILE = "optimizer.pt"
_STEP_FILE_NAME = "current_step.txt"
_ADAPTER_FILE_NAME = "adapter_model.safetensors"
_ADAPTER_CONFIG_NAME = "adapter_config.json"

# Mirrors krea2.state.WARM_START_FILENAME / WARM_START_FORMAT_VERSION. The two sides
# are in different interpreters (the hub has no torch), so the marker is the contract.
_WARM_START_FILE_NAME = "warm_start.json"
_WARM_START_FORMAT_VERSION = 1

# The two key layouts lora_io moves between: what `export_lora` writes for inference
# loaders, and what PEFT's `save_pretrained`/`set_peft_model_state_dict` use.
_EXPORT_PREFIX = "transformer."
_RESUME_PREFIX = "base_model.model."
_ADAPTER_NAME_SUFFIX = ".default.weight"
_ALPHA_SUFFIX = ".alpha"

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


def copy_resume_state(source_dir: Path, destination_dir: Path) -> int:
    """Copy a checkpoint's resumable state into a fresh output_dir, for forking.

    Unlike everything else in this module, this function raises — `OSError` on any
    filesystem failure — and the caller (`training_service.fork_training`) depends on
    that: a fork that silently produced a half-copied checkpoint would read back as
    "no checkpoint", which the trainer answers by training from step 0 without
    complaint. So the caller re-verifies with `checkpoint_step()` after this returns
    rather than trusting a clean return alone, but this at least fails loudly instead
    of leaving corrupt bytes.

    Copies exactly the three things `krea2.state.CheckpointManager.has_checkpoint()`
    requires: the full `resume_checkpoint/` directory (PEFT's `save_pretrained` writes
    `adapter_config.json` alongside the safetensors, and `lora_io.load_lora_weights`
    needs both), `optimizer.pt`, and `current_step.txt`. Deliberately NOT copied: the
    per-step `.safetensors` exports (their key layout can't be resumed from — see
    `lora_io.export_lora`), and the three CSVs (a forked branch gets fresh ones, which
    is the entire point of giving it a new output_dir instead of reusing the parent's).

    Each artifact lands via a `.incoming` staging path plus `os.replace`, mirroring
    `krea2/state.py`'s own commit protocol, and `current_step.txt` is written *last* —
    the same ordering `has_checkpoint`/`checkpoint_step` rely on to treat its presence
    as proof the other two are complete.

    Returns the copied step, read back from the destination's current_step.txt.
    """
    source_resume = source_dir / _RESUME_DIR_NAME
    source_optimizer = source_dir / _OPTIMIZER_STATE_FILE
    source_step = source_dir / _STEP_FILE_NAME

    destination_dir.mkdir(parents=True, exist_ok=True)

    dest_resume = destination_dir / _RESUME_DIR_NAME
    staging_resume = destination_dir / f".{_RESUME_DIR_NAME}.incoming"
    if staging_resume.exists():
        shutil.rmtree(staging_resume)
    shutil.copytree(source_resume, staging_resume)
    if dest_resume.exists():
        shutil.rmtree(dest_resume)
    os.replace(staging_resume, dest_resume)

    staging_optimizer = destination_dir / f".{_OPTIMIZER_STATE_FILE}.incoming"
    shutil.copyfile(source_optimizer, staging_optimizer)
    os.replace(staging_optimizer, destination_dir / _OPTIMIZER_STATE_FILE)

    staging_step = destination_dir / f".{_STEP_FILE_NAME}.incoming"
    shutil.copyfile(source_step, staging_step)
    os.replace(staging_step, destination_dir / _STEP_FILE_NAME)

    return int((destination_dir / _STEP_FILE_NAME).read_text(encoding="utf-8").strip())


# ── warm start: forking from a per-step export ──────────────────────────────
#
# Only the *last* step of a run is resumable: `krea2.state` rewrites
# resume_checkpoint/ and optimizer.pt in place on every save. The per-step exports are
# the only surviving record of the steps in between, so forking at step 900 of a
# finished 3000-step run means starting from one of those — weights alone, optimizer
# cold. `krea2.state.CheckpointManager._restore_warm_start` is the other half of this.


def _resume_layout_key(key: str) -> str:
    """Invert `lora_io.export_lora`'s key rewrite.

    That function maps `base_model.model.<X>.lora_A.default.weight` to
    `transformer.<X>.lora_A.default.weight` for inference loaders; PEFT's
    `save_pretrained` — the layout `set_peft_model_state_dict` actually accepts — writes
    `base_model.model.<X>.lora_A.weight`, dropping the adapter name. So the inverse is
    both a prefix swap and a `.default` strip, and it is a verified bijection on a real
    export: 528 tensors in, 528 out, shapes and dtypes matching the resume checkpoint
    written by the same run.
    """
    if not key.startswith(_EXPORT_PREFIX):
        raise ValueError(f"not an exported LoRA key: {key!r}")
    if key.endswith(_ALPHA_SUFFIX):
        # export_alpha_tensors adds these. They are not `lora_*` keys, so PEFT would
        # report them as unexpected and `load_lora_weights` would exit(1) — and dropping
        # them is not free, since their bytes sit inside the blob this copies verbatim.
        raise ValueError(
            "export carries per-module alpha tensors, which cannot be warm-started from; "
            "re-export with export_alpha_tensors off"
        )
    body = key[len(_EXPORT_PREFIX):]
    if body.endswith(_ADAPTER_NAME_SUFFIX):
        body = body[: -len(_ADAPTER_NAME_SUFFIX)] + ".weight"
    return _RESUME_PREFIX + body


def rewrite_export_to_resume_layout(source: Path, destination: Path) -> int:
    """Copy a per-step export into the layout PEFT loads, renaming keys only.

    A safetensors file is an 8-byte little-endian header length, that many bytes of
    JSON, then the tensor blob — and every `data_offsets` pair in the header is relative
    to the *start of the blob*, not to the file. Renaming keys therefore changes nothing
    but the header, so the blob is copied through byte-for-byte and the hub needs
    neither torch nor the safetensors package to do it. The bf16 weights arrive
    unchanged, which is the same precision `save_pretrained` writes the resume adapter
    at — a warm start loses the optimizer, not any weight fidelity.

    Raises `OSError` or `ValueError`; returns the number of tensors rewritten.
    """
    with source.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) < 8:
            raise ValueError(f"{source} is too short to hold a safetensors header")
        length = struct.unpack("<Q", raw_length)[0]
        if length <= 0 or length > _MAX_HEADER_BYTES:
            raise ValueError(f"{source} declares an implausible header length {length}")
        header = json.loads(handle.read(length))
        if not isinstance(header, dict):
            raise ValueError(f"{source} has a header that is not an object")

        rebuilt: dict[str, object] = {}
        if isinstance(header.get("__metadata__"), dict):
            # Kept for inspectability: ss_num_train_images and training_info are what
            # list_checkpoint_files reads, and nothing downstream is confused by them.
            rebuilt["__metadata__"] = header["__metadata__"]
        renamed = 0
        for key, value in header.items():
            if key == "__metadata__":
                continue
            rebuilt[_resume_layout_key(key)] = value
            renamed += 1
        if renamed == 0:
            raise ValueError(f"{source} carries no tensors")
        if len(rebuilt) - (1 if "__metadata__" in rebuilt else 0) != renamed:
            raise ValueError(f"{source} has keys that collide once renamed")

        encoded = json.dumps(rebuilt, separators=(",", ":")).encode("utf-8")
        # safetensors pads the header with spaces so the blob starts 8-byte aligned.
        encoded += b" " * (-(len(encoded) + 8) % 8)

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".{destination.name}.incoming"
        with staging.open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            shutil.copyfileobj(handle, out)
    os.replace(staging, destination)
    return renamed


def materialize_warm_start(
    *,
    step_export: Path,
    adapter_config: Path | None,
    destination_dir: Path,
    step: int,
    source_label: str = "",
) -> int:
    """Stage a branch's output_dir to start from `step_export`'s weights alone.

    Writes the renamed adapter into `resume_checkpoint/`, the parent's
    `adapter_config.json` beside it for inspectability (the trainer builds the adapter
    from its own config and never reads this back), and `warm_start.json` **last** — the
    same commit-last ordering `copy_resume_state` follows, for the same reason: that
    file is what makes the trainer take this path at all, so it must not exist until the
    weights it refers to are complete.

    Deliberately does *not* write `current_step.txt` or `optimizer.pt`. Their absence is
    what keeps `has_checkpoint()`/`checkpoint_step()` answering "no checkpoint here",
    which is exactly right: there is no resumable state, only weights.

    Raises `OSError` or `ValueError`; returns the step recorded in the marker.
    """
    if step <= 0:
        raise ValueError(f"warm start step must be positive, got {step}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    resume_dir = destination_dir / _RESUME_DIR_NAME
    resume_dir.mkdir(parents=True, exist_ok=True)

    rewrite_export_to_resume_layout(step_export, resume_dir / _ADAPTER_FILE_NAME)

    if adapter_config is not None and adapter_config.is_file():
        staging_config = resume_dir / f".{_ADAPTER_CONFIG_NAME}.incoming"
        shutil.copyfile(adapter_config, staging_config)
        os.replace(staging_config, resume_dir / _ADAPTER_CONFIG_NAME)

    marker = {
        "format_version": _WARM_START_FORMAT_VERSION,
        "step": int(step),
        "source": source_label or str(step_export),
    }
    staging_marker = destination_dir / f".{_WARM_START_FILE_NAME}.incoming"
    staging_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    os.replace(staging_marker, destination_dir / _WARM_START_FILE_NAME)
    return int(step)


def warm_start_step(output_dir: Path) -> int | None:
    """The step a staged warm start will begin at, or None if there isn't one.

    The counterpart to `checkpoint_step` for this path, and the same contract: it is the
    hub's read-only preview of what the trainer will decide, so it checks exactly what
    `_restore_warm_start` checks — a readable marker of the right version, and an adapter
    for it to refer to.
    """
    try:
        marker = json.loads(
            (output_dir / _WARM_START_FILE_NAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(marker, dict):
        return None
    if marker.get("format_version") != _WARM_START_FORMAT_VERSION:
        return None
    if not (output_dir / _RESUME_DIR_NAME / _ADAPTER_FILE_NAME).is_file():
        return None
    step = _int_or_none(marker.get("step"))
    return step if step is not None and step > 0 else None
