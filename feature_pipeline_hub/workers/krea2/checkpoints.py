"""Filesystem discipline for checkpoints: atomic writes and retention.

A training run is killed by SIGTERM, an OOM, or a closed laptop far more often than it
finishes cleanly, so every write here assumes it may be interrupted halfway.

Free of torch — these operate on paths and a caller-supplied writer, not on tensors.
"""
from __future__ import annotations

import os
import re
from typing import Callable

Logger = Callable[[str], None]

CHECKPOINT_PATTERN = re.compile(r"Krea2_LoRA_step_(\d+)\.safetensors")


def atomic_write(path: str, writer: Callable[[str], None]) -> None:
    """Write through a temp file then `os.replace` — atomic on POSIX and Windows.

    `writer` is called with the temp path. A crash mid-write leaves the previous file
    intact rather than a truncated one, which matters because a half-written adapter
    still loads and trains to garbage.
    """
    tmp = f"{path}.tmp"
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def rotate(output_dir: str, keep: int, log: Logger = print) -> list[str]:
    """Keep only the `keep` most recent per-step checkpoints; `keep <= 0` keeps all.

    Ordering comes from the step number parsed out of the filename, not ctime — ctime
    lies after a copy or an rsync, the step number does not. Never touches the FINAL
    export or any unrelated file. Returns the paths actually removed.
    """
    if keep <= 0:
        return []

    found: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        match = CHECKPOINT_PATTERN.fullmatch(name)
        if match:
            found.append((int(match.group(1)), os.path.join(output_dir, name)))
    found.sort(key=lambda pair: pair[0])

    removed: list[str] = []
    for _, path in found[:-keep]:
        try:
            os.remove(path)
            removed.append(path)
            log(f"  ↳ pruned old checkpoint / checkpoint antiguo eliminado: "
                f"{os.path.basename(path)}")
        except OSError as exc:
            log(f"  [!] Could not prune {os.path.basename(path)}: {exc}")
    return removed


def belongs_to_run(run_id_file: str, run_id: str) -> bool:
    """True when the checkpoint in this output dir belongs to the current pipeline run.

    Without a run id (standalone trainer) any checkpoint is accepted. With one, a
    checkpoint from an earlier run must be discarded: otherwise the phase would restore
    it with start_step == total_steps, ignore `init_lora_from`, and run an empty loop.
    """
    if not run_id:
        return True
    try:
        with open(run_id_file, "r", encoding="utf-8") as handle:
            return handle.read().strip() == run_id
    except Exception:
        return False
