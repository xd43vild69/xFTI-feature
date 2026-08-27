"""Filesystem discipline for LTX 2.3 checkpoints: atomic writes and rotation.

Free of torch: covered by mypy and pytest.
"""
from __future__ import annotations

import os
import re
from typing import Callable

Logger = Callable[[str], None]


def atomic_write(path: str, writer: Callable[[str], None]) -> None:
    """Write through a temp file then os.replace for atomic writes on POSIX and Windows."""
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


def rotate(output_dir: str, keep: int, prefix: str, log: Logger = print) -> list[str]:
    """Keep only the `keep` most recent per-step checkpoints; `keep <= 0` keeps all."""
    if keep <= 0 or not os.path.isdir(output_dir):
        return []

    pattern = re.compile(re.escape(prefix) + r"_step_(\d+)\.safetensors")
    found: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        match = pattern.fullmatch(name)
        if match:
            found.append((int(match.group(1)), os.path.join(output_dir, name)))
    found.sort(key=lambda pair: pair[0])

    removed: list[str] = []
    for _, path in found[:-keep]:
        try:
            os.remove(path)
            removed.append(path)
            log(f"  ↳ pruned old checkpoint / checkpoint antiguo eliminado: {os.path.basename(path)}")
        except OSError as exc:
            log(f"  [!] Could not prune {os.path.basename(path)}: {exc}")
    return removed


def belongs_to_run(run_id_file: str, run_id: str) -> bool:
    """True when the checkpoint in output dir belongs to the current pipeline run."""
    if not run_id:
        return True
    try:
        with open(run_id_file, "r", encoding="utf-8") as handle:
            return handle.read().strip() == run_id
    except Exception:
        return False
