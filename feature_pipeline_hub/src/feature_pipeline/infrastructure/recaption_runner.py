"""Launches the recaption worker in the LoRAlab virtualenv and streams its events.

Qwen3-VL lives in another checkout with its own (heavy) dependency set, so this
module owns the process boundary: locating that environment, feeding it a job, and
turning its JSON Lines output back into events the application layer can consume.
"""

import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Set FTI_LORALAB_ROOT to the AcademiaSD_LoRAlab-Krea2 checkout. There is no
# default on purpose: the path is machine-specific, and guessing wrong would fail
# with a confusing import error deep inside the worker.
LORALAB_ROOT_ENV = "FTI_LORALAB_ROOT"
PYTHON_ENV = "FTI_RECAPTION_PYTHON"

WORKER = Path(__file__).resolve().parents[3] / "workers" / "recaption_worker.py"


@dataclass(frozen=True)
class RecaptionEnvironment:
    """Everything needed to run the worker, already checked to exist on disk."""

    loralab_root: Path
    python: Path

    @property
    def model_dir(self) -> Path:
        return self.loralab_root / "Krea-2-NF4" / "text_encoder"


class RecaptionUnavailable(RuntimeError):
    """The recaption environment is not configured or is incomplete."""


def resolve_environment() -> RecaptionEnvironment:
    """Locate the LoRAlab checkout and interpreter, or explain what is missing."""
    root_value = os.environ.get(LORALAB_ROOT_ENV, "").strip()
    if not root_value:
        raise RecaptionUnavailable(
            f"AI recaptioning needs {LORALAB_ROOT_ENV}. Set it to your "
            "AcademiaSD_LoRAlab-Krea2 checkout before starting the app, e.g. "
            f'`export {LORALAB_ROOT_ENV}=/path/to/AcademiaSD_LoRAlab-Krea2`.'
        )

    root = Path(root_value).expanduser()
    if not root.is_dir():
        raise RecaptionUnavailable(f"{LORALAB_ROOT_ENV} points to a missing folder: {root}")

    python_value = os.environ.get(PYTHON_ENV, "").strip()
    python = Path(python_value).expanduser() if python_value else root / "venv" / "bin" / "python"
    if not python.is_file():
        raise RecaptionUnavailable(
            f"No Python interpreter at {python}. Set {PYTHON_ENV} if the LoRAlab "
            "virtualenv lives elsewhere."
        )

    model_dir = root / "Krea-2-NF4" / "text_encoder"
    if not (model_dir / "model.safetensors").is_file():
        raise RecaptionUnavailable(f"No Qwen3-VL weights found at {model_dir}")

    if not WORKER.is_file():
        raise RecaptionUnavailable(f"Recaption worker script is missing at {WORKER}")

    return RecaptionEnvironment(loralab_root=root, python=python)


def is_available() -> bool:
    try:
        resolve_environment()
    except RecaptionUnavailable:
        return False
    return True


def run_recaption(
    image_paths: list[str], detailed: bool, environment: RecaptionEnvironment | None = None
) -> Iterator[dict]:
    """Caption `image_paths`, yielding the worker's events as they arrive.

    Streaming rather than collecting lets the caller show progress: a batch takes
    a few seconds of model load plus roughly two seconds per image on GPU.
    """
    env = environment or resolve_environment()
    job = json.dumps(
        {"loralab_root": str(env.loralab_root), "images": image_paths, "detailed": detailed}
    )

    process = subprocess.Popen(
        [str(env.python), str(WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(job)
    process.stdin.close()

    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # The worker's dependencies print progress bars and warnings to stdout;
            # anything that is not an event is noise and gets dropped.
            continue

    process.wait()
    if process.returncode != 0:
        stderr = (process.stderr.read() if process.stderr else "").strip()
        tail = "\n".join(stderr.splitlines()[-8:])
        yield {
            "event": "failed",
            "message": f"Recaption worker exited with code {process.returncode}.\n{tail}",
        }
