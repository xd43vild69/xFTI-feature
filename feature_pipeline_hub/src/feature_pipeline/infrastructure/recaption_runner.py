"""Launches the recaption worker in the training runtime's virtualenv and streams its events.

Qwen3-VL is the same text encoder the training runtime already carries a local
copy of (see training_runner.py / scripts/setup_training_runtime.sh), so
recaptioning reuses that interpreter and model copy instead of reaching into an
external checkout. This module owns the process boundary: locating that
environment, feeding it a job, and turning its JSON Lines output back into
events the application layer can consume.
"""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from feature_pipeline.infrastructure import training_runner

WORKER = Path(__file__).resolve().parents[3] / "workers" / "recaption_worker.py"


@dataclass(frozen=True)
class RecaptionEnvironment:
    """Everything needed to run the worker, already checked to exist on disk."""

    runtime_dir: Path
    python: Path

    @property
    def model_dir(self) -> Path:
        return self.runtime_dir / "model" / "text_encoder"


class RecaptionUnavailable(RuntimeError):
    """The recaption environment is not configured or is incomplete."""


def resolve_environment() -> RecaptionEnvironment:
    """Locate the training runtime's interpreter and text encoder, or explain what is missing."""
    try:
        training_env = training_runner.resolve_environment()
    except training_runner.TrainingUnavailable as exc:
        raise RecaptionUnavailable(
            f"AI recaptioning reuses the training runtime, which isn't set up yet: {exc}"
        ) from exc

    model_dir = training_env.model_dir / "text_encoder"
    if not (model_dir / "model.safetensors").is_file():
        raise RecaptionUnavailable(f"No Qwen3-VL weights found at {model_dir}")

    if not WORKER.is_file():
        raise RecaptionUnavailable(f"Recaption worker script is missing at {WORKER}")

    return RecaptionEnvironment(runtime_dir=training_env.runtime_dir, python=training_env.python)


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
        {"text_encoder_dir": str(env.model_dir), "images": image_paths, "detailed": detailed}
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
