"""Schemas for the two process boundaries: worker settings in, worker events out.

Both boundaries used to be plain dicts crossing a process gap, which meant a typo
was never an error: `total_setps` simply never reached the worker, whose `_cfg()`
fallback chain silently used its own default — visible only hours into a run.

Settings models therefore set `extra="forbid"`: an unknown key fails at build time,
in the UI thread, before a process is spawned. Event models do the opposite and
ignore unknown fields, so a worker that starts reporting something new does not
break an older hub.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# --- worker settings (hub -> worker, written as settings.json) ----------------


class _StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrecacheSettings(_StrictSettings):
    """Keys this project sets for workers/precache_worker.py.

    Everything omitted falls through to that worker's own DEFAULTS, which is the
    intended way to drive the ported LoRAlab scripts.
    """

    model_id: str
    dataset_path: str
    cache_dir: str
    trigger_word: str = ""


class TrainSettings(_StrictSettings):
    """Keys this project sets for workers/train_worker.py."""

    model_id: str
    dataset_path: str
    cache_dir: str
    output_dir: str
    trigger_word: str = ""
    total_steps: int = Field(gt=0)
    lr: float = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    grad_accum_steps: int = Field(gt=0)
    save_every: int = Field(gt=0)
    seed: int = Field(ge=0)


# --- worker events (worker -> hub, one JSON object per stdout line) -----------
# Mirrors the protocol documented in workers/recaption_worker.py.


class LoadedEvent(BaseModel):
    event: Literal["loaded"]
    device: str = ""
    seconds: float = 0.0
    timestamp: float = 0.0
    run_id: str = ""


class CaptionEvent(BaseModel):
    event: Literal["caption"]
    path: str
    caption: str = ""
    seconds: float = 0.0
    timestamp: float = 0.0
    run_id: str = ""


class ErrorEvent(BaseModel):
    event: Literal["error"]
    path: str = ""
    message: str = "Unknown error"
    timestamp: float = 0.0
    run_id: str = ""


class DoneEvent(BaseModel):
    event: Literal["done"]
    captioned: int = 0
    failed: int = 0
    timestamp: float = 0.0
    run_id: str = ""


class FailedEvent(BaseModel):
    """Not emitted by the worker: synthesised by recaption_runner on a non-zero exit."""

    event: Literal["failed"]
    message: str = ""
    run_id: str = ""


RecaptionEvent = Annotated[
    Union[LoadedEvent, CaptionEvent, ErrorEvent, DoneEvent, FailedEvent],
    Field(discriminator="event"),
]

recaption_event_adapter: TypeAdapter[RecaptionEvent] = TypeAdapter(RecaptionEvent)


# --- worker lifecycle telemetry (precache_worker.py / train_worker.py) --------
# precache_worker.py and train_worker.py are vendored byte-for-byte from
# LoRAlab (see their own docstrings), so nothing inside their bodies emits
# structured events the way recaption_worker.py does. workers/_telemetry.py
# wraps only their `if __name__ == "__main__":` entrypoint — untouched
# otherwise — to emit these three events around the vendored call, on the
# same stdout stream that already carries the free-text log.


class WorkerLifecycleEvent(BaseModel):
    event: Literal["worker_started", "worker_finished", "worker_failed"]
    worker: str = ""
    run_id: str = ""
    timestamp: float = 0.0
    duration_seconds: float = 0.0
    gpu_seconds: float = 0.0
    error: str = ""


worker_lifecycle_event_adapter: TypeAdapter[WorkerLifecycleEvent] = TypeAdapter(WorkerLifecycleEvent)
