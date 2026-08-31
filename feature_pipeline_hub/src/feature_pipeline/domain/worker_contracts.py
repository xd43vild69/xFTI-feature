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

ModelArch = Literal["krea2", "ltx23"]


class _StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrecacheSettings(_StrictSettings):
    """Keys this project sets for workers/precache_worker.py (Krea 2)."""

    model_id: str
    dataset_path: str
    cache_dir: str
    trigger_word: str = ""


class LTX23PrecacheSettings(_StrictSettings):
    """Keys this project sets for workers/precache_ltx23_worker.py."""

    model_id: str
    dataset_path: str
    cache_dir: str
    trigger_word: str = ""
    target_area: int = 512 * 512
    multiple: int = 32
    max_seq_len: int = 1024
    frame_rate: float = 24.0
    num_frames: int = 1
    precache_offload: Literal["none", "model", "sequential", "cpu"] = "sequential"
    text_encoder_4bit: bool = True
    preview_custom_prompt: str = ""


class TrainSettings(_StrictSettings):
    """Keys this project sets for workers/train_worker.py (Krea 2)."""

    model_id: str
    dataset_path: str
    cache_dir: str
    output_dir: str
    trigger_word: str = ""
    checkpoint_prefix: str = ""
    total_steps: int = Field(gt=0)
    lr: float = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    grad_accum_steps: int = Field(gt=0)
    save_every: int = Field(gt=0)
    seed: int = Field(ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    lr_scheduler: Literal["cosine", "constant", "linear", "cosine_with_restarts", "step"] = "cosine"
    lr_num_cycles: int = Field(default=3, gt=0)
    timestep_weighting: Literal["none", "bell", "half_bell"] = "none"
    noise_offset: float = Field(default=0.0, ge=0.0)
    caption_dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class LTX23TrainSettings(_StrictSettings):
    """Keys this project sets for workers/train_ltx23_worker.py."""

    model_id: str
    dataset_path: str
    cache_dir: str
    output_dir: str
    trigger_word: str = ""
    project_name: str = ""
    total_steps: int = Field(gt=0)
    lr: float = Field(gt=0)
    lora_rank: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum_steps: int = Field(default=4, gt=0)
    save_every: int = Field(default=200, gt=0)
    seed: int = Field(default=314159, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    min_lr_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.001, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    frame_rate: float = 24.0
    max_text_tokens: int = 256
    lora_only_attn: bool = True
    cast_frozen_bf16: bool = True
    use_audio_loss: bool = False
    low_vram_12gb: bool = True
    activation_offload: bool = True
    loss_chunk_elements: int = 2000000
    lora_key_prefix: str = "diffusion_model."
    preview_every: int = 0
    preview_steps: int = 30
    preview_cfg: float = 3.0
    preview_mode: Literal["gen", "recon", "onestep"] = "gen"
    preview_vae_fp32: bool = True
    timestep_sampling: Literal["logit_normal", "uniform"] = "logit_normal"
    timestep_shift: float = Field(default=1.0, gt=0.0)
    use_loss_weighting: bool = False
    caption_dropout_prob: float = Field(default=0.05, ge=0.0, le=1.0)
    conditioning_mode: Literal["t2v", "i2v"] = "t2v"
    lr_schedule: Literal["constant_with_warmup", "cosine"] = "constant_with_warmup"
    cond_noise_prob: float = Field(default=0.15, ge=0.0, le=1.0)
    cond_noise_scale: float = Field(default=0.03, ge=0.0)
    use_ema: bool = True
    use_dora: bool = False


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
