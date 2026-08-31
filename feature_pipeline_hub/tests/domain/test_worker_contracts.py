"""The settings/event schemas are the only thing standing between a typo and a
silently wrong training run, so the failure cases matter more than the happy path."""

import pytest
from pydantic import ValidationError

from feature_pipeline.domain.worker_contracts import (
    CaptionEvent,
    FailedEvent,
    LTX23PrecacheSettings,
    LTX23TrainSettings,
    PrecacheSettings,
    TrainSettings,
    WorkerLifecycleEvent,
    recaption_event_adapter,
    worker_lifecycle_event_adapter,
)

VALID_TRAIN = {
    "model_id": "/models/krea2",
    "dataset_path": "/datasets/cats",
    "cache_dir": "/cache/cats",
    "output_dir": "/runs/train-1/checkpoints",
    "trigger_word": "sks_cat",
    "checkpoint_prefix": "cats",
    "total_steps": 1200,
    "lr": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "save_every": 25,
    "seed": 42,
    "warmup_steps": 100,
    "lr_scheduler": "cosine",
    "lr_num_cycles": 3,
    "timestep_weighting": "none",
    "noise_offset": 0.0,
    "caption_dropout_rate": 0.0,
}


def test_train_settings_accept_the_keys_the_worker_reads():
    assert TrainSettings(**VALID_TRAIN).model_dump() == VALID_TRAIN


def test_an_unknown_key_is_rejected_instead_of_silently_ignored():
    with pytest.raises(ValidationError):
        TrainSettings(**VALID_TRAIN | {"total_setps": 1200})


def test_a_run_that_would_do_nothing_is_rejected():
    with pytest.raises(ValidationError):
        TrainSettings(**VALID_TRAIN | {"total_steps": 0})
    with pytest.raises(ValidationError):
        TrainSettings(**VALID_TRAIN | {"lr": 0.0})


def test_precache_settings_require_the_paths():
    with pytest.raises(ValidationError):
        PrecacheSettings(model_id="/models/krea2", dataset_path="/datasets/cats")


def test_events_are_parsed_into_their_own_type():
    event = recaption_event_adapter.validate_python(
        {"event": "caption", "path": "/data/a.png", "caption": "a red car", "seconds": 2.0}
    )
    assert isinstance(event, CaptionEvent)
    assert event.caption == "a red car"

    assert isinstance(
        recaption_event_adapter.validate_python({"event": "failed", "message": "boom"}),
        FailedEvent,
    )


def test_worker_noise_does_not_parse_as_an_event():
    with pytest.raises(ValidationError):
        recaption_event_adapter.validate_python({"event": "progress_bar", "pct": 40})


def test_unknown_fields_on_a_known_event_are_tolerated():
    """A newer worker adding a field must not break an older hub."""
    event = recaption_event_adapter.validate_python(
        {"event": "caption", "path": "/data/a.png", "caption": "x", "model_revision": "abc"}
    )
    assert isinstance(event, CaptionEvent)


def test_recaption_events_carry_timestamp_and_run_id():
    event = recaption_event_adapter.validate_python(
        {
            "event": "caption",
            "path": "/data/a.png",
            "caption": "a red car",
            "seconds": 2.0,
            "timestamp": 1_700_000_000.0,
            "run_id": "batch-1",
        }
    )
    assert isinstance(event, CaptionEvent)
    assert event.timestamp == 1_700_000_000.0
    assert event.run_id == "batch-1"


def test_worker_lifecycle_events_parse():
    event = worker_lifecycle_event_adapter.validate_python(
        {
            "event": "worker_finished",
            "worker": "precache",
            "run_id": "precache-abc",
            "timestamp": 1_700_000_000.0,
            "duration_seconds": 42.5,
            "gpu_seconds": 42.5,
        }
    )
    assert isinstance(event, WorkerLifecycleEvent)
    assert event.worker == "precache"
    assert event.duration_seconds == 42.5


def test_worker_lifecycle_event_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        worker_lifecycle_event_adapter.validate_python({"event": "worker_paused"})


def test_ltx23_precache_settings_validation():
    valid = {
        "model_id": "/models/ltx23",
        "dataset_path": "/data",
        "cache_dir": "/cache",
        "multiple": 32,
        "precache_offload": "sequential",
    }
    settings = LTX23PrecacheSettings(**valid)
    assert settings.multiple == 32
    assert settings.precache_offload == "sequential"

    with pytest.raises(ValidationError):
        LTX23PrecacheSettings(**valid | {"extra_field": "invalid"})


def test_ltx23_train_settings_validation():
    valid = {
        "model_id": "/models/ltx23",
        "dataset_path": "/data",
        "cache_dir": "/cache",
        "output_dir": "/out",
        "total_steps": 500,
        "lr": 1e-4,
        "lora_rank": 32,
        "lora_alpha": 32,
    }
    settings = LTX23TrainSettings(**valid)
    assert settings.total_steps == 500
    assert settings.lora_rank == 32
    assert settings.lora_key_prefix == "diffusion_model."
    assert settings.timestep_shift == 1.0
    assert settings.use_loss_weighting is False
    assert settings.conditioning_mode == "i2v"
    assert settings.lr_schedule == "constant_with_warmup"
    assert settings.cond_noise_prob == 0.15

    with pytest.raises(ValidationError):
        LTX23TrainSettings(**valid | {"total_steps": 0})

