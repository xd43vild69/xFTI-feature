import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from feature_pipeline.application import training_service
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.training_runner import TrainingEnvironment

SUCCESSFUL_PRECACHE = """
import json, os
settings = json.load(open(os.environ["PRECACHE_SETTINGS_PATH"]))
assert settings["trigger_word"] == "sks_test"
print("Pre-caching finished! 2 encoded, 0 reused, VRAM freed", flush=True)
"""

FAILING_PRECACHE = """
print("something went wrong before finishing", flush=True)
"""

HANGING_PRECACHE = """
import time
time.sleep(30)
"""

STUB_TRAIN = """
import json, os
settings = json.load(open(os.environ["TRAIN_SETTINGS_PATH"]))
print(f"training with total_steps={settings['total_steps']}", flush=True)
"""


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


@pytest.fixture
def fake_model_dir(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    model = runtime / "model"
    for part in ("transformer", "text_encoder", "vae"):
        (model / part).mkdir(parents=True)
    (runtime / "venv" / "bin").mkdir(parents=True)
    (runtime / "venv" / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime))
    return runtime


def _write_stub(tmp_path: Path, script: str) -> Path:
    path = tmp_path / f"stub_{abs(hash(script))}.py"
    path.write_text(script)
    return path


def test_start_training_runs_precache_then_launches_train(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))

    training_run_id = training_service.start_training(
        conn,
        dataset_run_id="run-1",
        dataset_name="my_concept",
        trigger_word="sks_test",
        config=training_service.TrainingConfig(total_steps=20),
    )

    train_run = repo.get_training_run(conn, training_run_id)
    assert train_run.kind == "train"
    assert train_run.status == "running"
    assert train_run.config["total_steps"] == 20
    assert train_run.config["checkpoint_prefix"] == "my_concept"

    all_runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert sorted(r.kind for r in all_runs) == ["precache", "train"]
    precache_run = next(r for r in all_runs if r.kind == "precache")
    assert precache_run.status == "completed"

    from feature_pipeline.infrastructure import training_runner

    training_runner.stop_process(train_run.pid, grace_period_seconds=1)


def test_advanced_hyperparameters_flow_through_to_the_persisted_config(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    """warmup/scheduler/timestep-weighting/noise/caption-dropout used to fall
    through silently to the worker's own defaults because TrainingConfig didn't
    carry them — this confirms they now reach the launched worker's settings."""
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))

    training_run_id = training_service.start_training(
        conn,
        dataset_run_id="run-1",
        dataset_name="my_concept",
        trigger_word="sks_test",
        config=training_service.TrainingConfig(
            total_steps=20,
            warmup_steps=150,
            lr_scheduler="cosine_with_restarts",
            lr_num_cycles=2,
            timestep_weighting="bell",
            noise_offset=0.05,
            caption_dropout_rate=0.05,
        ),
    )

    train_run = repo.get_training_run(conn, training_run_id)
    assert train_run.config["warmup_steps"] == 150
    assert train_run.config["lr_scheduler"] == "cosine_with_restarts"
    assert train_run.config["lr_num_cycles"] == 2
    assert train_run.config["timestep_weighting"] == "bell"
    assert train_run.config["noise_offset"] == 0.05
    assert train_run.config["caption_dropout_rate"] == 0.05

    from feature_pipeline.infrastructure import training_runner

    training_runner.stop_process(train_run.pid, grace_period_seconds=1)


def test_an_invalid_timestep_weighting_is_rejected_instead_of_silently_falling_back():
    """Krea2's own resolver would silently fall back to 'none' on an unknown choice
    (e.g. a typo'd 'snr', which isn't implemented) — TrainingConfig must reject it
    at build time instead, in the UI thread, before a worker ever launches."""
    with pytest.raises(ValidationError):
        training_service.TrainingConfig(timestep_weighting="snr")


def test_an_invalid_lr_scheduler_is_rejected():
    with pytest.raises(ValidationError):
        training_service.TrainingConfig(lr_scheduler="warm_restarts")


def test_a_failing_precache_raises_and_never_launches_train(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, FAILING_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))

    with pytest.raises(training_service.PrecacheFailed):
        training_service.start_training(
            conn,
            dataset_run_id="run-1",
            dataset_name="my_concept",
            trigger_word="sks_test",
            config=training_service.TrainingConfig(),
        )

    runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert [r.kind for r in runs] == ["precache"]
    assert runs[0].status == "failed"


def test_a_hanging_precache_times_out(conn, fake_model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, HANGING_PRECACHE)
    )
    monkeypatch.setattr(training_service, "PRECACHE_TIMEOUT_SECONDS", 0.5)

    with pytest.raises(training_service.PrecacheFailed, match="did not finish"):
        training_service.start_training(
            conn,
            dataset_run_id="run-1",
            dataset_name="my_concept",
            trigger_word="sks",
            config=training_service.TrainingConfig(),
        )

    runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert runs[0].status == "failed"


def test_precache_telemetry_is_captured_when_the_worker_emits_it(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    """workers/_telemetry.py prints one more JSON line after the vendored
    script's own output; start_training should pick it up as duration/cost."""
    telemetry_line = (
        SUCCESSFUL_PRECACHE
        + '\nimport json as _j\n'
        'print(_j.dumps({"event": "worker_finished", "worker": "precache", '
        '"duration_seconds": 3.5, "gpu_seconds": 3.5}), flush=True)\n'
    )
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, telemetry_line)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))
    monkeypatch.setenv("FTI_GPU_HOURLY_RATE", "1.0")

    training_run_id = training_service.start_training(
        conn,
        dataset_run_id="run-1",
        dataset_name="my_concept",
        trigger_word="sks_test",
        config=training_service.TrainingConfig(total_steps=20),
    )

    from feature_pipeline.infrastructure import training_runner

    train_run = repo.get_training_run(conn, training_run_id)
    training_runner.stop_process(train_run.pid, grace_period_seconds=1)

    precache_run = next(
        r for r in repo.list_training_runs(conn, dataset_run_id="run-1") if r.kind == "precache"
    )
    assert precache_run.status == "completed"
    assert precache_run.duration_seconds == 3.5
    assert precache_run.gpu_seconds == 3.5
    assert precache_run.cost_estimate == round(3.5 / 3600, 4)


def test_finalize_dead_run_uses_the_lifecycle_event_when_present(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("FTI_GPU_HOURLY_RATE", "3.6")  # $3.6/hour == $0.001/second
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        '{"event": "worker_finished", "worker": "train", "duration_seconds": 100.0, '
        '"gpu_seconds": 100.0}\n'
    )
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="train", pid=99999, log_path=str(log_path), config={}
    )
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="failed")

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "completed"
    assert updated.duration_seconds == 100.0
    assert updated.cost_estimate == pytest.approx(0.1)


def test_finalize_dead_run_reports_the_workers_own_error(conn, tmp_path):
    log_path = tmp_path / "log.txt"
    log_path.write_text('{"event": "worker_failed", "worker": "train", "error": "CUDA OOM"}\n')
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="train", pid=99999, log_path=str(log_path), config={}
    )
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="completed")

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "failed"
    assert updated.error_message == "CUDA OOM"


def test_finalize_dead_run_falls_back_without_a_lifecycle_event(conn, tmp_path):
    """A process that died too hard to print anything (killed, crash) — falls
    back to the caller's status instead of claiming telemetry that doesn't exist."""
    log_path = tmp_path / "log.txt"
    log_path.write_text("some free-text log output, no JSON at all\n")
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="train", pid=99999, log_path=str(log_path), config={}
    )
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="failed")

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "failed"
    assert updated.duration_seconds is None


TRAIN_LOG_HEADER = (
    "step,update,epoch,loss,loss_avg,grad_norm,lr,sigma,bucket_h,bucket_w,secs,vram_peak_gb"
)


def _write_train_log(output_dir: Path, steps) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [TRAIN_LOG_HEADER] + [
        f"{step},{step // 4},1,0.100000,0.100000,1.0000,1.000e-04,0.5000,64,64,0.500,8.00"
        for step in steps
    ]
    (output_dir / "train_log.csv").write_text("\n".join(rows) + "\n")


def _train_run_with_log(conn, tmp_path, steps, *, log_name="log.txt"):
    output_dir = tmp_path / "checkpoints"
    if steps is not None:
        _write_train_log(output_dir, steps)
    log_path = tmp_path / log_name
    log_path.touch()
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="train", pid=99999,
        log_path=str(log_path), config={"output_dir": str(output_dir), "total_steps": 1200},
    )
    return training_run_id, log_path, output_dir


def test_read_train_log_summary_ignores_a_precache_run(conn, tmp_path):
    # precache_status finalizes pre-cache rows through the same path, and those have
    # no output_dir — the legacy fallback would point at an unrelated directory.
    _write_train_log(tmp_path / "checkpoints", [4, 8])
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="precache", pid=1,
        log_path=str(tmp_path / "log.txt"), config={"output_dir": str(tmp_path / "checkpoints")},
    )

    run = repo.get_training_run(conn, training_run_id)
    assert training_service.read_train_log_summary(run) is None


@pytest.mark.parametrize("content", [None, "", TRAIN_LOG_HEADER + "\n"])
def test_read_train_log_summary_returns_none_when_there_is_nothing_to_read(
    conn, tmp_path, content
):
    training_run_id, _, output_dir = _train_run_with_log(conn, tmp_path, None)
    if content is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_log.csv").write_text(content)

    run = repo.get_training_run(conn, training_run_id)
    assert training_service.read_train_log_summary(run) is None


def test_finalize_dead_run_records_real_steps_even_with_no_lifecycle_event(conn, tmp_path):
    """The killed-run case, and the reason the CSV is read before the early return.

    A process killed outright prints no worker_finished line, so there is no
    telemetry to recover — but krea2.metrics flushes every row as it writes it, so
    the CSV survives intact. That is exactly the run whose real step count matters.
    """
    training_run_id, log_path, _ = _train_run_with_log(conn, tmp_path, range(4, 401, 4))
    log_path.write_text("free-text output, killed before it could finish\n")
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="failed")

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "failed"
    assert updated.duration_seconds is None      # genuinely unknown
    assert updated.steps_executed == 400         # but this is not
    assert updated.metrics["updates_logged"] == 100


def test_finalize_dead_run_stores_metrics_alongside_the_lifecycle_telemetry(conn, tmp_path):
    training_run_id, log_path, _ = _train_run_with_log(conn, tmp_path, range(4, 1201, 4))
    log_path.write_text(
        '{"event": "worker_finished", "worker": "train", "duration_seconds": 600.0, '
        '"gpu_seconds": 600.0}\n'
    )
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="failed")

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "completed"
    assert updated.duration_seconds == 600.0
    assert updated.steps_executed == 1200


def test_finalize_dead_run_on_a_precache_row_leaves_the_metrics_alone(conn, tmp_path):
    log_path = tmp_path / "log.txt"
    log_path.write_text("nothing structured here\n")
    training_run_id = repo.create_training_run(
        conn, dataset_run_id="r1", kind="precache", pid=1, log_path=str(log_path), config={}
    )
    run = repo.get_training_run(conn, training_run_id)

    training_service.finalize_dead_run(conn, run, fallback_status="failed")

    assert repo.get_training_run(conn, training_run_id).steps_executed is None


def test_stopping_a_run_records_how_far_it_got(conn, tmp_path, monkeypatch):
    # A stopped run leaves 'running' at once, so finalize_dead_run never revisits
    # it — whatever is written here is all it will ever have.
    monkeypatch.setattr(training_service.training_runner, "stop_process", lambda pid: True)
    training_run_id, _, _ = _train_run_with_log(conn, tmp_path, range(4, 401, 4))

    training_service.stop_training(conn, training_run_id)

    updated = repo.get_training_run(conn, training_run_id)
    assert updated.status == "stopped"
    assert updated.steps_executed == 400
    assert updated.duration_seconds is not None and updated.duration_seconds >= 0


def test_dataset_and_cache_dirs_are_scoped_under_the_runtime(fake_model_dir):
    dataset_dir = training_service.dataset_dir_for("my_concept")
    cache_dir = training_service.cache_dir_for("my_concept")

    assert dataset_dir == fake_model_dir / "datasets" / "my_concept"
    assert cache_dir == fake_model_dir / "cache" / "my_concept"


def _write_dataset(root: Path, name: str, captions: list[str]) -> Path:
    """A minimal training_runtime/datasets/{name}/ with one .txt per caption."""
    dataset_dir = root / "datasets" / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for i, caption in enumerate(captions):
        (dataset_dir / f"img_{i}.txt").write_text(caption, encoding="utf-8")
    return dataset_dir


def test_detect_dataset_version_conflict_finds_a_stale_earlier_version(fake_model_dir):
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v2", ["vu_bd_v1, a photo", "vu_bd_v1, another"])

    conflict = training_service.detect_dataset_version_conflict("vu_bd_v2")

    assert conflict is not None
    assert conflict.current_name == "vu_bd_v2"
    assert conflict.stale_trigger_word == "vu_bd_v1"
    assert conflict.affected_files == 2
    assert conflict.suggested_next_version == "vu_bd_v3"


def test_detect_dataset_version_conflict_suggests_v3_even_when_v4_already_exists(
    fake_model_dir,
):
    """The next suggestion must be the first free slot after the current version,
    not the global max + 1 — otherwise a pre-existing v4 (e.g. a WIP folder) would
    make the suggestion skip straight past the actually-free v3."""
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v2", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v4", ["vu_bd_v4, a photo"])

    conflict = training_service.detect_dataset_version_conflict("vu_bd_v2")

    assert conflict is not None
    assert conflict.suggested_next_version == "vu_bd_v3"


def test_detect_dataset_version_conflict_skips_a_taken_next_version(fake_model_dir):
    """If v3 is already claimed by another dataset, the suggestion must skip past
    it to the next actually-free slot instead of colliding with it."""
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v2", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v3", ["vu_bd_v3, a photo"])

    conflict = training_service.detect_dataset_version_conflict("vu_bd_v2")

    assert conflict is not None
    assert conflict.suggested_next_version == "vu_bd_v4"


def test_detect_dataset_version_conflict_is_none_when_captions_are_already_current(
    fake_model_dir,
):
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v2", ["vu_bd_v2, a photo"])

    assert training_service.detect_dataset_version_conflict("vu_bd_v2") is None


def test_detect_dataset_version_conflict_is_none_without_a_versioned_name(fake_model_dir):
    _write_dataset(fake_model_dir, "my_concept", ["sks, a photo"])

    assert training_service.detect_dataset_version_conflict("my_concept") is None


def test_detect_dataset_version_conflict_is_none_without_sibling_versions(fake_model_dir):
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])

    assert training_service.detect_dataset_version_conflict("vu_bd_v1") is None


def test_detect_dataset_version_conflict_ignores_partial_word_matches(fake_model_dir):
    """vu_bd_v1 must not match inside vu_bd_v10 or vu_bd_v1_extra — whole-word only."""
    _write_dataset(fake_model_dir, "vu_bd_v1", ["vu_bd_v1, a photo"])
    _write_dataset(fake_model_dir, "vu_bd_v2", ["vu_bd_v1_extra, a photo, vu_bd_v10 style"])

    assert training_service.detect_dataset_version_conflict("vu_bd_v2") is None


def test_update_captions_in_dataset_dir_replaces_whole_words_only(tmp_path):
    dataset_dir = tmp_path / "vu_bd_v2"
    dataset_dir.mkdir()
    (dataset_dir / "a.txt").write_text("vu_bd_v1, a photo of a dog", encoding="utf-8")
    (dataset_dir / "b.txt").write_text("vu_bd_v1_extra style, unrelated", encoding="utf-8")
    (dataset_dir / "c.txt").write_text("no mention here", encoding="utf-8")

    updated = training_service.update_captions_in_dataset_dir(
        dataset_dir, "vu_bd_v1", "vu_bd_v2"
    )

    assert updated == 1
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "vu_bd_v2, a photo of a dog"
    assert (dataset_dir / "b.txt").read_text(
        encoding="utf-8"
    ) == "vu_bd_v1_extra style, unrelated"
    assert (dataset_dir / "c.txt").read_text(encoding="utf-8") == "no mention here"


def test_update_captions_in_dataset_dir_keeps_a_backup(tmp_path):
    dataset_dir = tmp_path / "vu_bd_v2"
    dataset_dir.mkdir()
    (dataset_dir / "a.txt").write_text("vu_bd_v1, a photo", encoding="utf-8")

    training_service.update_captions_in_dataset_dir(dataset_dir, "vu_bd_v1", "vu_bd_v2")

    assert (dataset_dir / "a.txt.bak").read_text(encoding="utf-8") == "vu_bd_v1, a photo"


def test_update_captions_in_dataset_dir_returns_zero_when_nothing_matches(tmp_path):
    dataset_dir = tmp_path / "vu_bd_v2"
    dataset_dir.mkdir()
    (dataset_dir / "a.txt").write_text("vu_bd_v2, already current", encoding="utf-8")

    updated = training_service.update_captions_in_dataset_dir(
        dataset_dir, "vu_bd_v1", "vu_bd_v2"
    )

    assert updated == 0


def _write_checkpoint(output_dir: Path, step: int) -> None:
    """The three files krea2.state.CheckpointManager.has_checkpoint() looks for."""
    (output_dir / "resume_checkpoint").mkdir(parents=True, exist_ok=True)
    (output_dir / "resume_checkpoint" / "adapter_model.safetensors").write_bytes(b"w")
    (output_dir / "optimizer.pt").write_bytes(b"o")
    (output_dir / "current_step.txt").write_text(str(step))


def _record_train_run(conn, output_dir: Path, *, total_steps: int = 1200, status: str = "stopped"):
    training_run_id = repo.create_training_run(
        conn,
        dataset_run_id="run-1",
        kind="train",
        pid=99999,
        log_path=str(output_dir.parent / "log.txt"),
        config={
            "model_id": "/m",
            "dataset_path": "/d",
            "cache_dir": "/c",
            "output_dir": str(output_dir),
            "trigger_word": "sks_test",
            "checkpoint_prefix": "my_concept",
            "total_steps": total_steps,
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
        },
    )
    repo.update_training_run_status(conn, training_run_id, status)
    return training_run_id


def test_checkpoint_step_reads_a_complete_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, 850)

    assert training_service.checkpoint_step(tmp_path) == 850


@pytest.mark.parametrize(
    "missing", ["optimizer.pt", "current_step.txt", "resume_checkpoint/adapter_model.safetensors"]
)
def test_checkpoint_step_rejects_an_incomplete_checkpoint(tmp_path, missing):
    """Any one of the three files absent means the trainer would refuse it too —
    the hub must not offer a resume the trainer will silently start from 0."""
    _write_checkpoint(tmp_path, 850)
    (tmp_path / missing).unlink()

    assert training_service.checkpoint_step(tmp_path) is None


def test_find_resume_points_lists_only_runs_with_a_checkpoint(conn, tmp_path):
    with_ckpt = tmp_path / "train-a" / "checkpoints"
    _write_checkpoint(with_ckpt, 400)
    without_ckpt = tmp_path / "train-b" / "checkpoints"
    without_ckpt.mkdir(parents=True)
    _record_train_run(conn, with_ckpt)
    _record_train_run(conn, without_ckpt)

    points = training_service.find_resume_points(conn, "run-1")

    assert [(p.step, p.output_dir) for p in points] == [(400, with_ckpt)]
    assert points[0].total_steps == 1200


def test_find_resume_points_ignores_a_still_running_run(conn, tmp_path):
    output_dir = tmp_path / "train-a" / "checkpoints"
    _write_checkpoint(output_dir, 400)
    _record_train_run(conn, output_dir, status="running")

    assert training_service.find_resume_points(conn, "run-1") == []


def test_resume_training_reuses_the_original_output_dir_and_hyperparameters(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    """The resumed launch must point at the same output_dir — that is where the
    trainer finds the checkpoint and appends train_log.csv — while getting a log
    of its own so the original run's log is not truncated."""
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))
    output_dir = tmp_path / "train-a" / "checkpoints"
    _write_checkpoint(output_dir, 400)
    original_id = _record_train_run(conn, output_dir)
    point = training_service.find_resume_points(conn, "run-1")[0]

    resumed_id = training_service.resume_training(
        conn, dataset_run_id="run-1", resume_point=point, total_steps=2000
    )

    resumed = repo.get_training_run(conn, resumed_id)
    assert resumed_id != original_id
    assert resumed.config["output_dir"] == str(output_dir)
    assert resumed.config["total_steps"] == 2000
    assert resumed.config["lora_rank"] == 16  # carried over untouched
    assert resumed.config["checkpoint_prefix"] == "my_concept"  # carried over untouched
    assert resumed.log_path != repo.get_training_run(conn, original_id).log_path

    from feature_pipeline.infrastructure import training_runner

    training_runner.stop_process(resumed.pid, grace_period_seconds=1)


def test_resume_training_refuses_a_total_below_the_checkpoint(conn, tmp_path):
    """The trainer would just report 'nothing to do' and exit — catch it here
    instead of burning a pre-cache and a launch to find out."""
    output_dir = tmp_path / "train-a" / "checkpoints"
    _write_checkpoint(output_dir, 1200)
    _record_train_run(conn, output_dir)
    point = training_service.find_resume_points(conn, "run-1")[0]

    with pytest.raises(training_service.ResumeUnavailable, match="step 1200"):
        training_service.resume_training(
            conn, dataset_run_id="run-1", resume_point=point, total_steps=1200
        )


def test_training_log_csv_path_follows_the_recorded_output_dir():
    """A resumed run's output_dir is the original run's, not a sibling of its log."""
    run = repo.TrainingRun(
        training_run_id="t2",
        dataset_run_id="r1",
        kind="train",
        status="running",
        pid=1,
        log_path="/x/runs/train-resumed/log.txt",
        config={"output_dir": "/x/runs/train-original/checkpoints"},
        started_at=None,
        finished_at=None,
    )

    assert training_service.training_log_csv_path(run) == Path(
        "/x/runs/train-original/checkpoints/train_log.csv"
    )


def test_training_log_csv_path_is_next_to_the_checkpoints():
    run = repo.TrainingRun(
        training_run_id="t1",
        dataset_run_id="r1",
        kind="train",
        status="running",
        pid=1,
        log_path="/x/runs/train-abc/log.txt",
        config={},
        started_at=None,
        finished_at=None,
    )

    assert training_service.training_log_csv_path(run) == Path(
        "/x/runs/train-abc/checkpoints/train_log.csv"
    )
