"""Orchestrates a training run: pre-cache (blocking, short) then train (detached, long).

Builds the settings dicts the ported workers (workers/precache_worker.py,
workers/train_worker.py) expect and launches them through training_runner. Only
the keys this project actually wants to control are set explicitly — every other
field falls back to the worker's own DEFAULTS, exactly as it does for a bare
train_settings.json in LoRAlab (see `_cfg()` in train_worker.py).
"""

import sqlite3
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from feature_pipeline.domain import cost
from feature_pipeline.domain.worker_contracts import PrecacheSettings, TrainSettings
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.storage import training_runtime_dir

WORKERS_DIR = Path(__file__).resolve().parents[3] / "workers"
PRECACHE_SCRIPT = WORKERS_DIR / "precache_worker.py"
TRAIN_SCRIPT = WORKERS_DIR / "train_worker.py"

PRECACHE_SETTINGS_ENV = "PRECACHE_SETTINGS_PATH"  # name LoRAlab's own script reads
TRAIN_SETTINGS_ENV = "TRAIN_SETTINGS_PATH"

PRECACHE_TIMEOUT_SECONDS = 20 * 60  # pre-cache is I/O + VAE-encode bound, minutes not hours


class TrainingConfig(BaseModel):
    """LoRA hyperparameters chosen in the UI.

    Bounds are enforced here rather than only by the Streamlit widgets, so a value
    that would waste a run (zero steps, a negative learning rate) is rejected at
    the point it is built.
    """

    model_config = ConfigDict(frozen=True)

    total_steps: int = Field(default=1200, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    lora_rank: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum_steps: int = Field(default=4, gt=0)
    save_every: int = Field(default=25, gt=0)
    seed: int = Field(default=42, ge=0)


class PrecacheFailed(RuntimeError):
    """Pre-cache exited non-zero or timed out; the caller should not launch training."""


def finalize_dead_run(conn: sqlite3.Connection, run: repo.TrainingRun, *, fallback_status: str) -> None:
    """Move a 'running' row whose process has exited to its real status, with telemetry.

    Reads the worker_finished/worker_failed line workers/_telemetry.py prints as
    its last line of output (see training_runner.read_lifecycle_event) and uses
    it as the source of truth for status, duration and cost — falling back to
    `fallback_status` with no telemetry if the process died too hard to print
    one (killed, machine restart, or a run launched before this existed).
    """
    event = training_runner.read_lifecycle_event(run.log_path)
    if event is None:
        repo.update_training_run_status(conn, run.training_run_id, fallback_status)
        return

    status = "completed" if event.get("event") == "worker_finished" else "failed"
    gpu_seconds = float(event.get("gpu_seconds") or 0.0)
    repo.update_training_run_status(
        conn,
        run.training_run_id,
        status,
        duration_seconds=float(event.get("duration_seconds") or 0.0),
        gpu_seconds=gpu_seconds,
        cost_estimate=cost.estimate_cost(gpu_seconds, cost.gpu_hourly_rate()),
        error_message=str(event.get("error", "")),
    )


def dataset_dir_for(dataset_name: str) -> Path:
    return training_runtime_dir() / "datasets" / dataset_name


def cache_dir_for(dataset_name: str) -> Path:
    return training_runtime_dir() / "cache" / dataset_name


def start_training(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    dataset_name: str,
    trigger_word: str,
    config: TrainingConfig,
) -> str:
    """Run pre-cache to completion, then launch training detached.

    Returns the training_run_id of the **train** job (the one the UI polls for
    progress) — the pre-cache job gets its own row too, but it's already
    finished by the time this returns.
    """
    dataset_path = dataset_dir_for(dataset_name)
    cache_dir = cache_dir_for(dataset_name)
    model_dir = training_runner.resolve_environment().model_dir

    _run_precache_blocking(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
    )

    return _launch_train(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
        config=config,
    )


def launch_precache(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    dataset_name: str,
    trigger_word: str,
) -> str:
    """Launch pre-cache detached and return its training_run_id immediately.

    The non-blocking counterpart to `start_training`'s first phase, for callers
    (like the MCP server) that cannot afford to block a request thread for up to
    `PRECACHE_TIMEOUT_SECONDS`. Poll completion with `precache_status`.
    """
    dataset_path = dataset_dir_for(dataset_name)
    cache_dir = cache_dir_for(dataset_name)
    model_dir = training_runner.resolve_environment().model_dir
    return _launch_precache(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
    )


def precache_status(conn: sqlite3.Connection, training_run_id: str) -> str:
    """Poll a pre-cache job launched via `launch_precache`.

    Returns "running", "completed", or "failed". The first call to observe the
    process has exited self-heals the stored row (same telemetry backfill as
    `finalize_dead_run`) so later calls just read back the settled status.
    """
    run = repo.get_training_run(conn, training_run_id)
    if run is None:
        raise ValueError(f"No such training run: {training_run_id}")
    if run.status != "running":
        return run.status
    if training_runner.is_process_alive(run.pid):
        return "running"

    log_text, _ = training_runner.read_log_tail(run.log_path)
    if "Pre-caching finished" in log_text:
        finalize_dead_run(conn, run, fallback_status="completed")
        return "completed"
    finalize_dead_run(conn, run, fallback_status="failed")
    return "failed"


def is_training_active(conn: sqlite3.Connection) -> bool:
    """Whether a training-runtime job (pre-cache/train/progressive/curate) is
    running right now — the GPU can only do one heavy job at a time.

    Self-healing: a 'running' row whose process actually died (crash, machine
    restart) is corrected to 'failed' here rather than blocking the GPU forever.
    """
    run = repo.find_running_training_run(conn)
    if run is None:
        return False
    if training_runner.is_process_alive(run.pid):
        return True
    finalize_dead_run(conn, run, fallback_status="failed")
    return False


def stop_training(conn: sqlite3.Connection, training_run_id: str) -> None:
    """Send SIGINT to a running job's process and mark its row 'stopped'."""
    run = repo.get_training_run(conn, training_run_id)
    if run is None:
        raise ValueError(f"No such training run: {training_run_id}")
    training_runner.stop_process(run.pid)
    repo.update_training_run_status(conn, training_run_id, "stopped")


def _launch_precache(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    model_dir: Path,
    dataset_path: Path,
    cache_dir: Path,
    trigger_word: str,
) -> str:
    precache_run_id = str(uuid.uuid4())
    run_dir = training_runtime_dir() / "runs" / f"precache-{precache_run_id}"
    settings = PrecacheSettings(
        model_id=str(model_dir),
        dataset_path=str(dataset_path),
        cache_dir=str(cache_dir),
        trigger_word=trigger_word,
    ).model_dump()

    pid, log_path = training_runner.launch(
        PRECACHE_SCRIPT,
        settings,
        run_dir,
        PRECACHE_SETTINGS_ENV,
        extra_env={"FTI_RUN_ID": precache_run_id},
    )
    return repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind="precache",
        pid=pid,
        log_path=log_path,
        config=settings,
    )


def _run_precache_blocking(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    model_dir: Path,
    dataset_path: Path,
    cache_dir: Path,
    trigger_word: str,
) -> None:
    training_run_id = _launch_precache(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
    )
    run = repo.get_training_run(conn, training_run_id)
    assert run is not None

    deadline = time.monotonic() + PRECACHE_TIMEOUT_SECONDS
    while training_runner.is_process_alive(run.pid):
        if time.monotonic() > deadline:
            training_runner.stop_process(run.pid)
            repo.update_training_run_status(conn, training_run_id, "failed")
            raise PrecacheFailed(f"Pre-cache did not finish within {PRECACHE_TIMEOUT_SECONDS}s")
        time.sleep(0.5)

    status = precache_status(conn, training_run_id)
    if status != "completed":
        log_text, _ = training_runner.read_log_tail(run.log_path)
        tail = "\n".join(log_text.splitlines()[-15:])
        raise PrecacheFailed(f"Pre-cache did not report success. Last log lines:\n{tail}")


def launch_train(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    dataset_name: str,
    trigger_word: str,
    config: TrainingConfig,
) -> str:
    """Launch the training phase detached, given pre-cache has already completed.

    The public counterpart to `_launch_train`'s internal use from `start_training`,
    for callers (like the MCP server) that split pre-cache and train into two
    separate steps instead of running them back-to-back in one blocking call.
    """
    dataset_path = dataset_dir_for(dataset_name)
    cache_dir = cache_dir_for(dataset_name)
    model_dir = training_runner.resolve_environment().model_dir
    return _launch_train(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
        config=config,
    )


def _launch_train(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    model_dir: Path,
    dataset_path: Path,
    cache_dir: Path,
    trigger_word: str,
    config: TrainingConfig,
) -> str:
    training_run_id_hint = str(uuid.uuid4())
    run_dir = training_runtime_dir() / "runs" / f"train-{training_run_id_hint}"
    output_dir = run_dir / "checkpoints"

    settings = TrainSettings(
        model_id=str(model_dir),
        dataset_path=str(dataset_path),
        cache_dir=str(cache_dir),
        output_dir=str(output_dir),
        trigger_word=trigger_word,
        **config.model_dump(),
    ).model_dump()

    pid, log_path = training_runner.launch(
        TRAIN_SCRIPT,
        settings,
        run_dir,
        TRAIN_SETTINGS_ENV,
        extra_env={"FTI_RUN_ID": training_run_id_hint},
    )

    return repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind="train",
        pid=pid,
        log_path=log_path,
        config=settings,
    )


def training_log_csv_path(training_run: repo.TrainingRun) -> Path:
    """train_worker.py writes train_log.csv next to the settings/log in its run_dir."""
    return Path(training_run.log_path).parent / "checkpoints" / "train_log.csv"
