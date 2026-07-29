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


def _run_precache_blocking(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    model_dir: Path,
    dataset_path: Path,
    cache_dir: Path,
    trigger_word: str,
) -> None:
    run_dir = training_runtime_dir() / "runs" / f"precache-{uuid.uuid4()}"
    settings = PrecacheSettings(
        model_id=str(model_dir),
        dataset_path=str(dataset_path),
        cache_dir=str(cache_dir),
        trigger_word=trigger_word,
    ).model_dump()

    pid, log_path = training_runner.launch(
        PRECACHE_SCRIPT, settings, run_dir, PRECACHE_SETTINGS_ENV
    )
    training_run_id = repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind="precache",
        pid=pid,
        log_path=log_path,
        config=settings,
    )

    deadline = time.monotonic() + PRECACHE_TIMEOUT_SECONDS
    while training_runner.is_process_alive(pid):
        if time.monotonic() > deadline:
            training_runner.stop_process(pid)
            repo.update_training_run_status(conn, training_run_id, "failed")
            raise PrecacheFailed(f"Pre-cache did not finish within {PRECACHE_TIMEOUT_SECONDS}s")
        time.sleep(0.5)

    log_text, _ = training_runner.read_log_tail(log_path)
    if "Pre-caching finished" not in log_text:
        repo.update_training_run_status(conn, training_run_id, "failed")
        tail = "\n".join(log_text.splitlines()[-15:])
        raise PrecacheFailed(f"Pre-cache did not report success. Last log lines:\n{tail}")

    repo.update_training_run_status(conn, training_run_id, "completed")


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

    pid, log_path = training_runner.launch(TRAIN_SCRIPT, settings, run_dir, TRAIN_SETTINGS_ENV)

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
