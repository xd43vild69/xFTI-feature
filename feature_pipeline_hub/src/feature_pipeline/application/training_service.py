"""Orchestrates a training run: pre-cache (blocking, short) then train (detached, long).

Builds the settings dicts the ported workers (workers/precache_worker.py,
workers/train_worker.py) expect and launches them through training_runner. Only
the keys this project actually wants to control are set explicitly — every other
field falls back to the worker's own DEFAULTS, exactly as it does for a bare
train_settings.json in LoRAlab (see `_cfg()` in train_worker.py).
"""

import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from feature_pipeline.domain import cost
from feature_pipeline.domain.worker_contracts import PrecacheSettings, TrainSettings
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.storage import training_runtime_dir, write_caption_sidecar

WORKERS_DIR = Path(__file__).resolve().parents[3] / "workers"
PRECACHE_SCRIPT = WORKERS_DIR / "precache_worker.py"
TRAIN_SCRIPT = WORKERS_DIR / "train_worker.py"

PRECACHE_SETTINGS_ENV = "PRECACHE_SETTINGS_PATH"  # name LoRAlab's own script reads
TRAIN_SETTINGS_ENV = "TRAIN_SETTINGS_PATH"

PRECACHE_TIMEOUT_SECONDS = 20 * 60  # pre-cache is I/O + VAE-encode bound, minutes not hours

# The three files krea2.state.CheckpointManager.has_checkpoint() requires before it will
# restore anything. `current_step.txt` is written last (see that module's commit
# protocol), so all three present means the checkpoint they describe is complete.
STEP_FILE = "current_step.txt"
OPTIMIZER_STATE_FILE = "optimizer.pt"
RESUME_ADAPTER = Path("resume_checkpoint") / "adapter_model.safetensors"


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
    warmup_steps: int = Field(default=100, ge=0)
    lr_scheduler: Literal["cosine", "constant", "linear", "cosine_with_restarts", "step"] = "cosine"
    lr_num_cycles: int = Field(default=3, gt=0)
    timestep_weighting: Literal["none", "bell", "half_bell"] = "none"
    noise_offset: float = Field(default=0.0, ge=0.0)
    caption_dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)


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


_VERSIONED_NAME = re.compile(r"^(.+)_v(\d+)$")


@dataclass(frozen=True)
class DatasetVersionConflict:
    """A versioned dataset (`{prefix}_v{N}`) whose captions still say an earlier
    sibling version's name — the usual cause is copying an older version's
    images/captions forward to seed the new one instead of recapturing them.
    """

    current_name: str
    stale_trigger_word: str
    affected_files: int
    suggested_next_version: str


def _caption_files_containing_word(dataset_dir: Path, word: str) -> list[Path]:
    pattern = re.compile(r"\b" + re.escape(word) + r"\b")
    return [
        p for p in sorted(dataset_dir.glob("*.txt"))
        if pattern.search(p.read_text(encoding="utf-8"))
    ]


def _next_available_version(taken_versions: set[int], start_from: int) -> int:
    """First version number >= start_from not already claimed by a sibling dataset.

    Walks forward one at a time rather than returning max(taken) + 1, so a gap
    (v1, v2, v4 on disk — v3 deleted or never used) is offered before jumping
    past it to v5.
    """
    candidate = start_from
    while candidate in taken_versions:
        candidate += 1
    return candidate


def detect_dataset_version_conflict(dataset_name: str) -> DatasetVersionConflict | None:
    """Look for a stale earlier-version name baked into `dataset_name`'s captions.

    Only versioned names (`{prefix}_v{N}`) are considered. Sibling versions are
    found on disk under training_runtime/datasets/ — same prefix, any version
    number — which is what catches a dataset duplicated forward from an older
    export: the folder is correctly named `{prefix}_v2`, but its .txt captions
    still read `{prefix}_v1` because that text was copied along with the images.
    Checks the most recent earlier sibling first, since that's the most likely
    source of a forward-copy.
    """
    match = _VERSIONED_NAME.match(dataset_name)
    if match is None:
        return None
    prefix, current_version_str = match.groups()
    current_version = int(current_version_str)

    dataset_dir = dataset_dir_for(dataset_name)
    if not dataset_dir.is_dir():
        return None

    datasets_root = training_runtime_dir() / "datasets"
    if not datasets_root.is_dir():
        return None
    sibling_pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)$")
    sibling_versions = [
        int(m.group(1))
        for entry in datasets_root.iterdir()
        if entry.is_dir() and (m := sibling_pattern.match(entry.name))
    ]
    if not sibling_versions:
        return None

    earlier_versions = sorted(
        (v for v in sibling_versions if v != current_version), reverse=True
    )
    for version in earlier_versions:
        candidate = f"{prefix}_v{version}"
        affected = _caption_files_containing_word(dataset_dir, candidate)
        if affected:
            next_version = _next_available_version(
                set(sibling_versions), current_version + 1
            )
            suggested_next_version = f"{prefix}_v{next_version}"
            return DatasetVersionConflict(
                current_name=dataset_name,
                stale_trigger_word=candidate,
                affected_files=len(affected),
                suggested_next_version=suggested_next_version,
            )
    return None


def update_captions_in_dataset_dir(dataset_dir: Path, old_word: str, new_word: str) -> int:
    """Replace `old_word` with `new_word` (whole-word match) across every caption
    under dataset_dir. Reuses storage.write_caption_sidecar so each rewrite gets
    the same atomic-write-plus-one-time-backup guarantee as any other caption
    edit — passing the .txt path itself as the "image path" is safe, since
    `.with_suffix(".txt")` on an already-.txt path is a no-op. Returns how many
    files actually changed.
    """
    pattern = re.compile(r"\b" + re.escape(old_word) + r"\b")
    updated = 0
    for txt_path in sorted(dataset_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        new_text = pattern.sub(new_word, text)
        if new_text != text:
            write_caption_sidecar(str(txt_path), new_text)
            updated += 1
    return updated


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
        dataset_name=dataset_name,
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
        dataset_name=dataset_name,
        model_dir=model_dir,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=trigger_word,
        config=config,
    )


@dataclass(frozen=True)
class ResumePoint:
    """A previous train run whose on-disk state can be continued.

    `step` is the last step the trainer committed, so a resumed run picks up at
    `step + 1`. `total_steps` is what that run was originally aiming for — the UI
    needs it because resuming at or past it is a no-op the trainer refuses.
    """

    training_run_id: str
    output_dir: Path
    step: int
    total_steps: int
    status: str
    started_at: datetime
    # Excluded from eq/hash: a frozen dataclass derives __hash__ from its compared
    # fields, and a dict in there would make instances unhashable — which breaks the
    # moment one is handed to a widget as an option. training_run_id identifies it.
    config: dict = field(compare=False)


class ResumeUnavailable(RuntimeError):
    """The requested run has no restorable checkpoint, or nothing left to run."""


def checkpoint_step(output_dir: Path) -> int | None:
    """The step a run's checkpoint sits at, or None if there isn't a complete one.

    Mirrors krea2.state.CheckpointManager.has_checkpoint() deliberately: this is
    the hub's read-only preview of the decision the trainer will make for itself
    once it starts, so the two must agree on what counts as restorable.
    """
    if not (output_dir / OPTIMIZER_STATE_FILE).is_file():
        return None
    if not (output_dir / RESUME_ADAPTER).is_file():
        return None
    try:
        return int((output_dir / STEP_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def find_resume_points(conn: sqlite3.Connection, dataset_run_id: str) -> list[ResumePoint]:
    """Every train run of this dataset that still has a checkpoint on disk, newest first.

    Driven by the `training_runs` rows rather than by scanning the runtime
    directory, so each entry keeps the hyperparameters it was launched with —
    resuming has to reuse them, since a changed rank or alpha makes the saved
    adapter fail to load.
    """
    points: list[ResumePoint] = []
    for run in repo.list_training_runs(conn, dataset_run_id=dataset_run_id):
        if run.kind != "train" or run.status == "running":
            continue
        raw_output_dir = run.config.get("output_dir")
        if not raw_output_dir:
            continue
        output_dir = Path(raw_output_dir)
        step = checkpoint_step(output_dir)
        if step is None:
            continue
        points.append(
            ResumePoint(
                training_run_id=run.training_run_id,
                output_dir=output_dir,
                step=step,
                total_steps=int(run.config.get("total_steps") or 0),
                status=run.status,
                started_at=run.started_at,
                config=run.config,
            )
        )
    return points


def resume_training(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    resume_point: ResumePoint,
    total_steps: int,
) -> str:
    """Continue a previous run from its checkpoint, into a fresh training_runs row.

    The relaunch keeps the original `output_dir`, which is what makes this a
    resume at all: that is where the trainer looks for the checkpoint, appends
    train_log.csv (krea2.metrics opens it in append mode for exactly this case)
    and rotates per-step exports. Only the settings/log pair is new, so the
    original run's log survives instead of being truncated by the relaunch.

    Every hyperparameter is carried over untouched apart from `total_steps` —
    the one thing you must be able to raise, since a checkpoint at the original
    target has nothing left to run.
    """
    step = checkpoint_step(resume_point.output_dir)
    if step is None:
        raise ResumeUnavailable(
            f"No complete checkpoint under {resume_point.output_dir} anymore."
        )
    if total_steps <= step:
        raise ResumeUnavailable(
            f"Checkpoint is at step {step}; total_steps must be greater than that "
            f"to have anything left to train (got {total_steps})."
        )

    config = dict(resume_point.config)
    _run_precache_blocking(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=Path(config["model_id"]),
        dataset_path=Path(config["dataset_path"]),
        cache_dir=Path(config["cache_dir"]),
        trigger_word=config.get("trigger_word", ""),
    )

    settings = TrainSettings(**{**config, "total_steps": total_steps}).model_dump()
    run_dir = training_runtime_dir() / "runs" / f"train-{uuid.uuid4()}"

    pid, log_path = training_runner.launch(
        TRAIN_SCRIPT,
        settings,
        run_dir,
        TRAIN_SETTINGS_ENV,
        extra_env={"FTI_RUN_ID": str(uuid.uuid4())},
    )
    return repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind="train",
        pid=pid,
        log_path=log_path,
        config=settings,
    )


def _launch_train(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    dataset_name: str,
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
        checkpoint_prefix=dataset_name,
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
    """Where train_worker.py writes train_log.csv: inside the run's output_dir.

    Read from the stored settings rather than derived from the log's location,
    because a resumed run keeps the *original* run's output_dir while getting a
    log of its own — the two stop being siblings. Falls back to the old layout
    for rows written before output_dir was recorded.
    """
    output_dir = training_run.config.get("output_dir")
    if output_dir:
        return Path(output_dir) / "train_log.csv"
    return Path(training_run.log_path).parent / "checkpoints" / "train_log.csv"
