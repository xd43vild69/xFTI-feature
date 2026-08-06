"""Orchestrates a training run: pre-cache (blocking, short) then train (detached, long).

Builds the settings dicts the ported workers (workers/precache_worker.py,
workers/train_worker.py) expect and launches them through training_runner. Only
the keys this project actually wants to control are set explicitly — every other
field falls back to the worker's own DEFAULTS, exactly as it does for a bare
train_settings.json in LoRAlab (see `_cfg()` in train_worker.py).
"""

import csv
import re
import shutil
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from feature_pipeline.domain import checkpoint_log, cost, train_log
from feature_pipeline.domain.naming import slugify
from feature_pipeline.domain.worker_contracts import PrecacheSettings, TrainSettings
from feature_pipeline.infrastructure import checkpoint_files
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

    checkpoint_name: str = ""
    total_steps: int = Field(default=3000, gt=0)
    lr: float = Field(default=1.5e-4, gt=0)
    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    batch_size: int = Field(default=1, gt=0)
    grad_accum_steps: int = Field(default=4, gt=0)
    save_every: int = Field(default=300, gt=0)
    seed: int = Field(default=42, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    lr_scheduler: Literal["cosine", "constant", "linear", "cosine_with_restarts", "step"] = "cosine"
    lr_num_cycles: int = Field(default=3, gt=0)
    timestep_weighting: Literal["none", "bell", "half_bell"] = "none"
    noise_offset: float = Field(default=0.0, ge=0.0)
    caption_dropout_rate: float = Field(default=0.05, ge=0.0, le=1.0)


def merge_training_config_overrides(
    current: dict, overrides: dict
) -> tuple[dict, list[str]]:
    """Merge a pasted JSON blueprint onto `current`, validated via TrainingConfig's own bounds.

    Returns (new_config_dict, ignored_extra_keys). Raises pydantic.ValidationError if a
    known field's value has the wrong type or is out of bounds — callers should reject the
    whole apply and surface that message rather than commit a partially-invalid config.
    """
    known = set(TrainingConfig.model_fields)
    extra_keys = sorted(set(overrides) - known)
    merged = {**current, **{k: v for k, v in overrides.items() if k in known}}
    return TrainingConfig(**merged).model_dump(), extra_keys


class PrecacheFailed(RuntimeError):
    """Pre-cache exited non-zero or timed out; the caller should not launch training."""


def finalize_dead_run(conn: sqlite3.Connection, run: repo.TrainingRun, *, fallback_status: str) -> None:
    """Move a 'running' row whose process has exited to its real status, with telemetry.

    Reads the worker_finished/worker_failed line workers/_telemetry.py prints as
    its last line of output (see training_runner.read_lifecycle_event) and uses
    it as the source of truth for status, duration and cost — falling back to
    `fallback_status` with no telemetry if the process died too hard to print
    one (killed, machine restart, or a run launched before this existed).

    The CSV is read first, and unconditionally. A killed run prints no lifecycle
    line but — because krea2.metrics flushes every row as it writes it — still
    leaves a complete train_log.csv, and that is exactly the run whose real step
    count you most want to know. Doing this after the early return below would
    skip it in precisely that case.
    """
    _persist_train_log_metrics(conn, run)

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
    """Send SIGINT to a running job's process and mark its row 'stopped', with telemetry.

    A stopped run leaves 'running' immediately, which means finalize_dead_run —
    the only other place telemetry is written — never revisits it: is_training_active
    and the progress panel both look for status 'running'. So whatever is recorded
    here is all this run will ever have, and a deliberately stopped run is the very
    case where "how far did it actually get" matters most.

    The CSV is read straight after the signal without waiting for the process to
    go: the per-row flush means everything but the single in-flight step is
    already on disk. `duration_seconds` is measured hub-side, from launch to stop
    request, rather than by the worker — a slightly different quantity from a
    completed run's, and a much better one than NULL.
    """
    run = repo.get_training_run(conn, training_run_id)
    if run is None:
        raise ValueError(f"No such training run: {training_run_id}")
    training_runner.stop_process(run.pid)

    _persist_train_log_metrics(conn, run)

    elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
    repo.update_training_run_status(
        conn,
        training_run_id,
        "stopped",
        duration_seconds=elapsed,
        gpu_seconds=elapsed,
        cost_estimate=cost.estimate_cost(elapsed, cost.gpu_hourly_rate()),
    )


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
    # "exact" — the run's own resume_checkpoint/optimizer.pt, restorable in full and the
    # only thing resume_training will ever touch. "warm" — one of the per-step exports,
    # weights only. Defaults to "exact" so every pre-existing construction site (and
    # find_resume_points itself) keeps its meaning unchanged.
    kind: str = "exact"
    # The `{prefix}_step_N.safetensors` a warm point starts from; None when kind is
    # "exact", where the state lives in output_dir itself.
    source_export: Path | None = None


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

    One entry per `output_dir`, not per row. A resume reuses its parent's output_dir by
    design, so N launches of one lineage share a single checkpoint — offering it N times
    would be N identical choices leading to the identical relaunch. Rows arrive
    newest-first, so the one kept is the latest launch, whose config carries whatever
    `total_steps` was last raised to.
    """
    points: list[ResumePoint] = []
    seen: set[Path] = set()
    for run in repo.list_training_runs(conn, dataset_run_id=dataset_run_id):
        if run.kind != "train" or run.status == "running":
            continue
        raw_output_dir = run.config.get("output_dir")
        if not raw_output_dir:
            continue
        output_dir = Path(raw_output_dir)
        if output_dir in seen:
            continue
        seen.add(output_dir)
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


class ForkUnavailable(RuntimeError):
    """A fork was requested that cannot be launched as asked."""


# Hyperparameters pinned to the parent on every fork, and never accepted as an
# override. lora_rank/lora_alpha/model_id are hard constraints — the copied adapter
# would fail to load against different ones (same reason resume_training carries
# them). The rest are the confounder this whole feature exists to avoid: lr,
# lr_scheduler, lr_num_cycles and warmup_steps jointly determine the learning rate at
# every step past the fork, so two sibling branches with different total_steps are
# already on different points of their cosine tail before either of them touches a
# single differing image. batch_size/grad_accum_steps/seed change what a "step" even
# samples. Only total_steps (unavoidable — it's what "further training" means) and
# save_every (only changes save cadence, not learning) are accepted overrides.
_FORK_PINNED_KEYS = (
    "model_id", "lr", "lr_scheduler", "lr_num_cycles", "warmup_steps",
    "lora_rank", "lora_alpha", "seed", "batch_size", "grad_accum_steps",
    "timestep_weighting", "noise_offset", "caption_dropout_rate",
)


@dataclass(frozen=True)
class BranchSpec:
    """What distinguishes one forked branch from its siblings."""

    label: str
    dataset_name: str
    trigger_word: str
    dataset_content_hash: str = ""
    weight_profile: dict = field(default_factory=dict)


def find_fork_points(conn: sqlite3.Connection, *, dataset_run_id: str) -> list[ResumePoint]:
    """Every step of this dataset's past runs that a branch can start from.

    Two kinds, and the difference is not cosmetic. The **exact** points are what
    find_resume_points returns: a run's `resume_checkpoint/` plus `optimizer.pt`, which
    restore the optimizer moments, RNG streams and sampler position along with the
    weights. There is at most one per run, because `krea2.state` rewrites both in place
    on every save — a finished 3000-step run offers step 3000 and nothing else.

    The **warm** points are the per-step `.safetensors` exports, which is why they exist
    here at all: they are the only surviving record of steps 300, 600, 900… and forking
    mid-run is the whole point of the feature. They carry weights alone, so a branch
    started from one has a cold optimizer. That does not invalidate a sibling comparison
    — every branch off the same point is equally cold, so the delta between them is
    still the data intervention — but it does mean such a branch cannot be read against
    the *parent's* own curve past that step, which is why the UI pushes a control branch.

    Ordered newest run first, and within a run by step descending, so the head of the
    list is the most recent state available. The exact point of a run sorts ahead of its
    own warm points at the same step: with both on offer the restorable one always wins.
    """
    points: list[ResumePoint] = []
    for exact in find_resume_points(conn, dataset_run_id):
        points.append(exact)
        points.extend(_warm_fork_points(exact))
    return points


def _warm_fork_points(exact: ResumePoint) -> list[ResumePoint]:
    """The per-step exports under a run's output_dir, newest step first.

    Skips any export at the exact point's own step: they are the same weights, and the
    restorable version is strictly better. Skips the FINAL export for the same reason —
    it duplicates the last per-step file (see `domain.checkpoint_log.reconstruct`, which
    merges the pair for exactly this reason).
    """
    warm: list[ResumePoint] = []
    for info in checkpoint_files.list_checkpoint_files(exact.output_dir):
        if info.is_final or info.step >= exact.step or info.step <= 0:
            continue
        warm.append(
            replace(exact, step=info.step, kind="warm", source_export=info.path)
        )
    return sorted(warm, key=lambda point: point.step, reverse=True)


def fork_training(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    fork_point: ResumePoint,
    branch: BranchSpec,
    total_steps: int,
    save_every: int | None = None,
) -> str:
    """Branch a previous run's checkpoint into an independent training lineage.

    Unlike resume_training, this mints a *fresh* output_dir and copies the parent's
    checkpoint into it rather than reusing the parent's in place. That is the whole
    difference, and it is what makes N sibling branches possible from one checkpoint:
    the parent is left untouched and re-forkable, and each branch gets its own
    train_log.csv/val_log.csv/checkpoint_log.csv instead of appending to a shared one.

    A `fork_point.kind` of "warm" starts from a per-step export instead: same fresh
    output_dir, same pinned hyperparameters, but the branch begins with a cold optimizer
    because that is all those files contain. See find_fork_points for what that costs.

    `branch.dataset_name` must point at a dataset the branch's own export already
    wrote (see export_service.export_branch_dataset) — this function launches
    training against it, it does not create it.
    """
    if fork_point.kind == "warm":
        if fork_point.source_export is None or not fork_point.source_export.is_file():
            raise ForkUnavailable(
                f"The export this branch would start from is gone: "
                f"{fork_point.source_export}."
            )
        step = fork_point.step
    else:
        step = checkpoint_step(fork_point.output_dir) or 0
        if step == 0:
            raise ForkUnavailable(
                f"No complete checkpoint under {fork_point.output_dir} anymore."
            )
    if total_steps <= step:
        raise ForkUnavailable(
            f"Checkpoint is at step {step}; total_steps must be greater than that "
            f"to have anything left to train (got {total_steps})."
        )

    parent_config = dict(fork_point.config)
    parent_dataset_name = Path(str(parent_config.get("dataset_path", ""))).name
    if branch.dataset_name == parent_dataset_name:
        raise ForkUnavailable(
            "A branch cannot reuse its parent's dataset folder — exporting into it "
            "would delete the parent's images out from under it."
        )

    run_dir = training_runtime_dir() / "runs" / f"train-{uuid.uuid4()}"
    output_dir = run_dir / "checkpoints"

    # Re-verify rather than trust a clean return: a half-staged start reads as "nothing
    # to restore" to the trainer, which would then train the branch from step 0 without
    # complaint — this is the entire safety story for the staging below.
    try:
        if fork_point.kind == "warm":
            assert fork_point.source_export is not None  # narrowed above
            staged_step = checkpoint_files.materialize_warm_start(
                step_export=fork_point.source_export,
                adapter_config=(
                    fork_point.output_dir / "resume_checkpoint" / "adapter_config.json"
                ),
                destination_dir=output_dir,
                step=step,
                source_label=str(fork_point.source_export),
            )
            verified = checkpoint_files.warm_start_step(output_dir)
        else:
            staged_step = checkpoint_files.copy_resume_state(
                fork_point.output_dir, output_dir
            )
            verified = checkpoint_step(output_dir)
    except (OSError, ValueError) as exc:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise ForkUnavailable(f"Could not stage the parent checkpoint: {exc}") from exc

    if staged_step != step or verified != step:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise ForkUnavailable("Checkpoint staging did not verify; aborting the fork.")

    dataset_path = dataset_dir_for(branch.dataset_name)
    cache_dir = cache_dir_for(branch.dataset_name)
    parent_prefix = str(parent_config.get("checkpoint_prefix") or dataset_run_id)

    overrides = {key: parent_config[key] for key in _FORK_PINNED_KEYS if key in parent_config}
    settings_dict = {
        **parent_config,
        **overrides,
        "dataset_path": str(dataset_path),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "trigger_word": branch.trigger_word,
        "checkpoint_prefix": f"{parent_prefix}__{slugify(branch.label)}",
        "total_steps": total_steps,
    }
    if save_every is not None:
        settings_dict["save_every"] = save_every
    settings = TrainSettings(**settings_dict).model_dump()

    _run_precache_blocking(
        conn,
        dataset_run_id=dataset_run_id,
        model_dir=Path(settings["model_id"]),
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        trigger_word=branch.trigger_word,
    )

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
        experiment=repo.ExperimentLineage(
            parent_training_run_id=fork_point.training_run_id,
            fork_step=step,
            branch_label=branch.label,
            dataset_content_hash=branch.dataset_content_hash,
            weight_profile=branch.weight_profile,
        ),
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
        checkpoint_prefix=config.checkpoint_name or dataset_name,
        **config.model_dump(exclude={"checkpoint_name"}),
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


def read_train_log_summary(
    training_run: repo.TrainingRun,
) -> train_log.TrainLogSummary | None:
    """Summarise this run's train_log.csv, or None if there is nothing usable there.

    The I/O half of domain.train_log: this finds and streams the file, the domain
    decides what its numbers mean. Streamed rather than read whole because a run at
    grad_accum_steps=1 leaves one row per step, which for a long run is six figures.

    Every failure returns None and none of them raise, because every caller is
    finishing a run and none of them can do anything useful with the exception: the
    file is absent for a run that died before its first optimizer update, or whose
    runtime directory has since been reclaimed, and the last line is routinely half
    written when the process was killed mid-row.
    """
    if training_run.kind != "train":
        # precache_status finalizes pre-cache runs through the same path, and those
        # have no output_dir at all — the fallback would point somewhere arbitrary.
        return None
    try:
        with training_log_csv_path(training_run).open(newline="", encoding="utf-8") as handle:
            records = train_log.parse_rows(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return train_log.summarize(records) if records else None


def checkpoint_log_csv_path(training_run: repo.TrainingRun) -> Path:
    """Where krea2.metrics.CheckpointLog writes checkpoint_log.csv.

    Sibling of train_log.csv inside output_dir, so it is resolved the same way and
    inherits the same reason for it: a resumed run appends to the original run's file
    while writing its own log elsewhere.
    """
    output_dir = training_run.config.get("output_dir")
    if output_dir:
        return Path(output_dir) / "checkpoint_log.csv"
    return Path(training_run.log_path).parent / "checkpoints" / "checkpoint_log.csv"


def validation_log_csv_path(training_run: repo.TrainingRun) -> Path:
    """Where train_worker.py writes val_log.csv — sibling of train_log.csv.

    No summarizer alongside this one: domain/train_log.py states outright that it
    never reads val_log.csv, and the branch-comparison view charts the raw column
    rather than a scalar summary.
    """
    output_dir = training_run.config.get("output_dir")
    if output_dir:
        return Path(output_dir) / "val_log.csv"
    return Path(training_run.log_path).parent / "checkpoints" / "val_log.csv"


def read_checkpoint_log_summary(
    training_run: repo.TrainingRun,
) -> checkpoint_log.CheckpointLogSummary | None:
    """Summarise this run's checkpoint_log.csv, or None if there is nothing usable.

    The I/O half of domain.checkpoint_log, and the same contract as
    read_train_log_summary: streamed, never raises, absent for any run that predates
    checkpoint logging or died before writing its first checkpoint.
    """
    if training_run.kind != "train":
        return None
    try:
        path = checkpoint_log_csv_path(training_run)
        with path.open(newline="", encoding="utf-8") as handle:
            records = checkpoint_log.parse_rows(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return checkpoint_log.summarize(records) if records else None


def reconstruct_checkpoint_log_summary(
    launches: Sequence[repo.TrainingRun],
) -> checkpoint_log.CheckpointLogSummary | None:
    """Infer a training's spans from the checkpoints it left, for runs with no log.

    The last resort behind read_checkpoint_log_summary and the stored snapshot: every
    run launched before the trainer logged its saves has nothing but the
    `.safetensors` files themselves. Their mtimes bracket the work between them, and
    their headers carry the step and the image count.

    The launches are passed whole because their `started_at` is what makes a resumed
    run come out right — anchoring each launch's first span to when that process
    started, instead of charging it the hours the previous one lay dead.
    """
    if not launches:
        return None
    latest = launches[-1]
    if latest.kind != "train":
        return None

    output_dir = latest.config.get("output_dir")
    if not output_dir:
        return None

    files = checkpoint_files.list_checkpoint_files(Path(output_dir))
    if not files:
        return None

    starts = [
        (run.training_run_id, run.started_at.timestamp())
        for run in launches
        if run.started_at is not None
    ]
    save_every = latest.config.get("save_every")
    records = checkpoint_log.reconstruct(
        [
            checkpoint_log.CheckpointFile(
                step=info.step,
                written_at=info.written_at,
                num_images=info.num_images,
                epoch=info.epoch,
                is_final=info.is_final,
            )
            for info in files
        ],
        launches=starts,
        save_every=int(save_every) if save_every else None,
    )
    if not records:
        return None
    return checkpoint_log.summarize(records, reconstructed=True)


def _persist_train_log_metrics(
    conn: sqlite3.Connection, training_run: repo.TrainingRun
) -> None:
    """Store what the run's CSVs say, if either says anything.

    Both snapshots go in one statement, but neither is required: a run killed before
    its first checkpoint has a train_log and no checkpoint_log, and a run launched
    before checkpoint logging existed has the first and never the second.
    """
    summary = read_train_log_summary(training_run)
    checkpoints = read_checkpoint_log_summary(training_run)
    if summary is None and checkpoints is None:
        return
    repo.record_train_log_metrics(
        conn,
        training_run.training_run_id,
        steps_executed=summary.steps_executed if summary is not None else None,
        metrics=summary.model_dump(mode="json") if summary is not None else {},
        checkpoint_metrics=(
            checkpoints.model_dump(mode="json") if checkpoints is not None else {}
        ),
    )
