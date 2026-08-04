"""CRUD for training runs: the record of every pre-cache/train/curation subprocess launched.

Rows persist across sessions so the GPU-lock check and the live progress panel
can find a job's PID and log file after the browser closes and reopens.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TrainingRun:
    training_run_id: str
    dataset_run_id: str
    kind: str  # "precache" | "train" | "progressive" | "curate"
    status: str  # "running" | "completed" | "failed" | "stopped"
    pid: int
    log_path: str
    config: dict
    started_at: datetime
    finished_at: datetime | None
    duration_seconds: float | None = None
    gpu_seconds: float | None = None
    cost_estimate: float | None = None
    error_message: str = ""
    # What the run's train_log.csv said when it was finalized. `metrics` is a plain
    # dict, not domain.train_log.TrainLogSummary, for three reasons: it keeps this
    # layer from importing the domain's shape, a blob written by an older version
    # can never fail validation and break the page it appears on, and
    # mcp_server._training_run_dict runs dataclasses.asdict over these rows —
    # which leaves a model untouched and then fails to serialize it.
    steps_executed: int | None = None
    metrics: dict = field(default_factory=dict)
    # And what its checkpoint_log.csv said — the per-checkpoint spans. Same plain-dict
    # treatment, for the same three reasons.
    checkpoint_metrics: dict = field(default_factory=dict)


def create_training_run(
    conn: sqlite3.Connection,
    *,
    dataset_run_id: str,
    kind: str,
    pid: int,
    log_path: str,
    config: dict,
) -> str:
    training_run_id = str(uuid.uuid4())
    with conn:
        conn.execute(
            """
            INSERT INTO training_runs
                (training_run_id, dataset_run_id, kind, status, pid, log_path,
                 config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            (
                training_run_id,
                dataset_run_id,
                kind,
                pid,
                log_path,
                json.dumps(config),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return training_run_id


def update_training_run_status(
    conn: sqlite3.Connection,
    training_run_id: str,
    status: str,
    *,
    duration_seconds: float | None = None,
    gpu_seconds: float | None = None,
    cost_estimate: float | None = None,
    error_message: str = "",
) -> None:
    """Move a run to `status`, optionally attaching the telemetry it finished with.

    The telemetry fields come from training_runner.read_lifecycle_event() (the
    worker_finished/worker_failed line workers/_telemetry.py prints) — left at
    their defaults when no such event was found (a run that predates this
    telemetry, or one that crashed too hard to print anything).
    """
    finished_at = (
        datetime.now(timezone.utc).isoformat() if status != "running" else None
    )
    with conn:
        conn.execute(
            """
            UPDATE training_runs
            SET status = ?, finished_at = ?, duration_seconds = ?, gpu_seconds = ?,
                cost_estimate = ?, error_message = ?
            WHERE training_run_id = ?
            """,
            (
                status,
                finished_at,
                duration_seconds,
                gpu_seconds,
                cost_estimate,
                error_message,
                training_run_id,
            ),
        )


def record_train_log_metrics(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    steps_executed: int | None,
    metrics: dict,
    checkpoint_metrics: dict | None = None,
) -> None:
    """Attach the CSV snapshots to a run, touching nothing else.

    Deliberately not folded into update_training_run_status as more keyword
    arguments: that one writes every telemetry column from its defaults on every
    call, so any later bare status change would erase whatever was stored here.
    A separate statement makes these columns immune to that by construction.
    """
    with conn:
        conn.execute(
            """
            UPDATE training_runs
            SET steps_executed = ?, metrics_json = ?, checkpoint_metrics_json = ?
            WHERE training_run_id = ?
            """,
            (
                steps_executed,
                json.dumps(metrics),
                json.dumps(checkpoint_metrics or {}),
                training_run_id,
            ),
        )


def get_training_run(conn: sqlite3.Connection, training_run_id: str) -> TrainingRun | None:
    row = conn.execute(
        "SELECT * FROM training_runs WHERE training_run_id = ?", (training_run_id,)
    ).fetchone()
    return _row_to_training_run(row) if row is not None else None


def list_training_runs(
    conn: sqlite3.Connection, dataset_run_id: str | None = None
) -> list[TrainingRun]:
    """All training runs, newest first — optionally scoped to one dataset run."""
    if dataset_run_id is None:
        rows = conn.execute(
            "SELECT * FROM training_runs ORDER BY started_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM training_runs WHERE dataset_run_id = ? ORDER BY started_at DESC",
            (dataset_run_id,),
        ).fetchall()
    return [_row_to_training_run(row) for row in rows]


def find_running_training_run(conn: sqlite3.Connection) -> TrainingRun | None:
    """The single row (if any) still marked 'running' — the GPU-lock check."""
    row = conn.execute(
        "SELECT * FROM training_runs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return _row_to_training_run(row) if row is not None else None


def _row_to_training_run(row: sqlite3.Row) -> TrainingRun:
    return TrainingRun(
        training_run_id=row["training_run_id"],
        dataset_run_id=row["dataset_run_id"],
        kind=row["kind"],
        status=row["status"],
        pid=row["pid"],
        log_path=row["log_path"],
        config=json.loads(row["config_json"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(
            datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
        ),
        duration_seconds=row["duration_seconds"],
        gpu_seconds=row["gpu_seconds"],
        cost_estimate=row["cost_estimate"],
        error_message=row["error_message"],
        steps_executed=row["steps_executed"],
        metrics=_load_metrics(row["metrics_json"]),
        checkpoint_metrics=_load_metrics(row["checkpoint_metrics_json"]),
    )


def _load_metrics(raw: str | None) -> dict:
    """The stored metrics blob, or an empty one if it cannot be read.

    Degrading to "no metrics" rather than raising keeps a single corrupt row from
    breaking list_training_runs for every run on the page.
    """
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
