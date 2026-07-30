"""CRUD for training runs: the record of every pre-cache/train/curation subprocess launched.

Rows persist across sessions so the GPU-lock check and the live progress panel
can find a job's PID and log file after the browser closes and reopens.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
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
    )
