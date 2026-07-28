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
    conn: sqlite3.Connection, training_run_id: str, status: str
) -> None:
    finished_at = (
        datetime.now(timezone.utc).isoformat() if status != "running" else None
    )
    with conn:
        conn.execute(
            "UPDATE training_runs SET status = ?, finished_at = ? WHERE training_run_id = ?",
            (status, finished_at, training_run_id),
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
    )
