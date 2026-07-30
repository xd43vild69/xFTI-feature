"""SQLite schema and connection handling for local metadata tracking during curation.

CRUD over these tables lives in `ingestion_repository`.
"""

import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    concept_name TEXT NOT NULL,
    trigger_word TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL,
    concept_name TEXT NOT NULL,
    trigger_word TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES ingestion_runs(run_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    caption TEXT NOT NULL,
    original_caption TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    aspect_ratio REAL NOT NULL,
    image_format TEXT NOT NULL DEFAULT '',
    phash TEXT NOT NULL,
    dhash TEXT NOT NULL DEFAULT '',
    colorhash TEXT NOT NULL DEFAULT '',
    sharpness REAL NOT NULL DEFAULT 0,
    is_duplicate BOOLEAN NOT NULL DEFAULT 0,
    is_excluded BOOLEAN NOT NULL DEFAULT 0,
    is_valid BOOLEAN NOT NULL DEFAULT 1,
    validation_errors TEXT NOT NULL DEFAULT '[]',
    is_flagged BOOLEAN NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_samples_run_id ON samples(run_id);

CREATE TABLE IF NOT EXISTS dataset_versions (
    version_id TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
    version_tag TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    exported_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- One row per launched training_runner subprocess: pre-cache, train, progressive
-- phase, or curation scoring. `pid` + `status` are how the GPU lock (only one
-- heavy job at a time) and the live log/loss-chart panel find their process
-- across reruns and browser reloads, without keeping anything in memory.
CREATE TABLE IF NOT EXISTS training_runs (
    training_run_id TEXT PRIMARY KEY,
    dataset_run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    pid INTEGER NOT NULL,
    log_path TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_seconds REAL,
    gpu_seconds REAL,
    cost_estimate REAL,
    error_message TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
"""


def default_db_path() -> str:
    """Path of the local curation database, overridable via FTI_DB_PATH."""
    override = os.environ.get("FTI_DB_PATH")
    if override:
        return override

    db_path = Path(__file__).resolve().parents[3] / "data" / "feature_pipeline.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled and the schema in place."""
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    create_tables(conn)
    return conn


# Columns added after the first databases were created. CREATE TABLE IF NOT EXISTS
# leaves existing tables untouched, so they are backfilled here instead.
SAMPLE_COLUMN_MIGRATIONS = {
    "dhash": "TEXT NOT NULL DEFAULT ''",
    "colorhash": "TEXT NOT NULL DEFAULT ''",
    "is_excluded": "BOOLEAN NOT NULL DEFAULT 0",
    "is_flagged": "BOOLEAN NOT NULL DEFAULT 0",
    "sharpness": "REAL NOT NULL DEFAULT 0",
}

# Telemetry columns added once workers/_telemetry.py started emitting
# worker_finished/worker_failed events (see workers/_telemetry.py and
# training_runner.read_lifecycle_event).
TRAINING_RUN_COLUMN_MIGRATIONS = {
    "duration_seconds": "REAL",
    "gpu_seconds": "REAL",
    "cost_estimate": "REAL",
    "error_message": "TEXT NOT NULL DEFAULT ''",
}


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the curation tables if they don't already exist, then migrate columns."""
    conn.executescript(SCHEMA)
    _migrate_columns(conn, "samples", SAMPLE_COLUMN_MIGRATIONS)
    _migrate_columns(conn, "training_runs", TRAINING_RUN_COLUMN_MIGRATIONS)
    conn.commit()


def _migrate_columns(conn: sqlite3.Connection, table: str, migrations: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    for column, definition in migrations.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
