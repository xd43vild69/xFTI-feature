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
}


def create_tables(conn: sqlite3.Connection) -> None:
    """Create the curation tables if they don't already exist, then migrate columns."""
    conn.executescript(SCHEMA)
    _migrate_sample_columns(conn)
    conn.commit()


def _migrate_sample_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}

    for column, definition in SAMPLE_COLUMN_MIGRATIONS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE samples ADD COLUMN {column} {definition}")
