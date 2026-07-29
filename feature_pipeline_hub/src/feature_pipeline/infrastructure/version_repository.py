"""CRUD for dataset versions: one row per export, holding its manifest snapshot.

A version is what makes dataset quality comparable over time — without a stored
manifest the metrics are recomputed and thrown away on every rerun, and "is this
export better than the last one?" has no answer. Rows are never updated: a new
export of the same concept adds a version.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from feature_pipeline.domain.models import ConceptGroup, DatasetManifest


@dataclass(frozen=True)
class DatasetVersion:
    """A stored export: its manifest plus where the files landed."""

    version_id: str
    concept_id: str
    version_tag: str
    manifest: DatasetManifest
    exported_path: str
    created_at: datetime


def create_dataset_version(
    conn: sqlite3.Connection,
    *,
    concept: ConceptGroup,
    version_tag: str,
    manifest: DatasetManifest,
    exported_path: str,
) -> str:
    """Record an export, returning the new version_id."""
    version_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with conn:
        # dataset_versions.concept_id is a real foreign key, and runs saved before
        # `concepts` started being written have no row there. Backfilling it here
        # keeps versioning available for those older runs instead of failing on a
        # constraint the user can do nothing about.
        conn.execute(
            """
            INSERT OR IGNORE INTO concepts (concept_id, concept_name, trigger_word, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (concept.concept_id, concept.concept_name, concept.trigger_word, now),
        )
        conn.execute(
            """
            INSERT INTO dataset_versions
                (version_id, concept_id, version_tag, manifest_json, exported_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                concept.concept_id,
                version_tag,
                manifest.model_dump_json(),
                exported_path,
                now,
            ),
        )
    return version_id


def list_versions_for_concept(
    conn: sqlite3.Connection, concept_id: str
) -> list[DatasetVersion]:
    """Every stored version of a concept, newest first."""
    rows = conn.execute(
        "SELECT * FROM dataset_versions WHERE concept_id = ? ORDER BY created_at DESC",
        (concept_id,),
    ).fetchall()
    return [_row_to_version(row) for row in rows]


def latest_version_for_concept(
    conn: sqlite3.Connection, concept_id: str
) -> DatasetVersion | None:
    """The most recent version of a concept, or None if it was never exported."""
    row = conn.execute(
        "SELECT * FROM dataset_versions WHERE concept_id = ? ORDER BY created_at DESC LIMIT 1",
        (concept_id,),
    ).fetchone()
    return _row_to_version(row) if row else None


def _row_to_version(row: sqlite3.Row) -> DatasetVersion:
    return DatasetVersion(
        version_id=row["version_id"],
        concept_id=row["concept_id"],
        version_tag=row["version_tag"],
        manifest=DatasetManifest.model_validate(json.loads(row["manifest_json"])),
        exported_path=row["exported_path"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
