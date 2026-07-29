"""CRUD for ingestion runs: persist, list, load, and delete curated datasets.

Runs survive across sessions and are identified by `run_id`; they are only
removed when the user explicitly deletes them.
"""

import json
import sqlite3
from datetime import datetime, timezone

from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetSample,
    ImageMetrics,
    IngestionRun,
    IngestionRunSummary,
)


def save_ingestion_run(conn: sqlite3.Connection, run: IngestionRun) -> None:
    """Insert (or replace) a run and all of its samples in a single transaction."""
    now = datetime.now(timezone.utc).isoformat()
    concept = run.concept

    with conn:
        # `concepts` is what dataset_versions' foreign key points at, so a run has
        # to leave a row there before it can ever be versioned. Upserted rather
        # than INSERT OR REPLACE: REPLACE deletes the row first, which a
        # dataset_versions row referencing it would (rightly) refuse.
        conn.execute(
            """
            INSERT INTO concepts (concept_id, concept_name, trigger_word, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(concept_id) DO UPDATE SET
                concept_name = excluded.concept_name,
                trigger_word = excluded.trigger_word
            """,
            (concept.concept_id, concept.concept_name, concept.trigger_word, now),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
                (run_id, concept_id, concept_name, trigger_word,
                 source_path, source_kind, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                concept.concept_id,
                concept.concept_name,
                concept.trigger_word,
                run.source_path,
                run.source_kind,
                run.created_at.isoformat(),
            ),
        )
        conn.execute("DELETE FROM samples WHERE run_id = ?", (run.run_id,))
        conn.executemany(
            """
            INSERT INTO samples
                (sample_id, run_id, file_path, caption, original_caption,
                 width, height, aspect_ratio, image_format, phash, dhash, colorhash,
                 sharpness, is_duplicate, is_excluded, is_flagged, is_valid,
                 validation_errors, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.sample_id,
                    run.run_id,
                    s.image_path,
                    s.caption,
                    s.original_caption,
                    s.metrics.width,
                    s.metrics.height,
                    s.metrics.aspect_ratio,
                    s.metrics.format,
                    s.metrics.phash,
                    s.metrics.dhash,
                    s.metrics.colorhash,
                    s.metrics.sharpness,
                    int(s.is_duplicate),
                    int(s.is_excluded),
                    int(s.is_flagged),
                    int(s.is_valid),
                    json.dumps(s.validation_errors),
                    now,
                )
                for s in concept.samples
            ],
        )


def list_ingestion_runs(conn: sqlite3.Connection) -> list[IngestionRunSummary]:
    """Summaries of every stored run, newest first."""
    rows = conn.execute(
        """
        SELECT r.run_id, r.concept_name, r.trigger_word, r.source_kind, r.created_at,
               COUNT(s.sample_id) AS sample_count
        FROM ingestion_runs r
        LEFT JOIN samples s ON s.run_id = r.run_id
        GROUP BY r.run_id
        ORDER BY r.created_at DESC
        """
    ).fetchall()

    return [
        IngestionRunSummary(
            run_id=row["run_id"],
            concept_name=row["concept_name"],
            trigger_word=row["trigger_word"],
            source_kind=row["source_kind"],
            created_at=datetime.fromisoformat(row["created_at"]),
            sample_count=row["sample_count"],
        )
        for row in rows
    ]


def load_ingestion_run(conn: sqlite3.Connection, run_id: str) -> IngestionRun | None:
    """Rebuild a full run (concept + samples) from storage, or None if it's gone."""
    run_row = conn.execute(
        "SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        return None

    sample_rows = conn.execute(
        "SELECT * FROM samples WHERE run_id = ? ORDER BY file_path", (run_id,)
    ).fetchall()

    samples = [
        DatasetSample(
            sample_id=row["sample_id"],
            image_path=row["file_path"],
            caption=row["caption"],
            original_caption=row["original_caption"],
            metrics=ImageMetrics(
                width=row["width"],
                height=row["height"],
                aspect_ratio=row["aspect_ratio"],
                format=row["image_format"],
                phash=row["phash"],
                dhash=row["dhash"],
                colorhash=row["colorhash"],
                sharpness=row["sharpness"],
            ),
            is_duplicate=bool(row["is_duplicate"]),
            is_excluded=bool(row["is_excluded"]),
            is_flagged=bool(row["is_flagged"]),
            is_valid=bool(row["is_valid"]),
            validation_errors=json.loads(row["validation_errors"]),
        )
        for row in sample_rows
    ]

    return IngestionRun(
        run_id=run_row["run_id"],
        source_path=run_row["source_path"],
        source_kind=run_row["source_kind"],
        created_at=datetime.fromisoformat(run_row["created_at"]),
        concept=ConceptGroup(
            concept_id=run_row["concept_id"],
            concept_name=run_row["concept_name"],
            trigger_word=run_row["trigger_word"],
            samples=samples,
        ),
    )


def update_sample_caption(conn: sqlite3.Connection, sample_id: str, caption: str) -> None:
    """Persist a caption edited from the gallery."""
    with conn:
        conn.execute(
            "UPDATE samples SET caption = ?, updated_at = ? WHERE sample_id = ?",
            (caption, datetime.now(timezone.utc).isoformat(), sample_id),
        )


def set_samples_excluded(
    conn: sqlite3.Connection, sample_ids: list[str], excluded: bool
) -> None:
    """Exclude (or restore) samples. The image files are left on disk."""
    if not sample_ids:
        return

    placeholders = ",".join("?" for _ in sample_ids)
    with conn:
        conn.execute(
            f"UPDATE samples SET is_excluded = ?, updated_at = ? "
            f"WHERE sample_id IN ({placeholders})",
            [int(excluded), datetime.now(timezone.utc).isoformat(), *sample_ids],
        )


def mark_duplicates(conn: sqlite3.Connection, run_id: str, sample_ids: list[str]) -> None:
    """Record the outcome of a duplicate scan for a run: only `sample_ids` are duplicates."""
    with conn:
        conn.execute("UPDATE samples SET is_duplicate = 0 WHERE run_id = ?", (run_id,))
        if sample_ids:
            placeholders = ",".join("?" for _ in sample_ids)
            conn.execute(
                f"UPDATE samples SET is_duplicate = 1 WHERE sample_id IN ({placeholders})",
                sample_ids,
            )


def delete_ingestion_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete a run and its samples. Files on disk are handled by the caller."""
    with conn:
        conn.execute("DELETE FROM samples WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM ingestion_runs WHERE run_id = ?", (run_id,))
