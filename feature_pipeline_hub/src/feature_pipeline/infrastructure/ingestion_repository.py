"""CRUD for ingestion runs: persist, list, load, and delete curated datasets.

Runs survive across sessions and are identified by `run_id`; they are only
removed when the user explicitly deletes them.
"""

import json
import sqlite3
from dataclasses import dataclass
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
                 validation_errors, updated_at,
                 source_file_path, rotation_degrees, derived_max_side)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    s.source_image_path,
                    s.rotation_degrees,
                    s.derived_max_side,
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
            source_image_path=row["source_file_path"],
            rotation_degrees=row["rotation_degrees"],
            derived_max_side=row["derived_max_side"],
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


def update_sample_image(conn: sqlite3.Connection, sample: DatasetSample) -> None:
    """Persist a sample whose file was rewritten: path, provenance, metrics, verdict.

    Everything the pixels decide, and nothing else — the caption, the curation flags
    and the run membership are left as they are, so normalizing an image mid-curation
    does not undo the curation.
    """
    with conn:
        conn.execute(
            """
            UPDATE samples SET
                file_path = ?, width = ?, height = ?, aspect_ratio = ?, image_format = ?,
                phash = ?, dhash = ?, colorhash = ?, sharpness = ?,
                is_valid = ?, validation_errors = ?, updated_at = ?,
                source_file_path = ?, rotation_degrees = ?, derived_max_side = ?
            WHERE sample_id = ?
            """,
            (
                sample.image_path,
                sample.metrics.width,
                sample.metrics.height,
                sample.metrics.aspect_ratio,
                sample.metrics.format,
                sample.metrics.phash,
                sample.metrics.dhash,
                sample.metrics.colorhash,
                sample.metrics.sharpness,
                int(sample.is_valid),
                json.dumps(sample.validation_errors),
                datetime.now(timezone.utc).isoformat(),
                sample.source_image_path,
                sample.rotation_degrees,
                sample.derived_max_side,
                sample.sample_id,
            ),
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


def inventory_fingerprint(conn: sqlite3.Connection) -> tuple:
    """Cheap value that changes whenever the dataset inventory would look different.

    Aggregates only, so it costs a scan and materialises no rows — the point is to
    key a cache whose miss path loads every sample of every run.

    `SUM(is_duplicate)` is in here deliberately: `mark_duplicates` rewrites that
    column without touching `updated_at`, so a timestamp-only fingerprint would
    miss every duplicate scan. `MAX` over ISO-8601 UTC strings orders correctly as
    text. The run count catches deleting a run that had no samples to begin with.
    """
    counts = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(is_duplicate), 0), COALESCE(SUM(is_excluded), 0),
               COALESCE(MAX(updated_at), '')
        FROM samples
        """
    ).fetchone()
    run_count = conn.execute("SELECT COUNT(*) FROM ingestion_runs").fetchone()[0]

    return (*counts, run_count)


def delete_ingestion_run(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete a run and its samples. Files on disk are handled by the caller."""
    with conn:
        conn.execute("DELETE FROM samples WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM ingestion_runs WHERE run_id = ?", (run_id,))


# --- Pipeline telemetry (Fase 2: observability panel) ---


def update_step_telemetry(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    step: str,  # "import" | "recaption" | "quality" | "export"
    duration_seconds: float,
    error_count: int = 0,
) -> None:
    """Record the duration and error count for one step of the curation pipeline.

    Called from the UI panels (import_panel.py, recaption_panel.py, etc.) right
    after a step completes, to populate the observability dashboard.
    """
    duration_col = f"{step}_duration_seconds"
    error_col = f"{step}_error_count"

    with conn:
        conn.execute(
            f"UPDATE ingestion_runs SET {duration_col} = ?, {error_col} = ? WHERE run_id = ?",
            (duration_seconds, error_count, run_id),
        )


@dataclass(frozen=True)
class StepTelemetry:
    """What `update_step_telemetry` recorded for one run, read back for the dashboard."""

    run_id: str
    import_duration_seconds: float | None
    import_error_count: int
    recaption_duration_seconds: float | None
    recaption_error_count: int
    quality_duration_seconds: float | None
    export_duration_seconds: float | None
    export_error_count: int
    cost_estimate: float | None


def get_step_telemetry(conn: sqlite3.Connection, run_id: str) -> StepTelemetry | None:
    """Per-step durations and error counts for one run, or None if the run is gone."""
    row = conn.execute(
        """
        SELECT run_id, import_duration_seconds, import_error_count,
               recaption_duration_seconds, recaption_error_count,
               quality_duration_seconds, export_duration_seconds, export_error_count,
               cost_estimate
        FROM ingestion_runs WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    return StepTelemetry(
        run_id=row["run_id"],
        import_duration_seconds=row["import_duration_seconds"],
        import_error_count=row["import_error_count"] or 0,
        recaption_duration_seconds=row["recaption_duration_seconds"],
        recaption_error_count=row["recaption_error_count"] or 0,
        quality_duration_seconds=row["quality_duration_seconds"],
        export_duration_seconds=row["export_duration_seconds"],
        export_error_count=row["export_error_count"] or 0,
        cost_estimate=row["cost_estimate"],
    )


def update_run_cost_estimate(conn: sqlite3.Connection, run_id: str, cost: float | None) -> None:
    """Update the aggregated cost estimate for a run (training cost summed with import/etc)."""
    with conn:
        conn.execute("UPDATE ingestion_runs SET cost_estimate = ? WHERE run_id = ?", (cost, run_id))


def update_run_trigger_word(conn: sqlite3.Connection, run_id: str, trigger_word: str) -> None:
    """Update trigger_word across ingestion_runs and concepts for the given run."""
    trigger_word = trigger_word.strip()
    with conn:
        conn.execute("UPDATE ingestion_runs SET trigger_word = ? WHERE run_id = ?", (trigger_word, run_id))
        conn.execute(
            """
            UPDATE concepts SET trigger_word = ?
            WHERE concept_id = (SELECT concept_id FROM ingestion_runs WHERE run_id = ?)
            """,
            (trigger_word, run_id),
        )


def find_trigger_word_collision(
    conn: sqlite3.Connection, current_run_id: str, trigger_word: str
) -> str | None:
    """Check if trigger_word is already used by another run or concept in SQLite.
    Returns the colliding concept_name if found, else None.
    """
    trigger_word = trigger_word.strip().lower()
    if not trigger_word:
        return None

    row = conn.execute(
        """
        SELECT concept_name FROM ingestion_runs
        WHERE LOWER(trigger_word) = ? AND run_id != ?
        LIMIT 1
        """,
        (trigger_word, current_run_id),
    ).fetchone()
    if row:
        return str(row["concept_name"])

    row = conn.execute(
        """
        SELECT c.concept_name FROM concepts c
        WHERE LOWER(c.trigger_word) = ?
          AND c.concept_id != (
              SELECT COALESCE(concept_id, '') FROM ingestion_runs WHERE run_id = ?
          )
        LIMIT 1
        """,
        (trigger_word, current_run_id),
    ).fetchone()
    if row:
        return str(row["concept_name"])

    return None


def rename_concept_and_run(
    conn: sqlite3.Connection, run_id: str, new_concept_name: str
) -> tuple[str, str]:
    """Update concept_name across ingestion_runs and concepts for the given run.
    Returns (old_concept_name, new_concept_name).
    """
    new_concept_name = new_concept_name.strip()
    with conn:
        row = conn.execute(
            "SELECT concept_name FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        old_name = str(row["concept_name"]) if row else ""
        conn.execute(
            "UPDATE ingestion_runs SET concept_name = ? WHERE run_id = ?",
            (new_concept_name, run_id),
        )
        conn.execute(
            """
            UPDATE concepts SET concept_name = ?
            WHERE concept_id = (SELECT concept_id FROM ingestion_runs WHERE run_id = ?)
            """,
            (new_concept_name, run_id),
        )
    return old_name, new_concept_name
