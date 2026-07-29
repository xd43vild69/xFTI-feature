"""Versioning is what makes two exports comparable, so the cases that matter are
the ones where a version cannot be recorded at all."""

import sqlite3

import pytest

from feature_pipeline.domain.models import ConceptGroup, DatasetManifest
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.version_repository import (
    create_dataset_version,
    latest_version_for_concept,
    list_versions_for_concept,
)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _concept(concept_id: str = "c1") -> ConceptGroup:
    return ConceptGroup(
        concept_id=concept_id, concept_name="cyberpunk", trigger_word="sks_style"
    )


def _manifest(version: str = "v1", total: int = 10, sharpness: float = 120.0) -> DatasetManifest:
    return DatasetManifest(
        dataset_name="cyberpunk",
        version=version,
        concept_name="cyberpunk",
        trigger_word="sks_style",
        total_samples=total,
        duplicate_count=1,
        aspect_ratio_distribution={"1:1": total},
        orientation_distribution={"square": total},
        median_sharpness=sharpness,
        caption_word_stats={"mean": 12.0},
        content_hash="abc123",
    )


def test_a_version_round_trips_with_its_manifest(conn):
    create_dataset_version(
        conn,
        concept=_concept(),
        version_tag="v1",
        manifest=_manifest(),
        exported_path="/runtime/datasets/cyberpunk",
    )

    stored = latest_version_for_concept(conn, "c1")

    assert stored is not None
    assert stored.version_tag == "v1"
    assert stored.exported_path == "/runtime/datasets/cyberpunk"
    assert stored.manifest.total_samples == 10
    assert stored.manifest.median_sharpness == 120.0
    assert stored.manifest.orientation_distribution == {"square": 10}


def test_a_version_can_be_recorded_for_a_run_saved_before_concepts_were_written(conn):
    """dataset_versions has a real FK to `concepts`, which older runs never populated.

    Without the defensive backfill this raises IntegrityError, and the user has no
    way to fix it from the UI.
    """
    assert conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0] == 0

    create_dataset_version(
        conn,
        concept=_concept("legacy"),
        version_tag="v1",
        manifest=_manifest(),
        exported_path="/runtime/datasets/legacy",
    )

    assert latest_version_for_concept(conn, "legacy") is not None


def test_versions_are_listed_newest_first(conn):
    for tag in ("v1", "v2", "v3"):
        create_dataset_version(
            conn,
            concept=_concept(),
            version_tag=tag,
            manifest=_manifest(tag),
            exported_path=f"/runtime/datasets/{tag}",
        )

    assert [v.version_tag for v in list_versions_for_concept(conn, "c1")][0] == "v3"
    assert len(list_versions_for_concept(conn, "c1")) == 3


def test_versions_of_other_concepts_are_not_returned(conn):
    create_dataset_version(
        conn,
        concept=_concept("c1"),
        version_tag="v1",
        manifest=_manifest(),
        exported_path="/a",
    )
    create_dataset_version(
        conn,
        concept=_concept("c2"),
        version_tag="v1",
        manifest=_manifest(),
        exported_path="/b",
    )

    assert len(list_versions_for_concept(conn, "c1")) == 1
    assert latest_version_for_concept(conn, "c3") is None


def test_re_saving_a_concept_does_not_break_its_existing_versions(conn):
    """A rename must not delete the concept row a stored version points at."""
    from feature_pipeline.infrastructure.ingestion_repository import save_ingestion_run
    from feature_pipeline.domain.models import IngestionRun

    concept = _concept()
    save_ingestion_run(
        conn, IngestionRun(run_id="r1", source_path="/raw", source_kind="folder", concept=concept)
    )
    create_dataset_version(
        conn, concept=concept, version_tag="v1", manifest=_manifest(), exported_path="/a"
    )

    renamed = ConceptGroup(
        concept_id="c1", concept_name="cyberpunk", trigger_word="new_trigger"
    )
    save_ingestion_run(
        conn, IngestionRun(run_id="r1", source_path="/raw", source_kind="folder", concept=renamed)
    )

    assert latest_version_for_concept(conn, "c1") is not None
    assert conn.execute(
        "SELECT trigger_word FROM concepts WHERE concept_id = 'c1'"
    ).fetchone()[0] == "new_trigger"


def test_foreign_keys_are_actually_enforced(conn):
    """Guards the assumption the backfill exists for: the FK is not decorative."""
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO dataset_versions
                (version_id, concept_id, version_tag, manifest_json, exported_path, created_at)
            VALUES ('v', 'ghost', 'v1', '{}', '/a', '2026-01-01T00:00:00+00:00')
            """
        )
