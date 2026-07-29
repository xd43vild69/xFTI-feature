import sqlite3
from datetime import datetime, timezone

import pytest

from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetSample,
    ImageMetrics,
    IngestionRun,
)
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.ingestion_repository import (
    delete_ingestion_run,
    list_ingestion_runs,
    load_ingestion_run,
    mark_duplicates,
    save_ingestion_run,
    set_samples_excluded,
    update_sample_caption,
)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _make_run(run_id: str, concept_name: str = "cyberpunk", n_samples: int = 1) -> IngestionRun:
    samples = [
        DatasetSample(
            sample_id=f"{run_id}-s{i}",
            image_path=f"/data/raw/{run_id}/img{i}.png",
            caption=f"sks_style, image {i}",
            original_caption=f"image {i}",
            metrics=ImageMetrics(
                width=512, height=768, aspect_ratio=512 / 768, format="PNG", phash="abcd1234"
            ),
            is_valid=False,
            validation_errors=["Caption is empty"],
        )
        for i in range(n_samples)
    ]
    return IngestionRun(
        run_id=run_id,
        source_path=f"/data/raw/{run_id}",
        source_kind="upload",
        created_at=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        concept=ConceptGroup(
            concept_id=f"c-{run_id}",
            concept_name=concept_name,
            trigger_word="sks_style",
            samples=samples,
        ),
    )


def test_save_and_load_round_trips_the_run(conn):
    original = _make_run("r1", n_samples=2)

    save_ingestion_run(conn, original)
    loaded = load_ingestion_run(conn, "r1")

    assert loaded is not None
    assert loaded.run_id == original.run_id
    assert loaded.source_kind == "upload"
    assert loaded.created_at == original.created_at
    assert loaded.concept.concept_name == "cyberpunk"
    assert loaded.concept.trigger_word == "sks_style"
    assert len(loaded.concept.samples) == 2

    sample = loaded.concept.samples[0]
    assert sample.caption == original.concept.samples[0].caption
    assert sample.original_caption == original.concept.samples[0].original_caption
    assert sample.metrics.width == 512
    assert sample.metrics.format == "PNG"
    assert sample.is_valid is False
    assert sample.validation_errors == ["Caption is empty"]


def test_runs_with_the_same_concept_coexist(conn):
    save_ingestion_run(conn, _make_run("r1", concept_name="cat"))
    save_ingestion_run(conn, _make_run("r2", concept_name="cat"))

    summaries = list_ingestion_runs(conn)

    assert {s.run_id for s in summaries} == {"r1", "r2"}


def test_list_reports_sample_counts_newest_first(conn):
    older = _make_run("r1", n_samples=1)
    newer = _make_run("r2", n_samples=3)
    newer.created_at = datetime(2026, 7, 28, 18, 0, tzinfo=timezone.utc)

    save_ingestion_run(conn, older)
    save_ingestion_run(conn, newer)

    summaries = list_ingestion_runs(conn)

    assert [s.run_id for s in summaries] == ["r2", "r1"]
    assert [s.sample_count for s in summaries] == [3, 1]


def test_saving_the_same_run_twice_replaces_its_samples(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=3))
    save_ingestion_run(conn, _make_run("r1", n_samples=1))

    loaded = load_ingestion_run(conn, "r1")

    assert len(loaded.concept.samples) == 1


def test_update_sample_caption_persists(conn):
    save_ingestion_run(conn, _make_run("r1"))

    update_sample_caption(conn, "r1-s0", "sks_style, an edited caption")

    loaded = load_ingestion_run(conn, "r1")
    assert loaded.concept.samples[0].caption == "sks_style, an edited caption"


def test_delete_removes_the_run_and_its_samples(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=2))
    save_ingestion_run(conn, _make_run("r2", n_samples=2))

    delete_ingestion_run(conn, "r1")

    assert load_ingestion_run(conn, "r1") is None
    assert [s.run_id for s in list_ingestion_runs(conn)] == ["r2"]
    remaining = conn.execute("SELECT COUNT(*) FROM samples WHERE run_id = 'r1'").fetchone()
    assert remaining[0] == 0


def test_load_missing_run_returns_none(conn):
    assert load_ingestion_run(conn, "nope") is None


def test_dhash_and_exclusion_survive_a_round_trip(conn):
    run = _make_run("r1")
    run.concept.samples[0].metrics.dhash = "ffeeddccbbaa9988"
    run.concept.samples[0].is_excluded = True

    save_ingestion_run(conn, run)
    loaded = load_ingestion_run(conn, "r1")

    assert loaded.concept.samples[0].metrics.dhash == "ffeeddccbbaa9988"
    assert loaded.concept.samples[0].is_excluded is True


def test_set_samples_excluded_toggles_in_both_directions(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=3))

    set_samples_excluded(conn, ["r1-s0", "r1-s1"], True)
    loaded = load_ingestion_run(conn, "r1")
    assert [s.is_excluded for s in loaded.concept.samples] == [True, True, False]

    set_samples_excluded(conn, ["r1-s0"], False)
    loaded = load_ingestion_run(conn, "r1")
    assert [s.is_excluded for s in loaded.concept.samples] == [False, True, False]


def test_set_samples_excluded_with_no_ids_is_a_no_op(conn):
    save_ingestion_run(conn, _make_run("r1"))

    set_samples_excluded(conn, [], True)

    assert load_ingestion_run(conn, "r1").concept.samples[0].is_excluded is False


def test_mark_duplicates_replaces_the_previous_scan_result(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=3))

    mark_duplicates(conn, "r1", ["r1-s0", "r1-s1"])
    assert [s.is_duplicate for s in load_ingestion_run(conn, "r1").concept.samples] == [
        True,
        True,
        False,
    ]

    mark_duplicates(conn, "r1", ["r1-s2"])
    assert [s.is_duplicate for s in load_ingestion_run(conn, "r1").concept.samples] == [
        False,
        False,
        True,
    ]


def test_mark_duplicates_leaves_other_runs_alone(conn):
    save_ingestion_run(conn, _make_run("r1"))
    save_ingestion_run(conn, _make_run("r2"))
    mark_duplicates(conn, "r1", ["r1-s0"])
    mark_duplicates(conn, "r2", [])

    assert load_ingestion_run(conn, "r1").concept.samples[0].is_duplicate is True


def test_a_database_created_before_dhash_and_exclusion_is_migrated(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE ingestion_runs (
            run_id TEXT PRIMARY KEY, concept_id TEXT NOT NULL, concept_name TEXT NOT NULL,
            trigger_word TEXT NOT NULL, source_path TEXT NOT NULL,
            source_kind TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE samples (
            sample_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, file_path TEXT NOT NULL,
            caption TEXT NOT NULL, original_caption TEXT NOT NULL DEFAULT '',
            width INTEGER NOT NULL, height INTEGER NOT NULL, aspect_ratio REAL NOT NULL,
            image_format TEXT NOT NULL DEFAULT '', phash TEXT NOT NULL,
            is_duplicate BOOLEAN NOT NULL DEFAULT 0, is_valid BOOLEAN NOT NULL DEFAULT 1,
            validation_errors TEXT NOT NULL DEFAULT '[]', is_flagged BOOLEAN NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        """
    )
    legacy.execute(
        "INSERT INTO ingestion_runs VALUES ('old', 'c', 'cat', 'sks', '/tmp/x', 'folder', ?)",
        (datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),),
    )
    legacy.execute(
        "INSERT INTO samples (sample_id, run_id, file_path, caption, width, height,"
        " aspect_ratio, phash, updated_at) VALUES ('s0','old','/tmp/x/a.png','sks, a cat',"
        " 512, 512, 1.0, 'abcd1234', 'now')"
    )
    legacy.commit()
    legacy.close()

    migrated = get_connection(db_path)
    try:
        loaded = load_ingestion_run(migrated, "old")
    finally:
        migrated.close()

    assert loaded is not None
    assert loaded.concept.samples[0].metrics.dhash == ""
    assert loaded.concept.samples[0].is_excluded is False
    # Never measured, so it must read back as "unknown" rather than "perfectly soft".
    assert loaded.concept.samples[0].metrics.sharpness == 0.0


def test_sharpness_round_trips_through_storage(conn):
    run = _make_run("r1", n_samples=1)
    run.concept.samples[0].metrics.sharpness = 412.5

    save_ingestion_run(conn, run)
    loaded = load_ingestion_run(conn, "r1")

    assert loaded.concept.samples[0].metrics.sharpness == 412.5


def test_saving_a_run_registers_its_concept(conn):
    """dataset_versions' foreign key points at `concepts`, which used to stay empty."""
    save_ingestion_run(conn, _make_run("r1", concept_name="cyberpunk"))

    row = conn.execute("SELECT concept_name, trigger_word FROM concepts").fetchone()
    assert row["concept_name"] == "cyberpunk"
