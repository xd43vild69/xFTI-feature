from datetime import datetime, timezone

import pytest

from feature_pipeline.application.inventory_service import load_dataset_inventory
from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetSample,
    ImageMetrics,
    IngestionRun,
)
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.ingestion_repository import (
    mark_duplicates,
    save_ingestion_run,
    set_samples_excluded,
)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _make_run(
    run_id: str,
    concept_name: str = "cyberpunk",
    n_samples: int = 1,
    is_valid: bool = True,
    sharpness: float = 0.0,
    created_at: datetime | None = None,
) -> IngestionRun:
    samples = [
        DatasetSample(
            sample_id=f"{run_id}-s{i}",
            image_path=f"/data/raw/{run_id}/img{i}.png",
            caption=f"sks_style, image {i}",
            original_caption=f"image {i}",
            metrics=ImageMetrics(
                width=512,
                height=768,
                aspect_ratio=512 / 768,
                format="PNG",
                phash="abcd1234",
                sharpness=sharpness,
            ),
            is_valid=is_valid,
        )
        for i in range(n_samples)
    ]
    return IngestionRun(
        run_id=run_id,
        source_path=f"/data/raw/{run_id}",
        source_kind="upload",
        created_at=created_at or datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        concept=ConceptGroup(
            concept_id=f"c-{run_id}",
            concept_name=concept_name,
            trigger_word="sks_style",
            samples=samples,
        ),
    )


def test_a_saved_run_reports_its_identity_and_counts(conn):
    save_ingestion_run(conn, _make_run("r1", concept_name="cyberpunk", n_samples=3))

    (health,) = load_dataset_inventory(conn)

    assert health.run_id == "r1"
    assert health.concept_name == "cyberpunk"
    assert health.trigger_word == "sks_style"
    assert health.source_kind == "upload"
    assert health.total_samples == 3
    assert health.active_samples == 3
    assert health.excluded_samples == 0
    assert health.invalid_count == 0
    assert health.has_issues is False


def test_runs_are_listed_newest_first_without_leaking_counts(conn):
    save_ingestion_run(conn, _make_run("older", n_samples=1))
    save_ingestion_run(
        conn,
        _make_run(
            "newer",
            n_samples=4,
            created_at=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
        ),
    )

    inventory = load_dataset_inventory(conn)

    assert [h.run_id for h in inventory] == ["newer", "older"]
    assert [h.total_samples for h in inventory] == [4, 1]


def test_a_run_without_samples_is_all_zeros(conn):
    save_ingestion_run(conn, _make_run("empty", n_samples=0))

    (health,) = load_dataset_inventory(conn)

    assert health.total_samples == 0
    assert health.active_samples == 0
    assert health.median_sharpness == 0.0
    assert health.has_issues is False


def test_excluding_moves_a_sample_from_active_to_excluded(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=3))

    set_samples_excluded(conn, ["r1-s0"], True)
    (health,) = load_dataset_inventory(conn)

    assert health.total_samples == 3
    assert health.active_samples == 2
    assert health.excluded_samples == 1


def test_duplicate_count_reflects_the_stored_quality_verdict(conn):
    """End-to-end proof of the stored-flag path: no clustering happens here."""
    save_ingestion_run(conn, _make_run("r1", n_samples=3))

    assert load_dataset_inventory(conn)[0].duplicate_count == 0

    mark_duplicates(conn, "r1", ["r1-s0", "r1-s1"])

    (health,) = load_dataset_inventory(conn)
    assert health.duplicate_count == 2
    assert health.has_issues is True


def test_invalid_samples_surface_as_an_issue(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=2, is_valid=False))

    (health,) = load_dataset_inventory(conn)

    assert health.invalid_count == 2
    assert health.has_issues is True


def test_median_sharpness_is_reported_per_run(conn):
    save_ingestion_run(conn, _make_run("r1", n_samples=3, sharpness=42.0))

    assert load_dataset_inventory(conn)[0].median_sharpness == 42.0


def test_an_empty_database_yields_an_empty_inventory(conn):
    assert load_dataset_inventory(conn) == []
