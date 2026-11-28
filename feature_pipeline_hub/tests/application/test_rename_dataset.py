import pytest

from feature_pipeline.application import dataset_service
from feature_pipeline.domain.models import ConceptGroup, DatasetSample, ImageMetrics, IngestionRun
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure.database import get_connection


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _save(conn, run_id, name, trigger, captions):
    samples = [
        DatasetSample(
            sample_id=f"{run_id}-{i}",
            image_path=f"/img/{run_id}{i}.png",
            caption=c,
            original_caption=c,
            metrics=ImageMetrics(
                width=512, height=512, aspect_ratio=1.0, format="PNG", phash=f"p{run_id}{i}"
            ),
        )
        for i, c in enumerate(captions)
    ]
    run = IngestionRun(
        run_id=run_id,
        source_path="/raw",
        source_kind="folder",
        concept=ConceptGroup(
            concept_id=f"c-{run_id}", concept_name=name, trigger_word=trigger, samples=samples
        ),
    )
    repo.save_ingestion_run(conn, run)


def test_rename_syncs_trigger_and_captions(conn):
    _save(conn, "r1", "old_name", "old_trg", ["old_trg, a cat", "a dog"])

    result = dataset_service.rename_dataset(conn, "r1", "new_name")

    run = repo.load_ingestion_run(conn, "r1")
    assert run.concept.concept_name == "new_name"
    assert run.concept.trigger_word == "new_name"
    assert sorted(s.caption for s in run.concept.samples) == ["new_name, a cat", "new_name, a dog"]
    assert result.captions_changed == 2


def test_rename_refuses_a_trigger_owned_by_another_dataset(conn):
    _save(conn, "r1", "a", "a", ["a, x"])
    _save(conn, "r2", "b", "taken", ["taken, y"])

    with pytest.raises(ValueError):
        dataset_service.rename_dataset(conn, "r1", "taken")

    run = repo.load_ingestion_run(conn, "r1")
    assert run.concept.concept_name == "a"
    assert run.concept.samples[0].caption == "a, x"
