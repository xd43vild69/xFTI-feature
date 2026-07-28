from datetime import datetime, timezone
from pathlib import Path

from feature_pipeline.application.export_service import export_active_samples
from feature_pipeline.domain.models import ConceptGroup, DatasetSample, ImageMetrics, IngestionRun
from feature_pipeline.infrastructure.storage import training_dataset_dir


def _sample(sample_id: str, image_path: Path, caption: str, is_excluded: bool = False) -> DatasetSample:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(f"pixels-{sample_id}".encode())
    return DatasetSample(
        sample_id=sample_id,
        image_path=str(image_path),
        caption=caption,
        original_caption=caption,
        metrics=ImageMetrics(width=512, height=512, aspect_ratio=1.0, format="PNG", phash="abcd"),
        is_excluded=is_excluded,
    )


def _run(samples: list[DatasetSample], concept_name: str = "my_concept") -> IngestionRun:
    return IngestionRun(
        run_id="r1",
        source_path="/data/raw/r1",
        source_kind="folder",
        created_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        concept=ConceptGroup(
            concept_id="c1", concept_name=concept_name, trigger_word="sks", samples=samples
        ),
    )


def test_exports_only_active_samples(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    run = _run(
        [
            _sample("a", tmp_path / "src" / "a.png", "sks, a red car"),
            _sample("b", tmp_path / "src" / "b.png", "sks, excluded", is_excluded=True),
        ]
    )

    result = export_active_samples(run, "my_concept")

    dest = training_dataset_dir("my_concept")
    assert sorted(p.name for p in dest.iterdir()) == ["a.png", "a.txt"]
    assert (dest / "a.txt").read_text() == "sks, a red car"
    assert result.exported_count == 1
    assert result.skipped_excluded == 1


def test_wipes_previous_export_before_writing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    dest = training_dataset_dir("my_concept")
    dest.mkdir(parents=True)
    (dest / "stale.png").write_bytes(b"old")
    (dest / "curation_report.json").write_text("{}")

    run = _run([_sample("a", tmp_path / "src" / "a.png", "sks, a red car")])
    export_active_samples(run, "my_concept")

    assert sorted(p.name for p in dest.iterdir()) == ["a.png", "a.txt"]


def test_disambiguates_a_basename_collision_within_the_same_run(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    run = _run(
        [
            _sample("a", tmp_path / "src1" / "same.png", "sks, first"),
            _sample("b", tmp_path / "src2" / "same.png", "sks, second"),
        ]
    )

    result = export_active_samples(run, "my_concept")

    dest = training_dataset_dir("my_concept")
    png_names = sorted(p.name for p in dest.glob("*.png"))
    assert len(png_names) == 2
    assert result.renamed_for_collision != []


def test_an_all_excluded_run_exports_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    run = _run([_sample("a", tmp_path / "src" / "a.png", "sks", is_excluded=True)])

    result = export_active_samples(run, "my_concept")

    assert result.exported_count == 0
    assert training_dataset_dir("my_concept").exists()
