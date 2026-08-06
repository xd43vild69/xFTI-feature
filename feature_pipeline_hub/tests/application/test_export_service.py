from datetime import datetime, timezone
from pathlib import Path

from feature_pipeline.application.export_service import export_active_samples, export_branch_dataset
from feature_pipeline.domain.curation_report import WeightProfile
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


# ── export_branch_dataset ────────────────────────────────────────────────────

def test_branch_export_without_tiers_writes_no_report(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    samples = [_sample("a", tmp_path / "src" / "a.png", "sks, a car")]

    result = export_branch_dataset(samples, "concept__control")

    dest = training_dataset_dir("concept__control")
    assert result.report_path is None
    assert result.weights_are_effective is False
    assert not (dest / "curation_report.json").exists()


def test_branch_export_with_tiers_writes_a_report_keyed_by_exported_stem(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    a = _sample("a", tmp_path / "src" / "a.png", "sks, a car")
    b = _sample("b", tmp_path / "src" / "b.png", "sks, a truck")

    result = export_branch_dataset(
        [a, b], "concept__variant",
        tiers={"a": "priority", "b": "bad"},
        profile=WeightProfile(),
    )

    dest = training_dataset_dir("concept__variant")
    assert result.report_path == dest / "curation_report.json"
    assert result.weights_are_effective is True
    assert set(result.stems) == {"a", "b"}

    import json
    report = json.loads((dest / "curation_report.json").read_text())
    assert set(report["images"]) == {"a", "b"}
    assert report["baselines"] == ["a.png"]


def test_branch_export_removes_a_stale_curation_overrides_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    dest = training_dataset_dir("concept__variant")
    dest.mkdir(parents=True)
    (dest / "curation_overrides.json").write_text('{"threshold": 0.9}')

    samples = [_sample("a", tmp_path / "src" / "a.png", "sks, a car")]
    export_branch_dataset(samples, "concept__variant")

    assert not (dest / "curation_overrides.json").exists()


def test_branch_export_with_all_neutral_weights_is_not_effective(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    samples = [_sample("a", tmp_path / "src" / "a.png", "sks, a car")]

    result = export_branch_dataset(
        samples, "concept__variant", tiers={}, profile=WeightProfile()
    )

    assert result.weights_are_effective is False
