from datetime import datetime, timezone

from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetHealth,
    DatasetManifest,
    DatasetSample,
    ImageMetrics,
)


def _sample_metrics() -> ImageMetrics:
    return ImageMetrics(width=512, height=512, aspect_ratio=1.0, format="PNG", phash="a" * 16)


def test_image_metrics_roundtrip():
    metrics = _sample_metrics()
    assert metrics.width == 512
    assert metrics.aspect_ratio == 1.0


def test_dataset_sample_defaults():
    sample = DatasetSample(
        sample_id="s1",
        image_path="data/raw/img1.png",
        caption="sks_style a cat",
        original_caption="a cat",
        metrics=_sample_metrics(),
    )
    assert sample.is_duplicate is False
    assert sample.is_valid is True
    assert sample.validation_errors == []


def test_concept_group_holds_samples():
    sample = DatasetSample(
        sample_id="s1",
        image_path="data/raw/img1.png",
        caption="sks_style a cat",
        original_caption="a cat",
        metrics=_sample_metrics(),
    )
    concept = ConceptGroup(
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
        samples=[sample],
    )
    assert len(concept.samples) == 1
    assert concept.samples[0].sample_id == "s1"


def test_dataset_manifest_defaults_created_at():
    manifest = DatasetManifest(
        dataset_name="cyberpunk_style",
        version="v1.0.0",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
        total_samples=10,
        duplicate_count=1,
        aspect_ratio_distribution={"1:1": 8, "16:9": 2},
        content_hash="deadbeef",
    )
    assert manifest.total_samples == 10
    assert manifest.created_at is not None


def _health(**overrides) -> DatasetHealth:
    fields = {
        "run_id": "r1",
        "concept_name": "cyberpunk",
        "trigger_word": "sks_style",
        "source_kind": "upload",
        "created_at": datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        "total_samples": 3,
        "active_samples": 3,
        "excluded_samples": 0,
        "duplicate_count": 0,
        "missing_caption_count": 0,
        "invalid_count": 0,
    }
    return DatasetHealth(**{**fields, **overrides})


def test_a_clean_dataset_has_no_issues():
    assert _health().has_issues is False
    # Excluding images is a curation decision, not a problem to flag.
    assert _health(active_samples=2, excluded_samples=1).has_issues is False


def test_any_single_problem_flags_the_dataset():
    assert _health(duplicate_count=1).has_issues is True
    assert _health(missing_caption_count=1).has_issues is True
    assert _health(invalid_count=1).has_issues is True
