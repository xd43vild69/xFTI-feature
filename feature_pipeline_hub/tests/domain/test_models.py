from feature_pipeline.domain.models import (
    ConceptGroup,
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
