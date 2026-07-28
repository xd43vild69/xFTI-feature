from pathlib import Path

from PIL import Image

from feature_pipeline.application.dataset_service import ingest_concept_from_folder


def _make_image(path: Path, size=(512, 512)) -> None:
    Image.new("RGB", size, color="blue").save(path)


def test_ingest_concept_from_folder_builds_samples_with_trigger_injected(tmp_path: Path):
    _make_image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("a cat sitting")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    assert concept.concept_id == "c1"
    assert len(concept.samples) == 1
    sample = concept.samples[0]
    assert sample.caption == "sks_style, a cat sitting"
    assert sample.original_caption == "a cat sitting"
    assert sample.is_valid is True
    assert sample.validation_errors == []


def test_ingest_concept_from_folder_falls_back_to_trigger_word_when_caption_missing(
    tmp_path: Path,
):
    _make_image(tmp_path / "a.png")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    sample = concept.samples[0]
    # No .txt file next to the image: original_caption stays empty so the UI
    # can flag it, but the trigger word alone becomes a valid caption.
    assert sample.original_caption == ""
    assert sample.caption == "sks_style"
    assert sample.is_valid is True


def test_ingest_concept_from_folder_flags_low_resolution(tmp_path: Path):
    _make_image(tmp_path / "a.png", size=(128, 128))
    (tmp_path / "a.txt").write_text("a cat sitting")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    sample = concept.samples[0]
    assert sample.is_valid is False
    assert any("Resolution" in error for error in sample.validation_errors)


def test_ingest_concept_from_folder_empty_folder_returns_no_samples(tmp_path: Path):
    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    assert concept.samples == []
