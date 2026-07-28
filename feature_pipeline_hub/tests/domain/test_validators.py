from feature_pipeline.domain.models import DatasetSample, ImageMetrics
from feature_pipeline.domain.validators import (
    find_unpaired_files,
    validate_caption,
    validate_extension,
    validate_resolution,
    validate_sample,
)


def _metrics(width=512, height=512) -> ImageMetrics:
    return ImageMetrics(width=width, height=height, aspect_ratio=width / height, format="PNG", phash="a" * 16)


def test_validate_extension_accepts_allowed():
    assert validate_extension("img.png") == []
    assert validate_extension("img.JPG") == []
    assert validate_extension("img.webp") == []


def test_validate_extension_rejects_unknown():
    errors = validate_extension("img.bmp")
    assert len(errors) == 1
    assert "bmp" in errors[0]


def test_validate_resolution_flags_too_small():
    errors = validate_resolution(_metrics(width=256, height=256))
    assert len(errors) == 1


def test_validate_resolution_passes_minimum():
    assert validate_resolution(_metrics(width=512, height=512)) == []


def test_validate_caption_flags_empty():
    assert validate_caption("   ") == ["Caption is empty"]


def test_validate_caption_flags_too_short():
    errors = validate_caption("ab")
    assert len(errors) == 1


def test_validate_caption_accepts_reasonable_text():
    assert validate_caption("sks_style a photo of a cat") == []


def test_validate_sample_aggregates_all_errors():
    sample = DatasetSample(
        sample_id="s1",
        image_path="img.bmp",
        caption="",
        original_caption="",
        metrics=_metrics(width=100, height=100),
    )
    errors = validate_sample(sample)
    assert len(errors) == 3


def test_find_unpaired_files():
    images = ["a.png", "b.png", "c.png"]
    captions = ["a.txt", "b.txt", "d.txt"]
    images_without_caption, captions_without_image = find_unpaired_files(images, captions)
    assert images_without_caption == ["c.png"]
    assert captions_without_image == ["d.txt"]
