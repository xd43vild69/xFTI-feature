from pathlib import Path

from PIL import Image

from feature_pipeline.application.image_service import (
    classify_aspect_ratio,
    compute_dhash,
    compute_image_metrics,
    hamming_distance,
)


def _make_image(path: Path, size: tuple[int, int], color: str) -> None:
    Image.new("RGB", size, color=color).save(path)


def _make_patterned_image(path: Path, size: tuple[int, int], seed: int) -> None:
    """A gradient image whose pixel values depend on `seed`, so pHash differs between seeds."""
    width, height = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = ((x + seed) % 256, (y * seed) % 256, (x * y + seed) % 256)
    img.save(path)


def test_compute_image_metrics_reads_dimensions_and_format(tmp_path: Path):
    image_path = tmp_path / "square.png"
    _make_image(image_path, (512, 512), "red")

    metrics = compute_image_metrics(str(image_path))

    assert metrics.width == 512
    assert metrics.height == 512
    assert metrics.aspect_ratio == 1.0
    assert metrics.format == "PNG"
    assert len(metrics.phash) > 0


def test_identical_images_have_zero_hamming_distance(tmp_path: Path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _make_image(image_a, (256, 256), "blue")
    _make_image(image_b, (256, 256), "blue")

    metrics_a = compute_image_metrics(str(image_a))
    metrics_b = compute_image_metrics(str(image_b))

    assert hamming_distance(metrics_a.phash, metrics_b.phash) == 0


def test_different_images_have_nonzero_hamming_distance(tmp_path: Path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _make_patterned_image(image_a, (256, 256), seed=1)
    _make_patterned_image(image_b, (256, 256), seed=97)

    metrics_a = compute_image_metrics(str(image_a))
    metrics_b = compute_image_metrics(str(image_b))

    assert hamming_distance(metrics_a.phash, metrics_b.phash) > 0


def test_compute_dhash_returns_nonempty_hash(tmp_path: Path):
    image_path = tmp_path / "square.png"
    _make_image(image_path, (256, 256), "green")

    dhash = compute_dhash(str(image_path))

    assert len(dhash) > 0


def test_classify_aspect_ratio_buckets():
    assert classify_aspect_ratio(1.0) == "1:1"
    assert classify_aspect_ratio(16 / 9) == "16:9"
    assert classify_aspect_ratio(9 / 16) == "9:16"


def test_classify_aspect_ratio_falls_back_to_other():
    assert classify_aspect_ratio(3.7) == "other"
