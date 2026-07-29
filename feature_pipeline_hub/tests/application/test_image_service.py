from pathlib import Path

from PIL import Image, ImageFilter

from feature_pipeline.application.image_service import (
    classify_aspect_ratio,
    classify_orientation,
    color_distance,
    compute_dhash,
    compute_image_metrics,
    compute_sharpness,
    hamming_distance,
    make_square_thumbnail,
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


def test_compute_image_metrics_records_every_perceptual_hash(tmp_path: Path):
    image_path = tmp_path / "pattern.png"
    _make_patterned_image(image_path, (512, 512), seed=7)

    metrics = compute_image_metrics(str(image_path))

    assert metrics.phash and metrics.dhash and metrics.colorhash
    assert metrics.dhash == compute_dhash(str(image_path))


def test_color_distance_separates_images_that_share_a_luminance_hash(tmp_path: Path):
    red_path, blue_path = tmp_path / "red.png", tmp_path / "blue.png"
    _make_image(red_path, (512, 512), "red")
    _make_image(blue_path, (512, 512), "blue")

    red = compute_image_metrics(str(red_path))
    blue = compute_image_metrics(str(blue_path))

    # Flat images are identical to pHash/dHash, which ignore colour entirely.
    assert hamming_distance(red.phash, blue.phash) == 0
    assert color_distance(red.colorhash, blue.colorhash) > 0


def test_color_distance_is_zero_for_the_same_image(tmp_path: Path):
    image_path = tmp_path / "red.png"
    _make_image(image_path, (512, 512), "red")

    metrics = compute_image_metrics(str(image_path))

    assert color_distance(metrics.colorhash, metrics.colorhash) == 0


def test_make_square_thumbnail_returns_a_square_regardless_of_source_aspect(tmp_path: Path):
    landscape = tmp_path / "landscape.png"
    portrait = tmp_path / "portrait.png"
    _make_image(landscape, (1024, 512), "red")
    _make_image(portrait, (512, 1024), "blue")

    assert make_square_thumbnail(str(landscape), size=256).size == (256, 256)
    assert make_square_thumbnail(str(portrait), size=256).size == (256, 256)


def test_make_square_thumbnail_does_not_crop_the_source_content(tmp_path: Path):
    """The whole image must fit inside the square: no dimension may be upscaled past `size`."""
    image_path = tmp_path / "wide.png"
    _make_image(image_path, (1024, 256), "green")

    thumb = make_square_thumbnail(str(image_path), size=200)

    # 1024x256 fit into 200x200 preserving aspect ratio -> 200x50, centred.
    assert thumb.size == (200, 200)
    opaque_rows = [y for y in range(200) if thumb.getpixel((100, y))[3] > 0]
    assert len(opaque_rows) == 50
    assert min(opaque_rows) == 75 and max(opaque_rows) == 124


def test_make_square_thumbnail_of_a_square_image_has_no_transparent_margin(tmp_path: Path):
    image_path = tmp_path / "square.png"
    _make_image(image_path, (400, 400), "red")

    thumb = make_square_thumbnail(str(image_path), size=200)

    corners = [thumb.getpixel((x, y)) for x in (0, 199) for y in (0, 199)]
    assert all(pixel[3] == 255 for pixel in corners)


def test_make_square_thumbnail_pads_a_narrow_image_with_transparency(tmp_path: Path):
    image_path = tmp_path / "portrait.png"
    _make_image(image_path, (256, 1024), "blue")

    thumb = make_square_thumbnail(str(image_path), size=200)

    # Content is centred horizontally as a narrow vertical strip; corners stay padding.
    assert thumb.getpixel((0, 0))[3] == 0
    assert thumb.getpixel((199, 0))[3] == 0
    assert thumb.getpixel((100, 100))[3] == 255


# --- sharpness ----------------------------------------------------------------


def _checkerboard(size: tuple[int, int], square: int = 16) -> Image.Image:
    """Hard-edged pattern: plenty of high-frequency detail for the Laplacian to find."""
    img = Image.new("L", size)
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            pixels[x, y] = 255 if ((x // square) + (y // square)) % 2 == 0 else 0
    return img


def test_a_blurred_image_scores_lower_than_the_sharp_original():
    sharp = _checkerboard((256, 256))
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))

    assert compute_sharpness(blurred) < compute_sharpness(sharp)


def test_a_flat_image_has_no_sharpness():
    assert compute_sharpness(Image.new("L", (128, 128), color=128)) == 0.0


def test_sharpness_is_comparable_across_resolutions():
    """The same pattern at 1024 and 2048 scores the same, because both are normalised.

    This is what the thumbnail step buys: on the raw variance these two differ by
    roughly 2x purely because of pixel count, which would let resolution masquerade
    as focus. Note the invariance holds between images that get resampled — a source
    already at or below SHARPNESS_SAMPLE_SIZE keeps its own edge response.
    """
    at_1024 = compute_sharpness(_checkerboard((1024, 1024), square=32))
    at_2048 = compute_sharpness(_checkerboard((2048, 2048), square=64))

    assert abs(at_1024 - at_2048) / max(at_1024, at_2048) < 0.05


def test_sharpness_is_recorded_alongside_the_hashes(tmp_path: Path):
    image_path = tmp_path / "pattern.png"
    _make_patterned_image(image_path, (512, 512), seed=7)

    metrics = compute_image_metrics(str(image_path))

    assert metrics.sharpness > 0


def test_a_tiny_image_does_not_break_the_laplacian():
    assert compute_sharpness(Image.new("L", (2, 2), color=200)) == 0.0


# --- orientation --------------------------------------------------------------


def test_classify_orientation_rolls_up_to_three_buckets():
    assert classify_orientation(16 / 9) == "landscape"
    assert classify_orientation(9 / 16) == "portrait"
    assert classify_orientation(1.0) == "square"
    assert classify_orientation(1.02) == "square"
