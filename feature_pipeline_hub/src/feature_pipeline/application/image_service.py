"""Image analysis: dimensions, perceptual hashing, sharpness, and aspect-ratio bucketing."""

from pathlib import Path
from typing import NamedTuple

import imagehash
import numpy as np
from PIL import Image

from feature_pipeline.domain.models import ImageMetrics

ASPECT_RATIO_BUCKETS: dict[str, float] = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
}
BUCKET_TOLERANCE = 0.05


SHARPNESS_SAMPLE_SIZE = 512


def compute_sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian: low means blurred or out of focus.

    The image is reduced to a fixed size first because the variance scales with
    resolution — without that, a large soft photo can outscore a small crisp one and
    the numbers are not comparable within a set.

    The 3x3 Laplacian is applied directly on a float array rather than through
    `ImageFilter.Kernel`, which clamps to 0-255 on 8-bit images and would throw away
    the negative half of the response.

    Note this is only meaningful as a *relative* ranking inside one concept: texture
    and subject matter move it as much as focus does, so there is no threshold that
    means "blurry" across datasets.
    """
    grey = img.convert("L")
    grey.thumbnail((SHARPNESS_SAMPLE_SIZE, SHARPNESS_SAMPLE_SIZE), Image.LANCZOS)

    pixels = np.asarray(grey, dtype=np.float64)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0

    laplacian = (
        pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
        - 4 * pixels[1:-1, 1:-1]
    )
    return float(laplacian.var())


def compute_image_metrics(image_path: str) -> ImageMetrics:
    """Open an image once and derive its dimensions, format, hashes, and sharpness."""
    with Image.open(image_path) as img:
        width, height = img.size
        image_format = (img.format or Path(image_path).suffix.lstrip(".")).upper()
        phash = str(imagehash.phash(img))
        dhash = str(imagehash.dhash(img))
        colorhash = str(imagehash.colorhash(img))
        sharpness = compute_sharpness(img)

    return ImageMetrics(
        width=width,
        height=height,
        aspect_ratio=width / height,
        format=image_format,
        phash=phash,
        dhash=dhash,
        colorhash=colorhash,
        sharpness=sharpness,
    )


THUMBNAIL_SIZE = 512


def make_square_thumbnail(image_path: str, size: int = THUMBNAIL_SIZE) -> Image.Image:
    """Fit an image into a size×size canvas without cropping (letterboxed, transparent margins).

    Transparent padding (rather than a solid colour) lets the surrounding card
    background show through, so the thumbnail adapts to light/dark theme for free.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        offset = ((size - img.width) // 2, (size - img.height) // 2)
        canvas.paste(img, offset, img)
        return canvas


class OriginalImageFacts(NamedTuple):
    """What an image is on disk, before any viewer gets a chance to resample it."""

    width: int
    height: int
    image_format: str
    byte_size: int


def describe_original(image_path: str) -> OriginalImageFacts:
    """Read an image's real dimensions, format, and file size without decoding pixels.

    `Image.open` is lazy, so `.size` and `.format` come from the header alone. This
    is the ground truth shown next to a full-size preview: the point is to let the
    user confirm what they actually imported rather than infer it from how the
    preview happens to render.
    """
    path = Path(image_path)
    with Image.open(path) as img:
        width, height = img.size
        image_format = (img.format or path.suffix.lstrip(".")).upper()
    return OriginalImageFacts(width, height, image_format, path.stat().st_size)


def compute_dhash(image_path: str) -> str:
    """Compute a difference hash (dHash), a secondary signal for duplicate detection."""
    with Image.open(image_path) as img:
        return str(imagehash.dhash(img))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Distance between two perceptual hashes: 0 means identical, higher means more different."""
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


COLORHASH_BITS = 42


def color_distance(hash_a: str, hash_b: str) -> int:
    """Distance between two colour hashes.

    Colour hashes are flat bit strings rather than square matrices, so they need a
    different decoder than `hamming_distance`.
    """
    return imagehash.hex_to_flathash(hash_a, COLORHASH_BITS) - imagehash.hex_to_flathash(
        hash_b, COLORHASH_BITS
    )


SQUARE_TOLERANCE = 0.05


def classify_orientation(aspect_ratio: float) -> str:
    """Coarse landscape / portrait / square rollup of a width/height ratio."""
    if abs(aspect_ratio - 1.0) <= SQUARE_TOLERANCE:
        return "square"
    return "landscape" if aspect_ratio > 1.0 else "portrait"


def classify_aspect_ratio(aspect_ratio: float) -> str:
    """Map a raw width/height ratio to the closest known bucket, or 'other' if none is close."""
    bucket_name, bucket_ratio = min(
        ASPECT_RATIO_BUCKETS.items(),
        key=lambda item: abs(item[1] - aspect_ratio),
    )
    if abs(bucket_ratio - aspect_ratio) / bucket_ratio <= BUCKET_TOLERANCE:
        return bucket_name
    return "other"
