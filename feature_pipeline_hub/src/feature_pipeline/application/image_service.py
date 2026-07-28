"""Image analysis: dimension reading, perceptual hashing, and aspect-ratio bucketing."""

from pathlib import Path

import imagehash
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


def compute_image_metrics(image_path: str) -> ImageMetrics:
    """Open an image and derive its dimensions, format, and perceptual hash (pHash)."""
    with Image.open(image_path) as img:
        width, height = img.size
        image_format = (img.format or Path(image_path).suffix.lstrip(".")).upper()
        phash = str(imagehash.phash(img))
        dhash = str(imagehash.dhash(img))
        colorhash = str(imagehash.colorhash(img))

    return ImageMetrics(
        width=width,
        height=height,
        aspect_ratio=width / height,
        format=image_format,
        phash=phash,
        dhash=dhash,
        colorhash=colorhash,
    )


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


def classify_aspect_ratio(aspect_ratio: float) -> str:
    """Map a raw width/height ratio to the closest known bucket, or 'other' if none is close."""
    bucket_name, bucket_ratio = min(
        ASPECT_RATIO_BUCKETS.items(),
        key=lambda item: abs(item[1] - aspect_ratio),
    )
    if abs(bucket_ratio - aspect_ratio) / bucket_ratio <= BUCKET_TOLERANCE:
        return bucket_name
    return "other"
