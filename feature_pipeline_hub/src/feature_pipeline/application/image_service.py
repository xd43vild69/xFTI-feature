"""Image analysis: dimensions, perceptual hashing, sharpness, and aspect-ratio bucketing."""

from pathlib import Path
from typing import NamedTuple

import imagehash
import numpy as np
from PIL import Image, ImageOps

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
    grey.thumbnail((SHARPNESS_SAMPLE_SIZE, SHARPNESS_SAMPLE_SIZE), Image.Resampling.LANCZOS)

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

    EXIF orientation is applied, which is not cosmetic here: the full-size preview
    hands the file to the browser, and browsers honour that tag. Without this the
    grid would show a phone photo lying on its side and the zoom view would show it
    upright — and an operator reaching for the rotate buttons would be correcting a
    discrepancy that only ever existed in the thumbnail.
    """
    with Image.open(image_path) as opened:
        img = ImageOps.exif_transpose(opened) or opened
        rgba = img.convert("RGBA")
        rgba.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        offset = ((size - rgba.width) // 2, (size - rgba.height) // 2)
        canvas.paste(rgba, offset, rgba)
        return canvas


MAX_TRAINING_SIDE = 1024

# Save settings per output format, chosen so the re-encode is not the thing that
# costs quality: the resample already discarded detail, and a default-quality JPEG
# (75, 4:2:0 chroma) would then throw away colour detail on top of it. 95 with full
# chroma is visually transparent at this size; PNG and WebP-lossless are exact.
_SAVE_OPTIONS: dict[str, dict[str, object]] = {
    "JPEG": {"quality": 95, "subsampling": 0, "optimize": True},
    "PNG": {"optimize": True, "compress_level": 9},
    "WEBP": {"lossless": True, "quality": 100, "method": 6},
}
_FORMAT_BY_SUFFIX = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}


def is_oversized(width: int, height: int, max_side: int = MAX_TRAINING_SIDE) -> bool:
    """Whether an image's longest side exceeds the training resolution."""
    return max(width, height) > max_side


def target_size(width: int, height: int, max_side: int = MAX_TRAINING_SIDE) -> tuple[int, int]:
    """Dimensions that put the longest side exactly on `max_side`, aspect preserved.

    The long side is assigned rather than computed so it lands on `max_side` exactly
    regardless of rounding; only the short side is scaled and rounded.
    """
    if width >= height:
        return max_side, max(1, round(height * max_side / width))
    return max(1, round(width * max_side / height)), max_side


class TransformResult(NamedTuple):
    """What `apply_transform` produced, and what it started from."""

    output_path: str
    source_width: int
    source_height: int
    width: int
    height: int
    rewritten: bool


# Quarter turns, counter-clockwise, as Pillow names them. `transpose` is an exact
# permutation of the pixel grid — no interpolation, no resampling — so the rotation
# itself never costs quality. Only the re-encode can, which is why callers derive
# from the original rather than from the last derivation.
_ROTATIONS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def apply_transform(
    image_path: str,
    destination_dir: str,
    *,
    rotation_degrees: int = 0,
    max_side: int | None = MAX_TRAINING_SIDE,
) -> TransformResult:
    """Rotate and/or downscale an image, writing the result beside the original.

    One pass, one re-encode, whatever combination is asked for — which is the whole
    point of taking both parameters instead of chaining two calls. A sample that is
    rotated three times and normalized once still costs its JPEG exactly one
    generation, because the caller always passes the untouched original here along
    with the *accumulated* rotation.

    Order matters and is fixed: EXIF orientation is resolved into the pixels, then the
    rotation, then the downscale. Scaling before rotating would fit the wrong axis to
    `max_side`, and rotating before resolving EXIF would compound the two.

    Lanczos is the resampling filter, matching what `precache_worker.vae_latent` uses
    when it fits an image to its latent bucket — the point of doing this earlier is to
    make the reduction visible and reviewable, not to reduce it differently. Pillow
    scales the filter's support window with the reduction ratio, so a single `resize`
    is already area-averaged: it reads every source pixel and cannot alias the way a
    naive subsample or a fixed-kernel filter does. No sharpening is applied afterwards
    — an unsharp pass would put halos into the pixels the VAE encodes, and a LoRA
    learns those halos as readily as it learns the subject.

    `max_side=None` skips the downscale entirely. When there is nothing to do at all —
    no rotation, and either no limit or an image already within it — the original path
    comes back untouched and `rewritten` is False, since a round trip through JPEG is
    lossless in name only.
    """
    source = Path(image_path)
    rotation = rotation_degrees % 360

    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened) or opened
        source_width, source_height = image.size

        if rotation:
            image = image.transpose(_ROTATIONS[rotation])

        needs_resize = max_side is not None and is_oversized(*image.size, max_side)
        if not rotation and not needs_resize:
            return TransformResult(
                str(source), source_width, source_height, source_width, source_height, False
            )

        image_format = _FORMAT_BY_SUFFIX.get(
            source.suffix.lower(), (opened.format or "PNG").upper()
        )
        # ICC survives so a wide-gamut source keeps meaning what it meant; EXIF
        # deliberately does not, since its orientation tag is now baked into the
        # pixels and re-attaching it would turn the image a second time.
        icc_profile = image.info.get("icc_profile")

        if image_format == "JPEG" and image.mode not in {"RGB", "L"}:
            # JPEG has no alpha channel; Pillow raises rather than dropping it.
            image = image.convert("RGB")

        if needs_resize:
            assert max_side is not None  # narrowed by needs_resize
            image = image.resize(target_size(*image.size, max_side), Image.Resampling.LANCZOS)

        output = image

    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / source.name

    save_options = dict(_SAVE_OPTIONS.get(image_format, {}))
    if icc_profile:
        save_options["icc_profile"] = icc_profile
    output.save(output_path, format=image_format, **save_options)

    return TransformResult(
        str(output_path), source_width, source_height, output.width, output.height, True
    )


def normalize_to_max_side(
    image_path: str, destination_dir: str, max_side: int = MAX_TRAINING_SIDE
) -> TransformResult:
    """`apply_transform` with no rotation: the downscale-only case."""
    return apply_transform(image_path, destination_dir, max_side=max_side)


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
