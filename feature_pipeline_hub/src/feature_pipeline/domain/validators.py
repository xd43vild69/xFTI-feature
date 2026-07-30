"""Pure validation functions for dataset samples: no I/O, no side effects."""

from pathlib import Path

from feature_pipeline.domain.models import DatasetSample, ImageMetrics

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_RESOLUTION_PX = 512
# A side reaching this clears the sample even if the other falls short of
# MIN_RESOLUTION_PX: a tall or wide crop can carry enough detail on one axis alone.
MIN_RESOLUTION_PX_ALT = 1024
MIN_CAPTION_LENGTH = 3
MAX_CAPTION_LENGTH = 500


def validate_extension(image_path: str) -> list[str]:
    suffix = Path(image_path).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return [
            f"Unsupported image extension '{suffix}'; expected one of "
            f"{sorted(ALLOWED_IMAGE_EXTENSIONS)}"
        ]
    return []


def validate_resolution(metrics: ImageMetrics) -> list[str]:
    both_sides_pass = metrics.width >= MIN_RESOLUTION_PX and metrics.height >= MIN_RESOLUTION_PX
    one_side_is_large = metrics.width >= MIN_RESOLUTION_PX_ALT or metrics.height >= MIN_RESOLUTION_PX_ALT

    if both_sides_pass or one_side_is_large:
        return []

    return [
        f"Resolution {metrics.width}x{metrics.height} is below the minimum "
        f"{MIN_RESOLUTION_PX}x{MIN_RESOLUTION_PX} "
        f"(and neither side reaches {MIN_RESOLUTION_PX_ALT})"
    ]


def validate_caption(caption: str) -> list[str]:
    stripped = caption.strip()
    if not stripped:
        return ["Caption is empty"]
    if len(stripped) < MIN_CAPTION_LENGTH:
        return [f"Caption is shorter than {MIN_CAPTION_LENGTH} characters"]
    if len(stripped) > MAX_CAPTION_LENGTH:
        return [f"Caption exceeds {MAX_CAPTION_LENGTH} characters"]
    return []


def validate_sample(sample: DatasetSample) -> list[str]:
    """Run every sample-level rule and return the combined list of errors."""
    errors: list[str] = []
    errors.extend(validate_extension(sample.image_path))
    errors.extend(validate_resolution(sample.metrics))
    errors.extend(validate_caption(sample.caption))
    return errors


def find_unpaired_files(
    image_paths: list[str], caption_paths: list[str]
) -> tuple[list[str], list[str]]:
    """Match image/caption files by filename stem and report the ones missing a counterpart."""
    image_stems = {Path(p).stem: p for p in image_paths}
    caption_stems = {Path(p).stem: p for p in caption_paths}

    images_without_caption = [
        path for stem, path in image_stems.items() if stem not in caption_stems
    ]
    captions_without_image = [
        path for stem, path in caption_stems.items() if stem not in image_stems
    ]
    return images_without_caption, captions_without_image
