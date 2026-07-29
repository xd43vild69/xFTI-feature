"""Dataset quality analysis: perceptual deduplication, pairing checks, and statistics.

Everything here is pure: it reads already-computed sample metrics and returns
plain values, so the UI can recompute freely without touching disk.
"""

from collections import Counter

from feature_pipeline.application.caption_service import strip_trigger_word
from feature_pipeline.application.image_service import (
    classify_aspect_ratio,
    classify_orientation,
    color_distance,
    hamming_distance,
)
from feature_pipeline.domain.models import DatasetSample, DuplicateCluster

DEFAULT_PHASH_THRESHOLD = 5
# pHash and dHash both work on luminance, so flat or low-detail images (renders,
# screenshots, plain backdrops) hash the same regardless of colour. A colour-hash
# check vetoes those false positives.
COLOR_GUARD_DISTANCE = 4
MAX_DISTANCE = 64
# Captions are encoded by Qwen3-VL, not CLIP: the budget that matters is the
# `max_seq_len` the pre-cache worker passes to `encode_prompt`, not CLIP's 77.
# The hub has no tokenizer of its own, so words are converted with a ratio typical
# of a subword tokenizer on English text. Both numbers are named rather than folded
# into a constant so the assumption is greppable when max_seq_len changes.
TEXT_ENCODER_MAX_TOKENS = 128  # workers/precache_worker.py DEFAULTS["max_seq_len"]
TOKENS_PER_WORD = 1.5
CAPTION_WORD_WARNING = int(TEXT_ENCODER_MAX_TOKENS / TOKENS_PER_WORD)


def perceptual_distance(left: DatasetSample, right: DatasetSample) -> int:
    """Distance between two samples: pHash, tightened by dHash and vetoed by colour.

    Taking the larger of the pHash and dHash distances makes a pair count as a
    duplicate only if both signals agree. Images whose colours clearly differ are
    pushed to the maximum distance so they never cluster.
    """
    if left.metrics.colorhash and right.metrics.colorhash:
        if color_distance(left.metrics.colorhash, right.metrics.colorhash) > COLOR_GUARD_DISTANCE:
            return MAX_DISTANCE

    distance = hamming_distance(left.metrics.phash, right.metrics.phash)

    if left.metrics.dhash and right.metrics.dhash:
        distance = max(distance, hamming_distance(left.metrics.dhash, right.metrics.dhash))

    return distance


def find_duplicate_clusters(
    samples: list[DatasetSample], threshold: int = DEFAULT_PHASH_THRESHOLD
) -> list[DuplicateCluster]:
    """Group near-identical images. Excluded samples are ignored.

    Greedy single pass: the first sample of a cluster is the one proposed to keep,
    and every later sample within `threshold` of it joins that cluster.
    """
    candidates = [s for s in samples if not s.is_excluded]
    clustered: set[str] = set()
    clusters: list[DuplicateCluster] = []

    for index, sample in enumerate(candidates):
        if sample.sample_id in clustered:
            continue

        duplicates: list[tuple[DatasetSample, int]] = []
        for other in candidates[index + 1 :]:
            if other.sample_id in clustered:
                continue
            distance = perceptual_distance(sample, other)
            if distance <= threshold:
                duplicates.append((other, distance))
                clustered.add(other.sample_id)

        if duplicates:
            clustered.add(sample.sample_id)
            clusters.append(DuplicateCluster(kept=sample, duplicates=duplicates))

    return clusters


def describes_nothing(caption: str, trigger_word: str) -> bool:
    """True when a caption carries no description beyond the trigger word.

    Ingestion injects the trigger word even when no .txt file existed, so checking
    the raw caption is not enough. Working off the effective caption (rather than
    `original_caption`) means a description typed by hand clears the alert without
    having to re-ingest.
    """
    remainder = strip_trigger_word(caption, trigger_word)
    return not remainder.strip(" ,.\n\t")


def samples_missing_caption(
    samples: list[DatasetSample], trigger_word: str = ""
) -> list[DatasetSample]:
    """Images still waiting for a description."""
    return [
        s for s in samples if not s.is_excluded and describes_nothing(s.caption, trigger_word)
    ]


def caption_word_count(caption: str) -> int:
    return len(caption.split())


def caption_length_stats(samples: list[DatasetSample]) -> dict[str, float]:
    """Min/max/mean caption length in words, plus how many risk CLIP truncation."""
    active = [s for s in samples if not s.is_excluded]
    counts = [caption_word_count(s.caption) for s in active]

    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0, "too_long": 0}

    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 1),
        "too_long": sum(1 for c in counts if c > CAPTION_WORD_WARNING),
    }


def resolution_distribution(samples: list[DatasetSample]) -> dict[str, int]:
    """How many samples share each WxH resolution, most common first."""
    counter = Counter(
        f"{s.metrics.width}x{s.metrics.height}" for s in samples if not s.is_excluded
    )
    return dict(counter.most_common())


def aspect_ratio_distribution(samples: list[DatasetSample]) -> dict[str, int]:
    """How many samples fall into each aspect-ratio bucket, most common first."""
    counter = Counter(
        classify_aspect_ratio(s.metrics.aspect_ratio) for s in samples if not s.is_excluded
    )
    return dict(counter.most_common())


def orientation_distribution(samples: list[DatasetSample]) -> dict[str, int]:
    """How many samples are landscape, portrait, or square.

    A coarser rollup than `aspect_ratio_distribution`: the bucket breakdown answers
    "which ratios do I have", this one answers "is the set balanced".
    """
    counter = Counter(
        classify_orientation(s.metrics.aspect_ratio) for s in samples if not s.is_excluded
    )
    return dict(counter.most_common())


BLURRIEST_SAMPLE_COUNT = 6


def blurriest_samples(
    samples: list[DatasetSample], count: int = BLURRIEST_SAMPLE_COUNT
) -> list[DatasetSample]:
    """The least sharp active samples, softest first.

    Deliberately a ranking and not a threshold: the Laplacian variance depends on
    subject and texture as much as on focus, so "the softest N in this set" is a
    claim that holds, while "sharpness < X is blurry" is not.

    Runs ingested before sharpness was recorded carry 0.0 everywhere and produce no
    ranking at all. A single 0.0 inside a measured run is a real reading, though —
    a perfectly flat image scores exactly zero — so it must not be filtered out, or
    the softest image in the set would be the one hidden.
    """
    active = [s for s in samples if not s.is_excluded]
    if not any(s.metrics.sharpness > 0 for s in active):
        return []
    return sorted(active, key=lambda s: s.metrics.sharpness)[:count]


def median_sharpness(samples: list[DatasetSample]) -> float:
    """Median sharpness across active samples, or 0.0 when none was recorded."""
    scores = sorted(s.metrics.sharpness for s in samples if not s.is_excluded)
    if not scores:
        return 0.0
    middle = len(scores) // 2
    if len(scores) % 2:
        return round(scores[middle], 2)
    return round((scores[middle - 1] + scores[middle]) / 2, 2)


def samples_without_source_caption(samples: list[DatasetSample]) -> list[DatasetSample]:
    """Active samples whose source folder had no .txt file at all.

    Read off `original_caption`, which ingestion leaves empty in exactly that case —
    so this needs no disk access, unlike the mirror check for orphan .txt files.
    """
    return [s for s in samples if not s.is_excluded and not s.original_caption.strip()]


def quality_summary(
    samples: list[DatasetSample],
    trigger_word: str = "",
    threshold: int = DEFAULT_PHASH_THRESHOLD,
) -> dict[str, int]:
    """Headline counts for the quality dashboard."""
    active = [s for s in samples if not s.is_excluded]
    clusters = find_duplicate_clusters(samples, threshold)

    return {
        "total": len(samples),
        "active": len(active),
        "excluded": len(samples) - len(active),
        "duplicate_clusters": len(clusters),
        "duplicates": sum(len(c.duplicates) for c in clusters),
        "missing_caption": len(samples_missing_caption(samples, trigger_word)),
        "invalid": sum(1 for s in active if not s.is_valid),
    }
