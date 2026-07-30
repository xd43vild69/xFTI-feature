from feature_pipeline.application.quality_service import (
    CAPTION_WORD_WARNING,
    TEXT_ENCODER_MAX_TOKENS,
    TOKENS_PER_WORD,
    aspect_ratio_distribution,
    blurriest_samples,
    caption_length_stats,
    caption_word_count,
    describes_nothing,
    find_duplicate_clusters,
    median_sharpness,
    orientation_distribution,
    perceptual_distance,
    quality_summary,
    resolution_distribution,
    samples_missing_caption,
    samples_without_source_caption,
    stored_quality_summary,
)
from feature_pipeline.domain.models import DatasetSample, ImageMetrics

# imagehash parses these as 64-bit hex hashes; they differ by a controlled number of bits.
HASH_A = "0000000000000000"
HASH_A_1BIT = "0000000000000001"
HASH_A_3BITS = "0000000000000007"
HASH_FAR = "ffffffffffffffff"


def _sample(
    sample_id: str,
    phash: str = HASH_A,
    dhash: str = "",
    colorhash: str = "",
    caption: str = "sks_style, a description",
    original_caption: str = "a description",
    width: int = 512,
    height: int = 512,
    sharpness: float = 0.0,
    is_excluded: bool = False,
    is_valid: bool = True,
    is_duplicate: bool = False,
) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        image_path=f"/data/{sample_id}.png",
        caption=caption,
        original_caption=original_caption,
        metrics=ImageMetrics(
            width=width,
            height=height,
            aspect_ratio=width / height,
            format="PNG",
            phash=phash,
            dhash=dhash,
            colorhash=colorhash,
            sharpness=sharpness,
        ),
        is_excluded=is_excluded,
        is_valid=is_valid,
        is_duplicate=is_duplicate,
    )


# --- sharpness, orientation, unpaired files ----------------------------------


def test_blurriest_samples_ranks_softest_first():
    samples = [
        _sample("sharp", sharpness=900.0),
        _sample("soft", sharpness=12.0),
        _sample("middling", sharpness=300.0),
    ]

    assert [s.sample_id for s in blurriest_samples(samples)] == ["soft", "middling", "sharp"]


def test_blurriest_samples_skips_excluded_samples():
    samples = [
        _sample("dropped", sharpness=5.0, is_excluded=True),
        _sample("soft", sharpness=40.0),
    ]

    assert [s.sample_id for s in blurriest_samples(samples)] == ["soft"]


def test_a_run_ingested_before_sharpness_existed_produces_no_ranking():
    samples = [_sample("a", sharpness=0.0), _sample("b", sharpness=0.0)]

    assert blurriest_samples(samples) == []


def test_a_flat_image_inside_a_measured_run_ranks_softest():
    """Zero is a real reading for a uniform image, not the 'unmeasured' sentinel."""
    samples = [_sample("flat", sharpness=0.0), _sample("detailed", sharpness=900.0)]

    assert [s.sample_id for s in blurriest_samples(samples)] == ["flat", "detailed"]


def test_median_sharpness_of_an_even_number_of_samples_averages_the_middle():
    samples = [_sample(str(i), sharpness=v) for i, v in enumerate([10.0, 20.0, 30.0, 40.0])]

    assert median_sharpness(samples) == 25.0


def test_median_sharpness_without_samples_is_zero():
    assert median_sharpness([]) == 0.0


def test_orientation_distribution_rolls_ratios_up_to_three_buckets():
    samples = [
        _sample("wide", width=1600, height=900),
        _sample("tall", width=900, height=1600),
        _sample("square", width=512, height=512),
        _sample("also_wide", width=1024, height=768),
    ]

    assert orientation_distribution(samples) == {"landscape": 2, "portrait": 1, "square": 1}


def test_samples_without_source_caption_reads_the_original_not_the_effective_caption():
    """Ingestion injects the trigger word even with no .txt, so `caption` is never empty."""
    samples = [
        _sample("had_txt", original_caption="a red car"),
        _sample("no_txt", caption="sks_style", original_caption=""),
        _sample("dropped", original_caption="", is_excluded=True),
    ]

    assert [s.sample_id for s in samples_without_source_caption(samples)] == ["no_txt"]


def test_identical_hashes_have_zero_distance():
    assert perceptual_distance(_sample("a"), _sample("b")) == 0


def test_dhash_tightens_the_distance_when_both_samples_have_one():
    left = _sample("a", phash=HASH_A, dhash=HASH_A)
    right = _sample("b", phash=HASH_A, dhash=HASH_FAR)

    # pHash alone says identical; dHash disagrees, and the stricter signal wins.
    assert perceptual_distance(left, right) == 64


def test_missing_dhash_falls_back_to_phash_only():
    left = _sample("a", phash=HASH_A, dhash=HASH_A)
    right = _sample("b", phash=HASH_A, dhash="")

    assert perceptual_distance(left, right) == 0


def test_finds_a_cluster_of_near_identical_images():
    samples = [
        _sample("a", phash=HASH_A),
        _sample("b", phash=HASH_A_1BIT),
        _sample("c", phash=HASH_FAR),
    ]

    clusters = find_duplicate_clusters(samples, threshold=5)

    assert len(clusters) == 1
    assert clusters[0].kept.sample_id == "a"
    assert [s.sample_id for s, _ in clusters[0].duplicates] == ["b"]
    assert clusters[0].size == 2


def test_threshold_zero_only_matches_identical_hashes():
    samples = [_sample("a", phash=HASH_A), _sample("b", phash=HASH_A_1BIT)]

    assert find_duplicate_clusters(samples, threshold=0) == []
    assert len(find_duplicate_clusters(samples, threshold=1)) == 1


def test_distance_is_reported_per_duplicate():
    samples = [
        _sample("a", phash=HASH_A),
        _sample("b", phash=HASH_A_3BITS),
    ]

    cluster = find_duplicate_clusters(samples, threshold=5)[0]

    assert cluster.duplicates[0][1] == 3


def test_a_sample_belongs_to_a_single_cluster():
    samples = [_sample(sid, phash=HASH_A) for sid in ("a", "b", "c")]

    clusters = find_duplicate_clusters(samples, threshold=5)

    assert len(clusters) == 1
    assert len(clusters[0].duplicates) == 2


def test_excluded_samples_are_ignored_when_clustering():
    samples = [
        _sample("a", phash=HASH_A),
        _sample("b", phash=HASH_A, is_excluded=True),
    ]

    assert find_duplicate_clusters(samples, threshold=5) == []


def test_unique_images_produce_no_clusters():
    samples = [_sample("a", phash=HASH_A), _sample("b", phash=HASH_FAR)]

    assert find_duplicate_clusters(samples, threshold=5) == []


def test_caption_with_only_the_trigger_word_describes_nothing():
    assert describes_nothing("sks_style", "sks_style") is True
    assert describes_nothing("sks_style, ", "sks_style") is True
    assert describes_nothing("", "sks_style") is True
    assert describes_nothing("sks_style, a red car", "sks_style") is False


def test_missing_caption_clears_once_a_description_is_typed():
    empty = _sample("a", caption="sks_style", original_caption="")
    described = _sample("b", caption="sks_style, a red car", original_caption="")

    pending = samples_missing_caption([empty, described], trigger_word="sks_style")

    assert [s.sample_id for s in pending] == ["a"]


def test_missing_caption_skips_excluded_samples():
    excluded = _sample("a", caption="sks_style", original_caption="", is_excluded=True)

    assert samples_missing_caption([excluded], trigger_word="sks_style") == []


def test_caption_word_count_counts_words():
    assert caption_word_count("sks_style, a red car") == 4
    assert caption_word_count("") == 0


def test_caption_length_stats_summarises_active_samples():
    samples = [
        _sample("a", caption="one two"),
        _sample("b", caption="one two three four"),
        _sample("c", caption="ignored because excluded " * 30, is_excluded=True),
    ]

    stats = caption_length_stats(samples)

    assert stats["min"] == 2
    assert stats["max"] == 4
    assert stats["mean"] == 3.0
    assert stats["too_long"] == 0


def test_caption_length_stats_flags_captions_that_risk_truncation():
    samples = [_sample("a", caption="word " * (CAPTION_WORD_WARNING + 5))]

    assert caption_length_stats(samples)["too_long"] == 1


def test_the_truncation_warning_tracks_the_text_encoder_budget():
    """The threshold is derived from max_seq_len, not from CLIP's unrelated 77."""
    assert CAPTION_WORD_WARNING == int(TEXT_ENCODER_MAX_TOKENS / TOKENS_PER_WORD)
    assert caption_length_stats([_sample("a", caption="word " * 60)])["too_long"] == 0


def test_caption_length_stats_handles_an_empty_dataset():
    assert caption_length_stats([])["mean"] == 0.0


def test_resolution_distribution_counts_by_size_most_common_first():
    samples = [
        _sample("a", width=512, height=512),
        _sample("b", width=512, height=512),
        _sample("c", width=1024, height=768),
    ]

    assert resolution_distribution(samples) == {"512x512": 2, "1024x768": 1}


def test_aspect_ratio_distribution_uses_named_buckets():
    samples = [
        _sample("a", width=512, height=512),
        _sample("b", width=1024, height=768),
    ]

    assert aspect_ratio_distribution(samples) == {"1:1": 1, "4:3": 1}


def test_distributions_ignore_excluded_samples():
    samples = [
        _sample("a", width=512, height=512),
        _sample("b", width=1024, height=768, is_excluded=True),
    ]

    assert resolution_distribution(samples) == {"512x512": 1}
    assert aspect_ratio_distribution(samples) == {"1:1": 1}


def test_quality_summary_reports_headline_counts():
    samples = [
        _sample("a", phash=HASH_A),
        _sample("b", phash=HASH_A),
        _sample("c", phash=HASH_FAR, caption="sks_style", original_caption=""),
        _sample("d", phash=HASH_FAR, is_excluded=True),
        _sample("e", phash="00000000000000ff", is_valid=False),
    ]

    summary = quality_summary(samples, trigger_word="sks_style", threshold=5)

    assert summary["total"] == 5
    assert summary["active"] == 4
    assert summary["excluded"] == 1
    assert summary["duplicate_clusters"] == 1
    assert summary["duplicates"] == 1
    assert summary["missing_caption"] == 1
    assert summary["invalid"] == 1


# Real colour hashes: two plain squares of different colours, and a copy of the first.
COLOR_RED = "00000038000"
COLOR_BLUE = "00000000038"


def test_colour_guard_rejects_images_that_only_match_on_luminance():
    """Flat images hash the same under pHash/dHash regardless of colour."""
    red = _sample("a", phash=HASH_A, dhash=HASH_A, colorhash=COLOR_RED)
    blue = _sample("b", phash=HASH_A, dhash=HASH_A, colorhash=COLOR_BLUE)

    assert perceptual_distance(red, blue) == 64
    assert find_duplicate_clusters([red, blue], threshold=5) == []


def test_matching_colours_still_cluster():
    red = _sample("a", phash=HASH_A, dhash=HASH_A, colorhash=COLOR_RED)
    red_copy = _sample("b", phash=HASH_A, dhash=HASH_A, colorhash=COLOR_RED)

    assert perceptual_distance(red, red_copy) == 0
    assert len(find_duplicate_clusters([red, red_copy], threshold=5)) == 1


def test_colour_guard_is_skipped_when_a_sample_predates_colour_hashing():
    red = _sample("a", phash=HASH_A, colorhash=COLOR_RED)
    legacy = _sample("b", phash=HASH_A, colorhash="")

    assert perceptual_distance(red, legacy) == 0


def test_stored_summary_never_reclusters():
    """The load-bearing guarantee: two samples that *would* cluster, but aren't flagged.

    quality_summary would report these as duplicates; the stored variant must not,
    because re-clustering is exactly the O(n²) cost it exists to avoid.
    """
    a = _sample("a", phash=HASH_A)
    b = _sample("b", phash=HASH_A_1BIT)

    assert len(find_duplicate_clusters([a, b], threshold=5)) == 1
    assert stored_quality_summary([a, b])["duplicates"] == 0


def test_stored_summary_trusts_the_flag_over_the_hashes():
    far_but_flagged = _sample("a", phash=HASH_FAR, is_duplicate=True)
    near_but_unflagged = _sample("b", phash=HASH_A)

    assert stored_quality_summary([far_but_flagged, near_but_unflagged])["duplicates"] == 1


def test_stored_summary_counts_excluded_samples_only_as_excluded():
    active = _sample("a")
    excluded = _sample("b", is_excluded=True, is_valid=False, is_duplicate=True, caption="sks")

    summary = stored_quality_summary([active, excluded], "sks")

    assert summary["total"] == 2
    assert summary["active"] == 1
    assert summary["excluded"] == 1
    assert summary["duplicates"] == 0
    assert summary["invalid"] == 0
    assert summary["missing_caption"] == 0


def test_stored_summary_of_an_empty_set_is_all_zeros():
    assert stored_quality_summary([]) == {
        "total": 0,
        "active": 0,
        "excluded": 0,
        "duplicates": 0,
        "missing_caption": 0,
        "invalid": 0,
    }
