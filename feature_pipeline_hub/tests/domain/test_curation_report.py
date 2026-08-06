"""Building curation_report.json from tiers an operator assigned by hand.

The encoding matters because it has to make krea2.curation.load_weights resolve
the groups intended, not merely produce valid JSON — tests/workers/test_krea2_curation.py
pins that half of the contract; this file pins the encoding itself.
"""

import pytest

from feature_pipeline.domain.curation_report import (
    CurationReportError,
    WeightProfile,
    build_curation_report,
    is_effective,
    resolved_weights,
    tier_counts,
)

FILES = {"hero": "hero.png", "good1": "good1.png", "bad1": "bad1.png"}


def test_every_exported_stem_gets_an_entry() -> None:
    report = build_curation_report(FILES, {"hero": "priority"}, WeightProfile())
    assert set(report.images) == set(FILES)


def test_unassigned_stems_default_to_good() -> None:
    report = build_curation_report(FILES, {"hero": "priority"}, WeightProfile())
    assert report.images["good1"].score == 1.0
    assert report.images["bad1"].score == 1.0  # unassigned, not "bad"


def test_bad_tier_scores_below_threshold() -> None:
    report = build_curation_report(FILES, {"bad1": "bad"}, WeightProfile())
    assert report.images["bad1"].score == 0.0
    assert report.auto_threshold == 0.5


def test_priority_tier_lands_in_baselines_by_filename() -> None:
    report = build_curation_report(FILES, {"hero": "priority"}, WeightProfile())
    assert report.baselines == ["hero.png"]
    assert report.images["hero"].file == "hero.png"


def test_stem_containing_a_dot_is_rejected() -> None:
    with pytest.raises(CurationReportError):
        build_curation_report({"a.b": "a.b.png"}, {}, WeightProfile())


def test_tier_for_a_stem_that_was_not_exported_is_rejected() -> None:
    with pytest.raises(CurationReportError):
        build_curation_report(FILES, {"not_exported": "bad"}, WeightProfile())


def test_is_effective_false_when_profile_is_untouched() -> None:
    report = build_curation_report(FILES, {}, WeightProfile())
    assert is_effective(report) is False


def test_is_effective_true_when_a_tier_differs() -> None:
    report = build_curation_report(FILES, {"bad1": "bad"}, WeightProfile())
    assert is_effective(report) is True


def test_is_effective_false_even_with_assignments_if_profile_values_are_all_one() -> None:
    """A priority/bad tier assigned, but with weights that make no difference."""
    report = build_curation_report(
        FILES, {"hero": "priority", "bad1": "bad"}, WeightProfile(priority=1.0, bad=1.0)
    )
    assert is_effective(report) is False


def test_resolved_weights_matches_the_profile_per_tier() -> None:
    report = build_curation_report(
        FILES, {"hero": "priority", "bad1": "bad"},
        WeightProfile(priority=2.0, good=1.0, bad=0.25),
    )
    assert resolved_weights(report) == {"hero": 2.0, "good1": 1.0, "bad1": 0.25}


def test_tier_counts_counts_unassigned_as_good() -> None:
    counts = tier_counts({"hero": "priority", "bad1": "bad"}, total=3)
    assert counts == {"priority": 1, "good": 1, "bad": 1}


def test_tier_counts_with_no_assignments() -> None:
    assert tier_counts({}, total=4) == {"priority": 0, "good": 4, "bad": 0}
