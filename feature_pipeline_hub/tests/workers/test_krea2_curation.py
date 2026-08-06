"""Curation weighting: which images get more gradient, and when the feature disappears.

Scaling a sample's loss is a per-image learning rate, so a wrong weight here degrades a
LoRA without ever failing. The most important property is the collapse to `(None, None)`
— with curation absent or neutral, training must be bit-for-bit what it was before the
feature existed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.curation import (  # noqa: E402
    format_summary, load_weights, resolve_group, source_stem,
)

REPORT = {
    "mode": "face",
    "auto_threshold": 0.6,
    "weights": {"priority": 1.5, "good": 1.0, "bad": 0.5},
    "baselines": ["hero.png"],
    "images": {
        "hero":  {"file": "hero.png",  "score": 0.9},
        "good1": {"file": "good1.png", "score": 0.8},
        "bad1":  {"file": "bad1.png",  "score": 0.2},
        "edge":  {"file": "edge.png",  "score": 0.6},
    },
}


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "curation_report.json").write_text(json.dumps(REPORT))
    return tmp_path


def weights(directory: Path, names: list[str], **kwargs: object) -> dict[str, float] | None:
    result, _ = load_weights(directory, names, log=lambda _m: None, **kwargs)  # type: ignore[arg-type]
    return result


# ── group resolution ────────────────────────────────────────────────────────

def test_score_at_the_threshold_counts_as_good() -> None:
    assert resolve_group(0.6, 0.6, None) == "good"
    assert resolve_group(0.59, 0.6, None) == "bad"


def test_override_beats_the_score() -> None:
    assert resolve_group(0.1, 0.6, "good") == "good"
    assert resolve_group(0.9, 0.6, "bad") == "bad"


def test_unrecognized_override_is_ignored() -> None:
    assert resolve_group(0.9, 0.6, "excellent") == "good"


@pytest.mark.parametrize("mode,expected", [("face", "good"), ("style", "bad")])
def test_missing_data_falls_back_by_mode(mode: str, expected: str) -> None:
    """Face datasets keep unscored images; style datasets set them aside."""
    assert resolve_group(None, 0.6, None, mode=mode) == expected
    assert resolve_group(0.9, None, None, mode=mode) == expected


# ── flip variants ───────────────────────────────────────────────────────────

def test_source_stem_strips_only_the_flip_suffix() -> None:
    assert source_stem("img_001__flip") == "img_001"
    assert source_stem("img_001") == "img_001"
    assert source_stem("my__flipper") == "my__flipper"


def test_mirrored_variants_inherit_the_source_weight(dataset: Path) -> None:
    """Curation scored the source image; without this half the dataset trains at 1.0."""
    result = weights(dataset, ["bad1", "bad1__flip"])
    assert result == {"bad1": 0.5, "bad1__flip": 0.5}


# ── weighting ───────────────────────────────────────────────────────────────

def test_baselines_get_priority_weight(dataset: Path) -> None:
    assert weights(dataset, ["hero"]) == {"hero": 1.5}


def test_good_and_bad_split_on_the_threshold(dataset: Path) -> None:
    assert weights(dataset, ["good1", "bad1", "edge"]) == {
        "good1": 1.0, "bad1": 0.5, "edge": 1.0}


def test_images_absent_from_the_report_train_at_full_weight(dataset: Path) -> None:
    """Never penalize an image for missing data — it was just added after the last scan."""
    assert weights(dataset, ["bad1", "brand_new"])["brand_new"] == 1.0


# ── overrides ───────────────────────────────────────────────────────────────

def test_manual_threshold_replaces_the_automatic_one(dataset: Path) -> None:
    (dataset / "curation_overrides.json").write_text(json.dumps({"threshold": 0.85}))
    result = weights(dataset, ["good1", "edge"])
    assert result == {"good1": 0.5, "edge": 0.5}


def test_group_override_moves_a_single_image(dataset: Path) -> None:
    # Two moves, not one: promoting bad1 alone would leave every weight at 1.0, and
    # an all-neutral result collapses to None by design.
    (dataset / "curation_overrides.json").write_text(
        json.dumps({"groups": {"bad1": "good", "good1": "bad"}}))
    assert weights(dataset, ["bad1", "good1"]) == {"bad1": 1.0, "good1": 0.5}


def test_baseline_priority_outranks_a_group_override(dataset: Path) -> None:
    (dataset / "curation_overrides.json").write_text(
        json.dumps({"groups": {"hero": "bad"}}))
    assert weights(dataset, ["hero"]) == {"hero": 1.5}


def test_unreadable_overrides_fall_back_to_the_automatic_threshold(dataset: Path) -> None:
    (dataset / "curation_overrides.json").write_text("{broken")
    assert weights(dataset, ["good1", "bad1"]) == {"good1": 1.0, "bad1": 0.5}


# ── the no-op paths ─────────────────────────────────────────────────────────

def test_disabled_returns_nothing(dataset: Path) -> None:
    assert weights(dataset, ["hero"], enabled=False) is None


def test_missing_report_returns_nothing(tmp_path: Path) -> None:
    assert weights(tmp_path, ["a"]) is None


def test_unreadable_report_returns_nothing(tmp_path: Path) -> None:
    (tmp_path / "curation_report.json").write_text("{broken")
    assert weights(tmp_path, ["a"]) is None


def test_empty_image_table_returns_nothing(tmp_path: Path) -> None:
    (tmp_path / "curation_report.json").write_text(json.dumps({"images": {}}))
    assert weights(tmp_path, ["a"]) is None


def test_all_neutral_weights_collapse(tmp_path: Path) -> None:
    """Weights that are all 1.0 must vanish, so the loss path stays untouched."""
    (tmp_path / "curation_report.json").write_text(json.dumps({
        "mode": "face", "auto_threshold": 0.5,
        "weights": {"priority": 1.0, "good": 1.0, "bad": 1.0},
        "images": {"a": {"file": "a.png", "score": 0.9}},
    }))
    assert weights(tmp_path, ["a"]) is None


# ── summary ─────────────────────────────────────────────────────────────────

def test_summary_counts_each_group(dataset: Path) -> None:
    _, summary = load_weights(dataset, ["hero", "good1", "bad1", "edge"],
                              log=lambda _m: None)
    assert summary is not None
    assert (summary["priority"], summary["good"], summary["bad"]) == (1, 2, 1)
    assert summary["threshold"] == 0.6


def test_format_summary_renders_an_automatic_threshold() -> None:
    text = format_summary({"priority": 0, "good": 3, "bad": 1, "w_priority": 1.5,
                           "w_good": 1.0, "w_bad": 0.5, "threshold": None})
    assert "auto" in text
    assert "Alta Prioridad" not in text  # hidden when there are no baselines


def test_format_summary_renders_a_percentage_and_baselines() -> None:
    text = format_summary({"priority": 2, "good": 3, "bad": 1, "w_priority": 1.5,
                           "w_good": 1.0, "w_bad": 0.5, "threshold": 0.6})
    assert "60%" in text
    assert "Alta Prioridad" in text


# ── hub-built reports feed load_weights correctly ──────────────────────────
# The hub (feature_pipeline.domain.curation_report) builds curation_report.json
# synthetically, from tiers an operator picked rather than a score. These tests
# run both sides — the hub's writer and this module's reader — in one process to
# pin that a hub-built report resolves exactly as build_curation_report/
# resolved_weights predict. Import is local to keep the two packages' sys.path
# setup (this file inserts workers/ above) from leaking into the rest of the
# domain test suite.

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from feature_pipeline.domain.curation_report import (  # noqa: E402
    WeightProfile, build_curation_report, resolved_weights,
)


def test_hub_report_resolves_to_the_same_weights_predicted_by_resolved_weights(
    tmp_path: Path,
) -> None:
    files_by_stem = {
        "hero": "hero.png", "good1": "good1.png", "bad1": "bad1.png",
    }
    tiers = {"hero": "priority", "bad1": "bad"}
    profile = WeightProfile(priority=1.5, good=1.0, bad=0.5)
    report = build_curation_report(files_by_stem, tiers, profile)
    (tmp_path / "curation_report.json").write_text(report.model_dump_json())

    from_trainer = weights(tmp_path, ["hero", "good1", "bad1"])
    assert from_trainer == resolved_weights(report)
    assert from_trainer == {"hero": 1.5, "good1": 1.0, "bad1": 0.5}


def test_hub_report_flip_variant_inherits_the_source_tier(tmp_path: Path) -> None:
    files_by_stem = {"img_0001": "img_0001.png"}
    tiers = {"img_0001": "bad"}
    report = build_curation_report(files_by_stem, tiers, WeightProfile())
    (tmp_path / "curation_report.json").write_text(report.model_dump_json())

    result = weights(tmp_path, ["img_0001", "img_0001__flip"])
    assert result == {"img_0001": 0.5, "img_0001__flip": 0.5}


def test_hub_report_with_default_profile_collapses_to_none(tmp_path: Path) -> None:
    """An all-good, untouched-profile report must be a true no-op for the trainer."""
    files_by_stem = {"a": "a.png", "b": "b.png"}
    report = build_curation_report(files_by_stem, {}, WeightProfile())
    (tmp_path / "curation_report.json").write_text(report.model_dump_json())

    assert weights(tmp_path, ["a", "b"]) is None
