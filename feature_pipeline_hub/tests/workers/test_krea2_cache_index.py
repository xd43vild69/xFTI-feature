"""Cache enumeration, orphan detection, and the validation holdout split.

The holdout split is the one that repays testing: it decides which images never receive
a gradient, and both failure modes are silent. Leaking a mirrored pair across the split
makes validation loss track training loss and look encouraging; holding out everything
leaves nothing to train on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.cache_index import (  # noqa: E402
    choose_holdout, entry_names, find_orphans, format_orphan_warning, has_entries,
    read_manifest,
)


def make_cache(directory: Path, names: list[str]) -> Path:
    for name in names:
        for suffix in ("_latent.pt", "_embed.pt", "_mask.pt"):
            (directory / f"{name}{suffix}").write_text("x")
    return directory


# ── enumeration ─────────────────────────────────────────────────────────────

def test_lists_sample_names_sorted(tmp_path: Path) -> None:
    make_cache(tmp_path, ["b_002", "a_001", "c_003"])
    assert entry_names(tmp_path) == ["a_001", "b_002", "c_003"]


def test_ignores_dotfiles(tmp_path: Path) -> None:
    """A syncing client's partial download must not become a training sample."""
    make_cache(tmp_path, ["real"])
    (tmp_path / ".partial_latent.pt").write_text("x")
    assert entry_names(tmp_path) == ["real"]


def test_ignores_non_latent_files(tmp_path: Path) -> None:
    make_cache(tmp_path, ["real"])
    (tmp_path / "cache_manifest.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("x")
    assert entry_names(tmp_path) == ["real"]


def test_has_entries_detects_a_populated_cache(tmp_path: Path) -> None:
    assert not has_entries(tmp_path)
    assert not has_entries(str(tmp_path / "missing"))
    make_cache(tmp_path, ["a"])
    assert has_entries(tmp_path)


# ── manifest ────────────────────────────────────────────────────────────────

def test_absent_manifest_reads_as_no_opinion(tmp_path: Path) -> None:
    """None must not be confused with an empty manifest, which orphans everything."""
    assert read_manifest(str(tmp_path)) is None


def test_unreadable_manifest_reads_as_no_opinion(tmp_path: Path) -> None:
    (tmp_path / "cache_manifest.json").write_text("{broken")
    assert read_manifest(str(tmp_path)) is None


def test_empty_manifest_is_an_opinion(tmp_path: Path) -> None:
    (tmp_path / "cache_manifest.json").write_text(json.dumps({"entries": {}}))
    assert read_manifest(str(tmp_path)) == set()


# ── orphans ─────────────────────────────────────────────────────────────────

def test_no_manifest_means_no_orphan_check(tmp_path: Path) -> None:
    assert find_orphans(["a", "b"], str(tmp_path)) is None


def test_finds_entries_missing_from_the_manifest(tmp_path: Path) -> None:
    (tmp_path / "cache_manifest.json").write_text(
        json.dumps({"entries": {"kept": {}}}))
    assert find_orphans(["kept", "deleted"], str(tmp_path)) == ["deleted"]


def test_mirrored_variants_are_matched_by_source_stem(tmp_path: Path) -> None:
    """The manifest lists source images; flips are derived and never listed."""
    (tmp_path / "cache_manifest.json").write_text(
        json.dumps({"entries": {"img_001": {}}}))
    assert find_orphans(["img_001", "img_001__flip"], str(tmp_path)) == []


def test_orphan_warning_truncates_a_long_list() -> None:
    text = format_orphan_warning([f"img_{i}" for i in range(25)], shown=10)
    assert "25 cached entries" in text
    assert "... and 15 more" in text
    assert "img_9" in text and "img_24" not in text


def test_orphan_warning_says_they_are_still_training() -> None:
    """The warning has to be unambiguous: these images still get gradient."""
    assert "ARE being trained on" in format_orphan_warning(["a"])


# ── holdout split ───────────────────────────────────────────────────────────

def test_no_split_when_disabled() -> None:
    assert choose_holdout([f"img_{i}" for i in range(10)], 0.0) == []
    assert choose_holdout([f"img_{i}" for i in range(10)], -1.0) == []


def test_empty_dataset_splits_to_nothing() -> None:
    assert choose_holdout([], 0.2) == []


def test_split_is_deterministic() -> None:
    """A resumed run must hold out the same images, or the curve stops being comparable."""
    names = [f"img_{i:03d}" for i in range(20)]
    assert choose_holdout(names, 0.25) == choose_holdout(list(reversed(names)), 0.25)


@pytest.mark.parametrize("split,expected", [
    (0.5, 10),   # stride 2
    (0.25, 5),   # stride 4
    (0.1, 2),    # stride 10
])
def test_split_size_follows_the_ratio(split: float, expected: int) -> None:
    assert len(choose_holdout([f"img_{i:03d}" for i in range(20)], split)) == expected


def test_mirrored_pairs_stay_on_the_same_side() -> None:
    """A flipped copy of a held-out image is not an independent validation sample."""
    names = []
    for i in range(10):
        names += [f"img_{i:03d}", f"img_{i:03d}__flip"]
    holdout = set(choose_holdout(names, 0.25))
    for i in range(10):
        base, flip = f"img_{i:03d}", f"img_{i:03d}__flip"
        assert (base in holdout) == (flip in holdout), f"pair {i} was split"


def test_holdout_never_swallows_the_whole_dataset() -> None:
    """Returning [] means "validation off" — better than leaving nothing to train on."""
    assert choose_holdout(["only_one"], 0.9) == []


@pytest.mark.parametrize("split", [0.5, 0.75, 0.9, 1.0, 2.0])
def test_at_most_half_the_dataset_is_ever_held_out(split: float) -> None:
    """`stride` has a floor of 2, so val_split is effectively capped at 50%.

    Worth knowing before trusting the setting: asking for 0.9 does not give a 90%
    holdout, it gives 50%.
    """
    names = [f"img_{i:03d}" for i in range(20)]
    assert len(choose_holdout(names, split)) <= len(names) // 2


def test_holdout_leaves_training_images_behind() -> None:
    names = [f"img_{i:03d}" for i in range(20)]
    holdout = choose_holdout(names, 0.5)
    assert 0 < len(holdout) < len(names)


def test_holdout_is_a_subset_of_the_input() -> None:
    names = [f"img_{i:03d}" for i in range(20)]
    assert set(choose_holdout(names, 0.3)) <= set(names)
