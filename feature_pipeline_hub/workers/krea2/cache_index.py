"""Reasoning about *which* cached samples exist, before any of them is loaded.

Enumeration, orphan detection and the validation split are decisions about names, not
tensors — so they live apart from `dataset.py` and stay covered by mypy and CI. The
holdout split in particular is worth testing: it decides which images never receive a
gradient, and getting it wrong is invisible except as a validation curve that quietly
tracks the training one.
"""
from __future__ import annotations

import json
import math
import os

from krea2.curation import source_stem

LATENT_SUFFIX = "_latent.pt"


def entry_names(directory: str) -> list[str]:
    """Sorted names of every cached sample in `directory`, from its `*_latent.pt` files.

    Dotfiles are skipped so an editor swapfile or a syncing client's partial download
    never becomes a training sample.
    """
    return sorted(name[:-len(LATENT_SUFFIX)] for name in os.listdir(directory)
                  if not name.startswith(".") and name.endswith(LATENT_SUFFIX))


def has_entries(directory: str) -> bool:
    """True when `directory` holds at least one cached latent."""
    return os.path.isdir(directory) and any(
        name.endswith(LATENT_SUFFIX) for name in os.listdir(directory))


def read_manifest(cache_dir: str) -> set[str] | None:
    """Names recorded in cache_manifest.json, or None when it is absent or unreadable.

    None means "no opinion" and must not be confused with an empty manifest, which would
    mark every cached entry an orphan.
    """
    path = os.path.join(cache_dir, "cache_manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return set(json.load(handle).get("entries", {}))
    except Exception:
        return None


def find_orphans(cache_names: list[str], cache_dir: str) -> list[str] | None:
    """Cached entries missing from the manifest, or None when there is no manifest.

    These are latents whose source image was deleted or renamed. Without a manifest they
    were never noticed and kept being trained on. Compared by source stem, since mirrored
    variants are derived from an entry rather than listed as their own.
    """
    listed = read_manifest(cache_dir)
    if listed is None:
        return None
    return sorted(name for name in cache_names if source_stem(name) not in listed)


def choose_holdout(names: list[str], val_split: float) -> list[str]:
    """Pick a deterministic validation slice, keeping original/flip pairs together.

    Deterministic over sorted names rather than random, so a resumed run holds out the
    same images and its validation curve stays comparable across restarts. A pair split
    across both sides would leak: the mirrored copy of a held-out image is not an
    independent sample.

    Returns [] when the split would consume the whole dataset, leaving nothing to train
    on — the caller treats that as "validation off", not as an error.
    """
    if val_split <= 0 or not names:
        return []
    ordered = sorted(names)
    stride = max(2, math.ceil(1.0 / val_split))
    picked = {source_stem(n) for n in ordered[::stride]}
    holdout = [n for n in ordered if source_stem(n) in picked]
    return [] if len(holdout) >= len(ordered) else holdout


def format_orphan_warning(orphans: list[str], shown: int = 10) -> str:
    """The startup warning naming orphaned cache entries, truncated to `shown`."""
    lines = [f"\n[!] {len(orphans)} cached entries are not in cache_manifest.json "
             f"(deleted or renamed source images?) / no están en el manifiesto:"]
    lines += [f"      {name}" for name in orphans[:shown]]
    if len(orphans) > shown:
        lines.append(f"      ... and {len(orphans) - shown} more")
    lines.append("[!] They ARE being trained on. Re-run Pre-Cache with "
                 "prune_orphans='delete' to remove them / SE están entrenando.")
    return "\n".join(lines)
