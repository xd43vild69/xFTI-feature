"""Cache indexing and path resolution for LTX 2.3 multimodal cached data.

Free of torch, so this module can be tested and type-checked in the hub environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CacheEntry:
    """Metadata and paths for one cached sample."""

    name: str
    video_path: str
    audio_path: str
    info_path: str | None
    prompt_structure_path: str | None
    video_text_path: str
    audio_text_path: str


def get_text_cache_paths(cache_dir: str, base: str, max_text_tokens: int = 0) -> tuple[str, str]:
    """Derive cached precomputed connector output paths for a sample prefix."""
    prefix = base if base.endswith("_") else f"{base}_"
    tag = f"mt{int(max_text_tokens or 0)}_reg128_v2"
    video_text_path = os.path.join(cache_dir, f"{prefix}video_text{tag}.pt")
    audio_text_path = os.path.join(cache_dir, f"{prefix}audio_text{tag}.pt")
    return video_text_path, audio_text_path


def scan_cache_directory(cache_dir: str, max_text_tokens: int = 0) -> list[CacheEntry]:
    """Scan cache directory and return all valid cached entries for LTX 2.3."""
    if not os.path.isdir(cache_dir):
        return []

    entries: list[CacheEntry] = []
    files = sorted(os.listdir(cache_dir))

    for filename in files:
        if not filename.endswith("_video_latent.pt"):
            continue

        base = filename[:-len("_video_latent.pt")]
        video_path = os.path.join(cache_dir, filename)
        audio_path = os.path.join(cache_dir, f"{base}_audio_latent.pt")

        if not os.path.exists(audio_path):
            continue

        raw_info = os.path.join(cache_dir, f"{base}_info.json")
        info_path: str | None = raw_info if os.path.exists(raw_info) else None

        raw_structure = os.path.join(cache_dir, f"{base}_prompt_structure.json")
        prompt_structure_path: str | None = raw_structure if os.path.exists(raw_structure) else None

        video_text_path, audio_text_path = get_text_cache_paths(cache_dir, base, max_text_tokens)

        entries.append(
            CacheEntry(
                name=base,
                video_path=video_path,
                audio_path=audio_path,
                info_path=info_path,
                prompt_structure_path=prompt_structure_path,
                video_text_path=video_text_path,
                audio_text_path=audio_text_path,
            )
        )

    return entries


def has_valid_cache(cache_dir: str) -> bool:
    """Return True if the cache directory contains at least one complete entry."""
    return len(scan_cache_directory(cache_dir)) > 0
