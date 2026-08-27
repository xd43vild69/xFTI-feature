"""Unit tests for ltx23.cache_index scanning and path resolution."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from ltx23.cache_index import (
    get_text_cache_paths,
    has_valid_cache,
    scan_cache_directory,
)


def test_get_text_cache_paths() -> None:
    video_p, audio_p = get_text_cache_paths("/tmp/cache", "sample_01", max_text_tokens=256)
    assert video_p == "/tmp/cache/sample_01_video_textmt256_reg128_v2.pt"
    assert audio_p == "/tmp/cache/sample_01_audio_textmt256_reg128_v2.pt"


def test_scan_cache_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    assert not has_valid_cache(str(cache_dir))

    # Create dummy video + audio files
    (cache_dir / "sample_01_video_latent.pt").write_bytes(b"dummy")
    (cache_dir / "sample_01_audio_latent.pt").write_bytes(b"dummy")
    (cache_dir / "sample_01_info.json").write_text("{}", encoding="utf-8")

    entries = scan_cache_directory(str(cache_dir))
    assert len(entries) == 1
    assert entries[0].name == "sample_01"
    assert has_valid_cache(str(cache_dir))
