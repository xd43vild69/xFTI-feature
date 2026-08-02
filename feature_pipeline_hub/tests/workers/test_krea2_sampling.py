"""Dataset samplers: coverage, batch integrity, and resume fidelity.

The properties worth defending are that no batch mixes resolution buckets (differing
shapes cannot be concatenated) and that a mid-run save/restore reproduces the exact
remaining sequence, since a resumed run silently re-drawing images is invisible in the
logs and only shows up as a worse LoRA hours later.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.sampling import (  # noqa: E402
    EpochSampler, LegacySampler, bias_warning, build_sampler,
)

BUCKETS = {
    (64, 64): [f"img_{i:03d}" for i in range(16)],
    (48, 80): [f"wide_{i:03d}" for i in range(5)],
    (80, 48): ["tall_000"],
}
TOTAL_IMAGES = sum(len(v) for v in BUCKETS.values())


# ── shared guarantees ───────────────────────────────────────────────────────

def test_batches_never_mix_buckets() -> None:
    for sampler in (EpochSampler(BUCKETS, 3, 1), LegacySampler(BUCKETS, 3, 1)):
        for _ in range(80):
            size, names = sampler.next()
            assert all(name in BUCKETS[size] for name in names), type(sampler).__name__


def test_batch_size_is_always_honored() -> None:
    for batch_size in (1, 2, 5):
        sampler = EpochSampler(BUCKETS, batch_size, 7)
        assert all(len(sampler.next()[1]) == batch_size for _ in range(40))


def test_same_seed_reproduces_the_same_stream() -> None:
    a = [EpochSampler(BUCKETS, 2, 99).next() for _ in range(20)]
    b = [EpochSampler(BUCKETS, 2, 99).next() for _ in range(20)]
    assert a == b


def test_different_seeds_diverge() -> None:
    a = [EpochSampler(BUCKETS, 1, 1).next() for _ in range(30)]
    b = [EpochSampler(BUCKETS, 1, 2).next() for _ in range(30)]
    assert a != b


# ── EpochSampler: the coverage guarantee ────────────────────────────────────

def test_one_epoch_covers_every_image_exactly_once() -> None:
    """The reason this class exists — the legacy sampler cannot promise this."""
    sampler = EpochSampler(BUCKETS, 1, 42)
    seen: Counter[str] = Counter()
    while sampler.epoch <= 1:
        _, names = sampler.next()
        if sampler.epoch > 1:
            break
        seen.update(names)
    assert set(seen) == {n for names in BUCKETS.values() for n in names}
    assert set(seen.values()) == {1}


def test_epoch_advances_only_when_the_queue_drains() -> None:
    sampler = EpochSampler(BUCKETS, 1, 42)
    assert sampler.epoch == 0
    sampler.next()
    assert sampler.epoch == 1
    for _ in range(TOTAL_IMAGES - 1):
        sampler.next()
    assert sampler.epoch == 1
    sampler.next()
    assert sampler.epoch == 2


def test_short_final_batch_is_padded_from_its_own_bucket() -> None:
    """Padding from another bucket would produce a batch torch.cat cannot build."""
    sampler = EpochSampler({(64, 64): ["a", "b", "c"]}, 2, 3)
    for _ in range(10):
        size, names = sampler.next()
        assert len(names) == 2
        assert all(n in ("a", "b", "c") for n in names)


def test_repeats_multiply_an_image_within_the_epoch() -> None:
    sampler = EpochSampler({(64, 64): ["a", "b"]}, 1, 5, repeats={"a": 3})
    seen: Counter[str] = Counter()
    for _ in range(4):
        seen.update(sampler.next()[1])
    assert seen == Counter({"a": 3, "b": 1})


def test_larger_buckets_receive_proportionally_more_steps() -> None:
    """The bias EpochSampler was written to remove."""
    sampler = EpochSampler(BUCKETS, 1, 11)
    seen: Counter[str] = Counter()
    for _ in range(TOTAL_IMAGES * 4):
        seen.update(sampler.next()[1])
    assert seen["img_000"] == seen["tall_000"]


# ── LegacySampler: kept only to reproduce old runs ──────────────────────────

def test_legacy_samples_buckets_uniformly() -> None:
    """A 1-image bucket gets the same mass as a 16-image one — the documented bug."""
    sampler = LegacySampler(BUCKETS, 1, 3)
    seen: Counter[str] = Counter()
    for _ in range(3000):
        seen.update(sampler.next()[1])
    # tall_000 is alone in its bucket, so it takes that bucket's whole third.
    assert seen["tall_000"] > seen["img_000"] * 10


def test_legacy_has_no_epoch_concept() -> None:
    sampler = LegacySampler(BUCKETS, 1, 3)
    for _ in range(50):
        sampler.next()
    assert sampler.epoch == 0


# ── resume ──────────────────────────────────────────────────────────────────

def test_restored_sampler_reproduces_the_remaining_sequence() -> None:
    for make in (lambda: EpochSampler(BUCKETS, 2, 21),
                 lambda: LegacySampler(BUCKETS, 2, 21)):
        original = make()
        for _ in range(25):
            original.next()
        state = original.state_dict()
        expected = [original.next() for _ in range(20)]

        restored = make()
        restored.load_state_dict(state)
        assert [restored.next() for _ in range(20)] == expected


def test_epoch_survives_a_restore() -> None:
    sampler = EpochSampler(BUCKETS, 1, 8)
    for _ in range(TOTAL_IMAGES + 3):
        sampler.next()
    restored = EpochSampler(BUCKETS, 1, 8)
    restored.load_state_dict(sampler.state_dict())
    assert restored.epoch == sampler.epoch == 2


def test_state_is_json_safe() -> None:
    """torch.save(weights_only=True) refuses arbitrary objects, so state must be plain."""
    import json

    state = EpochSampler(BUCKETS, 2, 4).state_dict()
    json.dumps(state)
    assert isinstance(state["rng"], str)
    assert isinstance(state["queue"], str)


# ── helpers ─────────────────────────────────────────────────────────────────

def test_build_sampler_selects_by_name() -> None:
    assert isinstance(build_sampler("epoch", BUCKETS, 1, 0), EpochSampler)
    assert isinstance(build_sampler("legacy", BUCKETS, 1, 0), LegacySampler)
    assert isinstance(build_sampler("anything-else", BUCKETS, 1, 0), LegacySampler)


def test_bias_warning_reports_the_real_ratio() -> None:
    warning = bias_warning(BUCKETS)
    assert warning is not None
    assert "16x" in warning  # 16-image bucket against the 1-image one


def test_bias_warning_is_silent_when_there_is_no_bias() -> None:
    assert bias_warning({(64, 64): ["a", "b"]}) is None
    assert bias_warning({(64, 64): ["a"], (32, 32): ["b"]}) is None
