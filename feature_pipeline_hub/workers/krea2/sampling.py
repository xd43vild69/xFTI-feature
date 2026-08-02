"""Dataset samplers: which images form the next batch, and in what order.

Resolution buckets are the constraint that shapes both classes — latents of different
sizes cannot be concatenated, so a batch never mixes buckets.

Free of torch: both samplers work on names and use only `random`, which keeps them
testable in the hub environment and makes their RNG state serializable as plain JSON.
"""
from __future__ import annotations

import json
import random
from typing import Any, Iterable, Mapping

Bucket = tuple[int, int]
Batch = tuple[Bucket, list[str]]


def _rng_to_json(rng: random.Random) -> str:
    """Serialize a Random's state as JSON so it can live in a torch checkpoint.

    `torch.save` with `weights_only=True` refuses arbitrary Python objects on load, so
    the state travels as a string rather than as a pickled tuple.
    """
    version, internal, gauss = rng.getstate()
    return json.dumps([version, list(internal), gauss])


def _rng_from_json(rng: random.Random, blob: str) -> None:
    version, internal, gauss = json.loads(blob)
    rng.setstate((version, tuple(internal), gauss))


class EpochSampler:
    """Epoch-based sampling: every image is seen exactly once per epoch.

    Replaces the original `random.choice(bucket)` + `random.choice(image)`, which gave
    every *bucket* equal probability regardless of how many images it held — a 1-image
    bucket got as much mass as a 16-image one. Measured on a real dataset, six stray
    images were taking 67% of all steps.
    """

    def __init__(self, buckets: Mapping[Bucket, Iterable[str]], batch_size: int,
                 seed: int, repeats: Mapping[str, int] | None = None) -> None:
        self.buckets: dict[Bucket, list[str]] = {k: sorted(v) for k, v in buckets.items()}
        self.batch_size = max(1, int(batch_size))
        self.repeats = dict(repeats or {})
        self.rng = random.Random(seed)
        self.queue: list[Batch] = []
        self.epoch = 0

    def _refill(self) -> None:
        """Rebuild the shuffled batch queue for one full epoch over every bucket.

        Images are repeated per `repeats`, and a bucket's trailing short batch is padded
        from that same bucket — pulling from another would break `torch.cat`. Batches
        are then shuffled so resolutions interleave across the epoch.
        """
        self.epoch += 1
        batches: list[Batch] = []
        for size, names in sorted(self.buckets.items()):
            pool: list[str] = []
            for name in names:
                pool.extend([name] * max(1, int(self.repeats.get(name, 1))))
            self.rng.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    chunk = chunk + self.rng.choices(pool, k=self.batch_size - len(chunk))
                batches.append((size, chunk))
        self.rng.shuffle(batches)
        self.queue = batches

    def next(self) -> Batch:
        """Return the next `(bucket_size, sample_names)` batch, starting a new epoch if needed."""
        if not self.queue:
            self._refill()
        return self.queue.pop()

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "rng": _rng_to_json(self.rng),
            "queue": json.dumps([[list(size), names] for size, names in self.queue]),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.epoch = int(state["epoch"])
        _rng_from_json(self.rng, state["rng"])
        self.queue = [((int(size[0]), int(size[1])), list(names))
                      for size, names in json.loads(state["queue"])]


class LegacySampler:
    """Historical sampling: uniform over buckets, with replacement.

    Kept only to reproduce old runs. See `EpochSampler` for the bias this introduces.
    """

    def __init__(self, buckets: Mapping[Bucket, Iterable[str]], batch_size: int,
                 seed: int) -> None:
        self.buckets: dict[Bucket, list[str]] = {k: list(v) for k, v in buckets.items()}
        self.batch_size = max(1, int(batch_size))
        self.rng = random.Random(seed)
        self.epoch = 0

    def next(self) -> Batch:
        """Return a `(bucket_size, sample_names)` batch drawn uniformly at random."""
        size = self.rng.choice(list(self.buckets))
        return size, [self.rng.choice(self.buckets[size]) for _ in range(self.batch_size)]

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": 0, "rng": _rng_to_json(self.rng), "queue": "[]"}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _rng_from_json(self.rng, state["rng"])


def bias_warning(buckets: Mapping[Bucket, Iterable[str]]) -> str | None:
    """Describe the legacy sampler's bucket bias for this dataset, or None if it has none.

    The historical default is a measurable bug, so it is worth reporting with the real
    number rather than a generic caveat.
    """
    counts = sorted(len(list(v)) for v in buckets.values())
    if len(counts) <= 1 or counts[-1] <= counts[0] or counts[0] == 0:
        return None
    bias = counts[-1] / counts[0]
    return (f"[!] sampler='legacy' samples buckets uniformly, so an image in the "
            f"smallest bucket ({counts[0]} img) gets {bias:.0f}x the gradient steps of "
            f"one in the largest ({counts[-1]} img).\n"
            f"[!] Muestreo sesgado {bias:.0f}x. Use sampler='epoch' (or preset "
            f"'stable_v2') for full coverage / para cobertura completa.")


def build_sampler(mode: str, buckets: Mapping[Bucket, Iterable[str]], batch_size: int,
                  seed: int) -> EpochSampler | LegacySampler:
    """Construct the sampler named by `mode`; anything but "epoch" means the legacy one."""
    if mode == "epoch":
        return EpochSampler(buckets, batch_size, seed)
    return LegacySampler(buckets, batch_size, seed)
