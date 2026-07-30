"""GPU-time cost estimate: the only real "cost" this pipeline has.

Every worker runs locally (precache, train, recaption all load their own model
copy — see workers/_telemetry.py's gpu_seconds), so there is no per-token or
per-request API bill to track. FTI_GPU_HOURLY_RATE lets a project attach a
$/hour figure — its own cloud on-demand rate, or an amortised on-prem
estimate — to that GPU time. Left unset, cost is simply not estimated rather
than guessed at with a made-up default.
"""

import os

GPU_HOURLY_RATE_ENV = "FTI_GPU_HOURLY_RATE"


def gpu_hourly_rate() -> float | None:
    """The configured $/hour, or None if unset/invalid."""
    raw = os.environ.get(GPU_HOURLY_RATE_ENV, "").strip()
    if not raw:
        return None
    try:
        rate = float(raw)
    except ValueError:
        return None
    return rate if rate > 0 else None


def estimate_cost(gpu_seconds: float, hourly_rate: float | None) -> float | None:
    """`gpu_seconds` of work at `hourly_rate` $/hour, or None if the rate isn't configured."""
    if hourly_rate is None or gpu_seconds <= 0:
        return None
    return round(gpu_seconds * (hourly_rate / 3600), 4)
