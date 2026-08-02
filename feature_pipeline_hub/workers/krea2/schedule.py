"""Learning-rate schedule: warmup, then one of five decay shapes.

Free of torch — it is arithmetic over a step counter — so this module is covered by
mypy and by tests/workers/ in the hub environment.

The unit confusion this file exists to contain: `total_steps` counts micro-steps, but
the optimizer only steps once per `grad_accum_steps`, so the schedule runs in *updates*.
Every function here takes micro-steps and converts, in one place.
"""
from __future__ import annotations

import math


def decay_factor(progress: float, scheduler: str, num_cycles: int = 3,
                 step_gamma: float = 0.5, step_count: int = 4) -> float:
    """Decay multiplier in [0, 1] at `progress` ∈ [0, 1] through the post-warmup run.

    Anything not named falls through to cosine, which is the historical default.
    """
    if scheduler == "constant":
        return 1.0
    if scheduler == "linear":
        return 1.0 - progress
    if scheduler == "cosine_with_restarts":
        return 0.5 * (1 + math.cos(math.pi * ((progress * max(1, num_cycles)) % 1.0)))
    if scheduler == "step":
        return step_gamma ** int(progress * max(1, step_count))
    return 0.5 * (1 + math.cos(math.pi * progress))


def lr_at_step(
    step: int,
    *,
    lr: float,
    grad_accum_steps: int,
    warmup_updates: float,
    total_updates: float,
    min_lr_ratio: float,
    scheduler: str,
    num_cycles: int = 3,
    step_gamma: float = 0.5,
    step_count: int = 4,
) -> float:
    """Learning rate for a given micro-step, after warmup and the configured decay.

    Warmup is linear from 0 to `lr`. After it, the rate decays toward
    `lr * min_lr_ratio` — never to zero, so late steps still move the weights.
    `warmup_updates` is expected already normalized to updates (see `config`).
    """
    update = step / max(1, grad_accum_steps)
    if update < warmup_updates:
        return lr * update / max(1e-9, warmup_updates)

    progress = min(1.0, (update - warmup_updates) / max(1e-9, total_updates - warmup_updates))
    factor = decay_factor(progress, scheduler, num_cycles, step_gamma, step_count)
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * factor)
