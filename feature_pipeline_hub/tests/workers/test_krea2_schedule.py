"""Learning-rate schedule.

`lr_at` was a closure inside train_krea2 reading seven module globals, so it could
never be called from a test. The first test here is the extraction's proof: it compares
the extracted function against a verbatim copy of the original formula across a dense
grid of steps and every scheduler mode. The rest pin the properties that matter.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.schedule import decay_factor, lr_at_step  # noqa: E402

SCHEDULERS = ["cosine", "constant", "linear", "cosine_with_restarts", "step"]


def original_lr_at(step, LR, GRAD_ACCUM_STEPS, WARMUP_UPDATES, TOTAL_UPDATES,
                   MIN_LR_RATIO, LR_SCHEDULER, LR_NUM_CYCLES, LR_STEP_GAMMA,
                   LR_STEP_COUNT):
    """Verbatim copy of the pre-refactor closure, globals turned into parameters.

    Do not tidy this up — its value is being a literal transcription of what shipped.
    """
    update = step / max(1, GRAD_ACCUM_STEPS)
    if update < WARMUP_UPDATES:
        return LR * update / max(1e-9, WARMUP_UPDATES)
    prog = min(1.0, (update - WARMUP_UPDATES) / max(1e-9, TOTAL_UPDATES - WARMUP_UPDATES))
    if LR_SCHEDULER == "constant":
        factor = 1.0
    elif LR_SCHEDULER == "linear":
        factor = 1.0 - prog
    elif LR_SCHEDULER == "cosine_with_restarts":
        factor = 0.5 * (1 + math.cos(math.pi * ((prog * max(1, LR_NUM_CYCLES)) % 1.0)))
    elif LR_SCHEDULER == "step":
        factor = LR_STEP_GAMMA ** int(prog * max(1, LR_STEP_COUNT))
    else:
        factor = 0.5 * (1 + math.cos(math.pi * prog))
    return LR * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * factor)


@pytest.mark.parametrize("scheduler", SCHEDULERS + ["unknown-falls-back-to-cosine"])
@pytest.mark.parametrize("grad_accum,warmup,total", [
    (4, 100.0, 300.0),   # the defaults
    (1, 0.0, 1200.0),    # no warmup, no accumulation
    (8, 30.0, 150.0),
    (2, 149.0, 150.0),   # warmup consuming all but one update
])
def test_matches_the_original_formula_exactly(scheduler, grad_accum, warmup, total):
    """The extraction's proof: identical output over a dense grid, for every mode."""
    for step in range(0, int(total * grad_accum) + 1, 7):
        assert lr_at_step(
            step, lr=1e-4, grad_accum_steps=grad_accum, warmup_updates=warmup,
            total_updates=total, min_lr_ratio=0.1, scheduler=scheduler,
            num_cycles=3, step_gamma=0.5, step_count=4,
        ) == original_lr_at(step, 1e-4, grad_accum, warmup, total, 0.1,
                            scheduler, 3, 0.5, 4), f"{scheduler} @ step {step}"


def schedule(step, **kwargs):
    base = dict(lr=1e-4, grad_accum_steps=4, warmup_updates=100.0, total_updates=300.0,
                min_lr_ratio=0.1, scheduler="cosine")
    return lr_at_step(step, **{**base, **kwargs})


# ── warmup ──────────────────────────────────────────────────────────────────

def test_warmup_starts_at_zero_and_ramps_linearly() -> None:
    assert schedule(0) == 0.0
    assert schedule(200) == pytest.approx(5e-5)   # 50 of 100 updates
    assert schedule(400) == pytest.approx(1e-4)   # warmup complete


def test_warmup_is_measured_in_updates_not_micro_steps() -> None:
    """The mistake this module exists to prevent: 400 micro-steps == 100 updates."""
    assert schedule(400, grad_accum_steps=4) == pytest.approx(1e-4)
    assert schedule(100, grad_accum_steps=1) == pytest.approx(1e-4)


def test_zero_warmup_does_not_divide_by_zero() -> None:
    assert schedule(0, warmup_updates=0.0) == pytest.approx(1e-4)


# ── decay ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_never_decays_below_the_floor(scheduler: str) -> None:
    """min_lr_ratio is a floor, not a target: late steps must still move the weights."""
    for step in range(400, 1201, 10):
        assert schedule(step, scheduler=scheduler) >= 1e-4 * 0.1 - 1e-12


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_never_exceeds_the_base_rate(scheduler: str) -> None:
    for step in range(0, 1201, 10):
        assert schedule(step, scheduler=scheduler) <= 1e-4 + 1e-12


def test_cosine_reaches_the_floor_at_the_end() -> None:
    assert schedule(1200) == pytest.approx(1e-4 * 0.1)


def test_constant_holds_the_base_rate_after_warmup() -> None:
    assert schedule(800, scheduler="constant") == pytest.approx(1e-4)
    assert schedule(1200, scheduler="constant") == pytest.approx(1e-4)


def test_linear_decays_evenly() -> None:
    mid = schedule(800, scheduler="linear")
    assert mid == pytest.approx(1e-4 * (0.1 + 0.9 * 0.5))


def test_cosine_is_monotonically_decreasing_after_warmup() -> None:
    rates = [schedule(s) for s in range(400, 1201, 10)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_restarts_climb_back_up() -> None:
    """The distinguishing behavior: cosine_with_restarts is not monotonic."""
    rates = [schedule(s, scheduler="cosine_with_restarts") for s in range(400, 1201, 10)]
    assert any(b > a for a, b in zip(rates, rates[1:]))


def test_step_schedule_descends_in_discrete_jumps() -> None:
    rates = {round(schedule(s, scheduler="step"), 12) for s in range(400, 1201, 10)}
    assert len(rates) <= 5  # step_count 4 -> at most 5 plateaus


def test_progress_is_clamped_past_the_end() -> None:
    """A resumed run can overshoot total_steps; the rate must not go negative."""
    assert schedule(5000, scheduler="linear") == pytest.approx(1e-4 * 0.1)


# ── decay_factor ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_decay_factor_stays_in_unit_range(scheduler: str) -> None:
    for i in range(101):
        assert 0.0 - 1e-12 <= decay_factor(i / 100, scheduler) <= 1.0 + 1e-12


def test_unknown_scheduler_falls_back_to_cosine() -> None:
    assert decay_factor(0.5, "nonsense") == decay_factor(0.5, "cosine")
