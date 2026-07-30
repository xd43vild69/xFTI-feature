"""Measure and record the duration of each curation step for the observability dashboard.

Each step (import, recaption, quality, export) calls record_step() at the end, passing
the run_id, step name, duration, and any error count that occurred. This populates
ingestion_runs with per-step telemetry so the observability panel can show a unified view.
"""

import time
from contextlib import contextmanager

from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure.database import get_connection


@contextmanager
def measure_step(run_id: str, step: str):
    """Context manager that records the duration of a step.

    Usage:
        with measure_step(run_id, "import"):
            # ... do work ...
    """
    started = time.time()
    try:
        yield
    finally:
        duration = time.time() - started
        record_step(run_id, step, duration, error_count=0)


def record_step(run_id: str, step: str, duration_seconds: float, error_count: int = 0) -> None:
    """Record a step's telemetry in the database.

    Meant to be called at the end of each curation step (import, recaption, quality, export)
    with the measured duration and any errors that occurred.
    """
    with get_connection() as conn:
        repo.update_step_telemetry(
            conn, run_id=run_id, step=step, duration_seconds=duration_seconds, error_count=error_count
        )
