"""MCP server: exposes the pipeline's application layer as tools for autonomous agents.

Runs as its own process (stdio transport), separate from the Streamlit UI — the same
"separate process talking to the shared SQLite DB" pattern already used by workers/.
No auth layer exists anywhere in this app (it was built as a local, single-user tool),
so this server is meant to be launched locally and trusted, e.g. as a subprocess of a
LangGraph agent via stdio. It must not be exposed over the network without adding one.

Every tool opens its own SQLite connection and closes it before returning, matching
the per-operation-connection convention documented in ui/state.py and enforced by
`database.get_connection` (SQLite connections are not safe to share across threads).
"""

import dataclasses
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from mcp.server.fastmcp import FastMCP

from feature_pipeline.application import (
    dataset_service,
    export_service,
    inventory_service,
    quality_service,
    training_service,
)
from feature_pipeline.domain.models import IngestionRun
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.database import get_connection

mcp = FastMCP("feature-pipeline-hub")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _require_run(conn: sqlite3.Connection, run_id: str) -> IngestionRun:
    run = repo.load_ingestion_run(conn, run_id)
    if run is None:
        raise ValueError(f"No such dataset run: {run_id}")
    return run


def _training_run_dict(run: training_repo.TrainingRun) -> dict:
    data = dataclasses.asdict(run)
    data["started_at"] = run.started_at.isoformat()
    data["finished_at"] = run.finished_at.isoformat() if run.finished_at else None
    return data


@mcp.tool()
def list_dataset_runs() -> list[dict]:
    """List every stored ingestion run (dataset), newest first — lightweight, no samples."""
    with _db() as conn:
        return [s.model_dump(mode="json") for s in repo.list_ingestion_runs(conn)]


@mcp.tool()
def get_dataset_health() -> list[dict]:
    """Fleet-wide health summary for every dataset: counts, duplicates, median sharpness."""
    with _db() as conn:
        return [h.model_dump(mode="json") for h in inventory_service.load_dataset_inventory(conn)]


@mcp.tool()
def get_run_detail(run_id: str) -> dict:
    """Full detail of one dataset run, including every sample (caption, metrics, flags)."""
    with _db() as conn:
        run = _require_run(conn, run_id)
        return run.model_dump(mode="json")


@mcp.tool()
def import_dataset(folder_path: str, concept_name: str, trigger_word: str) -> dict:
    """Ingest a new dataset by scanning a local folder of images (+ optional .txt captions).

    Creates a new, independently-selectable run — re-importing the same concept never
    overwrites a previous run. Returns the new run's summary.
    """
    with _db() as conn:
        run = dataset_service.create_ingestion_run(
            folder_path=folder_path,
            concept_name=concept_name,
            trigger_word=trigger_word,
            source_kind="folder",
        )
        repo.save_ingestion_run(conn, run)
        return {
            "run_id": run.run_id,
            "concept_name": run.concept.concept_name,
            "trigger_word": run.concept.trigger_word,
            "sample_count": len(run.concept.samples),
        }


@mcp.tool()
def revalidate_run(run_id: str | None = None) -> dict:
    """Re-check samples against the current validation rules.

    Pass `run_id` to revalidate just that dataset, or omit it to sweep every stored
    run. Only needed after a validation rule change — imported samples otherwise keep
    the verdict computed at import time. Returns how many samples' verdicts changed.
    """
    with _db() as conn:
        if run_id is None:
            return {"run_id": None, "changed": dataset_service.revalidate_all_runs(conn)}
        run = _require_run(conn, run_id)
        changed = dataset_service.revalidate_samples(run.concept.samples)
        if changed:
            repo.save_ingestion_run(conn, run)
        return {"run_id": run_id, "changed": changed}


@mcp.tool()
def export_dataset(run_id: str, destination_name: str) -> dict:
    """Materialize a run's active (non-excluded) samples as a flat training_runtime/datasets/<destination_name>/ folder.

    This is the dataset shape Step 5 (training) expects. Required before starting a
    training run against this dataset.
    """
    with _db() as conn:
        run = _require_run(conn, run_id)
        result = export_service.export_active_samples(run, destination_name)
        return dataclasses.asdict(result)


@mcp.tool()
def quality_summary(run_id: str) -> dict:
    """Headline quality counts for one run: active/excluded/duplicate/missing-caption/invalid."""
    with _db() as conn:
        run = _require_run(conn, run_id)
        return quality_service.stored_quality_summary(run.concept.samples, run.concept.trigger_word)


@mcp.tool()
def start_lora_training(
    dataset_run_id: str,
    total_steps: int = 1200,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    save_every: int = 25,
    seed: int = 42,
) -> dict:
    """Start a LoRA training run: launches pre-cache and returns immediately (does not block).

    The dataset must already be exported via `export_dataset` under its concept name.
    Only one training-runtime job (pre-cache or train) can run at a time; call fails
    if one is already active. Poll `get_training_status` with the returned
    `training_run_id` until phase="precache" reports status="completed", then call
    `continue_lora_training` with the SAME hyperparameters to launch the train phase.
    """
    with _db() as conn:
        if training_service.is_training_active(conn):
            raise RuntimeError("A training-runtime job is already running; wait for it to finish.")

        run = _require_run(conn, dataset_run_id)
        training_run_id = training_service.launch_precache(
            conn,
            dataset_run_id=dataset_run_id,
            dataset_name=run.concept.concept_name,
            trigger_word=run.concept.trigger_word,
        )
        return {"training_run_id": training_run_id, "phase": "precache", "status": "running"}


@mcp.tool()
def continue_lora_training(
    dataset_run_id: str,
    precache_training_run_id: str,
    total_steps: int = 1200,
    lr: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    save_every: int = 25,
    seed: int = 42,
) -> dict:
    """Launch the training phase after `start_lora_training`'s pre-cache has completed.

    Call `get_training_status(precache_training_run_id)` first and only call this once
    it reports phase="precache", status="completed". Hyperparameters should match the
    call to `start_lora_training` (they were not persisted between the two calls).
    """
    with _db() as conn:
        status = training_service.precache_status(conn, precache_training_run_id)
        if status != "completed":
            raise RuntimeError(f"Pre-cache is not complete yet (status={status!r}).")

        run = _require_run(conn, dataset_run_id)
        config = training_service.TrainingConfig(
            total_steps=total_steps,
            lr=lr,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            save_every=save_every,
            seed=seed,
        )
        training_run_id = training_service.launch_train(
            conn,
            dataset_run_id=dataset_run_id,
            dataset_name=run.concept.concept_name,
            trigger_word=run.concept.trigger_word,
            config=config,
        )
        return {"training_run_id": training_run_id, "phase": "train", "status": "running"}


@mcp.tool()
def get_training_status(training_run_id: str) -> dict:
    """Check the status of a pre-cache or train job launched via `start_lora_training`.

    Returns phase ("precache" | "train") and status ("running" | "completed" | "failed"
    | "stopped"), plus duration/GPU-seconds/cost once a train job finishes.
    """
    with _db() as conn:
        run = training_repo.get_training_run(conn, training_run_id)
        if run is None:
            raise ValueError(f"No such training run: {training_run_id}")

        if run.kind == "precache":
            status = training_service.precache_status(conn, training_run_id)
            return {"training_run_id": training_run_id, "phase": "precache", "status": status}

        if run.status == "running" and not training_runner.is_process_alive(run.pid):
            training_service.finalize_dead_run(conn, run, fallback_status="failed")
            refreshed = training_repo.get_training_run(conn, training_run_id)
            if refreshed is not None:
                run = refreshed

        return {"phase": "train", **_training_run_dict(run)}


@mcp.tool()
def stop_training(training_run_id: str) -> dict:
    """Stop a running pre-cache or train job (SIGINT, graceful checkpoint save for train)."""
    with _db() as conn:
        training_service.stop_training(conn, training_run_id)
        return {"training_run_id": training_run_id, "status": "stopped"}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
