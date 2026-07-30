"""Step 5: launch and monitor a training run against the exported dataset.

Pre-cache is short (I/O + VAE-encode bound) and runs synchronously, blocking
the Start button while it finishes; training itself is hours long and runs
detached, so progress is read back from disk/SQLite on every refresh — this
page works the same whether it's the tab that launched the run or a fresh
browser session hours later.
"""

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

import state
from feature_pipeline.application import training_service
from feature_pipeline.domain.models import IngestionRun
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.database import get_connection

LOG_TAIL_BYTES = 8000


@contextmanager
def _db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    with _db() as conn:
        train_runs = training_repo.list_training_runs(conn, dataset_run_id=run.run_id)
    train_runs = [r for r in train_runs if r.kind == "train"]
    latest = train_runs[0] if train_runs else None  # newest first

    if latest is not None and latest.status == "running":
        _render_progress(latest.training_run_id)
        return

    _render_launch_form(run, latest)


def _render_launch_form(run: IngestionRun, latest: training_repo.TrainingRun | None) -> None:
    try:
        training_runner.resolve_environment()
    except training_runner.TrainingUnavailable as exc:
        st.caption(f":material/info: {exc}")
        return

    dataset_dir = training_service.dataset_dir_for(run.concept.concept_name)
    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        st.caption(
            f":material/info: No exported dataset at training_runtime/datasets/"
            f"{run.concept.concept_name}/. Export it in Step 4 first."
        )
        return

    if latest is not None:
        st.caption(f"Last run: {latest.status} · {latest.finished_at or 'in progress'}")

    with st.container(horizontal=True):
        total_steps = st.number_input("Total steps", min_value=1, value=1200, step=100)
        lr = st.number_input("Learning rate", min_value=0.0, value=1e-4, step=1e-5, format="%.6f")
        lora_rank = st.number_input("LoRA rank", min_value=1, value=16, step=1)
        lora_alpha = st.number_input("LoRA alpha", min_value=1, value=32, step=1)

    with st.container(horizontal=True):
        batch_size = st.number_input("Batch size", min_value=1, value=1, step=1)
        grad_accum_steps = st.number_input("Grad accumulation steps", min_value=1, value=4, step=1)
        save_every = st.number_input("Save every", min_value=1, value=25, step=25)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

    if st.button("Start training", icon=":material/play_arrow:"):
        config = training_service.TrainingConfig(
            total_steps=int(total_steps),
            lr=float(lr),
            lora_rank=int(lora_rank),
            lora_alpha=int(lora_alpha),
            batch_size=int(batch_size),
            grad_accum_steps=int(grad_accum_steps),
            save_every=int(save_every),
            seed=int(seed),
        )
        with st.spinner("Pre-caching dataset…"):
            with _db() as conn:
                try:
                    training_service.start_training(
                        conn,
                        dataset_run_id=run.run_id,
                        dataset_name=run.concept.concept_name,
                        trigger_word=run.concept.trigger_word,
                        config=config,
                    )
                except training_service.PrecacheFailed as exc:
                    st.error(str(exc))
                    return
        st.rerun()


@st.fragment(run_every="5s")
def _render_progress(training_run_id: str) -> None:
    with _db() as conn:
        run = training_repo.get_training_run(conn, training_run_id)
        if run is None:
            return

        alive = training_runner.is_process_alive(run.pid)
        if not alive and run.status == "running":
            training_service.finalize_dead_run(conn, run, fallback_status="failed")
            run = training_repo.get_training_run(conn, training_run_id)

    status_line = f"Training · {run.status} · started {run.started_at}"
    if run.duration_seconds:
        status_line += f" · {run.duration_seconds / 60:.1f} min"
    if run.cost_estimate is not None:
        status_line += f" · ~${run.cost_estimate:.2f}"
    st.caption(status_line)
    if run.status == "failed" and run.error_message:
        st.caption(f"⚠ {run.error_message}")

    csv_path = training_service.training_log_csv_path(run)
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            if "step" in df.columns and "loss" in df.columns:
                st.line_chart(df.set_index("step")["loss"])
        except pd.errors.EmptyDataError:
            pass

    log_text = _tail_log(run.log_path)
    st.code(log_text or "Waiting for output…", language=None, height=240)

    if run.status == "running":
        if st.button("Stop training", icon=":material/stop:"):
            training_runner.stop_process(run.pid)
            with _db() as conn:
                training_repo.update_training_run_status(conn, training_run_id, "stopped")
            st.rerun()
    else:
        if st.button("Back to config"):
            st.rerun()


def _tail_log(log_path: str) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    size = path.stat().st_size
    offset = max(0, size - LOG_TAIL_BYTES)
    text, _ = training_runner.read_log_tail(log_path, since_offset=offset)
    return text
