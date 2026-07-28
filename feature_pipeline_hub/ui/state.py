"""Session state and persistence helpers shared by the UI components.

Connections are opened per operation rather than cached: Streamlit reruns can land
on different threads, and a SQLite connection is not shared across them safely.
"""

from contextlib import contextmanager

import streamlit as st

from feature_pipeline.domain.models import IngestionRun, IngestionRunSummary
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.storage import delete_managed_folder

ACTIVE_RUN_KEY = "active_run_id"


@contextmanager
def _db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def list_runs() -> list[IngestionRunSummary]:
    with _db() as conn:
        return repo.list_ingestion_runs(conn)


def load_run(run_id: str) -> IngestionRun | None:
    with _db() as conn:
        return repo.load_ingestion_run(conn, run_id)


def save_run(run: IngestionRun) -> None:
    with _db() as conn:
        repo.save_ingestion_run(conn, run)


def save_caption(sample_id: str, caption: str) -> None:
    with _db() as conn:
        repo.update_sample_caption(conn, sample_id, caption)


def delete_run(run_id: str) -> None:
    """Drop a run from the database, plus its image folder when we own it."""
    with _db() as conn:
        run = repo.load_ingestion_run(conn, run_id)
        repo.delete_ingestion_run(conn, run_id)

    if run is not None and run.source_kind == "upload":
        delete_managed_folder(run.source_path)

    if st.session_state.get(ACTIVE_RUN_KEY) == run_id:
        st.session_state.pop(ACTIVE_RUN_KEY, None)


def set_active_run(run_id: str) -> None:
    st.session_state[ACTIVE_RUN_KEY] = run_id


def active_run() -> IngestionRun | None:
    """The run currently selected in the sidebar, reloaded from storage."""
    run_id = st.session_state.get(ACTIVE_RUN_KEY)
    if not run_id:
        return None

    run = load_run(run_id)
    if run is None:  # deleted from another session
        st.session_state.pop(ACTIVE_RUN_KEY, None)
    return run


def format_run_label(summary: IngestionRunSummary) -> str:
    icon = "📤" if summary.source_kind == "upload" else "📁"
    when = summary.created_at.astimezone().strftime("%d %b %H:%M")
    return f"{icon} {summary.concept_name} · {summary.sample_count} imgs · {when}"
