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


CAPTION_VERSIONS_KEY = "caption_widget_versions"


def caption_widget_key(run_id: str, sample_id: str) -> str:
    """Key for a caption editor, versioned so external edits are picked up.

    A keyed Streamlit widget ignores its `value` argument on later reruns, so a
    caption written from another tab would otherwise be reverted by the stale text
    still held in the widget's state. Bumping the version mints a new widget that
    reads the stored caption afresh.
    """
    version = st.session_state.get(CAPTION_VERSIONS_KEY, {}).get(sample_id, 0)
    return f"caption_{run_id}_{sample_id}_v{version}"


def persist_caption(sample_id: str, widget_key: str) -> None:
    """on_change callback: store whatever the user typed into a caption editor."""
    save_caption(sample_id, st.session_state[widget_key])


def persist_description(sample_id: str, widget_key: str, trigger_word: str) -> None:
    """on_change callback for the quality panel: turn a description into a caption."""
    description = st.session_state[widget_key].strip()
    if not description:
        return

    save_caption(sample_id, f"{trigger_word}, {description}" if trigger_word else description)
    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})
    versions[sample_id] = versions.get(sample_id, 0) + 1


def set_excluded(sample_ids: list[str], excluded: bool) -> None:
    with _db() as conn:
        repo.set_samples_excluded(conn, sample_ids, excluded)


def mark_duplicates(run_id: str, sample_ids: list[str]) -> None:
    with _db() as conn:
        repo.mark_duplicates(conn, run_id, sample_ids)


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
