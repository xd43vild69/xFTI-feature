"""Session state and persistence helpers shared by the UI components.

Connections are opened per operation rather than cached: Streamlit reruns can land
on different threads, and a SQLite connection is not shared across them safely.
"""

from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from PIL import Image

from feature_pipeline.application import caption_service, image_service
from feature_pipeline.domain.models import DatasetSample, IngestionRun, IngestionRunSummary
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.storage import delete_managed_folder, write_caption_sidecar

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


def preview_replace_counts(samples: list[DatasetSample], old_word: str) -> tuple[int, int]:
    """Calculate matching samples count and total exact word occurrences (case-sensitive)."""
    if not old_word:
        return 0, 0
    matching_samples = 0
    total_matches = 0
    for sample in samples:
        _, count = caption_service.replace_exact_word(sample.caption, old_word, "")
        if count > 0:
            matching_samples += 1
            total_matches += count
    return matching_samples, total_matches


def batch_replace_caption_word(run: IngestionRun, old_word: str, new_word: str) -> tuple[int, int]:
    """Batch replace exact occurrences of old_word with new_word in run's samples.

    Persists to SQLite and bumps caption widget versions so text areas refresh cleanly.
    Returns (samples_updated_count, total_replacements_count).
    """
    if not old_word:
        return 0, 0

    samples_updated = 0
    total_replacements = 0
    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})

    with _db() as conn:
        for sample in run.concept.samples:
            new_caption, count = caption_service.replace_exact_word(
                sample.caption, old_word, new_word
            )
            if count > 0:
                repo.update_sample_caption(conn, sample.sample_id, new_caption)
                sample.caption = new_caption
                versions[sample.sample_id] = versions.get(sample.sample_id, 0) + 1
                samples_updated += 1
                total_replacements += count

    return samples_updated, total_replacements


def selection_key(run_id: str, sample_id: str) -> str:
    return f"select_{run_id}_{sample_id}"


def selected_samples(run: IngestionRun) -> list[DatasetSample]:
    """Samples currently ticked in the curation grid."""
    return [
        sample
        for sample in run.concept.samples
        if st.session_state.get(selection_key(run.run_id, sample.sample_id))
    ]


def set_selection(run_id: str, samples: list[DatasetSample], selected: bool) -> None:
    """Tick or untick a group of samples.

    Only safe to call before the checkboxes are instantiated on this run —
    Streamlit rejects writes to a widget's key once its widget exists.
    """
    for sample in samples:
        st.session_state[selection_key(run_id, sample.sample_id)] = selected


def apply_recaption(sample: DatasetSample, caption: str) -> None:
    """Store an AI caption in the database and alongside the image.

    Writing the .txt sidecar keeps the source folder interoperable with LoRAlab,
    which reads captions from disk; the pre-AI text is kept as .txt.bak. Bumping
    the widget version makes the grid's editor pick the new text up.
    """
    save_caption(sample.sample_id, caption)
    write_caption_sidecar(sample.image_path, caption)

    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})
    versions[sample.sample_id] = versions.get(sample.sample_id, 0) + 1


def set_excluded(sample_ids: list[str], excluded: bool) -> None:
    with _db() as conn:
        repo.set_samples_excluded(conn, sample_ids, excluded)


def is_training_active() -> bool:
    """Whether a training-runtime job (pre-cache/train/progressive/curate) is
    running right now — the GPU can only do one heavy job at a time.

    Self-healing: a 'running' row whose process actually died (crash, machine
    restart) is corrected to 'failed' here rather than blocking the GPU forever.
    """
    with _db() as conn:
        run = training_repo.find_running_training_run(conn)
        if run is None:
            return False
        if training_runner.is_process_alive(run.pid):
            return True
        training_repo.update_training_run_status(conn, run.training_run_id, "failed")
        return False


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


@st.cache_data(max_entries=256)
def _cached_thumbnail(image_path: str, mtime: float, size: int) -> Image.Image:
    return image_service.make_square_thumbnail(image_path, size)


def render_thumbnail(image_path: str) -> None:
    """Render a square, theme-adaptive thumbnail, or an error if it's missing/unreadable.

    `mtime` is part of the cache key so an image edited or re-ingested on the same
    path invalidates its cached thumbnail automatically.
    """
    path = Path(image_path)
    if not path.exists():
        st.error(f"Missing file: {path.name}")
        return

    try:
        thumbnail = _cached_thumbnail(str(path), path.stat().st_mtime, image_service.THUMBNAIL_SIZE)
    except Exception:
        st.error(f"Could not read image: {path.name}")
        return

    st.image(thumbnail, width="stretch")


IMPORT_STEP = "steps/import_step.py"
CURATE_STEP = "steps/curate_step.py"
QUALITY_STEP = "steps/quality_step.py"
EXPORT_STEP = "steps/export_step.py"


def require_active_run() -> IngestionRun | None:
    """The active run, or None after drawing the shared 'nothing selected' state.

    Steps 2-4 all need a dataset before they can show anything; routing that
    through one helper keeps the empty state identical everywhere.
    """
    run = active_run()
    if run is not None and run.concept.samples:
        return run

    if run is not None:
        st.warning("This dataset has no images.")
        return None

    st.info("No dataset selected yet.")
    if st.button("Go to import", icon=":material/arrow_forward:"):
        st.switch_page(IMPORT_STEP)
    return None


def format_run_label(summary: IngestionRunSummary) -> str:
    icon = "📤" if summary.source_kind == "upload" else "📁"
    when = summary.created_at.astimezone().strftime("%d %b %H:%M")
    return f"{icon} {summary.concept_name} · {summary.sample_count} imgs · {when}"
