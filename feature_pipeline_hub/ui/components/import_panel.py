"""Step 1: import a raw folder, a set of uploads, or a copy of an existing dataset.

Concept name and trigger word live here rather than in a persistent sidebar —
they only matter while creating an ingestion.
"""

import time
import uuid
from pathlib import Path

import streamlit as st

import state
import step_telemetry
from feature_pipeline.application.dataset_service import clone_ingestion_run, create_ingestion_run
from feature_pipeline.domain.naming import slugify
from feature_pipeline.infrastructure.storage import run_upload_dir, save_uploaded_files

SOURCE_FOLDER = "From folder"
SOURCE_UPLOAD = "Upload files"
SOURCE_CLONE = "Clone dataset"

# Set once the user edits Trigger word by hand, so their edit sticks instead of
# being overwritten the next time Concept name changes.
_TRIGGER_TOUCHED_KEY = "trigger_word_touched"


def render() -> None:
    _render_form()
    _render_result()


def _sync_trigger_word() -> None:
    """Mirror Concept name into Trigger word, until the user has typed one directly.

    Runs on Concept name's on_change, so it can write session_state before Trigger
    word's own widget renders on the same script run — Streamlit forbids setting a
    keyed widget's value after it has already been instantiated.
    """
    if st.session_state.get(_TRIGGER_TOUCHED_KEY):
        return
    st.session_state["trigger_word"] = slugify(st.session_state.get("concept_name", ""))


def _mark_trigger_word_touched() -> None:
    st.session_state[_TRIGGER_TOUCHED_KEY] = True


def _render_form() -> None:
    with st.container(horizontal=True):
        st.text_input(
            "Concept name",
            key="concept_name",
            placeholder="cyberpunk_style",
            on_change=_sync_trigger_word,
        )
        st.text_input(
            "Trigger word",
            key="trigger_word",
            placeholder="sks_style",
            help="Auto-filled from Concept name until you edit it directly.",
            on_change=_mark_trigger_word_touched,
        )

    source = st.segmented_control(
        "Source",
        [SOURCE_FOLDER, SOURCE_UPLOAD, SOURCE_CLONE],
        default=SOURCE_FOLDER,
        required=True,
        label_visibility="collapsed",
    )

    if source == SOURCE_CLONE:
        _render_clone_form()
        return

    from_folder = source != SOURCE_UPLOAD

    typed_path = None
    uploaded_files = None

    if from_folder:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            typed_path = st.text_input(
                "Folder path",
                key="raw_folder_path",
                placeholder="/path/to/your/raw/images",
                help="Local folder with images (PNG/JPG/WebP) and optional matching .txt captions",
                label_visibility="collapsed",
            )
            scan_clicked = st.button("Scan folder", icon=":material/search:")
    else:
        uploaded_files = st.file_uploader(
            "Upload images (PNG, JPG, WebP) and optional .txt captions",
            type=["png", "jpg", "jpeg", "webp", "txt"],
            accept_multiple_files=True,
            key="uploaded_files",
            label_visibility="collapsed",
        )
        scan_clicked = st.button("Process files", icon=":material/upload:")

    if scan_clicked:
        _ingest(from_folder, typed_path, uploaded_files)


def _render_clone_form() -> None:
    """Pick an existing dataset to copy; Concept name / Trigger word above name the copy."""
    summaries = state.list_runs()
    if not summaries:
        st.info("No datasets to clone yet — import one first.")
        return

    by_run_id = {summary.run_id: summary for summary in summaries}

    with st.container(horizontal=True, vertical_alignment="bottom"):
        source_run_id = st.selectbox(
            "Dataset to clone",
            list(by_run_id),
            format_func=lambda run_id: state.format_run_label(by_run_id[run_id]),
            key="clone_source_run_id",
            label_visibility="collapsed",
        )
        clone_clicked = st.button("Clone dataset", icon=":material/content_copy:")

    include_excluded = st.checkbox(
        "Include excluded images",
        key="clone_include_excluded",
        help=(
            "Off by default: images excluded during curation stay out of the copy. "
            "The clone gets its own images and captions — editing or deleting it never "
            "touches the original."
        ),
    )

    if clone_clicked and source_run_id:
        _clone(source_run_id, include_excluded)


def _clone(source_run_id: str, include_excluded: bool) -> None:
    concept_name = st.session_state.get("concept_name", "").strip()
    trigger_word = st.session_state.get("trigger_word", "").strip()

    # Same order as `_ingest`: nothing is written to disk before the name that every
    # copied file is named after is known to exist.
    if not concept_name or not trigger_word:
        st.warning("Set concept name and trigger word above first.")
        return

    source_run = state.load_run(source_run_id)
    if source_run is None:  # deleted from another session while this page sat open
        st.error("That dataset no longer exists.")
        return

    run_id = str(uuid.uuid4())
    started = time.time()
    with st.spinner("Copying images..."):
        run = clone_ingestion_run(
            source=source_run,
            destination=run_upload_dir(run_id),
            concept_name=concept_name,
            trigger_word=trigger_word,
            run_id=run_id,
            include_excluded=include_excluded,
        )

    if not run.concept.samples:
        st.warning("Nothing was copied — the source dataset's images are no longer on disk.")
        state.save_run(run)
        step_telemetry.record_step(run_id, "import", time.time() - started, error_count=1)
        return

    state.save_run(run)
    state.set_active_run(run.run_id)

    error_count = sum(1 for s in run.concept.samples if not s.is_valid)
    step_telemetry.record_step(run_id, "import", time.time() - started, error_count=error_count)
    st.session_state["ingest_message"] = (
        f"Cloned {len(run.concept.samples)} image(s) from "
        f"'{source_run.concept.concept_name}' into '{concept_name}'."
    )
    st.rerun()


def _ingest(from_folder: bool, typed_path: str | None, uploaded_files: list | None) -> None:
    concept_name = st.session_state.get("concept_name", "").strip()
    trigger_word = st.session_state.get("trigger_word", "").strip()

    # Checked before writing anything: standardized filenames are derived from
    # concept_name, so an upload can't be saved to disk without it, and a run row
    # never gets written for a rejected submission — nothing to leave orphaned.
    if not from_folder and not uploaded_files:
        st.warning("Upload at least one image.")
        return
    if from_folder and not typed_path:
        st.warning("Enter a folder path.")
        return
    if not concept_name or not trigger_word:
        st.warning("Set concept name and trigger word above first.")
        return

    # A fresh run per click: previous ingestions stay selectable instead of being
    # overwritten. Uploads are copied into data/raw/<run_id>/ so they survive reloads.
    run_id = str(uuid.uuid4())
    folder_path = typed_path if from_folder else save_uploaded_files(
        uploaded_files, concept_name, destination=run_upload_dir(run_id)
    )

    started = time.time()
    try:
        with st.spinner("Analyzing images..."):
            run = create_ingestion_run(
                folder_path=folder_path,
                concept_name=concept_name,
                trigger_word=trigger_word,
                source_kind="folder" if from_folder else "upload",
                run_id=run_id,
            )
    except NotADirectoryError as exc:
        st.error(str(exc))
        return

    if not run.concept.samples:
        st.warning("No PNG/JPG/WebP images found.")
        state.save_run(run)
        step_telemetry.record_step(run_id, "import", time.time() - started, error_count=1)
        return

    state.save_run(run)
    state.set_active_run(run.run_id)

    # Record import telemetry after saving the run to DB
    error_count = sum(1 for s in run.concept.samples if not s.is_valid)
    step_telemetry.record_step(run_id, "import", time.time() - started, error_count=error_count)
    # Rerun so the context bar picks up the new dataset right away.
    st.session_state["ingest_message"] = f"Imported {len(run.concept.samples)} image(s)."
    st.rerun()


def _render_result() -> None:
    if message := st.session_state.pop("ingest_message", None):
        st.success(message)

    run = state.active_run()
    if run is None or not run.concept.samples:
        return

    with st.expander(f"Files ({len(run.concept.samples)})"):
        st.dataframe(
            [
                {
                    "file": Path(s.image_path).name,
                    "resolution": f"{s.metrics.width}x{s.metrics.height}",
                    "caption": s.caption,
                    "valid": s.is_valid,
                    "errors": "; ".join(s.validation_errors) or "-",
                }
                for s in run.concept.samples
            ],
            width="stretch",
        )

    if st.button("Continue to curate", icon=":material/arrow_forward:"):
        st.switch_page(state.CURATE_STEP)
