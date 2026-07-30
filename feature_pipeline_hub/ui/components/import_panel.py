"""Step 1: import a raw folder or a set of uploads into a new dataset.

Concept name and trigger word live here rather than in a persistent sidebar —
they only matter while creating an ingestion.
"""

import time
import uuid
from pathlib import Path

import streamlit as st

import state
import step_telemetry
from feature_pipeline.application.dataset_service import create_ingestion_run
from feature_pipeline.infrastructure.storage import run_upload_dir, save_uploaded_files

SOURCE_FOLDER = "From folder"
SOURCE_UPLOAD = "Upload files"


def render() -> None:
    _render_form()
    _render_result()


def _render_form() -> None:
    with st.container(horizontal=True):
        st.text_input("Concept name", key="concept_name", placeholder="cyberpunk_style")
        st.text_input("Trigger word", key="trigger_word", placeholder="sks_style")

    source = st.segmented_control(
        "Source",
        [SOURCE_FOLDER, SOURCE_UPLOAD],
        default=SOURCE_FOLDER,
        required=True,
        label_visibility="collapsed",
    )
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


def _ingest(from_folder: bool, typed_path: str | None, uploaded_files: list | None) -> None:
    concept_name = st.session_state.get("concept_name", "").strip()
    trigger_word = st.session_state.get("trigger_word", "").strip()

    # A fresh run per click: previous ingestions stay selectable instead of being
    # overwritten. Uploads are copied into data/raw/<run_id>/ so they survive reloads.
    run_id = str(uuid.uuid4())
    folder_path = typed_path if from_folder else (
        save_uploaded_files(uploaded_files, destination=run_upload_dir(run_id))
        if uploaded_files
        else None
    )

    if not folder_path:
        st.warning(
            "Enter a folder path." if from_folder else "Upload at least one image."
        )
        return

    if not concept_name or not trigger_word:
        st.warning("Set concept name and trigger word above first.")
        return

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
