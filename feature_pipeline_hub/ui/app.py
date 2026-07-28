"""Streamlit entry point: FTI Feature Pipeline Hub.

Ingestion and Gallery are wired to the real pipeline. Quality and Export
tabs still delegate to placeholder components (Iteraciones 3-4).
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import state
import streamlit as st
from components import concept_selector, export_modal, gallery, quality_panel

from feature_pipeline.application.dataset_service import create_ingestion_run
from feature_pipeline.infrastructure.storage import run_upload_dir, save_uploaded_files

st.set_page_config(page_title="FTI Feature Pipeline Hub", layout="wide")

st.title("🖼️ FTI Feature Pipeline Hub")

with st.sidebar:
    st.header("Concept Configuration")
    st.text_input("Concept Name", key="concept_name")
    st.text_input("Trigger word", key="trigger_word")
    st.divider()
    concept_selector.render()

tab_ingest, tab_gallery, tab_quality, tab_export = st.tabs(
    ["Ingestion", "Gallery", "Quality", "Export & Versioning"]
)

with tab_ingest:
    st.subheader("Import Raw Dataset")

    ingest_source = st.radio(
        "Choose input method:",
        ["📁 From local folder", "📤 Upload files"],
        horizontal=True,
    )

    from_folder = ingest_source == "📁 From local folder"
    typed_path = None
    uploaded_files = None

    if from_folder:
        typed_path = st.text_input(
            "Folder path",
            key="raw_folder_path",
            placeholder="/path/to/your/raw/images",
            help="Local folder with images (PNG/JPG/WebP) and optional matching .txt captions",
        )
        scan_clicked = st.button("🔍 Scan folder", type="primary")
    else:
        uploaded_files = st.file_uploader(
            "Upload images (PNG, JPG, WebP) and optional .txt captions",
            type=["png", "jpg", "jpeg", "webp", "txt"],
            accept_multiple_files=True,
            key="uploaded_files",
        )
        scan_clicked = st.button("📤 Process uploaded files", type="primary")

    if scan_clicked:
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
                "Please enter a folder path."
                if from_folder
                else "Please upload at least one image."
            )
        elif not concept_name or not trigger_word:
            st.warning("Set Concept Name and Trigger word in the sidebar first.")
        else:
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
            else:
                if run.concept.samples:
                    state.save_run(run)
                    state.set_active_run(run.run_id)
                    # Rerun so the sidebar selector picks up the new run right away.
                    st.session_state["ingest_message"] = (
                        f"Found {len(run.concept.samples)} image(s). "
                        "Saved as a new ingestion — preview it in the Gallery tab."
                    )
                    st.rerun()
                else:
                    st.warning("No PNG/JPG/WebP images found.")

    if message := st.session_state.pop("ingest_message", None):
        st.success(message)

    active_run = state.active_run()
    concept = active_run.concept if active_run else None
    if concept is not None and concept.samples:
        st.divider()
        st.caption(f"Active ingestion: **{concept.concept_name}** · `{active_run.run_id}`")
        total = len(concept.samples)
        invalid = sum(1 for s in concept.samples if not s.is_valid)
        without_caption = sum(1 for s in concept.samples if not s.original_caption)

        col1, col2, col3 = st.columns(3)
        col1.metric("Images found", total)
        col2.metric("Missing caption", without_caption)
        col3.metric("Validation errors", invalid)

        st.dataframe(
            [
                {
                    "file": Path(s.image_path).name,
                    "resolution": f"{s.metrics.width}x{s.metrics.height}",
                    "caption": s.caption,
                    "valid": s.is_valid,
                    "errors": "; ".join(s.validation_errors) or "-",
                }
                for s in concept.samples
            ],
            width="stretch",
        )
    else:
        st.info("No dataset loaded yet.")

with tab_gallery:
    gallery.render()

with tab_quality:
    quality_panel.render()

with tab_export:
    export_modal.render()
