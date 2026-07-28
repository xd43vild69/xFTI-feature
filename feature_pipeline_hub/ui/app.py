"""Streamlit entry point: FTI Feature Pipeline Hub.

Ingestion tab is wired to the real pipeline. Gallery, Quality, and
Export tabs still delegate to placeholder components (Iteraciones 3-4).
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from components import concept_selector, export_modal, gallery, quality_panel

from feature_pipeline.application.dataset_service import ingest_concept_from_folder
from feature_pipeline.infrastructure.storage import save_uploaded_files

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

        # Uploads are staged to a temp folder only on click, so reruns don't pile up copies.
        folder_path = typed_path if from_folder else (
            save_uploaded_files(uploaded_files) if uploaded_files else None
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
            if "concept_id" not in st.session_state:
                st.session_state["concept_id"] = str(uuid.uuid4())
            try:
                with st.spinner("Analyzing images..."):
                    concept = ingest_concept_from_folder(
                        folder_path=folder_path,
                        concept_id=st.session_state["concept_id"],
                        concept_name=concept_name,
                        trigger_word=trigger_word,
                    )
            except NotADirectoryError as exc:
                st.error(str(exc))
            else:
                st.session_state["concept"] = concept
                if concept.samples:
                    st.success(f"Found {len(concept.samples)} image(s).")
                else:
                    st.warning("No PNG/JPG/WebP images found.")

    concept = st.session_state.get("concept")
    if concept is not None and concept.samples:
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
            use_container_width=True,
        )
    else:
        st.info("No dataset loaded yet.")

with tab_gallery:
    gallery.render()

with tab_quality:
    quality_panel.render()

with tab_export:
    export_modal.render()
