"""Streamlit entry point: FTI Feature Pipeline Hub.

Layout-only skeleton for this step. Each tab currently delegates to a
placeholder component; real behavior lands in Iteraciones 2-4.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from components import concept_selector, export_modal, gallery, quality_panel

st.set_page_config(page_title="FTI Feature Pipeline Hub", layout="wide")

st.title("🖼️ FTI Feature Pipeline Hub")

with st.sidebar:
    st.header("Configuración de Concepto")
    st.text_input("Nombre del concepto", key="concept_name")
    st.text_input("Trigger word", key="trigger_word")
    st.divider()
    concept_selector.render()

tab_ingest, tab_gallery, tab_quality, tab_export = st.tabs(
    ["Ingesta", "Galería", "Calidad", "Versionado y Publicación"]
)

with tab_ingest:
    st.info("Ingestion manager: implemented in Iteración 2.")

with tab_gallery:
    gallery.render()

with tab_quality:
    quality_panel.render()

with tab_export:
    export_modal.render()
