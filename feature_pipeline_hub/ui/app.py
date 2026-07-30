"""Streamlit entry point: FTI Feature Pipeline Hub.

The dataset curation flow is sequential — import, curate, check quality, export,
train — so the steps are pages of a top navigation rather than free-floating
tabs. The body of this script runs before the selected page, which is what puts
the context bar on every step without repeating it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import state
import streamlit as st
from components import context_bar

st.set_page_config(page_title="FTI Feature Pipeline Hub", layout="wide")

context_bar.render()

st.navigation(
    [
        st.Page(state.OBSERVABILITY_STEP, title="📊 Metrics"),
        st.Page(state.IMPORT_STEP, title="1 · Import", default=True),
        st.Page(state.CURATE_STEP, title="2 · Curate"),
        st.Page(state.QUALITY_STEP, title="3 · Quality"),
        st.Page(state.EXPORT_STEP, title="4 · Export"),
        st.Page(state.TRAIN_STEP, title="5 · Train"),
    ],
    position="top",
).run()
