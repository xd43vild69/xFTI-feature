"""Step 4: manifest.json preview, version tagging, and the "Freeze Feature Dataset" action.

Placeholder for Iteración 4 (Versionado y Publicación).
"""

import streamlit as st

import state


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    st.caption("Export & versioning: implemented in Iteration 4.")
