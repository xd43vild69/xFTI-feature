"""Step 4: materialize the curated dataset for training.

Writes the Active (non-excluded) samples of the current run as image+caption
pairs into training_runtime/datasets/<name>/ — exactly the flat layout the
training scripts (Step 5) expect. The recaption checkbox selection is a
per-session, ephemeral thing; export always uses the same "Active" criterion
already visible as a filter in Curate, so what you excluded there is what
stays out of the trained dataset.

Confirmation is an inline two-step (not st.dialog): exporting overwrites a
folder, and a plain widget flow is what AppTest can actually drive end to end.
"""

import streamlit as st

import state
from feature_pipeline.application.export_service import export_active_samples
from feature_pipeline.domain.models import IngestionRun
from feature_pipeline.infrastructure.storage import training_dataset_dir

PENDING_KEY = "export_pending_destination"


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    if message := st.session_state.pop("export_message", None):
        st.success(message)

    active_count = sum(1 for s in run.concept.samples if not s.is_excluded)
    excluded_count = len(run.concept.samples) - active_count

    st.caption(
        f"{active_count} image(s) will be exported"
        + (f" · {excluded_count} excluded, left out" if excluded_count else "")
    )

    destination_name = st.text_input(
        "Destination folder name",
        value=run.concept.concept_name,
        help="Written to training_runtime/datasets/<name>/. If that folder "
        "already has content, it is replaced entirely.",
    )

    pending = st.session_state.get(PENDING_KEY)

    if pending == destination_name and destination_name:
        _render_confirmation(run, destination_name)
        return

    if pending is not None and pending != destination_name:
        # The name changed after asking for confirmation — the old confirmation
        # no longer refers to what's about to be written, so drop it silently.
        st.session_state.pop(PENDING_KEY, None)

    existing = training_dataset_dir(destination_name) if destination_name else None
    if existing and existing.is_dir() and any(existing.iterdir()):
        st.caption(
            f":material/warning: '{destination_name}' already has content — it will be replaced."
        )

    if st.button(
        "Export dataset",
        icon=":material/upload_file:",
        disabled=not destination_name or active_count == 0,
    ):
        st.session_state[PENDING_KEY] = destination_name
        st.rerun()


def _render_confirmation(run: IngestionRun, destination_name: str) -> None:
    target = training_dataset_dir(destination_name)
    if target.is_dir() and any(target.iterdir()):
        st.warning(f"This replaces everything currently in '{destination_name}'.")
    st.write(f"Export to `training_runtime/datasets/{destination_name}/`?")

    with st.container(horizontal=True):
        if st.button("Confirm export", icon=":material/upload_file:"):
            result = export_active_samples(run, destination_name)
            st.session_state.pop(PENDING_KEY, None)
            st.session_state["export_message"] = (
                f"Exported {result.exported_count} image(s) to "
                f"training_runtime/datasets/{destination_name}/"
            )
            st.rerun()
        if st.button("Cancel"):
            st.session_state.pop(PENDING_KEY, None)
            st.rerun()
