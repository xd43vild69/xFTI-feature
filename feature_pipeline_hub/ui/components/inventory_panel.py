"""Fleet view: health counts for every imported dataset, side by side.

Sits above the active-run pipeline detail on the Metrics page. Answers "what have
I imported and which set is cleanest", which no single-run view can.

The table is deliberately just a sortable table: clicking a column header ranks
the datasets on a number the user picked, rather than on a composite score this
module would have had to invent.
"""

import streamlit as st

import state
from feature_pipeline.domain.models import DatasetHealth

DUPLICATE_HELP = (
    "Read from the verdict the Quality step stored, not recomputed. A dataset that "
    "has never been through step 3 shows 0 even if it does contain duplicates."
)
SHARPNESS_HELP = (
    "Variance of the Laplacian. Only meaningful as a ranking inside one dataset — "
    "texture and subject move it as much as focus does — so these numbers are not "
    "comparable between rows."
)


def render() -> None:
    inventory = state.dataset_inventory()

    st.subheader("📦 Dataset inventory")

    if not inventory:
        st.info("No datasets imported yet.")
        if st.button("Go to import", icon=":material/arrow_forward:"):
            st.switch_page(state.IMPORT_STEP)
        return

    _render_totals(inventory)
    _render_table(inventory)


def _render_totals(inventory: list[DatasetHealth]) -> None:
    col_sets, col_images, col_active, col_issues = st.columns(4)
    col_sets.metric("Datasets", len(inventory))
    col_images.metric("Images", sum(h.total_samples for h in inventory))
    col_active.metric("Active images", sum(h.active_samples for h in inventory))
    col_issues.metric(
        "With issues",
        sum(1 for h in inventory if h.has_issues),
        help="Datasets carrying at least one duplicate, missing description, or invalid image.",
    )


def _render_table(inventory: list[DatasetHealth]) -> None:
    active_id = st.session_state.get(state.ACTIVE_RUN_KEY)

    rows = [
        {
            "Dataset": f"{'▶ ' if h.run_id == active_id else ''}{h.concept_name}",
            "Imported": h.created_at.astimezone().strftime("%d %b %H:%M"),
            "Images": h.total_samples,
            "Active": h.active_samples,
            "Excluded": h.excluded_samples,
            "Duplicates": h.duplicate_count,
            "No description": h.missing_caption_count,
            "Invalid": h.invalid_count,
            "Median sharpness": h.median_sharpness,
        }
        for h in inventory
    ]

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "Duplicates": st.column_config.NumberColumn(help=DUPLICATE_HELP),
            "Median sharpness": st.column_config.NumberColumn(help=SHARPNESS_HELP),
        },
    )

    st.caption(
        "Click a column header to rank the datasets by it. ▶ marks the active one. "
        f"Duplicates: {DUPLICATE_HELP} Median sharpness: {SHARPNESS_HELP}"
    )
