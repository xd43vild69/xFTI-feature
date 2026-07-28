"""Interactive curation grid: image cards with editable captions.

Renders whichever ingestion run is selected in the sidebar.
"""

from pathlib import Path

import streamlit as st

import state
from feature_pipeline.domain.models import DatasetSample, IngestionRun

FILTERS = {
    "All": lambda s: True,
    "Validation errors": lambda s: not s.is_valid,
    "Missing caption": lambda s: not s.original_caption,
}


def render() -> None:
    run = state.active_run()

    if run is None:
        st.info(
            "No ingestion selected. Import a dataset in the **Ingestion** tab, "
            "then pick one under **Stored ingestions** in the sidebar."
        )
        return

    if not run.concept.samples:
        st.warning("This ingestion has no images.")
        return

    st.caption(
        f"**{run.concept.concept_name}** · trigger `{run.concept.trigger_word}` · "
        f"{len(run.concept.samples)} images · source `{run.source_path}`"
    )

    col_filter, col_size = st.columns([3, 1])
    chosen_filter = col_filter.radio(
        "Show", list(FILTERS), horizontal=True, key=f"gallery_filter_{run.run_id}"
    )
    columns_per_row = col_size.select_slider(
        "Columns", options=[2, 3, 4, 5, 6], value=4, key=f"gallery_cols_{run.run_id}"
    )

    samples = [s for s in run.concept.samples if FILTERS[chosen_filter](s)]
    if not samples:
        st.info(f"No images match '{chosen_filter}'.")
        return

    for row_start in range(0, len(samples), columns_per_row):
        row = samples[row_start : row_start + columns_per_row]
        for column, sample in zip(st.columns(columns_per_row), row):
            with column:
                _render_card(sample, run)


def _render_card(sample: DatasetSample, run: IngestionRun) -> None:
    image_path = Path(sample.image_path)

    if image_path.exists():
        st.image(str(image_path), width="stretch")
    else:
        st.error(f"Missing file: {image_path.name}")

    st.caption(
        f"{image_path.name} · {sample.metrics.width}×{sample.metrics.height}"
        f"{'' if sample.is_valid else ' · ⚠️'}"
    )

    edited = st.text_area(
        "Caption",
        value=sample.caption,
        key=f"caption_{run.run_id}_{sample.sample_id}",
        height=80,
        label_visibility="collapsed",
    )
    if edited != sample.caption:
        state.save_caption(sample.sample_id, edited)

    if not sample.is_valid:
        st.caption("⚠️ " + "; ".join(sample.validation_errors))
