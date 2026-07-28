"""Interactive curation grid: image cards with editable captions.

Renders whichever ingestion run is selected in the sidebar.
"""

from pathlib import Path

import streamlit as st

import state
from feature_pipeline.application import quality_service as quality
from feature_pipeline.domain.models import DatasetSample, IngestionRun

# Each filter takes the sample and the concept's trigger word.
FILTERS = {
    "Active": lambda s, trigger: not s.is_excluded,
    "All": lambda s, trigger: True,
    "Duplicates": lambda s, trigger: s.is_duplicate and not s.is_excluded,
    "Validation errors": lambda s, trigger: not s.is_valid and not s.is_excluded,
    "Missing caption": lambda s, trigger: not s.is_excluded
    and quality.describes_nothing(s.caption, trigger),
    "Excluded": lambda s, trigger: s.is_excluded,
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

    trigger = run.concept.trigger_word
    samples = [s for s in run.concept.samples if FILTERS[chosen_filter](s, trigger)]
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

    state.render_thumbnail(sample.image_path)

    badges = ""
    if sample.is_excluded:
        badges += " · 🚫 excluded"
    elif sample.is_duplicate:
        badges += " · 👯 duplicate"
    if not sample.is_valid:
        badges += " · ⚠️"

    st.caption(f"{image_path.name} · {sample.metrics.width}×{sample.metrics.height}{badges}")

    caption_key = state.caption_widget_key(run.run_id, sample.sample_id)
    st.text_area(
        "Caption",
        value=sample.caption,
        key=caption_key,
        height=80,
        label_visibility="collapsed",
        on_change=state.persist_caption,
        args=(sample.sample_id, caption_key),
    )

    action = "Restore" if sample.is_excluded else "Exclude"
    if st.button(
        action, key=f"toggle_{run.run_id}_{sample.sample_id}", width="stretch"
    ):
        state.set_excluded([sample.sample_id], not sample.is_excluded)
        st.rerun()

    if not sample.is_valid:
        st.caption("⚠️ " + "; ".join(sample.validation_errors))
