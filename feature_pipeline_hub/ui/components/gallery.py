"""Step 2: curation grid — square thumbnails with editable captions.

Renders whichever dataset is active in the context bar.
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
    "Invalid": lambda s, trigger: not s.is_valid and not s.is_excluded,
    "No caption": lambda s, trigger: not s.is_excluded
    and quality.describes_nothing(s.caption, trigger),
    "Excluded": lambda s, trigger: s.is_excluded,
}


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    with st.container(horizontal=True, vertical_alignment="center"):
        chosen_filter = st.segmented_control(
            "Show",
            list(FILTERS),
            default="Active",
            required=True,
            label_visibility="collapsed",
            key=f"gallery_filter_{run.run_id}",
        )
        with st.popover("", icon=":material/tune:", help="View options"):
            columns_per_row = st.select_slider(
                "Columns", options=[2, 3, 4, 5, 6], value=4, key=f"gallery_cols_{run.run_id}"
            )

    trigger = run.concept.trigger_word
    samples = [s for s in run.concept.samples if FILTERS[chosen_filter](s, trigger)]
    if not samples:
        st.caption(f"No images match '{chosen_filter}'.")
        return

    for row_start in range(0, len(samples), columns_per_row):
        row = samples[row_start : row_start + columns_per_row]
        for column, sample in zip(st.columns(columns_per_row), row):
            with column:
                _render_card(sample, run)


def _render_card(sample: DatasetSample, run: IngestionRun) -> None:
    state.render_thumbnail(sample.image_path)

    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"{Path(sample.image_path).name} · {_status(sample)}")

        excluded = sample.is_excluded
        if st.button(
            "",
            icon=":material/undo:" if excluded else ":material/block:",
            type="tertiary",
            help="Restore into the dataset" if excluded else "Exclude from the dataset",
            key=f"toggle_{run.run_id}_{sample.sample_id}",
        ):
            state.set_excluded([sample.sample_id], not excluded)
            st.rerun()

    caption_key = state.caption_widget_key(run.run_id, sample.sample_id)
    st.text_area(
        "Caption",
        value=sample.caption,
        key=caption_key,
        height=68,
        label_visibility="collapsed",
        on_change=state.persist_caption,
        args=(sample.sample_id, caption_key),
    )

    if not sample.is_valid:
        st.caption("⚠️ " + "; ".join(sample.validation_errors))


def _status(sample: DatasetSample) -> str:
    if sample.is_excluded:
        return "excluded"
    if sample.is_duplicate:
        return "duplicate"
    return f"{sample.metrics.width}×{sample.metrics.height}"
