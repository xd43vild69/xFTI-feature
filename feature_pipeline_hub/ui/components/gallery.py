"""Step 2: curation grid — square thumbnails with editable captions.

Renders whichever dataset is active in the context bar.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import state
from components import image_zoom, recaption_panel
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


@st.dialog("Renombrar palabra en captions")
def _render_rename_dialog(run: IngestionRun) -> None:
    st.write("Busca una palabra exacta en los captions de las imágenes y reemplázala.")

    old_word = st.text_input(
        "Palabra exacta a buscar (case-sensitive)",
        placeholder="ej. cat",
        key=f"rename_old_{run.run_id}",
    )
    new_word = st.text_input(
        "Reemplazar por",
        placeholder="ej. dog",
        key=f"rename_new_{run.run_id}",
    )

    matching_samples, total_matches = state.preview_replace_counts(
        run.concept.samples, old_word
    )

    if old_word:
        if total_matches > 0:
            st.info(
                f"Se encontraron **{total_matches}** coincidencia(s) en **{matching_samples}** imagen(es)."
            )
        else:
            st.warning(f"No se encontró la palabra exacta '{old_word}'.")

    if st.button(
        f"Reemplazar {total_matches} coincidencia(s)",
        type="primary",
        disabled=(total_matches == 0),
        use_container_width=True,
    ):
        samples_cnt, total_cnt = state.batch_replace_caption_word(
            run, old_word, new_word
        )
        st.success(f"¡Se reemplazaron {total_cnt} coincidencia(s) en {samples_cnt} imagen(es)!")
        st.rerun()


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    image_zoom.inject_styles()

    # Keyboard shortcut listener for F2
    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc._f2ListenerAdded) {
            doc._f2ListenerAdded = true;
            doc.addEventListener('keydown', function(e) {
                if (e.key === 'F2') {
                    e.preventDefault();
                    const buttons = Array.from(doc.querySelectorAll('button'));
                    const btn = buttons.find(b => b.innerText && b.innerText.includes('Renombrar (F2)'));
                    if (btn) btn.click();
                }
            });
        }
        </script>
        """,
        height=0,
        width=0,
    )

    toolbar = st.container(horizontal=True, vertical_alignment="center")
    with toolbar:
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
                "Columns", options=[2, 3, 4, 5, 6], value=6, key=f"gallery_cols_{run.run_id}"
            )

    trigger = run.concept.trigger_word
    samples = [s for s in run.concept.samples if FILTERS[chosen_filter](s, trigger)]
    if not samples:
        st.caption(f"No images match '{chosen_filter}'.")
        return

    # Rendered into the same toolbar row as the filters above. Also needs to run
    # before the grid: selecting all writes the checkboxes' session keys, which
    # Streamlit only allows while those widgets do not yet exist on this run.
    recaption_panel.render_toolbar(run, samples, container=toolbar)

    with toolbar:
        if st.button(
            "Renombrar (F2)",
            icon=":material/find_replace:",
            help="Reemplazar palabra en los captions (F2)",
            key=f"btn_rename_{run.run_id}",
        ):
            _render_rename_dialog(run)

    thumbnail_size = state.thumbnail_size_for_columns(columns_per_row)
    for row_start in range(0, len(samples), columns_per_row):
        row = samples[row_start : row_start + columns_per_row]
        for column, sample in zip(st.columns(columns_per_row), row):
            with column:
                _render_card(sample, run, thumbnail_size)


def _render_card(sample: DatasetSample, run: IngestionRun, thumbnail_size: int) -> None:
    image_zoom.clickable_thumbnail(
        sample.image_path,
        f"gallery_{run.run_id}_{sample.sample_id}",
        size=thumbnail_size,
    )

    with st.container(horizontal=True, vertical_alignment="center"):
        st.checkbox(
            Path(sample.image_path).name,
            key=state.selection_key(run.run_id, sample.sample_id),
            help="Select for batch recaptioning",
            label_visibility="collapsed",
        )
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
