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
    # The companion to the normalize dialog: pick this, select what it shows, resize.
    "Oversized": lambda s, trigger: not s.is_excluded
    and max(s.metrics.width, s.metrics.height) > state.MAX_TRAINING_SIDE,
    "Excluded": lambda s, trigger: s.is_excluded,
}


@st.dialog("Editar captions")
def _render_caption_dialog(run: IngestionRun, visible: list[DatasetSample]) -> None:
    """Batch caption edits: replace a word, or append one to the end.

    Both halves share the scope selector, which is the part that used to be
    missing — replacing always hit every image in the dataset, filter and
    selection alike, without saying so.
    """
    selected = state.selected_samples(run)
    scopes = {
        "Seleccionadas": selected,
        "Filtro actual": visible,
        "Todas": run.concept.samples,
    }

    mode = st.radio(
        "Acción",
        ["Reemplazar", "Añadir"],
        horizontal=True,
        key=f"caption_mode_{run.run_id}",
        captions=[
            "Cambia una palabra exacta por otra.",
            "Añade una palabra al final, si no está ya.",
        ],
    )

    scope = st.radio(
        "Aplicar a",
        list(scopes),
        horizontal=True,
        # Defaults to the selection when there is one, since that is the more
        # deliberate choice; falls back to the whole dataset when there is not.
        index=0 if selected else 2,
        key=f"caption_scope_{run.run_id}",
    )
    targets = scopes[scope]

    if not targets:
        st.warning("No hay imágenes en este alcance. Selecciona algunas en la galería.")
        return

    st.caption(f"{len(targets)} imagen(es) en el alcance.")

    if mode == "Reemplazar":
        _render_replace_form(run, targets)
    else:
        _render_append_form(run, targets)


def _render_replace_form(run: IngestionRun, targets: list[DatasetSample]) -> None:
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

    matching_samples, total_matches = state.preview_replace_counts(targets, old_word)

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
        samples_cnt, total_cnt = state.batch_replace_caption_word(targets, old_word, new_word)
        st.success(f"¡Se reemplazaron {total_cnt} coincidencia(s) en {samples_cnt} imagen(es)!")
        st.rerun()


def _render_append_form(run: IngestionRun, targets: list[DatasetSample]) -> None:
    word = st.text_input(
        "Palabra o frase a añadir al final",
        placeholder="ej. outdoors",
        key=f"append_word_{run.run_id}",
        help="Se salta las imágenes cuyo caption ya la contenga, sin importar mayúsculas "
        "ni en qué posición esté.",
    )

    pending, already = state.preview_append_counts(targets, word.strip())

    if word.strip():
        if pending:
            note = f"Se añadirá a **{pending}** imagen(es)."
            if already:
                note += f" **{already}** ya la tiene(n) y no se tocarán."
            st.info(note)
        else:
            st.warning("Todas las imágenes del alcance ya contienen esa palabra.")

    if st.button(
        f"Añadir a {pending} imagen(es)",
        type="primary",
        disabled=(pending == 0),
        use_container_width=True,
    ):
        updated = state.batch_append_caption_word(targets, word.strip())
        st.success(f"¡Se añadió '{word.strip()}' a {updated} imagen(es)!")
        st.rerun()


@st.dialog("Normalizar resolución")
def _render_normalize_dialog(run: IngestionRun, visible: list[DatasetSample]) -> None:
    """Downscale oversized images to 1024px on the longest side.

    Same scope selector as the caption dialog, filtered down to the images that
    actually exceed the limit: an image already within it is not offered, since
    re-encoding it would cost quality and gain nothing.
    """
    st.write(
        f"Reduce las imágenes cuyo lado mayor supera **{state.MAX_TRAINING_SIDE}px**, "
        "conservando la proporción (filtro Lanczos, sin recortar). Los archivos "
        "originales no se tocan: las copias van a la carpeta del dataset y las "
        "imágenes pasan a apuntar ahí."
    )

    scopes = {
        "Seleccionadas": state.selected_samples(run),
        "Filtro actual": visible,
        "Todas": run.concept.samples,
    }
    scope = st.radio(
        "Aplicar a",
        list(scopes),
        horizontal=True,
        index=0 if scopes["Seleccionadas"] else 2,
        key=f"normalize_scope_{run.run_id}",
    )

    targets = state.oversized_samples(scopes[scope])
    if not targets:
        st.info("No hay imágenes por encima del límite en este alcance.")
        return

    biggest = max(max(s.metrics.width, s.metrics.height) for s in targets)
    st.caption(
        f"{len(targets)} imagen(es) por encima del límite · lado mayor máximo: {biggest}px."
    )
    with st.expander("Ver cuáles"):
        for sample in targets:
            st.caption(
                f"{Path(sample.image_path).name} · "
                f"{sample.metrics.width}×{sample.metrics.height}"
            )

    if st.button(
        f"Normalizar {len(targets)} imagen(es)",
        type="primary",
        use_container_width=True,
        key=f"btn_normalize_apply_{run.run_id}",
    ):
        with st.spinner("Redimensionando…"):
            outcomes = state.normalize_samples(run, targets)

        failed = [o for o in outcomes if o.error]
        resized = len(outcomes) - len(failed)
        message = f"{resized} imagen(es) normalizada(s) a {state.MAX_TRAINING_SIDE}px."
        if failed:
            message += f" {len(failed)} fallaron: " + "; ".join(
                f"{o.filename} ({o.error})" for o in failed
            )
        st.session_state["append_message"] = message
        st.rerun()


@st.dialog("Añadir imágenes")
def _render_append_dialog(run: IngestionRun) -> None:
    st.write(
        "Añade imágenes que falten a este dataset. La curación ya hecha "
        "(exclusiones, captions editados) se conserva."
    )

    uploaded_files = st.file_uploader(
        "Imágenes (PNG, JPG, WebP) y captions .txt opcionales",
        type=["png", "jpg", "jpeg", "webp", "txt"],
        accept_multiple_files=True,
        key=f"append_files_{run.run_id}",
        label_visibility="collapsed",
    )

    if st.button(
        "Añadir al dataset",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        added = state.append_images(run, uploaded_files)
        if not added:
            st.warning("No se añadió ninguna imagen nueva.")
            return

        duplicates = sum(1 for s in added if s.is_duplicate)
        invalid = sum(1 for s in added if not s.is_valid)
        notes = [f"{len(added)} imagen(es) añadida(s)"]
        if duplicates:
            notes.append(f"{duplicates} marcada(s) como duplicada(s)")
        if invalid:
            notes.append(f"{invalid} con errores de validación")

        st.session_state["append_message"] = " · ".join(notes) + "."
        st.rerun()


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    if message := st.session_state.pop("append_message", None):
        st.success(message)

    image_zoom.inject_styles()

    # Keyboard shortcut listener for F2
    components.html(
        """
        <script>
        (function() {
            try {
                const parentWin = window.parent;
                const parentDoc = parentWin.document;

                if (parentWin._f2Handler) {
                    parentDoc.removeEventListener('keydown', parentWin._f2Handler, true);
                    parentWin.removeEventListener('keydown', parentWin._f2Handler, true);
                }

                parentWin._f2Handler = function(e) {
                    if (e.key === 'F2' || e.code === 'F2' || e.keyCode === 113) {
                        // 1. Selector directo por clase Streamlit del botón
                        let btn = parentDoc.querySelector('[class*="st-key-btn_rename_"] button');

                        // 2. Fallback: Búsqueda exhaustiva en todos los botones
                        if (!btn) {
                            const buttons = Array.from(parentDoc.querySelectorAll('button'));
                            btn = buttons.find(b => {
                                const txt = (b.innerText || b.textContent || '').toLowerCase();
                                const aria = (b.getAttribute('aria-label') || '').toLowerCase();
                                return txt.includes('captions') || txt.includes('f2') || aria.includes('captions') || aria.includes('f2');
                            });
                        }

                        if (btn) {
                            e.preventDefault();
                            e.stopPropagation();
                            btn.click();
                            btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: parentWin }));
                        }
                    }
                };

                parentDoc.addEventListener('keydown', parentWin._f2Handler, true);
                parentWin.addEventListener('keydown', parentWin._f2Handler, true);
            } catch (err) {
                console.error('Error attaching F2 shortcut listener:', err);
            }
        })();
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

    # Rendered into the same toolbar row as the filters above. Also needs to run
    # before the grid: selecting all writes the checkboxes' session keys, which
    # Streamlit only allows while those widgets do not yet exist on this run.
    recaption_panel.render_toolbar(run, samples, container=toolbar)

    with toolbar:
        if st.button(
            "Captions (F2)",
            icon=":material/find_replace:",
            help="Reemplazar o añadir una palabra en los captions (F2)",
            key=f"btn_rename_{run.run_id}",
        ):
            _render_caption_dialog(run, samples)

        if st.button(
            "Añadir",
            icon=":material/add_photo_alternate:",
            help="Añadir imágenes a este dataset sin perder la curación",
            key=f"btn_append_{run.run_id}",
        ):
            _render_append_dialog(run)

        oversized = state.oversized_samples(run.concept.samples)
        if st.button(
            f"Normalizar ({len(oversized)})" if oversized else "Normalizar",
            icon=":material/photo_size_select_large:",
            help=f"Reducir a {state.MAX_TRAINING_SIDE}px el lado mayor de las imágenes "
            "que lo superan",
            disabled=not oversized,
            key=f"btn_normalize_{run.run_id}",
        ):
            _render_normalize_dialog(run, samples)

    # After the toolbar, not before it: a filter matching nothing is exactly when
    # adding images is most likely to be what the user wants.
    if not samples:
        st.caption(f"No images match '{chosen_filter}'.")
        return

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
        sample=sample,
        run_id=run.run_id,
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
