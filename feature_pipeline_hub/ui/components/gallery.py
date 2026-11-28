"""Step 2: curation grid — square thumbnails with editable captions.

Renders whichever dataset is active in the context bar.
"""

import hashlib
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


_DROPZONE_CSS = """
<style>
[class*="st-key-curate_dropzone_"] [data-testid="stFileUploaderDropzone"] {
    padding: 0.45rem 0.9rem !important;
    min-height: 2.6rem !important;
    border: 1.5px dashed rgba(128, 128, 128, 0.45) !important;
    border-radius: 0.55rem !important;
    transition: border-color 0.15s ease, background-color 0.15s ease !important;
}
[class*="st-key-curate_dropzone_"] [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4b9eff !important;
    background-color: rgba(75, 158, 255, 0.06) !important;
}
/* Hide uploaded file pills because files are ingested immediately on drop */
[class*="st-key-curate_dropzone_"] [data-testid="stFileUploaderFile"] {
    display: none !important;
}
</style>
"""


def _dropzone_version_key(run_id: str) -> str:
    return f"curate_drop_version_{run_id}"


def _distinct_uploads(uploaded_files: list) -> list:
    """Drop repeats of the same file within one batch.

    The page-wide drop handler below hands a drop to the uploader twice (a synthetic
    `drop` event, then a fallback that assigns the input's files), and the uploader
    appends both. The two copies are the same name, size and bytes, so keying on that
    removes exactly them without needing to know which path fired.
    """
    seen: set[tuple[str, str]] = set()
    distinct = []
    for f in uploaded_files:
        key = (f.name, hashlib.sha1(f.getvalue()).hexdigest())
        if key not in seen:
            seen.add(key)
            distinct.append(f)
    return distinct


def _ingest_dropped_files(run: IngestionRun, uploaded_files: list) -> None:
    """Immediately integrate dropped/selected files into `run` just like Import, then rerun."""
    uploaded_files = _distinct_uploads(uploaded_files)
    st.session_state[_dropzone_version_key(run.run_id)] = (
        st.session_state.get(_dropzone_version_key(run.run_id), 0) + 1
    )

    with st.spinner("Integrando imagen(es) al paquete del dataset…"):
        added = state.append_images(run, uploaded_files)

    if not added:
        st.session_state["append_warning"] = (
            "No se añadió ninguna imagen nueva (formato no soportado o archivo sin imagen asociada)."
        )
        st.rerun()

    duplicates = sum(1 for s in added if s.is_duplicate)
    invalid = sum(1 for s in added if not s.is_valid)
    sample_names = [Path(s.image_path).name for s in added[:3]]
    preview_names = ", ".join(sample_names) + ("…" if len(added) > 3 else "")

    notes = [f"{len(added)} imagen(es) integrada(s) ({preview_names})"]
    if duplicates:
        notes.append(f"{duplicates} marcada(s) como duplicada(s)")
    if invalid:
        notes.append(f"{invalid} con errores de validación")

    st.session_state["append_message"] = " · ".join(notes) + "."
    st.rerun()


@st.dialog("Añadir imágenes")
def _render_append_dialog(run: IngestionRun) -> None:
    st.write(
        "Suelta o selecciona imágenes para integrarlas inmediatamente a este dataset. "
        "La curación ya hecha (exclusiones, captions editados) se conserva."
    )

    dialog_v = st.session_state.get(f"append_dialog_v_{run.run_id}", 0)
    uploaded_files = st.file_uploader(
        "Imágenes (PNG, JPG, WebP) y captions .txt opcionales",
        type=["png", "jpg", "jpeg", "webp", "txt"],
        accept_multiple_files=True,
        key=f"append_files_{run.run_id}_v{dialog_v}",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.session_state[f"append_dialog_v_{run.run_id}"] = dialog_v + 1
        _ingest_dropped_files(run, uploaded_files)


def render() -> None:
    run = state.active_run()
    if run is None:
        st.info("No dataset selected yet.")
        if st.button("Go to import", icon=":material/arrow_forward:"):
            st.switch_page(state.IMPORT_STEP)
        return

    if message := st.session_state.pop("append_message", None):
        st.toast(message, icon=":material/check_circle:")
        st.success(message)
    if warning_msg := st.session_state.pop("append_warning", None):
        st.warning(warning_msg)

    st.html(_DROPZONE_CSS)
    image_zoom.inject_styles()

    # Keyboard shortcut listener (F2) + Full-page drag-and-drop forwarding into Curate dropzone
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
                        let btn = parentDoc.querySelector('[class*="st-key-btn_rename_"] button');
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

                // Clean up previous full-page drag-and-drop listeners if present
                if (parentWin._curateCleanupDnD) {
                    parentWin._curateCleanupDnD();
                }

                let dragDepth = 0;

                function getCurateDropzone() {
                    return parentDoc.querySelector('[class*="st-key-curate_dropzone_"] [data-testid="stFileUploaderDropzone"]');
                }

                function hasFiles(e) {
                    if (!e.dataTransfer || !e.dataTransfer.types) return false;
                    return Array.from(e.dataTransfer.types).includes('Files');
                }

                function ensureOverlay() {
                    let el = parentDoc.getElementById('fti-curate-drop-overlay');
                    if (!el) {
                        el = parentDoc.createElement('div');
                        el.id = 'fti-curate-drop-overlay';
                        el.style.cssText = [
                            'position: fixed',
                            'inset: 12px',
                            'z-index: 999999',
                            'display: none',
                            'align-items: center',
                            'justify-content: center',
                            'background: rgba(15, 23, 42, 0.58)',
                            'backdrop-filter: blur(3px)',
                            'border: 3px dashed #4b9eff',
                            'border-radius: 16px',
                            'pointer-events: none',
                            'color: #ffffff',
                            'font-family: system-ui, -apple-system, sans-serif',
                            'font-size: 1.25rem',
                            'font-weight: 600',
                            'box-shadow: 0 12px 40px rgba(0,0,0,0.35)'
                        ].join(';');
                        el.innerHTML = '<div style="padding: 1.2rem 2rem; background: rgba(30,41,59,0.92); border-radius: 12px; border: 1px solid rgba(75,158,255,0.5);">📥 Suelta tus imágenes aquí para integrarlas al dataset</div>';
                        parentDoc.body.appendChild(el);
                    }
                    return el;
                }

                function hideOverlay() {
                    dragDepth = 0;
                    const el = parentDoc.getElementById('fti-curate-drop-overlay');
                    if (el) el.style.display = 'none';
                }

                function onDragEnter(e) {
                    const dz = getCurateDropzone();
                    if (!dz || !hasFiles(e)) return;
                    dragDepth++;
                    if (!dz.contains(e.target)) {
                        ensureOverlay().style.display = 'flex';
                    }
                }

                function onDragOver(e) {
                    const dz = getCurateDropzone();
                    if (!dz || !hasFiles(e)) return;
                    if (!dz.contains(e.target)) {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'copy';
                        ensureOverlay().style.display = 'flex';
                    } else {
                        hideOverlay();
                    }
                }

                function onDragLeave(e) {
                    if (!getCurateDropzone()) {
                        hideOverlay();
                        return;
                    }
                    dragDepth = Math.max(0, dragDepth - 1);
                    if (dragDepth === 0 || e.clientX <= 0 || e.clientY <= 0 ||
                        e.clientX >= parentWin.innerWidth || e.clientY >= parentWin.innerHeight) {
                        hideOverlay();
                    }
                }

                function onDrop(e) {
                    const dz = getCurateDropzone();
                    if (!dz || !hasFiles(e)) return;
                    hideOverlay();
                    // Let native Streamlit dropzone handle direct drops on itself
                    if (dz.contains(e.target)) return;

                    const files = e.dataTransfer.files;
                    if (!files || files.length === 0) return;

                    e.preventDefault();
                    e.stopPropagation();

                    const dt = new DataTransfer();
                    for (let i = 0; i < files.length; i++) {
                        dt.items.add(files[i]);
                    }

                    // Primary path: react-dropzone onDrop handler on Streamlit's dropzone
                    const dropEvt = new DragEvent('drop', {
                        bubbles: true,
                        cancelable: true,
                        dataTransfer: dt
                    });
                    dz.dispatchEvent(dropEvt);

                    // Fallback: assign to underlying file input if drop event was ignored
                    const input = dz.querySelector('input[type="file"]') ||
                                  parentDoc.querySelector('[class*="st-key-curate_dropzone_"] input[type="file"]');
                    if (input && (!input.files || input.files.length === 0)) {
                        try {
                            input.files = dt.files;
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                        } catch (_) {}
                    }
                }

                parentDoc.addEventListener('dragenter', onDragEnter, true);
                parentDoc.addEventListener('dragover', onDragOver, true);
                parentDoc.addEventListener('dragleave', onDragLeave, true);
                parentDoc.addEventListener('drop', onDrop, true);

                parentWin._curateCleanupDnD = function() {
                    parentDoc.removeEventListener('dragenter', onDragEnter, true);
                    parentDoc.removeEventListener('dragover', onDragOver, true);
                    parentDoc.removeEventListener('dragleave', onDragLeave, true);
                    parentDoc.removeEventListener('drop', onDrop, true);
                    hideOverlay();
                };
            } catch (err) {
                console.error('Error attaching Curate listeners:', err);
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
                "Columns", options=[2, 3, 4, 5, 6], value=4, key=f"gallery_cols_{run.run_id}"
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

    drop_v = st.session_state.get(_dropzone_version_key(run.run_id), 0)
    dropped_files = st.file_uploader(
        "Arrastra y suelta imágenes en cualquier parte de Curate para integrarlas al paquete",
        type=["png", "jpg", "jpeg", "webp", "txt"],
        accept_multiple_files=True,
        key=f"curate_dropzone_{run.run_id}_v{drop_v}",
        label_visibility="collapsed",
    )
    if dropped_files:
        _ingest_dropped_files(run, dropped_files)

    # After the toolbar, not before it: a filter matching nothing is exactly when
    # adding images is most likely to be what the user wants.
    if not samples:
        if not run.concept.samples:
            st.info("Este dataset aún no tiene imágenes. Suelta tus imágenes arriba o en cualquier parte de esta pantalla para importarlas.")
        else:
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
        tooltip=Path(sample.image_path).name,
        size=thumbnail_size,
    )

    with st.container(horizontal=True, vertical_alignment="center"):
        st.checkbox(
            Path(sample.image_path).name,
            key=state.selection_key(run.run_id, sample.sample_id),
            help="Select for batch recaptioning",
            label_visibility="collapsed",
        )
        st.caption(_status(sample))

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
