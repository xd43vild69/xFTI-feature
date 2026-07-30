"""Global context bar: which dataset is active and how healthy it is.

Rendered by the entrypoint before the page body, so it shows on every step. This
is the single place that answers "which dataset am I on" — previously that was
repeated in the sidebar, the ingestion caption and the gallery caption, and the
counters were duplicated between the ingestion and quality panels.
"""

import streamlit as st

import state
from feature_pipeline.application import quality_service as quality
from feature_pipeline.domain.models import IngestionRun


def render() -> None:
    runs = state.list_runs()
    if not runs:
        return

    run_ids = [r.run_id for r in runs]
    labels = {r.run_id: state.format_run_label(r) for r in runs}

    active_id = st.session_state.get(state.ACTIVE_RUN_KEY)
    if active_id not in run_ids:
        active_id = run_ids[0]
        state.set_active_run(active_id)

    with st.container(horizontal=True, vertical_alignment="center"):
        # No widget key on purpose: ACTIVE_RUN_KEY is the single source of truth,
        # and a keyed widget would override `index` and keep showing the previous
        # run after a fresh ingestion selects a new one.
        selected = st.selectbox(
            "Active dataset",
            options=run_ids,
            index=run_ids.index(active_id),
            format_func=lambda run_id: labels[run_id],
            label_visibility="collapsed",
            width=320,
        )
        if selected != active_id:
            state.set_active_run(selected)
            st.rerun()

        run = state.active_run()
        if run is not None:
            _render_counters(run)
            _render_details(run)

        if st.button(
            "Revalidate",
            icon=":material/rule:",
            type="tertiary",
            help="Re-check every dataset against the current validation rules "
            "(fixes stale verdicts after a rule change)",
        ):
            changed = state.revalidate_all_runs()
            st.session_state["revalidate_message"] = (
                f"{changed} sample(s) updated." if changed else "Nothing changed."
            )
            st.rerun()

        if st.button(
            "Delete", icon=":material/delete:", type="tertiary", help="Delete this dataset"
        ):
            _confirm_delete(selected, labels[selected])

    if message := st.session_state.pop("revalidate_message", None):
        st.toast(message, icon=":material/rule:")


def _render_counters(run: IngestionRun) -> None:
    """Headline counts, read from stored flags rather than recomputed.

    Duplicate detection is O(n²); it runs in the quality step and persists its
    verdict, so the bar can stay cheap on every rerun of every page.
    """
    samples = run.concept.samples
    active = [s for s in samples if not s.is_excluded]
    duplicates = sum(1 for s in active if s.is_duplicate)
    excluded = len(samples) - len(active)
    missing = len(quality.samples_missing_caption(samples, run.concept.trigger_word))

    st.caption(f"{len(active)} images")
    if duplicates:
        st.badge(f"{duplicates} duplicate", color="orange")
    if missing:
        st.badge(f"{missing} without description", color="orange")
    if excluded:
        st.badge(f"{excluded} excluded", color="gray")


def _render_details(run: IngestionRun) -> None:
    with st.popover("", icon=":material/info:", help="Dataset details"):
        st.caption("Concept")
        st.write(run.concept.concept_name)
        st.caption("Trigger word")
        st.code(run.concept.trigger_word, language=None)
        st.caption("Source")
        st.code(run.source_path, language=None)
        st.caption("Run id")
        st.code(run.run_id, language=None)


@st.dialog("Delete dataset")
def _confirm_delete(run_id: str, label: str) -> None:
    st.write(f"Delete **{label}**?")
    st.caption(
        "Uploaded images are removed from disk. Folders imported by path are left "
        "untouched — only the dataset record is deleted."
    )

    with st.container(horizontal=True):
        if st.button("Delete", icon=":material/delete:"):
            state.delete_run(run_id)
            st.rerun()
        if st.button("Cancel"):
            st.rerun()
