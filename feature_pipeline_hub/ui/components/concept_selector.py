"""Sidebar selector for the active ingestion run.

Lives in the sidebar rather than inside a tab: Gallery, Quality, and Export all
read the same active run, so a single global selector keeps them in sync.
"""

import streamlit as st

import state


def render() -> None:
    st.subheader("Stored ingestions")

    runs = state.list_runs()
    if not runs:
        st.caption("No ingestions stored yet. Import a dataset in the Ingestion tab.")
        return

    run_ids = [r.run_id for r in runs]
    labels = {r.run_id: state.format_run_label(r) for r in runs}

    active_id = st.session_state.get(state.ACTIVE_RUN_KEY)
    if active_id not in run_ids:
        active_id = run_ids[0]
        state.set_active_run(active_id)

    # No widget key on purpose: ACTIVE_RUN_KEY is the single source of truth, and a
    # keyed widget would override `index` and keep showing the previous run after a
    # fresh ingestion selects a new one.
    selected = st.selectbox(
        "Active dataset",
        options=run_ids,
        index=run_ids.index(active_id),
        format_func=lambda run_id: labels[run_id],
    )
    if selected != active_id:
        state.set_active_run(selected)
        st.rerun()

    st.caption(f"`{selected}`")

    confirm_key = f"confirm_delete_{selected}"
    if st.session_state.get(confirm_key):
        st.warning("Delete this ingestion permanently?")
        col_yes, col_no = st.columns(2)
        if col_yes.button("Yes, delete", type="primary", width="stretch"):
            state.delete_run(selected)
            st.session_state.pop(confirm_key, None)
            st.rerun()
        if col_no.button("Cancel", width="stretch"):
            st.session_state.pop(confirm_key, None)
            st.rerun()
    elif st.button("🗑️ Delete ingestion", width="stretch"):
        st.session_state[confirm_key] = True
        st.rerun()
