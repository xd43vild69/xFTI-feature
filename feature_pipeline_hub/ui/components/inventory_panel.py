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

    if message := st.session_state.pop("inventory_message", None):
        st.success(message)
    if error := st.session_state.pop("inventory_error", None):
        st.error(error)

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

    event = st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="inventory_table",
        column_config={
            "Duplicates": st.column_config.NumberColumn(help=DUPLICATE_HELP),
            "Median sharpness": st.column_config.NumberColumn(help=SHARPNESS_HELP),
        },
    )

    selected = [inventory[i] for i in event.selection.rows if i < len(inventory)]
    if selected:
        _render_cleanup(selected)

    st.caption(
        "Click a column header to rank the datasets by it. ▶ marks the active one. "
        f"Duplicates: {DUPLICATE_HELP} Median sharpness: {SHARPNESS_HELP}"
    )


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _render_cleanup(selected: list[DatasetHealth]) -> None:
    plans = {h.run_id: state.plan_lora_cleanup(h.run_id) for h in selected}
    blocked = [h for h in selected if plans[h.run_id].blocked_reason]
    cleanable = [h for h in selected if not plans[h.run_id].blocked_reason]
    file_count = sum(len(plans[h.run_id].files) for h in cleanable)
    total_bytes = sum(plans[h.run_id].total_bytes for h in cleanable)

    with st.container(border=True):
        st.markdown(f"**Clean up {len(selected)} selected dataset(s)**")
        if blocked:
            st.warning(
                "Training still running, skipped: "
                + ", ".join(h.concept_name for h in blocked)
                + ". Stop it first."
            )
        if cleanable:
            st.caption(
                f"Removes {sum(h.total_samples for h in cleanable)} images across "
                f"{len(cleanable)} dataset(s) and {file_count} LoRA .safetensors file(s), "
                f"freeing {_format_size(total_bytes)} of weights."
            )
        if st.button(
            f"Clean up {len(cleanable)} selected…" if cleanable else "Clean up…",
            icon=":material/delete_sweep:",
            disabled=not cleanable,
            key="inventory_cleanup",
        ):
            _confirm_cleanup(cleanable, file_count, total_bytes)


@st.dialog("Delete datasets and their LoRAs?")
def _confirm_cleanup(datasets: list[DatasetHealth], file_count: int, total_bytes: int) -> None:
    st.write(
        f"This permanently deletes **{len(datasets)}** dataset(s) (images, captions and "
        f"curation) and **{file_count}** LoRA .safetensors file(s) "
        f"({_format_size(total_bytes)}). It cannot be undone."
    )
    st.markdown("\n".join(f"- {h.concept_name} · {h.total_samples} images" for h in datasets))
    st.caption(
        "Training logs and metrics are kept. A dataset imported from a folder of your "
        "own leaves that folder untouched."
    )
    if st.button("Delete", type="primary", icon=":material/delete_forever:"):
        done = 0
        files = 0
        freed = 0
        failures: list[str] = []
        for h in datasets:
            deleted, size, errors = state.clean_up_dataset(h.run_id)
            files += deleted
            freed += size
            if errors:
                failures.append(f"{h.concept_name}: " + "; ".join(errors))
            else:
                done += 1
        message = (
            f"Deleted {done} dataset(s): {files} LoRA file(s), {_format_size(freed)} freed."
        )
        if failures:
            st.session_state["inventory_error"] = (
                "Kept because some files could not be deleted: " + " | ".join(failures)
            )
        st.session_state["inventory_message"] = message
        st.rerun()
