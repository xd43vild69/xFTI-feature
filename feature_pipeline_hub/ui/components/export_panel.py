"""Step 4: materialize the curated dataset for training.

Writes the Active (non-excluded) samples of the current run as image+caption
pairs into training_runtime/datasets/<name>/ — exactly the flat layout the
training scripts (Step 5) expect. The recaption checkbox selection is a
per-session, ephemeral thing; export always uses the same "Active" criterion
already visible as a filter in Curate, so what you excluded there is what
stays out of the trained dataset.

Confirmation is an inline two-step (not st.dialog): exporting overwrites a
folder, and a plain widget flow is what AppTest can actually drive end to end.
"""

import time

import streamlit as st

import state
import step_telemetry
from feature_pipeline.application.dataset_service import build_manifest
from feature_pipeline.application.export_service import export_active_samples
from feature_pipeline.domain.models import DatasetManifest, IngestionRun
from feature_pipeline.infrastructure.storage import training_dataset_dir

PENDING_KEY = "export_pending_destination"


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    if message := st.session_state.pop("export_message", None):
        st.success(message)
    if delta := st.session_state.pop("export_delta", None):
        st.caption("\n\n".join(delta))

    active_count = sum(1 for s in run.concept.samples if not s.is_excluded)
    excluded_count = len(run.concept.samples) - active_count

    st.caption(
        f"{active_count} image(s) will be exported"
        + (f" · {excluded_count} excluded, left out" if excluded_count else "")
    )

    col_destination, col_version = st.columns([3, 1])
    with col_destination:
        destination_name = st.text_input(
            "Destination folder name",
            value=run.concept.concept_name,
            help="Written to training_runtime/datasets/<name>/. If that folder "
            "already has content, it is replaced entirely.",
        )
    with col_version:
        version_tag = st.text_input(
            "Version tag",
            value=_suggested_version_tag(run.concept.concept_id),
            help="Labels this export so its quality snapshot can be compared "
            "against earlier ones.",
        )

    _render_previous_version(run.concept.concept_id)

    pending = st.session_state.get(PENDING_KEY)

    if pending == destination_name and destination_name:
        _render_confirmation(run, destination_name, version_tag)
        return

    if pending is not None and pending != destination_name:
        # The name changed after asking for confirmation — the old confirmation
        # no longer refers to what's about to be written, so drop it silently.
        st.session_state.pop(PENDING_KEY, None)

    existing = training_dataset_dir(destination_name) if destination_name else None
    if existing and existing.is_dir() and any(existing.iterdir()):
        st.caption(
            f":material/warning: '{destination_name}' already has content — it will be replaced."
        )

    if st.button(
        "Export dataset",
        icon=":material/upload_file:",
        disabled=not destination_name or active_count == 0,
    ):
        st.session_state[PENDING_KEY] = destination_name
        st.rerun()


def _render_confirmation(run: IngestionRun, destination_name: str, version_tag: str) -> None:
    target = training_dataset_dir(destination_name)
    if target.is_dir() and any(target.iterdir()):
        st.warning(f"This replaces everything currently in '{destination_name}'.")
    st.write(f"Export to `training_runtime/datasets/{destination_name}/`?")

    with st.container(horizontal=True):
        if st.button("Confirm export", icon=":material/upload_file:"):
            _export_and_version(run, destination_name, version_tag)
            st.rerun()
        if st.button("Cancel"):
            st.session_state.pop(PENDING_KEY, None)
            st.rerun()


def _export_and_version(run: IngestionRun, destination_name: str, version_tag: str) -> None:
    """Write the dataset, then record what was written as a comparable snapshot.

    The manifest is built from the same in-memory run that was just exported, so the
    snapshot describes exactly those files. Failing to record a version must not make
    a successful export look failed, so that half is reported separately.
    """
    started = time.time()
    previous = state.latest_dataset_version(run.concept.concept_id)
    result = export_active_samples(run, destination_name)

    st.session_state.pop(PENDING_KEY, None)
    message = (
        f"Exported {result.exported_count} image(s) to "
        f"training_runtime/datasets/{destination_name}/"
    )

    if version_tag:
        manifest = build_manifest(run.concept, version_tag, destination_name)
        state.save_dataset_version(run.concept, version_tag, manifest, result.destination)
        message += f" · recorded as `{version_tag}`"
        st.session_state["export_delta"] = _describe_delta(
            previous.manifest if previous else None, manifest
        )

    st.session_state["export_message"] = message
    step_telemetry.record_step(run.run_id, "export", time.time() - started, error_count=0)


def _describe_delta(previous: DatasetManifest | None, current: DatasetManifest) -> list[str]:
    """Plain-language differences against the last export of this concept."""
    if previous is None:
        return [f"First recorded version · content hash `{current.content_hash[:12]}`"]

    if previous.content_hash == current.content_hash:
        return [f"Identical to `{previous.version}` — same images, same captions."]

    lines = [f"Compared with `{previous.version}`:"]
    for label, before, after, fmt in (
        ("Samples", previous.total_samples, current.total_samples, "{:+d}"),
        ("Duplicates", previous.duplicate_count, current.duplicate_count, "{:+d}"),
        ("Median sharpness", previous.median_sharpness, current.median_sharpness, "{:+.0f}"),
        (
            "Mean caption words",
            previous.caption_word_stats.get("mean", 0.0),
            current.caption_word_stats.get("mean", 0.0),
            "{:+.1f}",
        ),
    ):
        if before != after:
            lines.append(f"· {label}: {before} → {after} ({fmt.format(after - before)})")

    return lines


def _suggested_version_tag(concept_id: str) -> str:
    """Next tag in the vN sequence, so repeated exports don't all share one label."""
    previous = state.latest_dataset_version(concept_id)
    if previous is None:
        return "v1"

    tag = previous.version_tag
    if tag.startswith("v") and tag[1:].isdigit():
        return f"v{int(tag[1:]) + 1}"
    return tag


def _render_previous_version(concept_id: str) -> None:
    previous = state.latest_dataset_version(concept_id)
    if previous is None:
        return

    stamp = previous.created_at.strftime("%Y-%m-%d %H:%M")
    st.caption(
        f":material/history: Last exported as `{previous.version_tag}` on {stamp} · "
        f"{previous.manifest.total_samples} image(s)"
    )
