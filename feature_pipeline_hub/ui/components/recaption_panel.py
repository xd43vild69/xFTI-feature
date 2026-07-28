"""Batch AI recaptioning from the curation grid.

Captions come from the same Qwen3-VL-4B that Krea 2 uses as its text encoder,
run in the LoRAlab checkout as a subprocess. A batch costs a few seconds of model
load plus roughly two seconds per image on GPU, so the run is shown with live
progress rather than a spinner.
"""

import streamlit as st

import state
from feature_pipeline.application import recaption_service
from feature_pipeline.domain.models import DatasetSample, IngestionRun
from feature_pipeline.infrastructure import recaption_runner

DETAILED_KEY = "recaption_detailed"


def render_toolbar(run: IngestionRun, visible: list[DatasetSample]) -> None:
    """Selection controls plus the recaption action, above the grid."""
    selected = state.selected_samples(run)

    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f"{len(selected)} selected")

        if st.button("All", type="tertiary", help="Select every image shown"):
            state.set_selection(run.run_id, visible, True)
            st.rerun()

        if st.button("None", type="tertiary", help="Clear the selection"):
            state.set_selection(run.run_id, run.concept.samples, False)
            st.rerun()

        try:
            recaption_runner.resolve_environment()
        except recaption_runner.RecaptionUnavailable as exc:
            st.caption(f":material/info: {exc}")
            return

        with st.popover("", icon=":material/tune:", help="Recaption settings"):
            st.checkbox(
                "Detailed captions",
                key=DETAILED_KEY,
                help="2-4 sentences covering pose, clothing, lighting and setting, "
                "instead of a single factual sentence. Slower (~5s per image).",
            )

        if st.button(
            f"Recaption {len(selected)}",
            icon=":material/auto_awesome:",
            disabled=not selected,
            help="Select images above, then click to regenerate captions with Qwen3-VL. ~2s per image on GPU.",
        ):
            _run_batch(run, selected)


def _run_batch(run: IngestionRun, samples: list[DatasetSample]) -> None:
    total = len(samples)
    detailed = bool(st.session_state.get(DETAILED_KEY))

    progress = st.progress(0.0, text="Loading Qwen3-VL…")
    done = failed = 0

    with st.status(f"Recaptioning {total} image(s)", expanded=True) as status:
        for event in recaption_service.recaption_samples(
            samples, run.concept.trigger_word, detailed=detailed
        ):
            if event.kind == "loaded":
                status.write(f"Model loaded on {event.device} in {event.seconds}s.")

            elif event.kind == "caption":
                sample = _find(samples, event.sample_id)
                if sample is not None:
                    state.apply_recaption(sample, event.caption)
                done += 1
                progress.progress(done / total, text=f"{done}/{total} captioned")

            elif event.kind == "error":
                failed += 1
                status.write(f":material/warning: {event.message}")

            elif event.kind == "failed":
                status.update(label="Recaption failed", state="error")
                st.error(event.message)
                return

        label = f"Recaptioned {done}/{total}"
        if failed:
            label += f" · {failed} failed"
        status.update(label=label, state="complete", expanded=bool(failed))

    # st.rerun() below refreshes the caption widgets (their keys are versioned by
    # apply_recaption), but it also wipes whatever the status/progress widgets
    # above just rendered — a plain rerun happens fast enough that the summary
    # never gets read. st.toast is the one widget documented to survive exactly
    # one rerun, so it's what carries the result across.
    st.toast(label, icon=":material/auto_awesome:" if not failed else ":material/warning:")
    st.rerun()


def _find(samples: list[DatasetSample], sample_id: str) -> DatasetSample | None:
    return next((s for s in samples if s.sample_id == sample_id), None)
