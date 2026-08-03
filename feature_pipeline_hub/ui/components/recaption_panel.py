"""Batch AI recaptioning from the curation grid.

Captions come from the same Qwen3-VL-4B that Krea 2 uses as its text encoder,
run as a subprocess against the training runtime's own venv and model copy
(see recaption_runner.py). A batch costs a few seconds of model load plus
roughly two seconds per image on GPU, so the run is shown with live progress
rather than a spinner.
"""

import time

import streamlit as st

import state
import step_telemetry
from feature_pipeline.application import recaption_service
from feature_pipeline.domain.models import DatasetSample, IngestionRun
from feature_pipeline.infrastructure import recaption_runner

MODE_KEY = "recaption_mode"

# What each mode leaves *out* is the point — that is what the trigger word ends up
# carrying. See domain/caption_schema.py.
MODE_HELP = {
    "subject": "Shot, pose, clothing, background and light. Says nothing about the "
    "face, hair or build, so the trigger word learns those.",
    "location": "Time of day, weather, light and whatever is passing through. Says "
    "nothing about the architecture or style, so the trigger word learns those.",
}


def render_toolbar(
    run: IngestionRun,
    visible: list[DatasetSample],
    container: "st.delta_generator.DeltaGenerator | None" = None,
) -> None:
    """Selection controls plus the recaption action.

    Renders into `container` when given, so it shares a row with the
    gallery's own filter/rename buttons instead of a separate one.
    """
    selected = state.selected_samples(run)

    ctx = container if container is not None else st.container(
        horizontal=True, vertical_alignment="center"
    )
    with ctx:
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

        training_active = state.is_training_active()
        if training_active:
            st.caption(":material/info: GPU busy training — recaption is paused until it finishes.")

        with st.popover("", icon=":material/tune:", help="Recaption settings"):
            st.radio(
                "Caption mode",
                list(MODE_HELP),
                key=MODE_KEY,
                format_func=lambda mode: mode.capitalize(),
                captions=list(MODE_HELP.values()),
            )

        if st.button(
            f"Recaption {len(selected)}",
            icon=":material/auto_awesome:",
            disabled=not selected or training_active,
            help="Select images above, then click to regenerate captions with Qwen3-VL. ~2s per image on GPU.",
        ):
            _run_batch(run, selected)


def _run_batch(run: IngestionRun, samples: list[DatasetSample]) -> None:
    started = time.time()
    total = len(samples)
    mode = st.session_state.get(MODE_KEY) or "subject"

    progress = st.progress(0.0, text="Loading Qwen3-VL…")
    done = failed = 0

    with st.status(f"Recaptioning {total} image(s)", expanded=True) as status:
        for event in recaption_service.recaption_samples(
            samples, run.concept.trigger_word, mode=mode
        ):
            if event.kind == "loaded":
                status.write(f"Model loaded on {event.device} in {event.seconds}s.")
                if event.device == "cpu":
                    # The worker falls back to CPU when the GPU is too full to load
                    # the model. It finishes rather than failing the batch, but at
                    # minutes per image instead of ~2s — worth saying out loud, since
                    # otherwise the only symptom is that nothing seems to happen.
                    st.warning(
                        "Running on CPU because the GPU was busy — expect minutes per "
                        "image instead of ~2s. Free the VRAM (close ComfyUI or another "
                        "model) and re-run for the fast path.",
                        icon=":material/warning:",
                    )

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
                step_telemetry.record_step(run.run_id, "recaption", time.time() - started, error_count=failed + 1)
                return

        label = f"Recaptioned {done}/{total}"
        if failed:
            label += f" · {failed} failed"
        status.update(label=label, state="complete", expanded=bool(failed))
        step_telemetry.record_step(run.run_id, "recaption", time.time() - started, error_count=failed)

    # st.rerun() below refreshes the caption widgets (their keys are versioned by
    # apply_recaption), but it also wipes whatever the status/progress widgets
    # above just rendered — a plain rerun happens fast enough that the summary
    # never gets read. st.toast is the one widget documented to survive exactly
    # one rerun, so it's what carries the result across.
    st.toast(label, icon=":material/auto_awesome:" if not failed else ":material/warning:")
    st.rerun()


def _find(samples: list[DatasetSample], sample_id: str) -> DatasetSample | None:
    return next((s for s in samples if s.sample_id == sample_id), None)
