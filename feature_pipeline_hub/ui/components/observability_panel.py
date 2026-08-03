"""Dashboard of pipeline metrics: durations, throughput, errors, and cost across all 5 steps.

The first four steps are populated by step_telemetry.record_step() from each step's
UI component. Training is different in kind: its numbers come from what the trainer
wrote while it ran, read back through training_metrics_service.

That distinction is why the Training row reports steps *executed* rather than the
`total_steps` it was configured with. A run that died at step 400 of 3000 was
reported here as "3000 steps" — the target restated as though it were an
achievement. Every figure below is measured, and anything unmeasured shows as "—"
rather than borrowing a number from the configuration.
"""

import streamlit as st

import state

# Diffusion loss is dominated by the noise level sampled at each step, not by how
# good the adapter is, so it says whether a run converged on its own objective and
# nothing at all about output quality. Repeated wherever a loss figure appears.
LOSS_CAVEAT = (
    "Diffusion loss is dominated by the noise level sampled each step, so it "
    "tracks convergence, not image quality. Compare within a run, never across."
)


def render() -> None:
    """Show pipeline metrics for the active run, if any."""
    run = state.active_run()
    if run is None:
        st.info("Select a dataset first to see pipeline metrics.")
        return

    telemetry = state.step_telemetry(run.run_id)
    lineages = state.training_lineages(run.run_id)
    # Newest training first. group_training_lineages drops pre-cache rows, which
    # otherwise shadowed the training whenever one had run more recently.
    lineage = lineages[0] if lineages else None

    if telemetry is None:
        st.warning("No pipeline data for this run yet.")
        return

    st.subheader("📊 Pipeline Metrics")

    ingest_cost = telemetry.cost_estimate
    total_items = len(run.concept.samples)

    rows = []
    total_duration = 0.0
    total_errors = 0

    steps = [
        ("Import", telemetry.import_duration_seconds, telemetry.import_error_count),
        ("Recaption", telemetry.recaption_duration_seconds, telemetry.recaption_error_count),
        ("Quality", telemetry.quality_duration_seconds, 0),
        ("Export", telemetry.export_duration_seconds, telemetry.export_error_count),
    ]

    for step_name, dur, err in steps:
        if dur is None:
            rows.append({"Step": step_name, "Items": "-", "Duration": "-", "Errors": "-"})
            continue

        total_duration += dur
        total_errors += err
        throughput = total_items / dur if dur > 0 else 0
        rows.append(
            {
                "Step": step_name,
                "Items": total_items,
                "Duration": f"{dur:.1f}s",
                "Throughput": f"{throughput:.1f} items/s" if throughput > 0 else "-",
                "Errors": err if err > 0 else "-",
            }
        )

    if lineage is not None:
        total_duration += lineage.wall_clock_seconds or 0.0
        rows.append(
            {
                "Step": "Training",
                "Items": _steps_label(lineage),
                "Duration": (
                    f"{lineage.wall_clock_seconds:.1f}s"
                    if lineage.wall_clock_seconds is not None
                    else "-"
                ),
                "Throughput": (
                    f"{lineage.seconds_per_step:.2f} s/step"
                    if lineage.seconds_per_step is not None
                    else "-"
                ),
                "Errors": "✓" if lineage.status == "completed" else "✗",
            }
        )

    rows.append(
        {
            "Step": "TOTAL",
            "Items": "-",
            "Duration": f"{total_duration:.1f}s ({total_duration/60:.1f} min)",
            "Throughput": "-",
            "Errors": total_errors if total_errors > 0 else "-",
        }
    )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    train_cost = lineage.cost_estimate if lineage else None
    with col1:
        st.metric("Ingestion Cost", f"${ingest_cost:.4f}" if ingest_cost else "N/A")
    with col2:
        st.metric(
            "Training Cost",
            f"${train_cost:.4f}" if train_cost else "N/A",
            help="FTI_GPU_HOURLY_RATE applied to the process's elapsed time. The "
                 "workers report wall clock, not measured GPU utilisation, so this "
                 "prices how long the GPU was held rather than how hard it worked.",
        )
    with col3:
        total_cost = (ingest_cost or 0.0) + (train_cost or 0.0)
        st.metric("Total Cost", f"${total_cost:.4f}" if total_cost > 0 else "N/A")

    if lineage is not None:
        _render_training_detail(lineage, total_items)


def _steps_label(lineage) -> str:
    """Steps executed against the target — or an honest dash.

    Never the target on its own: a run nobody measured has to look unmeasured,
    which is the whole point of this panel's rewrite.
    """
    if lineage.steps_executed is None:
        return "—"
    if lineage.total_steps_target > 0:
        return f"{lineage.steps_executed:,} / {lineage.total_steps_target:,} steps"
    return f"{lineage.steps_executed:,} steps"


def _render_training_detail(lineage, active_images: int) -> None:
    """The training's own numbers, below the pipeline table."""
    st.divider()
    st.subheader("🎛️ Training")

    st.caption(_provenance(lineage))

    log = lineage.train_log
    if log is None:
        st.caption(
            "No train_log.csv was found for this run, and nothing was stored when it "
            "finished — there is nothing to report beyond the row above."
        )
        return

    progress, timing, dataset, health = st.tabs(
        ["Progress", "Time", "Dataset & config", "Health"]
    )

    with progress:
        cols = st.columns(4)
        cols[0].metric(
            "Steps executed",
            f"{log.steps_executed:,}",
            help="Highest step the trainer reached. One CSV row is one optimizer "
                 "update, covering grad_accum_steps of these.",
        )
        cols[1].metric(
            "Completed",
            f"{lineage.completion_fraction:.0%}" if lineage.completion_fraction else "—",
            help=f"Against the {lineage.total_steps_target:,}-step target of the "
                 "most recent launch.",
        )
        cols[2].metric("Epochs completed", f"{log.epochs_completed:,}",
                       help="The highest epoch seen is still in progress, so it is not counted.")
        cols[3].metric("Optimizer updates", f"{log.updates_logged:,}",
                       help="Includes work redone after a resume.")

        cols = st.columns(4)
        cols[0].metric("Final loss", _fmt(log.final_loss_avg, 4), help=LOSS_CAVEAT)
        best_help = (
            f"Reached at step {log.best_loss_avg_step:,}. " if log.best_loss_avg_step else ""
        )
        cols[1].metric("Best loss", _fmt(log.best_loss_avg, 4), help=best_help + LOSS_CAVEAT)
        cols[2].metric("First loss", _fmt(log.first_loss_avg, 4), help=LOSS_CAVEAT)
        cols[3].metric(
            "Trend",
            _fmt(log.loss_delta, 4),
            help="Mean loss over the last rows minus the first. Negative means it "
                 "fell. " + LOSS_CAVEAT,
        )

    with timing:
        cols = st.columns(4)
        cols[0].metric(
            "Wall clock",
            f"{lineage.wall_clock_seconds / 60:.1f} min" if lineage.wall_clock_seconds else "—",
            help="Summed across every launch of this training. Excludes idle time "
                 "between them.",
        )
        cols[1].metric(
            "Seconds / step",
            f"{lineage.seconds_per_step:.2f}" if lineage.seconds_per_step else "—",
            help="Wall clock divided by steps executed — model loading and "
                 "checkpointing included, because that time was really spent.",
        )
        cols[2].metric(
            "Median step time",
            f"{log.median_seconds_per_logged_step:.2f}s"
            if log.median_seconds_per_logged_step else "—",
            help="From the CSV, which times a single micro-step and only on update "
                 "steps — biased high, and not summable into total compute time.",
        )
        cols[3].metric(
            "Peak VRAM",
            f"{log.peak_vram_gb:.1f} GB" if log.peak_vram_gb else "—",
            help="Allocator-tracked peak, cumulative over the process. Reads lower "
                 "than nvidia-smi, which counts reserved memory.",
        )

    with dataset:
        cols = st.columns(4)
        cols[0].metric("Active images (now)", f"{active_images:,}",
                       help="The dataset's current state, which may have been edited "
                            "since this training ran.")
        cols[1].metric(
            "Images / epoch (est.)",
            f"{lineage.images_per_epoch_estimate:,.0f}"
            if lineage.images_per_epoch_estimate else "—",
            help="Derived from the sampler's epoch boundaries. Bucket padding "
                 "inflates it slightly; a large gap from the count on the left means "
                 "the trained dataset differed from the current one.",
        )
        cols[2].metric("Steps / epoch", _fmt(log.steps_per_epoch, 0))
        cols[3].metric(
            "LR peak → final",
            f"{log.lr_peak:.1e} → {log.lr_final:.1e}" if log.lr_peak else "—",
            help="Equal values mean the schedule never actually moved.",
        )

        if log.bucket_step_counts:
            st.caption("Resolution buckets actually trained")
            st.dataframe(
                [{"Bucket": name, "Updates": count}
                 for name, count in log.bucket_step_counts.items()],
                use_container_width=True, hide_index=True,
            )

        config = lineage.latest.config
        st.caption("Hyperparameters of the most recent launch")
        st.dataframe(
            [
                {"Parameter": key, "Value": config.get(key, "—")}
                for key in ("lr", "lora_rank", "lora_alpha", "batch_size",
                            "grad_accum_steps", "warmup_steps", "lr_scheduler",
                            "lr_num_cycles", "seed", "noise_offset",
                            "caption_dropout_rate", "timestep_weighting")
            ],
            use_container_width=True, hide_index=True,
        )

    with health:
        cols = st.columns(4)
        skipped_share = (
            log.skipped_updates / log.updates_logged if log.updates_logged else 0.0
        )
        cols[0].metric(
            "Skipped updates",
            f"{log.skipped_updates:,}",
            delta=f"{skipped_share:.1%}" if log.skipped_updates else None,
            delta_color="inverse",
            help="Updates the trainer discarded for a non-finite gradient — the run "
                 "advanced without learning. A tiny genuine norm also rounds to zero "
                 "in this column, so a handful is not alarming.",
        )
        cols[1].metric(
            "Non-finite losses", f"{log.nonfinite_loss_count:,}",
            help="NaN or infinite loss. Anything above zero means the run was "
                 "diverging somewhere.",
        )
        cols[2].metric(
            "Grad norm p50 / p95",
            f"{_fmt(log.grad_norm_p50, 2)} / {_fmt(log.grad_norm_p95, 2)}",
            help="Zeros excluded. A p95 far above the p50 is the exploding-gradient "
                 "fingerprint.",
        )
        cols[3].metric(
            "Loss spikes", f"{log.loss_spike_count:,}",
            help="Heuristic: losses far above the run's median. A legitimately "
                 "high-noise step can trip it.",
        )


def _provenance(lineage) -> str:
    """One line on where these numbers came from and what happened to the run."""
    parts = [
        lineage.status,
        f"{lineage.launch_count} launch{'es' if lineage.launch_count != 1 else ''}",
    ]
    log = lineage.train_log
    if log is not None and log.restart_count:
        parts.append(
            f"resumed {log.restart_count}×, {log.rewound_steps:,} steps rewound and redone"
        )
    if log is not None and log.final_step != log.steps_executed:
        parts.append(f"ended at step {log.final_step:,}")
    if log is not None and not lineage.train_log_is_live:
        parts.append("train_log.csv is gone — showing the snapshot stored when it finished")
    return " · ".join(parts)


def _fmt(value: float | None, places: int) -> str:
    return f"{value:.{places}f}" if value is not None else "—"
