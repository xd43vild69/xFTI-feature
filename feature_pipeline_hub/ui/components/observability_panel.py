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

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import state
from feature_pipeline.domain import cost

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
    # Newest training first. group_training_lineages drops pre-cache rows, which
    # otherwise shadowed the training whenever one had run more recently.
    lineages = state.training_lineages(run.run_id)

    if telemetry is None:
        st.warning("No pipeline data for this run yet.")
        return

    st.subheader("📊 Pipeline Metrics")

    # Chosen before the table is built, so the Training row, the cost metric and the
    # detail below all describe the same training rather than the table showing the
    # newest while the detail shows whatever was picked.
    lineage = _select_lineage(lineages, run.run_id)

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

        group_key = _fork_group_key(lineage, lineages)
        if group_key is not None:
            _render_branch_comparison(run.run_id, group_key)


def _select_lineage(lineages, run_id: str):
    """Which of this dataset's trainings the page describes. Newest by default.

    Every launch of `start_training` mints its own output_dir, so a dataset trained
    five times has five independent records — and the page used to show only the last
    one, with no way to reach the rest. A resume is deliberately *not* a separate
    entry: it continues the same training, and appears here as one.
    """
    if not lineages:
        return None
    if len(lineages) == 1:
        return lineages[0]

    # Keyed per dataset so switching datasets does not carry a selection across to a
    # run whose list of trainings is entirely different.
    return st.selectbox(
        "Training run",
        lineages,
        index=0,
        format_func=_lineage_label,
        key=f"metrics_lineage_{run_id}",
        help="Each launch of a training on this dataset is its own record. Resumes "
             "are folded into the training they continue.",
    )


def _lineage_label(lineage) -> str:
    """One training as a single line: when it started, what it made, how it went."""
    started = lineage.launches[0].started_at
    parts = [
        started.strftime("%Y-%m-%d %H:%M") if started is not None else "unknown date",
        str(lineage.latest.config.get("checkpoint_prefix") or "—"),
        _steps_label(lineage),
        lineage.status,
    ]
    if lineage.launch_count > 1:
        parts.append(f"{lineage.launch_count} launches")
    return " · ".join(parts)


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

    progress, timing, checkpoints, dataset, health = st.tabs(
        ["Progress", "Time", "Checkpoints", "Dataset & config", "Health"]
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

    with checkpoints:
        _render_checkpoints(lineage)

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


def _render_checkpoints(lineage) -> None:
    """What each `.safetensors` cost: wall clock per save_every span, and per image.

    The point of the normalised columns: with the rest of the configuration held
    steady, seconds-per-image is the figure that carries across datasets of different
    sizes, so a checkpoint here can be compared to a checkpoint of another run.
    """
    log = lineage.checkpoint_log
    if log is None or not log.intervals:
        st.info(
            "No checkpoint timings for this training, and no checkpoints left on disk "
            "to infer them from — so it either stopped before its first save, or its "
            "runtime directory has since been reclaimed."
        )
        # Still worth showing: what it was launched with survives the checkpoints.
        _render_launch_config(lineage)
        return

    save_every = lineage.latest.config.get("save_every")

    if log.is_reconstructed:
        st.warning(
            "**Estimated.** This training ran before the trainer logged its saves, so "
            "these spans were inferred from when each `.safetensors` was written. Two "
            "things follow: the first span of each launch also contains the model and "
            "cache load, and a file that was ever copied or moved carries the wrong "
            "timestamp. Runs launched from now on are measured, not inferred.",
            icon="🧮",
        )

    cols = st.columns(5)
    cols[0].metric(
        "Checkpoints written",
        f"{log.checkpoint_count:,}",
        help=(f"One per {save_every} micro-steps, plus any written on the way out."
              if save_every else "Includes any written on the way out."),
    )
    cols[4].metric(
        "Launches",
        f"{log.launch_count:,}",
        help="Processes that contributed to this training. More than one means it "
             "was resumed — the same training continued, each launch timing its own "
             "spans from the moment it came back up.",
    )
    cols[1].metric(
        "Median per checkpoint",
        _duration(log.median_seconds_per_checkpoint),
        help="Wall clock of a complete save_every span, measured between consecutive "
             "saves. Spans cut short by a stop, the final leftover span and the "
             "resume seam are excluded — they are partial by construction.",
    )
    cols[2].metric(
        "Seconds / image",
        _fmt(log.median_seconds_per_image, 2),
        help=f"Median span divided by the {log.images_trained:,} unique training "
             "images. Comparable across runs on datasets of different sizes."
             if log.images_trained else
             "Median span divided by the run's unique training images.",
    )
    cols[3].metric(
        "Interrupted spans",
        f"{log.interrupted_count:,}",
        help="Checkpoints written by a stop, an OOM or a non-finite gradient rather "
             "than by the save_every cadence. Their time is partial by definition.",
    )

    rate = cost.gpu_hourly_rate()
    st.dataframe(
        [
            {
                "Launch": interval.launch_ordinal,
                "Step": f"{interval.step:,}",
                "Reason": interval.reason,
                "Span": _duration(interval.elapsed_seconds),
                "Cumulative": _duration(interval.cumulative_seconds),
                "Δ steps": f"{interval.steps_delta:,}",
                "Images": f"{interval.num_images:,}" if interval.num_images else "—",
                "s / image": _fmt(interval.seconds_per_image, 2),
                "s / step": _fmt(interval.seconds_per_step, 2),
                "Cost": _money(cost.estimate_cost(interval.elapsed_seconds, rate)),
            }
            for interval in log.intervals
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Each span is timed from the previous checkpoint of the same launch — a "
        "resume starts its clock when the process came back, so the hours a stopped "
        "run spent waiting are never counted as training time. The Launch column is "
        "what separates one process's spans from the next one's."
        + ("" if lineage.checkpoint_log_is_live else
           " · checkpoint_log.csv is gone — showing the snapshot stored when it finished")
    )

    _render_launch_config(lineage)


def _render_launch_config(lineage) -> None:
    """The exact settings this training was launched with, ready to copy or reuse.

    Read from the stored `config_json` rather than from the run's settings.json on
    disk: it is the same content — training_runner writes one from the other — but the
    row outlives the runtime directory, and the numbers above are worth nothing
    without the configuration that produced them.

    Per launch, because a resume can differ from the launch it continues — it carries
    everything over except `total_steps`, which only has to clear the checkpoint's
    step, not the original target — and because it writes its settings to a run
    directory of its own, so the path below is not the one the checkpoints are in.
    """
    launches = lineage.launches
    run = launches[-1]

    with st.expander("Launch configuration (JSON)"):
        if len(launches) > 1:
            run = st.selectbox(
                "Launch",
                launches,
                index=len(launches) - 1,
                format_func=lambda item: _launch_label(item, launches),
                key=f"ckpt_config_{lineage.latest.training_run_id}",
                help="Every launch carried its own settings. A resume copies them "
                     "except for total_steps, which it may raise — and it writes them "
                     "to a run directory of its own while keeping the original's "
                     "output_dir, which is what makes it a resume.",
            )

        payload = json.dumps(run.config, indent=2, sort_keys=True)
        # st.code renders a copy button of its own in the corner.
        st.code(payload, language="json")
        st.download_button(
            "Download settings.json",
            payload,
            file_name=f"settings-{run.training_run_id[:8]}.json",
            mime="application/json",
            key=f"ckpt_config_dl_{run.training_run_id}",
        )
        st.caption(f"Launched from `{Path(run.log_path).parent / 'settings.json'}`")


def _launch_label(run, launches) -> str:
    """One launch as a line: its position, when it started, how it ended."""
    position = launches.index(run) + 1
    started = run.started_at
    return " · ".join([
        f"Launch {position}",
        started.strftime("%Y-%m-%d %H:%M") if started is not None else "unknown date",
        f"{int(run.config.get('total_steps') or 0):,} steps",
        run.status,
    ])


def _money(value: float | None) -> str:
    """A dash when FTI_GPU_HOURLY_RATE is unset — an unpriced run is not a free one."""
    return f"${value:.4f}" if value is not None else "—"


def _duration(seconds: float | None) -> str:
    """Seconds as h:mm:ss or m:ss — the scale a checkpoint span actually lands on."""
    if seconds is None:
        return "—"
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


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


def _fork_group_key(lineage, lineages) -> tuple[str, int] | None:
    """(parent_training_run_id, fork_step) if the selected training is part of a
    fork lineage — either a branch itself, or a checkpoint that has been forked.

    Fork identity (fork_parent_run_id/fork_step) is fixed at creation and never
    changes, so reading it off the already-fetched `lineages` list is safe even
    though that list is cached on a fingerprint that can lag a sibling's CSV
    growth — that staleness only matters for the chart data, fetched separately
    below through `state.experiment_branches`/`branch_curves`.
    """
    if lineage.latest.fork_parent_run_id and lineage.latest.fork_step is not None:
        return (lineage.latest.fork_parent_run_id, lineage.latest.fork_step)
    for other in lineages:
        if other.latest.fork_parent_run_id == lineage.latest.training_run_id:
            return (lineage.latest.training_run_id, other.latest.fork_step)
    return None


def _render_branch_comparison(dataset_run_id: str, group_key: tuple[str, int]) -> None:
    """Overlay every sibling branch forked from the same checkpoint.

    Reads train_log.csv/val_log.csv straight off each branch's own output_dir
    rather than the persisted metrics_json: a branch still training has no
    persisted snapshot at all (that only gets written when a launch finishes), and
    a scalar summary has no per-step series to chart in the first place. Each
    branch owns its output_dir — unlike a resume, nothing here is shared, so
    reading the live CSV is safe and describes that branch alone.
    """
    parent_training_run_id, fork_step = group_key
    branches = state.experiment_branches(dataset_run_id, parent_training_run_id, fork_step)
    if not branches:
        return

    st.divider()
    st.subheader("🌿 Branch comparison")
    st.caption(f"Forked from checkpoint step {fork_step:,} · {len(branches)} branch(es)")
    st.caption(
        LOSS_CAVEAT + " Siblings share a seed and a fork point, which makes a "
        "cross-branch comparison less invalid than comparing unrelated runs — but "
        "a lower curve still is not 'better images', only 'lower loss'."
    )

    control = next(
        (b for b in branches if b.latest.branch_label == "control"),
        branches[0],
    )
    total_steps_by_branch = {b.latest.branch_label: b.total_steps_target for b in branches}
    if len(set(total_steps_by_branch.values())) > 1:
        st.caption(
            ":material/warning: Branches target different total_steps — their LR "
            "schedules diverge from the fork point onward, on top of whatever the "
            "dataset intervention changed."
        )

    mismatched_hash = any(
        b.latest.dataset_content_hash != control.latest.dataset_content_hash for b in branches
    )

    curves = state.branch_curves(dataset_run_id, parent_training_run_id, fork_step)

    train_series = {}
    for branch in branches:
        label = branch.latest.branch_label or branch.latest.training_run_id[:8]
        df = curves.get(label, {}).get("train")
        if df is not None and "step" in df.columns and "loss_avg" in df.columns:
            train_series[label] = df.set_index("step")["loss_avg"]
    if train_series:
        st.line_chart(pd.concat(train_series, axis=1))
    else:
        st.caption("No train_log.csv yet for any branch.")

    if mismatched_hash:
        st.caption(
            ":material/info: Validation curve hidden — branches hold out different "
            "images (choose_holdout is a stride over the sorted dataset, so adding "
            "or removing an image moves it), so val_loss is not comparable across "
            "them here."
        )
    else:
        val_series = {}
        for branch in branches:
            label = branch.latest.branch_label or branch.latest.training_run_id[:8]
            df = curves.get(label, {}).get("val")
            if df is not None and "step" in df.columns and "val_loss" in df.columns:
                val_series[label] = df.set_index("step")["val_loss"]
        if val_series:
            st.line_chart(pd.concat(val_series, axis=1))

    running_labels = [
        b.latest.branch_label or b.latest.training_run_id[:8]
        for b in branches if b.status == "running"
    ]
    if running_labels:
        st.caption(
            "Still training: " + ", ".join(running_labels) + " — their series stop "
            "short on the chart above; that means in progress, not diverged."
        )

    rows = []
    for branch in branches:
        label = branch.latest.branch_label or branch.latest.training_run_id[:8]
        log = branch.train_log
        content_hash = branch.latest.dataset_content_hash
        rows.append(
            {
                "Branch": label,
                "Weights": (
                    "/".join(
                        f"{k}×{v}" for k, v in branch.latest.weight_profile.items()
                    )
                    if branch.latest.weight_profile else "—"
                ),
                "Content hash": (
                    f"{content_hash[:8]} ✓"
                    if content_hash and content_hash == control.latest.dataset_content_hash
                    else (content_hash[:8] if content_hash else "—")
                ),
                "Steps": _steps_label(branch),
                "Final loss": _fmt(log.final_loss_avg, 4) if log else "—",
                "Best loss": _fmt(log.best_loss_avg, 4) if log else "—",
                "Status": branch.status,
                "Cost": _money(branch.cost_estimate),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        ":material/info: A fingerprint warning in each branch's log is expected — "
        "it's the new cache directory, not a corrupt checkpoint. Loss is logged "
        "before any curation weight is applied, by design, so the chart will not "
        "visibly react to a weight-only change even though training does."
    )
