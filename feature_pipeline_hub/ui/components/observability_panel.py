"""Dashboard of curation pipeline metrics: durations, throughput, errors, and cost across all 5 steps.

Populated by step_telemetry.record_step() calls from each step's UI component (import, recaption,
quality, export) and training_service.finalize_dead_run() for the training step. This panel
consolidates them into a single view so the operator can see the full pipeline at a glance.
"""

import streamlit as st

import state
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure.database import get_connection


def render() -> None:
    """Show pipeline metrics for the active run, if any."""
    run = state.active_run()
    if run is None:
        st.info("Select a dataset first to see pipeline metrics.")
        return

    conn = get_connection()
    try:
        ingest_row = conn.execute(
            """
            SELECT run_id, import_duration_seconds, import_error_count,
                   recaption_duration_seconds, recaption_error_count,
                   quality_duration_seconds, export_duration_seconds, export_error_count,
                   cost_estimate
            FROM ingestion_runs WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()

        # Find the training run for this dataset (if any)
        training_runs = training_repo.list_training_runs(conn, dataset_run_id=run.run_id)
        train_run = training_runs[0] if training_runs else None
    finally:
        conn.close()

    if ingest_row is None:
        st.warning("No pipeline data for this run yet.")
        return

    st.subheader("📊 Pipeline Metrics")

    # Unpack ingestion telemetry
    (
        _,
        import_dur,
        import_err,
        recaption_dur,
        recaption_err,
        quality_dur,
        export_dur,
        export_err,
        ingest_cost,
    ) = ingest_row

    # Build rows for the table
    rows = []
    total_duration = 0.0
    total_errors = 0

    steps = [
        ("Import", import_dur, import_err, len(run.concept.samples)),
        ("Recaption", recaption_dur, recaption_err, len(run.concept.samples)),
        ("Quality", quality_dur, 0, len(run.concept.samples)),  # quality doesn't count errors the same way
        ("Export", export_dur, export_err, len(run.concept.samples)),
    ]

    for step_name, dur, err, total_items in steps:
        if dur is None:
            rows.append({"Step": step_name, "Items": "-", "Duration": "-", "Errors": "-", "Cost": "-"})
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

    # Add training row if available
    if train_run is not None and train_run.duration_seconds is not None:
        total_duration += train_run.duration_seconds
        rows.append(
            {
                "Step": "Training",
                "Items": f"{train_run.config.get('total_steps', '?')} steps",
                "Duration": f"{train_run.duration_seconds:.1f}s",
                "Throughput": "-",
                "Errors": "✓" if train_run.status == "completed" else "✗",
            }
        )

    # Totals row
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

    # Cost summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ingestion Cost", f"${ingest_cost:.4f}" if ingest_cost else "N/A")
    with col2:
        train_cost = train_run.cost_estimate if train_run and train_run.cost_estimate else 0.0
        st.metric("Training Cost", f"${train_cost:.4f}" if train_cost else "N/A")
    with col3:
        total_cost = (ingest_cost or 0.0) + (train_cost or 0.0)
        st.metric("Total Cost", f"${total_cost:.4f}" if total_cost > 0 else "N/A")

    # Status summary
    if train_run is not None:
        st.caption(
            f"Training: {train_run.status} · "
            f"{train_run.duration_seconds / 60:.1f} min "
            f"({train_run.gpu_seconds / 3600:.2f} GPU-hours)"
            if train_run.duration_seconds
            else f"Training: {train_run.status}"
        )
