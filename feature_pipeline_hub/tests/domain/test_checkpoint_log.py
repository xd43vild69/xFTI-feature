"""Reading checkpoint_log.csv.

Two things carry weight here. The parser must survive a file written by a process
that was killed — a torn last row, a repeated header, a missing image count — because
that is the normal way a training run ends. And the medians must exclude the resume
seam, where `steps_delta` goes non-positive: those spans are real time, so they belong
in the totals, but dividing by them would produce a per-step figure that is negative
or infinite.
"""

from feature_pipeline.domain.checkpoint_log import (
    CHECKPOINT_LOG_COLUMNS,
    CheckpointFile,
    CheckpointRecord,
    parse_rows,
    reconstruct,
    summarize,
)

HEADER = list(CHECKPOINT_LOG_COLUMNS)


def row(step, *, epoch=1, reason="periodic", timestamp=1000.0,
        elapsed=120.0, delta=100, images=40, launch="L1"):
    return [str(step), str(epoch), reason, f"{timestamp:.3f}",
            f"{elapsed:.3f}", str(delta), "" if images is None else str(images), launch]


def record(step, *, reason="periodic", elapsed=120.0, delta=100, images=40, launch="L1"):
    return CheckpointRecord(step=step, epoch=1, reason=reason, timestamp=1000.0,
                            elapsed_seconds=elapsed, steps_delta=delta,
                            num_images=images, launch_id=launch)


# ── parsing ─────────────────────────────────────────────────────────────────

def test_columns_are_located_by_header_name():
    reordered = ["reason", "step", "epoch", "timestamp", "elapsed_seconds",
                 "steps_delta", "num_images"]
    records = parse_rows([reordered, ["interrupt", "137", "2", "1000.0", "30.0", "37", "40"]])

    assert len(records) == 1
    assert records[0].step == 137
    assert records[0].reason == "interrupt"


def test_a_log_with_no_header_falls_back_to_writer_order():
    records = parse_rows([row(100)])

    assert records[0].step == 100
    assert records[0].elapsed_seconds == 120.0


def test_a_row_torn_mid_write_is_skipped_and_the_rest_survive():
    records = parse_rows([HEADER, row(100), ["200", "2", "periodic", "1120."], row(300)])

    assert [r.step for r in records] == [100, 300]


def test_a_repeated_header_is_not_read_as_data():
    records = parse_rows([HEADER, row(100), HEADER, row(200)])

    assert [r.step for r in records] == [100, 200]


def test_blank_lines_are_ignored():
    assert parse_rows([HEADER, [], ["", "", ""], row(100)])[0].step == 100


def test_a_missing_image_count_parses_as_none():
    records = parse_rows([HEADER, row(100, images=None)])

    assert records[0].num_images is None


def test_a_non_finite_span_is_dropped():
    """It would otherwise poison every cumulative figure after it."""
    assert parse_rows([HEADER, ["100", "1", "periodic", "1000.0", "nan", "100", "40"]]) == []


def test_a_backwards_clock_clamps_rather_than_losing_the_checkpoint():
    records = parse_rows([HEADER, row(100, elapsed=-30.0)])

    assert records[0].elapsed_seconds == 0.0


def test_a_row_missing_its_reason_is_skipped():
    assert parse_rows([HEADER, ["100", "1", "", "1000.0", "120.0", "100", "40"]]) == []


# ── summarizing ─────────────────────────────────────────────────────────────

def test_an_empty_log_summarizes_to_zeros():
    summary = summarize([])

    assert summary.checkpoint_count == 0
    assert summary.total_elapsed_seconds == 0.0
    assert summary.median_seconds_per_checkpoint is None
    assert summary.intervals == ()


def test_spans_accumulate_in_file_order():
    summary = summarize([record(100, elapsed=100.0),
                         record(200, elapsed=140.0),
                         record(300, elapsed=120.0)])

    assert [i.cumulative_seconds for i in summary.intervals] == [100.0, 240.0, 360.0]
    assert summary.total_elapsed_seconds == 360.0
    assert summary.last_step == 300


def test_the_normalised_ratios_use_unique_images_and_steps():
    summary = summarize([record(100, elapsed=120.0, delta=100, images=40)])
    interval = summary.intervals[0]

    assert interval.seconds_per_image == 3.0   # 120 / 40 unique images
    assert interval.seconds_per_step == 1.2    # 120 / 100 steps


def test_the_median_span_ignores_the_resume_seam():
    """The rewound span is real time, but its step delta says nothing about pace."""
    summary = summarize([record(100, elapsed=100.0),
                         record(50, elapsed=900.0, delta=-50),   # seam
                         record(150, elapsed=100.0)])

    assert summary.median_seconds_per_checkpoint == 100.0
    assert summary.total_elapsed_seconds == 1100.0   # still counted


def test_a_zero_denominator_reports_no_ratio_rather_than_zero():
    summary = summarize([record(100, delta=0, images=None)])
    interval = summary.intervals[0]

    assert interval.seconds_per_step is None
    assert interval.seconds_per_image is None
    assert summary.median_seconds_per_image is None


def test_partial_spans_stay_out_of_the_per_checkpoint_medians():
    """A stop lands mid-span; averaging it in would understate a checkpoint's cost."""
    summary = summarize([record(100, elapsed=120.0),
                         record(137, reason="interrupt", elapsed=30.0, delta=37),
                         record(200, elapsed=120.0),
                         record(240, reason="final", elapsed=40.0, delta=40)])

    assert summary.median_seconds_per_checkpoint == 120.0
    assert summary.median_seconds_per_image == 3.0
    assert summary.total_elapsed_seconds == 310.0   # all four still counted


def test_a_run_with_no_complete_span_reports_no_median():
    summary = summarize([record(37, reason="interrupt", elapsed=30.0, delta=37)])

    assert summary.median_seconds_per_checkpoint is None
    assert summary.total_elapsed_seconds == 30.0


def test_interrupted_spans_are_counted_separately():
    summary = summarize([record(100), record(137, reason="interrupt"), record(200)])

    assert summary.checkpoint_count == 3
    assert summary.interrupted_count == 1
    assert [i.is_interrupted for i in summary.intervals] == [False, True, False]


def test_launches_are_numbered_by_first_appearance():
    """File order is launch order — a resume can only append."""
    summary = summarize([record(100, launch="L1"),
                         record(200, launch="L1"),
                         record(150, launch="L2", delta=-50),   # the resume seam
                         record(250, launch="L2")])

    assert [i.launch_ordinal for i in summary.intervals] == [1, 1, 2, 2]
    assert summary.launch_count == 2


def test_a_log_without_the_launch_column_reads_as_one_launch():
    """Files written before the column existed must not become N unknown launches."""
    records = parse_rows([HEADER[:7], row(100)[:7], row(200)[:7]])
    summary = summarize(records)

    assert summary.launch_count == 1
    assert {i.launch_ordinal for i in summary.intervals} == {1}
    assert {i.launch_id for i in summary.intervals} == {""}


def test_the_launch_id_survives_parsing():
    records = parse_rows([HEADER, row(100, launch="run-abc")])

    assert records[0].launch_id == "run-abc"


def test_the_image_count_reported_is_the_last_one_seen():
    """A resume can train a dataset that has since been edited."""
    summary = summarize([record(100, images=40), record(200, images=55)])

    assert summary.images_trained == 55


def test_the_summary_round_trips_through_json():
    """It is stored on the training_runs row and validated back out of it."""
    from feature_pipeline.domain.checkpoint_log import CheckpointLogSummary

    summary = summarize([record(100), record(137, reason="interrupt")])
    restored = CheckpointLogSummary.model_validate(summary.model_dump(mode="json"))

    assert restored == summary


# ── reconstructing a run that predates the log ──────────────────────────────
#
# Modelled on train-e2aae046, a real run whose timeline is known independently:
# launch 1 started at t=0 and saved every 300 steps in ~731s, was stopped at step
# 1852, lay dead for 744s, and launch 2 resumed and finished. See the module
# docstring for what the inferred numbers can and cannot say.

def saved(step, at, *, images=21, final=False):
    return CheckpointFile(step=step, written_at=at, num_images=images, is_final=final)


def test_consecutive_files_bracket_the_work_between_them():
    records = reconstruct([saved(300, 1_000.0), saved(600, 1_731.0)],
                          launches=[("L1", 773.0)])

    assert [r.elapsed_seconds for r in records] == [227.0, 731.0]
    assert [r.steps_delta for r in records] == [300, 300]


def test_the_first_span_of_a_launch_is_anchored_to_its_start():
    """Otherwise the very first checkpoint of a run has nothing to measure against."""
    records = reconstruct([saved(300, 1_000.0)], launches=[("L1", 227.0)])

    assert records[0].elapsed_seconds == 773.0


def test_the_gap_where_the_process_was_dead_is_not_billed_as_training():
    """The whole reason the launch start times are passed in."""
    records = reconstruct(
        [saved(1852, 5_000.0), saved(2100, 6_387.0)],
        launches=[("L1", 0.0), ("L2", 5_744.0)],
    )

    # 6387 - 5744, not 6387 - 5000: the 744s in between, nobody was training.
    assert records[1].elapsed_seconds == 643.0
    assert records[1].launch_id == "L2"


def test_a_resumed_launch_counts_only_the_steps_it_ran():
    """A resume restarts from the newest checkpoint, so that file is where it began."""
    records = reconstruct(
        [saved(1852, 5_000.0), saved(2100, 6_387.0)],
        launches=[("L1", 0.0), ("L2", 5_744.0)],
    )

    assert records[1].steps_delta == 248


def test_the_final_export_does_not_invent_a_second_checkpoint():
    """FINAL is the same weights at the same step, written seconds later."""
    records = reconstruct([saved(3000, 9_000.0), saved(3000, 9_000.4, final=True)])

    assert len(records) == 1
    assert records[0].reason == "final"


def test_a_step_off_the_cadence_reads_as_an_interruption():
    records = reconstruct([saved(1800, 1_000.0), saved(1852, 1_126.0)], save_every=300)

    assert [r.reason for r in records] == ["periodic", "interrupt"]


def test_without_save_every_nothing_is_called_an_interruption():
    """A run with no stored settings cannot be second-guessed."""
    records = reconstruct([saved(1852, 1_126.0)])

    assert records[0].reason == "periodic"


def test_files_out_of_order_are_sorted_by_when_they_were_written():
    records = reconstruct([saved(600, 1_731.0), saved(300, 1_000.0)])

    assert [r.step for r in records] == [300, 600]


def test_reconstructing_nothing_yields_nothing():
    assert reconstruct([]) == []


def test_the_summary_says_the_spans_were_inferred():
    summary = summarize(reconstruct([saved(300, 1_000.0), saved(600, 1_731.0)]),
                        reconstructed=True)

    assert summary.is_reconstructed is True
    assert summary.checkpoint_count == 2


def test_a_measured_summary_is_not_marked_as_inferred():
    assert summarize([record(100)]).is_reconstructed is False
