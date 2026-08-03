import csv
import math
from pathlib import Path

from feature_pipeline.domain.train_log import (
    TRAIN_LOG_COLUMNS,
    parse_rows,
    summarize,
)

HEADER = list(TRAIN_LOG_COLUMNS)

# The 10-column log the golden smoke test recorded before `secs` and
# `vram_peak_gb` were appended to the writer.
GOLDEN_LOG = (
    Path(__file__).resolve().parent.parent
    / "workers" / "golden" / "expected" / "smoke_train_log.csv"
)


def _row(
    step: int,
    *,
    update: int | None = None,
    epoch: int = 1,
    loss: float = 0.10,
    loss_avg: float = 0.10,
    grad_norm: float = 1.0,
    lr: float = 1e-4,
    sigma: float = 0.5,
    bucket_h: int = 64,
    bucket_w: int = 64,
    secs: float = 0.5,
    vram: float = 8.0,
) -> list[str]:
    """One row formatted exactly as krea2.metrics.CsvLogs.log_step writes it."""
    return [
        str(step),
        str(update if update is not None else step // 4),
        str(epoch),
        f"{loss:.6f}",
        f"{loss_avg:.6f}",
        f"{grad_norm:.4f}",
        f"{lr:.3e}",
        f"{sigma:.4f}",
        str(bucket_h),
        str(bucket_w),
        f"{secs:.3f}",
        f"{vram:.2f}",
    ]


# --- parsing ------------------------------------------------------------------


def test_a_full_row_populates_every_field():
    (record,) = parse_rows([HEADER, _row(8, epoch=2, secs=1.25, vram=9.5)])

    assert (record.step, record.update, record.epoch) == (8, 2, 2)
    assert record.secs == 1.25
    assert record.vram_peak_gb == 9.5


def test_a_log_without_the_trailing_columns_is_still_read():
    # The two newest columns are genuinely optional — a log predating them must
    # not be discarded, it just has nothing to say about time or memory.
    records = parse_rows([HEADER[:10], _row(4)[:10], _row(8)[:10]])

    assert [r.step for r in records] == [4, 8]
    assert all(r.secs is None and r.vram_peak_gb is None for r in records)


def test_the_golden_smoke_log_reads_as_twelve_steps_not_three_rows():
    # The committed 10-column baseline, read end to end. Its three rows are three
    # optimizer updates covering twelve micro-steps — the distinction this whole
    # module exists to keep straight.
    with GOLDEN_LOG.open(newline="", encoding="utf-8") as handle:
        records = parse_rows(csv.reader(handle))

    assert len(records) == 3
    assert all(r.secs is None and r.vram_peak_gb is None for r in records)

    summary = summarize(records)
    assert summary.steps_executed == 12
    assert summary.updates_logged == 3
    assert summary.logged_seconds is None
    assert summary.peak_vram_gb is None


def test_columns_are_located_by_name_not_position():
    # Guards against a column ever being inserted rather than appended: reading by
    # position would silently shift every field one across.
    shuffled = ["epoch", "step", "update", "loss", "loss_avg", "grad_norm",
                "lr", "sigma", "bucket_h", "bucket_w"]
    row = ["7", "40", "10", "0.5", "0.5", "1.0", "1e-4", "0.5", "64", "64"]

    (record,) = parse_rows([shuffled, row])

    assert (record.step, record.update, record.epoch) == (40, 10, 7)


def test_a_log_with_no_header_falls_back_to_positional_order():
    (record,) = parse_rows([_row(12)])

    assert record.step == 12


def test_a_row_torn_mid_write_is_skipped_and_the_rest_survive():
    # SIGKILL between two writes leaves a partial final line; everything flushed
    # before it is intact and must not be thrown away with it.
    records = parse_rows([HEADER, _row(4), ["8", "2", "1"]])

    assert [r.step for r in records] == [4]


def test_blank_lines_and_a_repeated_header_are_ignored():
    records = parse_rows([HEADER, _row(4), [], ["", "", ""], HEADER, _row(8)])

    assert [r.step for r in records] == [4, 8]


def test_a_row_with_an_unreadable_step_is_skipped():
    records = parse_rows([HEADER, _row(4), ["oops"] + _row(8)[1:]])

    assert [r.step for r in records] == [4]


def test_an_empty_log_parses_and_summarizes_without_raising():
    assert parse_rows([]) == []
    assert parse_rows([HEADER]) == []

    summary = summarize([])
    assert summary.steps_executed == 0
    assert summary.updates_logged == 0
    assert summary.final_loss_avg is None
    assert summary.bucket_step_counts == {}


# --- steps --------------------------------------------------------------------


def test_steps_executed_counts_steps_not_rows():
    # The core misconception: a row is one optimizer update, which covers
    # grad_accum_steps micro-steps. Three rows here are twelve steps.
    summary = summarize(parse_rows([HEADER, _row(4), _row(8), _row(12)]))

    assert summary.steps_executed == 12
    assert summary.updates_logged == 3
    assert summary.unique_steps_covered == 3


# --- resume -------------------------------------------------------------------


def _resumed_log() -> list[list[str]]:
    """A run that reached step 900, then resumed from a checkpoint at 600."""
    rows = [HEADER]
    rows += [_row(step) for step in range(4, 901, 4)]
    rows += [_row(step) for step in range(604, 701, 4)]
    return rows


def test_a_resume_reports_how_far_it_got_and_where_it_ended_up():
    summary = summarize(parse_rows(_resumed_log()))

    assert summary.steps_executed == 900          # furthest point reached
    assert summary.final_step == 700              # where it actually left off
    assert summary.restart_count == 1
    assert summary.rewound_steps == 296           # 900 -> 604
    assert summary.updates_logged == 250          # redone work still ran
    assert summary.unique_steps_covered == 225    # ...but covered no new ground


def test_the_final_loss_comes_from_the_last_row_not_the_highest_step():
    # Sorting by step would report the pre-crash tail as the run's ending state,
    # when the resume has since overwritten it.
    rows = [HEADER]
    rows += [_row(step, loss_avg=0.9) for step in range(4, 901, 4)]
    rows += [_row(step, loss_avg=0.2) for step in range(604, 701, 4)]

    summary = summarize(parse_rows(rows))

    assert summary.final_loss_avg == 0.2


def test_two_seams_are_both_counted():
    rows = [HEADER]
    rows += [_row(s) for s in range(4, 101, 4)]
    rows += [_row(s) for s in range(52, 121, 4)]
    rows += [_row(s) for s in range(80, 141, 4)]

    assert summarize(parse_rows(rows)).restart_count == 2


def test_steps_per_epoch_ignores_the_span_the_resume_rewound():
    # The epoch counter is restored from the checkpoint too, so a seam produces a
    # negative span that would otherwise drag the median below zero.
    rows = [HEADER]
    rows += [_row(s, epoch=1) for s in range(4, 101, 4)]
    rows += [_row(s, epoch=2) for s in range(104, 201, 4)]
    rows += [_row(s, epoch=1) for s in range(52, 101, 4)]

    steps_per_epoch = summarize(parse_rows(rows)).steps_per_epoch

    assert steps_per_epoch is not None and steps_per_epoch > 0


# --- loss ---------------------------------------------------------------------


def test_the_best_loss_is_reported_with_the_step_it_happened_at():
    rows = [HEADER, _row(4, loss_avg=0.5), _row(8, loss_avg=0.1), _row(12, loss_avg=0.3)]

    summary = summarize(parse_rows(rows))

    assert summary.best_loss_avg == 0.1
    assert summary.best_loss_avg_step == 8


def test_a_falling_loss_reports_a_negative_delta():
    rows = [HEADER]
    rows += [_row(s, loss=1.0) for s in range(4, 401, 4)]
    rows += [_row(s, loss=0.2) for s in range(404, 801, 4)]

    summary = summarize(parse_rows(rows))

    assert summary.mean_loss_head == 1.0
    assert summary.mean_loss_tail == 0.2
    assert summary.loss_delta is not None and summary.loss_delta < 0


def test_the_loss_windows_never_overlap_on_a_short_run():
    # With two rows the window is one each; a naive fixed window would average the
    # same rows at both ends and report every short run as perfectly flat.
    summary = summarize(parse_rows([HEADER, _row(4, loss=1.0), _row(8, loss=0.4)]))

    assert summary.mean_loss_head == 1.0
    assert summary.mean_loss_tail == 0.4


def test_a_single_row_has_no_trend_at_all():
    summary = summarize(parse_rows([HEADER, _row(4)]))

    assert summary.mean_loss_head is None
    assert summary.loss_delta is None


# --- health -------------------------------------------------------------------


def test_a_zero_grad_norm_counts_as_a_skipped_update():
    rows = [HEADER, _row(4, grad_norm=1.0), _row(8, grad_norm=0.0), _row(12, grad_norm=2.0)]

    assert summarize(parse_rows(rows)).skipped_updates == 1


def test_non_finite_losses_are_counted_and_kept_out_of_the_averages():
    # A NaN is kept as a record — it is the most useful thing the log can report —
    # but one of them would poison every mean it took part in.
    nan_row = ["20", "5", "1", "nan", "0.5", "1.0", "1e-4", "0.5", "64", "64", "0.5", "8.0"]
    rows = [HEADER] + [_row(4 * i, loss=1.0) for i in range(1, 5)] + [nan_row]

    summary = summarize(parse_rows(rows))

    assert summary.nonfinite_loss_count == 1
    assert summary.updates_logged == 5
    assert summary.mean_loss_head is not None
    assert math.isfinite(summary.mean_loss_head)
    assert summary.mean_loss_tail == 1.0


def test_grad_norm_percentiles_ignore_the_skipped_updates():
    # Zeros mean "this update did not happen"; averaging them in would make an
    # unstable run look calm.
    rows = [HEADER] + [_row(4 * i, grad_norm=0.0) for i in range(1, 20)]
    rows += [_row(100, grad_norm=5.0), _row(104, grad_norm=50.0)]

    summary = summarize(parse_rows(rows))

    assert summary.grad_norm_p50 is not None and summary.grad_norm_p50 >= 5.0
    assert summary.grad_norm_max == 50.0


def test_a_spike_far_above_the_median_is_flagged():
    rows = [HEADER] + [_row(4 * i, loss=0.1) for i in range(1, 21)]
    rows += [_row(100, loss=9.0)]

    assert summarize(parse_rows(rows)).loss_spike_count == 1


# --- shape --------------------------------------------------------------------


def test_the_epoch_in_progress_is_not_counted_as_completed():
    rows = [HEADER, _row(4, epoch=1), _row(8, epoch=2), _row(12, epoch=3)]

    summary = summarize(parse_rows(rows))

    assert summary.epochs_started == 3
    assert summary.epochs_completed == 2


def test_buckets_are_counted_and_ordered_by_share():
    rows = [HEADER, _row(4, bucket_w=64, bucket_h=64),
            _row(8, bucket_w=96, bucket_h=48), _row(12, bucket_w=64, bucket_h=64)]

    counts = summarize(parse_rows(rows)).bucket_step_counts

    assert counts == {"64x64": 2, "96x48": 1}
    assert list(counts) == ["64x64", "96x48"]


def test_the_learning_rate_schedule_is_visible_from_its_endpoints():
    # peak == final is how a scheduler that never actually ran shows up.
    rows = [HEADER, _row(4, lr=1e-5), _row(8, lr=1e-4), _row(12, lr=2e-5)]

    summary = summarize(parse_rows(rows))

    assert summary.lr_peak == 1e-4
    assert summary.lr_final == 2e-5


def test_time_and_memory_are_summarized_when_the_columns_are_present():
    rows = [HEADER, _row(4, secs=1.0, vram=8.0), _row(8, secs=3.0, vram=9.0)]

    summary = summarize(parse_rows(rows))

    assert summary.logged_seconds == 4.0
    assert summary.median_seconds_per_logged_step == 2.0
    assert summary.peak_vram_gb == 9.0
