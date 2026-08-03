"""Reading train_log.csv: what a training run actually did, as opposed to what it was told to do.

The trainer writes one row per real optimizer update (`workers/krea2/metrics.py`,
`CsvLogs.log_step`, guarded by `if did_update:`), flushed per row so a run killed
outright still leaves a complete log. That file is the only record of what happened
inside the loop — the `training_runs` row knows the *target* `total_steps` and the
process's wall clock, and nothing else.

Which is why the headline number here is `steps_executed = max(step)`, not
`len(records)`. A row is an update, and an update covers `grad_accum_steps`
micro-steps, so 750 rows of a default run mean 3000 steps. Reporting the row count
as steps would understate the run fourfold; reporting the configured target would
overstate a run that died early by however far it got. Neither is a measurement.

Pure, in the same split as `caption_schema`: this module knows the file's *format*
and what its numbers mean, the application layer opens the file. It takes rows
already split into cells rather than the file's text, so the caller can stream a
`csv.reader` over a log that may run to six figures of rows.

Two things this deliberately does not do. It never reads `val_log.csv`: the
trainer's `validate_every` defaults to 0 and `val_split` to 0.0, and neither is
reachable through `TrainSettings` (`extra="forbid"`), so that file is header-only
in every run this hub can launch — fields that are structurally always empty would
be the same lie in a new place. And `summarize` takes only records, never config:
a summary that mixed in `total_steps` could assert something the log never said.
Completion percentages belong one layer up, where config lives.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

# Mirror of krea2.metrics.TRAIN_COLUMNS. Duplicated rather than imported because
# `workers/` is not on the hub's sys.path and the training runtime is a different
# interpreter entirely — the same reason caption_schema ships prompts as text.
# tests/workers/test_krea2_metrics.py pins the two copies together.
TRAIN_LOG_COLUMNS = (
    "step", "update", "epoch", "loss", "loss_avg", "grad_norm",
    "lr", "sigma", "bucket_h", "bucket_w", "secs", "vram_peak_gb",
)

# The first ten arrived together; `secs` and `vram_peak_gb` were appended later, so
# logs written before that are still perfectly readable and simply have neither.
_REQUIRED_COLUMNS = TRAIN_LOG_COLUMNS[:10]

# Rows averaged at each end of the run to estimate whether loss moved at all. Small
# enough to still mean something on a short run, large enough to survive the
# per-step noise that sigma sampling injects.
LOSS_WINDOW = 50

# A step whose loss exceeds this multiple of the run's median counts as a spike. A
# heuristic: a legitimately high-sigma step can trip it, so it flags "worth a look",
# not "broken".
SPIKE_FACTOR = 5.0


@dataclass(frozen=True)
class TrainStepRecord:
    """One logged optimizer update.

    A plain dataclass rather than a model: there is one of these per row, up to six
    figures of them, and they validate nothing that `float()` has not already
    decided. `secs` and `vram_peak_gb` are None for logs written before those
    columns existed.
    """

    step: int
    update: int
    epoch: int
    loss: float
    loss_avg: float
    grad_norm: float
    lr: float
    sigma: float
    bucket_h: int
    bucket_w: int
    secs: float | None
    vram_peak_gb: float | None


class TrainLogSummary(BaseModel):
    """Everything the log can say about a run, computed once.

    A model rather than a dataclass because this one crosses layers: it is stored
    as JSON on the training_runs row and read back into the UI, the same job
    DatasetHealth does.
    """

    model_config = ConfigDict(frozen=True)

    # --- progress -------------------------------------------------------------
    steps_executed: int
    final_step: int
    updates_logged: int
    unique_steps_covered: int
    restart_count: int
    rewound_steps: int

    # --- loss -----------------------------------------------------------------
    first_loss_avg: float | None = None
    final_loss_avg: float | None = None
    best_loss_avg: float | None = None
    best_loss_avg_step: int | None = None
    mean_loss_head: float | None = None
    mean_loss_tail: float | None = None
    loss_delta: float | None = None

    # --- time -----------------------------------------------------------------
    logged_seconds: float | None = None
    median_seconds_per_logged_step: float | None = None
    peak_vram_gb: float | None = None

    # --- shape ----------------------------------------------------------------
    epochs_started: int = 0
    epochs_completed: int = 0
    steps_per_epoch: float | None = None
    bucket_step_counts: dict[str, int] = {}
    lr_peak: float | None = None
    lr_final: float | None = None

    # --- health ---------------------------------------------------------------
    skipped_updates: int = 0
    nonfinite_loss_count: int = 0
    grad_norm_p50: float | None = None
    grad_norm_p95: float | None = None
    grad_norm_max: float | None = None
    loss_spike_count: int = 0


def parse_rows(rows: Iterable[Sequence[str]]) -> list[TrainStepRecord]:
    """Read CSV cells into records, skipping whatever cannot be read.

    Columns are located by *header name*, not position, so a log missing the two
    trailing columns needs no special case and a column inserted in the middle one
    day would not silently shift every field. A log with no header at all falls
    back to positional order.

    Nothing here raises. A row torn in half by a SIGKILL mid-write, a stray blank
    line, a header repeated by some future append — all are skipped, and the rows
    around them survive. Non-finite losses are *kept*: a NaN is the single most
    useful thing this file can tell you, and discarding it would hide the failure.
    """
    columns: dict[str, int] | None = None
    records: list[TrainStepRecord] = []

    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue

        if any(cell.strip().lower() == "step" for cell in row):
            # The header — or, after some future append, a second copy of it.
            # Matched anywhere in the row rather than in the first cell, so a
            # reordered header is still recognised as one instead of being read
            # as data through the positional fallback.
            columns = {name.strip().lower(): i for i, name in enumerate(row)}
            continue

        index = columns if columns is not None else _positional_columns(row)
        record = _parse_row(row, index)
        if record is not None:
            records.append(record)

    return records


def summarize(records: Sequence[TrainStepRecord]) -> TrainLogSummary:
    """Reduce the log to one summary. Never raises; an empty log summarizes to zeros.

    Written throughout to survive a **resume**, where the trainer reopens the same
    CSV in append mode and starts again from its last checkpoint. `step` is
    therefore not monotonic: it can jump backwards at the seam, and the rows after
    it can repeat work the rows before already covered.

    So the quantities split into three kinds. How far the run *got* is `max(step)`,
    ignoring order. Where it *ended up* is the last row, ignoring magnitude — which
    is why the loss figures walk the rows in file order rather than sorting by
    step, or a rewound tail would be reported as the run's final state. And what it
    *cost* (time, skipped updates, gradient behaviour) covers every row, because
    redone work still ran.
    """
    if not records:
        return TrainLogSummary(
            steps_executed=0,
            final_step=0,
            updates_logged=0,
            unique_steps_covered=0,
            restart_count=0,
            rewound_steps=0,
        )

    restart_count, rewound_steps = _count_restarts(records)
    head, tail = _loss_endpoints(records)
    finite_losses = [r.loss for r in records if math.isfinite(r.loss)]
    finite_avg = [r for r in records if math.isfinite(r.loss_avg)]
    best = min(finite_avg, key=lambda r: r.loss_avg, default=None)

    secs = [r.secs for r in records if r.secs is not None]
    vram = [r.vram_peak_gb for r in records if r.vram_peak_gb is not None]
    grad_norms = sorted(
        r.grad_norm for r in records if r.grad_norm > 0.0 and math.isfinite(r.grad_norm)
    )
    epochs_started = max(r.epoch for r in records)

    return TrainLogSummary(
        steps_executed=max(r.step for r in records),
        final_step=records[-1].step,
        updates_logged=len(records),
        unique_steps_covered=len({r.step for r in records}),
        restart_count=restart_count,
        rewound_steps=rewound_steps,
        first_loss_avg=finite_avg[0].loss_avg if finite_avg else None,
        final_loss_avg=finite_avg[-1].loss_avg if finite_avg else None,
        best_loss_avg=best.loss_avg if best is not None else None,
        best_loss_avg_step=best.step if best is not None else None,
        mean_loss_head=head,
        mean_loss_tail=tail,
        loss_delta=(tail - head) if head is not None and tail is not None else None,
        logged_seconds=sum(secs) if secs else None,
        median_seconds_per_logged_step=_percentile(sorted(secs), 0.5),
        peak_vram_gb=max(vram) if vram else None,
        epochs_started=epochs_started,
        # The highest epoch seen is the one in progress, so it is not finished.
        epochs_completed=max(0, epochs_started - 1),
        steps_per_epoch=_steps_per_epoch(records),
        bucket_step_counts=_bucket_counts(records),
        lr_peak=max(r.lr for r in records),
        lr_final=records[-1].lr,
        # The trainer writes 0.0 when it skipped a non-finite gradient. It also
        # writes 0.0 for a genuinely tiny norm, since the column is formatted to
        # four decimals — rare enough to note rather than engineer around.
        skipped_updates=sum(1 for r in records if r.grad_norm <= 0.0),
        nonfinite_loss_count=sum(1 for r in records if not math.isfinite(r.loss)),
        grad_norm_p50=_percentile(grad_norms, 0.5),
        grad_norm_p95=_percentile(grad_norms, 0.95),
        grad_norm_max=grad_norms[-1] if grad_norms else None,
        loss_spike_count=_count_spikes(finite_losses),
    )


# --- parsing helpers ----------------------------------------------------------


def _positional_columns(row: Sequence[str]) -> dict[str, int]:
    """Column map for a log with no header, read in the order the writer emits."""
    return {name: i for i, name in enumerate(TRAIN_LOG_COLUMNS) if i < len(row)}


def _cell(row: Sequence[str], index: dict[str, int], name: str) -> str | None:
    position = index.get(name)
    if position is None or position >= len(row):
        return None
    text = row[position].strip()
    return text or None


def _parse_row(row: Sequence[str], index: dict[str, int]) -> TrainStepRecord | None:
    """One row into a record, or None if it cannot be trusted.

    All-or-nothing on the ten original columns: a row missing one of them was torn
    mid-write, and half a step is worse than no step. The two later columns are
    genuinely optional.
    """
    try:
        values = {}
        for name in _REQUIRED_COLUMNS:
            text = _cell(row, index, name)
            if text is None:
                return None
            values[name] = float(text)
    except ValueError:
        return None

    optional: dict[str, float | None] = {}
    for name in ("secs", "vram_peak_gb"):
        text = _cell(row, index, name)
        try:
            optional[name] = float(text) if text is not None else None
        except ValueError:
            optional[name] = None

    return TrainStepRecord(
        step=int(values["step"]),
        update=int(values["update"]),
        epoch=int(values["epoch"]),
        loss=values["loss"],
        loss_avg=values["loss_avg"],
        grad_norm=values["grad_norm"],
        lr=values["lr"],
        sigma=values["sigma"],
        bucket_h=int(values["bucket_h"]),
        bucket_w=int(values["bucket_w"]),
        secs=optional["secs"],
        vram_peak_gb=optional["vram_peak_gb"],
    )


# --- summary helpers ----------------------------------------------------------


def _count_restarts(records: Sequence[TrainStepRecord]) -> tuple[int, int]:
    """Seams where `step` failed to advance, and how many steps each threw away.

    Each is a resume: the trainer restored a checkpoint that was necessarily at or
    behind wherever the previous launch died, so everything between the two is
    about to be run a second time.
    """
    restarts = 0
    rewound = 0
    for earlier, later in zip(records, records[1:]):
        if later.step <= earlier.step:
            restarts += 1
            rewound += earlier.step - later.step
    return restarts, rewound


def _loss_endpoints(
    records: Sequence[TrainStepRecord],
) -> tuple[float | None, float | None]:
    """Mean loss over the run's first and last rows, as a crude trend.

    The window shrinks to half the log so the two ends can never overlap and report
    a run as flat purely because it was short. Non-finite losses are dropped here —
    they are counted separately, and one NaN would otherwise poison both means.
    """
    finite = [r.loss for r in records if math.isfinite(r.loss)]
    window = min(LOSS_WINDOW, len(finite) // 2)
    if window == 0:
        return None, None
    return (
        sum(finite[:window]) / window,
        sum(finite[-window:]) / window,
    )


def _steps_per_epoch(records: Sequence[TrainStepRecord]) -> float | None:
    """Median distance between the starts of consecutive epochs.

    Read off the log rather than derived from the image count, so it reflects what
    the sampler actually did — bucket padding included. Spans that come out
    non-positive are the resume seam rewinding the epoch counter along with the
    step, and are dropped rather than allowed to drag the median negative.
    """
    first_step: dict[int, int] = {}
    for record in records:
        previous = first_step.get(record.epoch)
        if previous is None or record.step < previous:
            first_step[record.epoch] = record.step

    epochs = sorted(first_step)
    spans = [
        first_step[later] - first_step[earlier]
        for earlier, later in zip(epochs, epochs[1:])
        if first_step[later] - first_step[earlier] > 0
    ]
    if not spans:
        return None
    return _percentile(sorted(spans), 0.5)


def _bucket_counts(records: Sequence[TrainStepRecord]) -> dict[str, int]:
    """How many updates ran at each resolution bucket, largest share first."""
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.bucket_w}x{record.bucket_h}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _count_spikes(losses: Sequence[float]) -> int:
    """Losses far above the run's median. Heuristic: high-sigma steps trip it too."""
    if not losses:
        return 0
    middle = _percentile(sorted(losses), 0.5)
    if middle is None or middle <= 0:
        return 0
    return sum(1 for loss in losses if loss > SPIKE_FACTOR * middle)


def _percentile(ordered: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile of an already-sorted sequence, or None if empty."""
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
