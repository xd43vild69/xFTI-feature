"""Reading checkpoint_log.csv: what each `.safetensors` cost to reach.

`train_log.csv` answers what the loop *did* — losses, gradients, buckets — but not
what it took. Its only temporal column is `secs`, which times a single micro-step and
is written only on update steps, so summing it undercounts by roughly
`grad_accum_steps` and misses everything outside the step itself. `domain/train_log.py`
says as much in `TrainLogSummary.logged_seconds`.

This file is the other half. The trainer stamps one row every time it actually writes
a checkpoint (`workers/krea2/metrics.py`, `CheckpointLog.record`, called from
`CheckpointManager.save` after the write succeeded), carrying wall-clock elapsed since
the previous one and the number of unique training images. With `save_every` fixed and
the rest of the hyperparameters standardized, that turns a run into a sequence of
comparable spans: seconds per checkpoint, and seconds per image, across datasets of
different sizes.

Two things shape the arithmetic here, both of them consequences of how the writer
works:

**The clock is per launch.** A resumed run's first row is timed from the moment the
process came back up, not from the previous launch's last save. So spans are additive
within a launch and across the lineage, and never include the hours a stopped run spent
waiting to be resumed. `launch_id` is what makes that visible: a resume appends to the
file the earlier process wrote, and without the column two launches' spans would read as
one launch's.

**`steps_delta` can be non-positive.** That is the resume seam: the trainer restored a
checkpoint at or behind where it died, so the first row after it measures against a
step the log already passed. Those spans are real time that was really spent — they
stay in the totals — but they are dropped from the per-step and per-image medians,
where a negative or zero denominator would be meaningless.

Pure, in the same split as `train_log`: this knows the file's format, the application
layer opens it.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from feature_pipeline.domain.train_log import percentile

# Mirror of krea2.metrics.CHECKPOINT_COLUMNS. Duplicated rather than imported for the
# same reason TRAIN_LOG_COLUMNS is: `workers/` is not on the hub's sys.path and the
# training runtime is a different interpreter.
# tests/workers/test_krea2_metrics.py pins the two copies together.
CHECKPOINT_LOG_COLUMNS = (
    "step", "epoch", "reason", "timestamp",
    "elapsed_seconds", "steps_delta", "num_images", "launch_id",
)

# Everything but num_images, which the standalone workflow can legitimately leave
# blank, and launch_id, which files written before that column existed do not have.
_REQUIRED_COLUMNS = CHECKPOINT_LOG_COLUMNS[:6]

# Reasons the writer emits. Anything else is kept verbatim rather than rejected: a
# newer trainer writing a reason this version has not heard of should not cost the
# reader a row.
INTERRUPTED_REASON = "interrupt"


@dataclass(frozen=True)
class CheckpointRecord:
    """One checkpoint the trainer wrote, and the span that led to it."""

    step: int
    epoch: int
    reason: str
    timestamp: float
    elapsed_seconds: float
    steps_delta: int
    num_images: int | None
    # The process that wrote the row. Empty for a log written before the column
    # existed, which reads as "one launch" rather than as an unknown.
    launch_id: str = ""


class CheckpointInterval(BaseModel):
    """One row, with the derived figures the UI shows.

    `seconds_per_image` divides by the dataset's *unique* image count rather than by
    the images the span consumed, deliberately: the question it answers is "what does
    one checkpoint of a dataset this size cost", which is what makes two runs over
    different datasets comparable. Both ratios are None when their denominator is not
    usable, rather than zero — a missing measurement and a fast one are different
    things.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    epoch: int
    reason: str
    elapsed_seconds: float
    cumulative_seconds: float
    steps_delta: int
    num_images: int | None = None
    seconds_per_image: float | None = None
    seconds_per_step: float | None = None
    launch_id: str = ""
    # 1-based, in the order the launches appear in the file. The id is opaque; this
    # is what an operator can actually read off a table.
    launch_ordinal: int = 1

    @property
    def is_interrupted(self) -> bool:
        return self.reason == INTERRUPTED_REASON


class CheckpointLogSummary(BaseModel):
    """Every checkpoint of one training, and the medians across them.

    Crosses layers like TrainLogSummary does — stored as JSON on the training_runs row
    so the figures survive the runtime directory being reclaimed — so it is a model
    rather than a dataclass.
    """

    model_config = ConfigDict(frozen=True)

    intervals: tuple[CheckpointInterval, ...] = ()
    checkpoint_count: int = 0
    # How many separate processes contributed. More than one means the training was
    # resumed — the same training, continued, not a second one.
    launch_count: int = 0
    interrupted_count: int = 0
    # Every span, partial ones included — this is how long the run really took.
    total_elapsed_seconds: float = 0.0
    # These three describe complete save_every spans only. See `summarize`.
    median_seconds_per_checkpoint: float | None = None
    median_seconds_per_image: float | None = None
    median_seconds_per_step: float | None = None
    images_trained: int | None = None
    last_step: int = 0
    # True when these spans were inferred from the checkpoint files' timestamps
    # rather than measured by the trainer. See `reconstruct`.
    is_reconstructed: bool = False


def parse_rows(rows: Iterable[Sequence[str]]) -> list[CheckpointRecord]:
    """Read CSV cells into records, skipping whatever cannot be read.

    Columns are located by header name with a positional fallback, and nothing raises
    — the same contract as `train_log.parse_rows`, and for the same reason: this file
    is written by a process that is expected to be killed.
    """
    columns: dict[str, int] | None = None
    records: list[CheckpointRecord] = []

    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue

        if any(cell.strip().lower() == "step" for cell in row):
            columns = {name.strip().lower(): i for i, name in enumerate(row)}
            continue

        index = columns if columns is not None else _positional_columns(row)
        record = _parse_row(row, index)
        if record is not None:
            records.append(record)

    return records


def summarize(
    records: Sequence[CheckpointRecord], *, reconstructed: bool = False
) -> CheckpointLogSummary:
    """Reduce the log to one summary. Never raises; an empty log summarizes to zeros.

    `reconstructed` rides through to the summary so the UI can say where these came
    from. It changes no arithmetic — inferred spans are summarized exactly like
    measured ones, they are simply less certain.
    """
    if not records:
        return CheckpointLogSummary(is_reconstructed=reconstructed)

    ordinals = _launch_ordinals(records)

    intervals: list[CheckpointInterval] = []
    cumulative = 0.0
    for record in records:
        cumulative += record.elapsed_seconds
        intervals.append(
            CheckpointInterval(
                step=record.step,
                epoch=record.epoch,
                reason=record.reason,
                elapsed_seconds=record.elapsed_seconds,
                cumulative_seconds=cumulative,
                steps_delta=record.steps_delta,
                num_images=record.num_images,
                seconds_per_image=_ratio(record.elapsed_seconds, record.num_images),
                seconds_per_step=_ratio(record.elapsed_seconds, record.steps_delta),
                launch_id=record.launch_id,
                launch_ordinal=ordinals[record.launch_id],
            )
        )

    images = [r.num_images for r in records if r.num_images]
    # The two per-checkpoint medians describe what a *complete* span costs, so they
    # see only complete spans. An interrupt was cut short by a stop or a failure and
    # a `final` closes whatever was left over after the last cadence boundary —
    # averaging those in would report the typical checkpoint as cheaper than any
    # checkpoint ever was. Both still count towards the totals: the time was spent.
    full = [i for i in intervals if i.reason == "periodic" and i.steps_delta > 0]

    return CheckpointLogSummary(
        intervals=tuple(intervals),
        checkpoint_count=len(intervals),
        launch_count=len(ordinals),
        interrupted_count=sum(1 for i in intervals if i.is_interrupted),
        total_elapsed_seconds=cumulative,
        median_seconds_per_checkpoint=_median([i.elapsed_seconds for i in full]),
        median_seconds_per_image=_median(
            [i.seconds_per_image for i in full if i.seconds_per_image is not None]
        ),
        median_seconds_per_step=_median(
            [i.seconds_per_step for i in intervals if i.seconds_per_step is not None]
        ),
        # The last one seen rather than the max: the dataset a resume trained on is
        # the one that describes where the run ended up.
        images_trained=images[-1] if images else None,
        last_step=max(r.step for r in records),
        is_reconstructed=reconstructed,
    )


@dataclass(frozen=True)
class CheckpointFile:
    """A checkpoint found on disk, for a run that predates the log."""

    step: int
    written_at: float          # mtime, epoch seconds
    num_images: int | None     # ss_num_train_images from the safetensors header
    epoch: int = 0
    is_final: bool = False


def reconstruct(
    files: Sequence[CheckpointFile],
    launches: Sequence[tuple[str, float]] = (),
    save_every: int | None = None,
) -> list[CheckpointRecord]:
    """Infer the spans of a run that finished before the trainer logged them.

    A checkpoint's mtime is when it was written, so consecutive files bracket the
    work between them. That reconstructs everything the trainer would have recorded,
    with two differences worth stating rather than hiding:

    **The first span of each launch includes startup.** Loading the model and the
    latent cache lands inside it — measurably, about 40 extra seconds on a run whose
    other spans were 12 minutes. It is real time the launch spent, but it is not the
    cost of training those steps.

    **The gap where the process was dead is removed, if we know about it.** Given
    `launches` as (id, started_at) pairs, a span is anchored to its launch's start
    rather than to the previous launch's last save, which is the same rule the live
    writer follows. Without them, a resumed run bills its downtime as training —
    twelve minutes of nothing, on the run this was built against.

    `save_every` decides which saves look like the cadence and which were forced: a
    step off the cadence was written by a stop, an OOM or a failing gradient. Without
    it every non-final save is assumed periodic, which is what a run with no stored
    settings looks like.
    """
    ordered = _dedupe_by_step(sorted(files, key=lambda f: (f.written_at, f.step)))
    starts = sorted(launches, key=lambda pair: pair[1])

    records: list[CheckpointRecord] = []
    previous_time: dict[str, float] = {}
    previous_step: dict[str, int] = {}
    last_step_overall: int | None = None

    for entry in ordered:
        launch_id, launch_start = _owning_launch(starts, entry.written_at)

        # Anchored to this launch's previous save, else to where the launch began,
        # else — with no launch information at all — to the save before it.
        anchor = previous_time.get(launch_id)
        if anchor is None:
            anchor = launch_start if launch_start is not None else entry.written_at
        elapsed = max(0.0, entry.written_at - anchor)

        # Steps come from the previous save of this launch, falling back to the last
        # save anywhere: a resume restarts from the newest checkpoint, so that file
        # is exactly where the launch picked up.
        base = previous_step.get(launch_id, last_step_overall or 0)
        records.append(
            CheckpointRecord(
                step=entry.step,
                epoch=entry.epoch,
                reason=_infer_reason(entry, save_every),
                timestamp=entry.written_at,
                elapsed_seconds=elapsed,
                steps_delta=entry.step - base,
                num_images=entry.num_images,
                launch_id=launch_id,
            )
        )
        previous_time[launch_id] = entry.written_at
        previous_step[launch_id] = entry.step
        last_step_overall = entry.step

    return records


def _dedupe_by_step(files: Sequence[CheckpointFile]) -> list[CheckpointFile]:
    """One entry per step, keeping the earliest.

    `{prefix}_FINAL.safetensors` is the same weights at the same step as the last
    `_step_N` file, written seconds later — two entries would invent a checkpoint
    that took no time and covered no steps. The final flag survives the merge so the
    row still reads as the run's last.
    """
    merged: dict[int, CheckpointFile] = {}
    for entry in files:
        existing = merged.get(entry.step)
        if existing is None:
            merged[entry.step] = entry
            continue
        merged[entry.step] = CheckpointFile(
            step=existing.step,
            written_at=min(existing.written_at, entry.written_at),
            num_images=existing.num_images or entry.num_images,
            epoch=max(existing.epoch, entry.epoch),
            is_final=existing.is_final or entry.is_final,
        )
    return sorted(merged.values(), key=lambda f: (f.written_at, f.step))


def _owning_launch(
    starts: Sequence[tuple[str, float]], written_at: float
) -> tuple[str, float | None]:
    """The launch a file was written by: the last one that had started by then."""
    owner: tuple[str, float | None] = ("", None)
    for launch_id, started in starts:
        if started <= written_at:
            owner = (launch_id, started)
    return owner


def _infer_reason(entry: CheckpointFile, save_every: int | None) -> str:
    """Which kind of save produced this file, as far as its step number shows."""
    if entry.is_final:
        return "final"
    if save_every and entry.step % save_every != 0:
        return INTERRUPTED_REASON
    return "periodic"


# --- parsing helpers ----------------------------------------------------------


def _positional_columns(row: Sequence[str]) -> dict[str, int]:
    """Column map for a log with no header, read in the order the writer emits."""
    return {name: i for i, name in enumerate(CHECKPOINT_LOG_COLUMNS) if i < len(row)}


def _cell(row: Sequence[str], index: dict[str, int], name: str) -> str | None:
    position = index.get(name)
    if position is None or position >= len(row):
        return None
    text = row[position].strip()
    return text or None


def _parse_row(
    row: Sequence[str], index: dict[str, int]
) -> CheckpointRecord | None:
    """One row into a record, or None if it cannot be trusted.

    All-or-nothing on the six columns that describe the span. `num_images` is optional
    — the standalone workflow can leave it blank — and a row whose elapsed time is
    non-finite is dropped rather than allowed to poison every cumulative figure
    downstream of it.
    """
    reason = _cell(row, index, "reason")
    if reason is None:
        return None

    try:
        values: dict[str, float] = {}
        for name in _REQUIRED_COLUMNS:
            if name == "reason":
                continue
            text = _cell(row, index, name)
            if text is None:
                return None
            values[name] = float(text)
    except ValueError:
        return None

    if not math.isfinite(values["elapsed_seconds"]):
        return None

    images: int | None = None
    text = _cell(row, index, "num_images")
    if text is not None:
        try:
            images = int(float(text))
        except ValueError:
            images = None

    launch_id = _cell(row, index, "launch_id") or ""

    return CheckpointRecord(
        step=int(values["step"]),
        epoch=int(values["epoch"]),
        reason=reason,
        timestamp=values["timestamp"],
        # Clamped rather than dropped: a clock stepping backwards mid-run is a machine
        # problem, not a reason to lose the checkpoint's existence.
        elapsed_seconds=max(0.0, values["elapsed_seconds"]),
        steps_delta=int(values["steps_delta"]),
        num_images=images,
        launch_id=launch_id,
    )


# --- summary helpers ----------------------------------------------------------


def _launch_ordinals(records: Sequence[CheckpointRecord]) -> dict[str, int]:
    """Launch id → 1-based position, in order of first appearance.

    File order is launch order: a resume can only append. Numbered rather than shown
    raw because the id is a uuid — "launch 2" is what an operator can read, and the id
    is still there for cross-referencing the lifecycle events in log.txt.
    """
    ordinals: dict[str, int] = {}
    for record in records:
        if record.launch_id not in ordinals:
            ordinals[record.launch_id] = len(ordinals) + 1
    return ordinals


def _ratio(elapsed: float, denominator: int | None) -> float | None:
    """Elapsed per unit, or None when the denominator says nothing."""
    if not denominator or denominator <= 0:
        return None
    return elapsed / denominator


def _median(values: Sequence[float]) -> float | None:
    return percentile(sorted(values), 0.5)
