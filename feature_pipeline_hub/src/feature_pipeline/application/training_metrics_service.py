"""What one training actually did, assembled from however many launches it took.

A resume is not a new training. `resume_training` reuses the previous run's
`output_dir` — that reuse is the *only* thing that makes it a resume, since that
is where the trainer looks for a checkpoint and where krea2.metrics appends
train_log.csv — while allocating a fresh run directory for its own settings and
log. So the database ends up with N rows describing one continuous piece of work,
and the CSV that describes it belongs to all of them equally.

Grouping on `output_dir` is therefore grouping on "the same training", which is
what an operator means by the word. Everything here reads; nothing launches.

Sits alongside inventory_service for the same reason that one exists: the Metrics
page needs a read-only view *across* runs that no single step can answer, and
training_service is already long enough without it.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from feature_pipeline.application import training_service
from feature_pipeline.domain.checkpoint_log import CheckpointLogSummary
from feature_pipeline.domain.train_log import TrainLogSummary
from feature_pipeline.infrastructure import training_repository as repo


@dataclass(frozen=True)
class TrainingLineage:
    """One training, and the measurements that describe it.

    `train_log` is the whole lineage's log, not one launch's. `train_log_is_live`
    says whether it was just read off disk or restored from the snapshot stored
    when a launch finished — the stored one covers only that launch, so the
    distinction is worth showing rather than hiding.

    `checkpoint_log` is the same arrangement for the per-checkpoint spans, resolved
    independently: a run can perfectly well have one file and not the other, since
    checkpoint logging arrived later and a run killed before its first save never
    writes one at all.
    """

    output_dir: str
    launches: tuple[repo.TrainingRun, ...]  # oldest first
    train_log: TrainLogSummary | None
    train_log_is_live: bool
    checkpoint_log: CheckpointLogSummary | None
    checkpoint_log_is_live: bool
    total_steps_target: int
    wall_clock_seconds: float | None
    cost_estimate: float | None
    steps_executed: int | None
    completion_fraction: float | None
    images_per_epoch_estimate: float | None

    @property
    def latest(self) -> repo.TrainingRun:
        return self.launches[-1]

    @property
    def launch_count(self) -> int:
        return len(self.launches)

    @property
    def resume_count(self) -> int:
        return len(self.launches) - 1

    @property
    def status(self) -> str:
        return self.latest.status

    @property
    def seconds_per_step(self) -> float | None:
        """Wall-clock seconds per step — the honest throughput figure.

        Derived from the process's elapsed time rather than from the CSV's `secs`
        column, which times a single micro-step and is sampled only on update
        steps: summing it would undercount by roughly grad_accum_steps. This one
        includes model loading and checkpointing, which is the point — that time
        was really spent.
        """
        if not self.wall_clock_seconds or not self.steps_executed:
            return None
        return self.wall_clock_seconds / self.steps_executed


def group_training_lineages(
    runs: Sequence[repo.TrainingRun],
) -> list[list[repo.TrainingRun]]:
    """Group train runs by the output_dir they share, newest lineage first.

    Pure, so it can be tested without a database. Pre-cache rows are dropped here
    rather than by each caller: they share a dataset_run_id with the training they
    preceded, and picking "the newest row" without filtering silently returns a
    pre-cache row whenever one ran more recently than the train it belongs to.

    A row from before output_dir was recorded gets a key of its own, so old rows
    stay separate runs instead of collapsing into one fictional lineage under "".
    """
    groups: dict[str, list[repo.TrainingRun]] = {}
    for run in runs:
        if run.kind != "train":
            continue
        key = run.config.get("output_dir") or f"\x00run:{run.training_run_id}"
        groups.setdefault(key, []).append(run)
    # list_training_runs hands them over newest first, so each group is reversed to
    # read as the launches happened.
    return [list(reversed(group)) for group in groups.values()]


def load_training_lineages(
    conn: sqlite3.Connection, dataset_run_id: str
) -> list[TrainingLineage]:
    """Every training this dataset has been through, newest first."""
    runs = repo.list_training_runs(conn, dataset_run_id=dataset_run_id)
    return [_build_lineage(group) for group in group_training_lineages(runs)]


def _build_lineage(launches: list[repo.TrainingRun]) -> TrainingLineage:
    latest = launches[-1]
    summary, is_live = _resolve_summary(launches)
    checkpoints, checkpoints_live = _resolve_checkpoint_summary(launches)

    # From the latest launch: a resume must raise total_steps above the checkpoint
    # to have anything left to run, so the first launch's target is stale.
    total_steps = int(latest.config.get("total_steps") or 0)
    steps_executed = summary.steps_executed if summary is not None else latest.steps_executed

    completion: float | None = None
    if total_steps > 0 and steps_executed is not None:
        completion = min(1.0, steps_executed / total_steps)

    images_per_epoch: float | None = None
    if summary is not None and summary.steps_per_epoch is not None:
        batch_size = int(latest.config.get("batch_size") or 0)
        if batch_size > 0:
            images_per_epoch = summary.steps_per_epoch * batch_size

    return TrainingLineage(
        output_dir=str(latest.config.get("output_dir") or ""),
        launches=tuple(launches),
        train_log=summary,
        train_log_is_live=is_live,
        checkpoint_log=checkpoints,
        checkpoint_log_is_live=checkpoints_live,
        total_steps_target=total_steps,
        wall_clock_seconds=_total(run.duration_seconds for run in launches),
        cost_estimate=_total(run.cost_estimate for run in launches),
        steps_executed=steps_executed,
        completion_fraction=completion,
        images_per_epoch_estimate=images_per_epoch,
    )


def _resolve_summary(
    launches: Sequence[repo.TrainingRun],
) -> tuple[TrainLogSummary | None, bool]:
    """The lineage's log: read live if the file is still there, stored if not.

    Live wins because it covers the whole lineage, including a launch still in
    flight, where a stored snapshot only ever describes the launch that wrote it.
    Falling back at all is why the columns are persisted: the runtime directory is
    large and gets reclaimed, and losing the history with it would be a shame.
    """
    live = training_service.read_train_log_summary(launches[-1])
    if live is not None:
        return live, True

    for run in reversed(launches):
        if not run.metrics:
            continue
        try:
            return TrainLogSummary.model_validate(run.metrics), False
        except ValidationError:
            # Written by an older version whose fields no longer fit. Not worth
            # failing the page over; try the launch before it.
            continue
    return None, False


def _resolve_checkpoint_summary(
    launches: Sequence[repo.TrainingRun],
) -> tuple[CheckpointLogSummary | None, bool]:
    """The lineage's checkpoint spans, live if the file survives, stored if not.

    Same policy as _resolve_summary, kept as a second function rather than a generic
    one: the two differ in which reader and which stored column they use, and the
    shared part is four lines of control flow.
    """
    live = training_service.read_checkpoint_log_summary(launches[-1])
    if live is not None:
        return live, True

    for run in reversed(launches):
        if not run.checkpoint_metrics:
            continue
        try:
            return CheckpointLogSummary.model_validate(run.checkpoint_metrics), False
        except ValidationError:
            continue

    # Last resort: every run launched before the trainer logged its saves has none of
    # the above, but the checkpoints themselves are still on disk and their mtimes say
    # when each one was written. Inferred rather than measured, and marked as such.
    return training_service.reconstruct_checkpoint_log_summary(launches), False


def _total(values: Iterable[float | None]) -> float | None:
    """Sum what is present, or None if nothing is."""
    present = [value for value in values if value is not None]
    return sum(present) if present else None
