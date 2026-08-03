import pytest

from feature_pipeline.application import training_metrics_service as metrics
from feature_pipeline.domain.train_log import TRAIN_LOG_COLUMNS
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure.database import get_connection

HEADER = ",".join(TRAIN_LOG_COLUMNS)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _write_log(output_dir, steps, *, loss=0.1):
    """A train_log.csv with one row per given step, as the trainer would leave it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [HEADER]
    for step in steps:
        rows.append(
            f"{step},{step // 4},1,{loss:.6f},{loss:.6f},1.0000,"
            f"1.000e-04,0.5000,64,64,0.500,8.00"
        )
    (output_dir / "train_log.csv").write_text("\n".join(rows) + "\n")


def _create(conn, *, output_dir, kind="train", total_steps=1200, dataset_run_id="run-1"):
    return repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind=kind,
        pid=1234,
        log_path="/tmp/log.txt",
        config={
            "output_dir": str(output_dir),
            "total_steps": total_steps,
            "batch_size": 2,
        },
    )


# --- grouping (pure) ----------------------------------------------------------


def _run(training_run_id, *, kind="train", output_dir="/out", total_steps=1200):
    from datetime import datetime, timezone

    return repo.TrainingRun(
        training_run_id=training_run_id,
        dataset_run_id="run-1",
        kind=kind,
        status="completed",
        pid=1,
        log_path="/tmp/log.txt",
        config={"output_dir": output_dir, "total_steps": total_steps} if output_dir else {},
        started_at=datetime.now(timezone.utc),
        finished_at=None,
    )


def test_launches_sharing_an_output_dir_are_one_training():
    # list_training_runs hands them over newest first; the lineage reads oldest first.
    groups = metrics.group_training_lineages([_run("second"), _run("first")])

    assert len(groups) == 1
    assert [run.training_run_id for run in groups[0]] == ["first", "second"]


def test_different_output_dirs_are_different_trainings():
    groups = metrics.group_training_lineages(
        [_run("a", output_dir="/out/a"), _run("b", output_dir="/out/b")]
    )

    assert len(groups) == 2


def test_a_precache_row_never_shadows_the_training_it_preceded():
    # A pre-cache shares the dataset_run_id and often has the newer started_at, so
    # "take the newest row" quietly reported pre-cache data as the training's.
    groups = metrics.group_training_lineages(
        [_run("precache", kind="precache", output_dir=""), _run("train")]
    )

    assert len(groups) == 1
    assert groups[0][0].training_run_id == "train"


def test_legacy_rows_without_an_output_dir_stay_separate():
    # They predate output_dir being recorded; keying on "" would merge every one of
    # them into a single fictional training.
    groups = metrics.group_training_lineages(
        [_run("a", output_dir=""), _run("b", output_dir="")]
    )

    assert len(groups) == 2


# --- loading ------------------------------------------------------------------


def test_a_lineage_reports_the_steps_its_log_actually_recorded(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, range(4, 401, 4))
    _create(conn, output_dir=output_dir, total_steps=1200)

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.train_log is not None
    assert lineage.steps_executed == 400
    assert lineage.total_steps_target == 1200
    assert lineage.completion_fraction == pytest.approx(400 / 1200)
    assert lineage.train_log_is_live is True


def test_the_target_comes_from_the_latest_launch_because_a_resume_raises_it(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, range(4, 1601, 4))
    _create(conn, output_dir=output_dir, total_steps=1200)
    _create(conn, output_dir=output_dir, total_steps=2000)

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.launch_count == 2
    assert lineage.resume_count == 1
    assert lineage.total_steps_target == 2000


def test_wall_clock_and_cost_are_summed_across_launches(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, [4, 8])
    first = _create(conn, output_dir=output_dir)
    second = _create(conn, output_dir=output_dir)
    repo.update_training_run_status(
        conn, first, "completed", duration_seconds=100.0, cost_estimate=1.0
    )
    repo.update_training_run_status(
        conn, second, "completed", duration_seconds=50.0, cost_estimate=0.5
    )

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.wall_clock_seconds == 150.0
    assert lineage.cost_estimate == pytest.approx(1.5)


def test_launches_with_no_telemetry_do_not_make_the_totals_zero(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, [4])
    _create(conn, output_dir=output_dir)

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.wall_clock_seconds is None
    assert lineage.seconds_per_step is None


def test_the_stored_snapshot_takes_over_when_the_log_is_gone(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, range(4, 401, 4))
    training_run_id = _create(conn, output_dir=output_dir)
    run = repo.get_training_run(conn, training_run_id)
    from feature_pipeline.application import training_service

    summary = training_service.read_train_log_summary(run)
    repo.record_train_log_metrics(
        conn, training_run_id,
        steps_executed=summary.steps_executed, metrics=summary.model_dump(mode="json"),
    )
    (output_dir / "train_log.csv").unlink()

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.train_log is not None
    assert lineage.train_log_is_live is False
    assert lineage.steps_executed == 400


def test_a_lineage_with_neither_log_nor_snapshot_reports_nothing_rather_than_guessing(
    conn, tmp_path
):
    _create(conn, output_dir=tmp_path / "gone")

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.train_log is None
    assert lineage.steps_executed is None
    assert lineage.completion_fraction is None


def test_a_zero_target_does_not_divide_by_zero(conn, tmp_path):
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, [4])
    _create(conn, output_dir=output_dir, total_steps=0)

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.completion_fraction is None


def test_completion_never_exceeds_one(conn, tmp_path):
    # A resume that overshot its own target, or a target lowered afterwards.
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, range(4, 2001, 4))
    _create(conn, output_dir=output_dir, total_steps=1200)

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.completion_fraction == 1.0


def test_seconds_per_step_uses_wall_clock_not_the_csv_column(conn, tmp_path):
    # The CSV's `secs` times one micro-step and is sampled only on update steps;
    # summing it would undercount by roughly grad_accum_steps.
    output_dir = tmp_path / "checkpoints"
    _write_log(output_dir, range(4, 401, 4))
    training_run_id = _create(conn, output_dir=output_dir)
    repo.update_training_run_status(
        conn, training_run_id, "completed", duration_seconds=800.0
    )

    (lineage,) = metrics.load_training_lineages(conn, "run-1")

    assert lineage.seconds_per_step == pytest.approx(2.0)
