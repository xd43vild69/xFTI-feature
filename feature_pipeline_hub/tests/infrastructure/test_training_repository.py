import pytest

from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.training_repository import (
    create_training_run,
    find_running_training_run,
    get_training_run,
    list_training_runs,
    record_train_log_metrics,
    update_training_run_status,
)


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


def _create(conn, dataset_run_id="run-1", kind="train", pid=1234):
    return create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind=kind,
        pid=pid,
        log_path="/tmp/log.txt",
        config={"total_steps": 3000, "lr": 0.0003},
    )


def test_create_and_get_round_trips(conn):
    training_run_id = _create(conn)

    run = get_training_run(conn, training_run_id)

    assert run is not None
    assert run.dataset_run_id == "run-1"
    assert run.kind == "train"
    assert run.status == "running"
    assert run.pid == 1234
    assert run.log_path == "/tmp/log.txt"
    assert run.config == {"total_steps": 3000, "lr": 0.0003}
    assert run.finished_at is None


def test_get_missing_run_returns_none(conn):
    assert get_training_run(conn, "nope") is None


def test_update_status_sets_finished_at_when_leaving_running(conn):
    training_run_id = _create(conn)

    update_training_run_status(conn, training_run_id, "completed")

    run = get_training_run(conn, training_run_id)
    assert run.status == "completed"
    assert run.finished_at is not None


def test_update_status_back_to_running_clears_finished_at(conn):
    training_run_id = _create(conn)
    update_training_run_status(conn, training_run_id, "failed")

    update_training_run_status(conn, training_run_id, "running")

    assert get_training_run(conn, training_run_id).finished_at is None


def test_list_training_runs_newest_first(conn):
    first = _create(conn, pid=1)
    second = _create(conn, pid=2)

    ids = [r.training_run_id for r in list_training_runs(conn)]

    assert ids == [second, first]


def test_list_training_runs_can_scope_to_a_dataset_run(conn):
    _create(conn, dataset_run_id="run-a")
    _create(conn, dataset_run_id="run-b")

    scoped = list_training_runs(conn, dataset_run_id="run-a")

    assert [r.dataset_run_id for r in scoped] == ["run-a"]


def test_find_running_training_run_returns_none_when_nothing_is_running(conn):
    training_run_id = _create(conn)
    update_training_run_status(conn, training_run_id, "completed")

    assert find_running_training_run(conn) is None


def test_find_running_training_run_finds_the_active_one(conn):
    stopped = _create(conn, pid=1)
    update_training_run_status(conn, stopped, "stopped")
    running = _create(conn, pid=2)

    found = find_running_training_run(conn)

    assert found is not None
    assert found.training_run_id == running


def test_a_fresh_run_has_no_telemetry_yet(conn):
    training_run_id = _create(conn)

    run = get_training_run(conn, training_run_id)

    assert run.duration_seconds is None
    assert run.gpu_seconds is None
    assert run.cost_estimate is None
    assert run.error_message == ""
    assert run.steps_executed is None
    assert run.metrics == {}


def test_status_update_can_attach_telemetry(conn):
    training_run_id = _create(conn)

    update_training_run_status(
        conn,
        training_run_id,
        "completed",
        duration_seconds=125.4,
        gpu_seconds=125.4,
        cost_estimate=0.05,
    )

    run = get_training_run(conn, training_run_id)
    assert run.status == "completed"
    assert run.duration_seconds == 125.4
    assert run.gpu_seconds == 125.4
    assert run.cost_estimate == 0.05


def test_status_update_can_attach_an_error_message(conn):
    training_run_id = _create(conn)

    update_training_run_status(conn, training_run_id, "failed", error_message="CUDA OOM")

    assert get_training_run(conn, training_run_id).error_message == "CUDA OOM"


def test_train_log_metrics_round_trip(conn):
    training_run_id = _create(conn)

    record_train_log_metrics(
        conn, training_run_id, steps_executed=1180, metrics={"final_loss_avg": 0.08}
    )

    run = get_training_run(conn, training_run_id)
    assert run.steps_executed == 1180
    assert run.metrics == {"final_loss_avg": 0.08}


def test_a_later_status_change_does_not_erase_the_metrics(conn):
    # update_training_run_status writes every telemetry column from its keyword
    # defaults, so folding these two in as more arguments would have meant any bare
    # status change silently nulled them. They get their own statement instead.
    training_run_id = _create(conn)
    record_train_log_metrics(
        conn, training_run_id, steps_executed=1180, metrics={"final_loss_avg": 0.08}
    )

    update_training_run_status(conn, training_run_id, "stopped")

    run = get_training_run(conn, training_run_id)
    assert run.steps_executed == 1180
    assert run.metrics == {"final_loss_avg": 0.08}


def test_an_unreadable_metrics_blob_degrades_to_empty_rather_than_raising(conn):
    # One corrupt row must not break list_training_runs for every run on the page.
    training_run_id = _create(conn)
    with conn:
        conn.execute(
            "UPDATE training_runs SET metrics_json = ? WHERE training_run_id = ?",
            ("not json at all", training_run_id),
        )

    assert get_training_run(conn, training_run_id).metrics == {}
    assert len(list_training_runs(conn)) == 1
