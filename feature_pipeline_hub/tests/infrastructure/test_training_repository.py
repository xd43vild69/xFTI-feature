import pytest

from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.training_repository import (
    create_training_run,
    find_running_training_run,
    get_training_run,
    list_training_runs,
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
