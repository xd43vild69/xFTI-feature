import sys
from pathlib import Path

import pytest

from feature_pipeline.application import training_service
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.training_runner import TrainingEnvironment

SUCCESSFUL_PRECACHE = """
import json, os
settings = json.load(open(os.environ["PRECACHE_SETTINGS_PATH"]))
assert settings["trigger_word"] == "sks_test"
print("Pre-caching finished! 2 encoded, 0 reused, VRAM freed", flush=True)
"""

FAILING_PRECACHE = """
print("something went wrong before finishing", flush=True)
"""

HANGING_PRECACHE = """
import time
time.sleep(30)
"""

STUB_TRAIN = """
import json, os
settings = json.load(open(os.environ["TRAIN_SETTINGS_PATH"]))
print(f"training with total_steps={settings['total_steps']}", flush=True)
"""


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


@pytest.fixture
def fake_model_dir(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    model = runtime / "model"
    for part in ("transformer", "text_encoder", "vae"):
        (model / part).mkdir(parents=True)
    (runtime / "venv" / "bin").mkdir(parents=True)
    (runtime / "venv" / "bin" / "python").symlink_to(sys.executable)
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime))
    return runtime


def _write_stub(tmp_path: Path, script: str) -> Path:
    path = tmp_path / f"stub_{abs(hash(script))}.py"
    path.write_text(script)
    return path


def test_start_training_runs_precache_then_launches_train(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))

    training_run_id = training_service.start_training(
        conn,
        dataset_run_id="run-1",
        dataset_name="my_concept",
        trigger_word="sks_test",
        config=training_service.TrainingConfig(total_steps=20),
    )

    train_run = repo.get_training_run(conn, training_run_id)
    assert train_run.kind == "train"
    assert train_run.status == "running"
    assert train_run.config["total_steps"] == 20

    all_runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert sorted(r.kind for r in all_runs) == ["precache", "train"]
    precache_run = next(r for r in all_runs if r.kind == "precache")
    assert precache_run.status == "completed"

    from feature_pipeline.infrastructure import training_runner

    training_runner.stop_process(train_run.pid, grace_period_seconds=1)


def test_a_failing_precache_raises_and_never_launches_train(
    conn, fake_model_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, FAILING_PRECACHE)
    )
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))

    with pytest.raises(training_service.PrecacheFailed):
        training_service.start_training(
            conn,
            dataset_run_id="run-1",
            dataset_name="my_concept",
            trigger_word="sks_test",
            config=training_service.TrainingConfig(),
        )

    runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert [r.kind for r in runs] == ["precache"]
    assert runs[0].status == "failed"


def test_a_hanging_precache_times_out(conn, fake_model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(
        training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, HANGING_PRECACHE)
    )
    monkeypatch.setattr(training_service, "PRECACHE_TIMEOUT_SECONDS", 0.5)

    with pytest.raises(training_service.PrecacheFailed, match="did not finish"):
        training_service.start_training(
            conn,
            dataset_run_id="run-1",
            dataset_name="my_concept",
            trigger_word="sks",
            config=training_service.TrainingConfig(),
        )

    runs = repo.list_training_runs(conn, dataset_run_id="run-1")
    assert runs[0].status == "failed"


def test_dataset_and_cache_dirs_are_scoped_under_the_runtime(fake_model_dir):
    dataset_dir = training_service.dataset_dir_for("my_concept")
    cache_dir = training_service.cache_dir_for("my_concept")

    assert dataset_dir == fake_model_dir / "datasets" / "my_concept"
    assert cache_dir == fake_model_dir / "cache" / "my_concept"


def test_training_log_csv_path_is_next_to_the_checkpoints():
    run = repo.TrainingRun(
        training_run_id="t1",
        dataset_run_id="r1",
        kind="train",
        status="running",
        pid=1,
        log_path="/x/runs/train-abc/log.txt",
        config={},
        started_at=None,
        finished_at=None,
    )

    assert training_service.training_log_csv_path(run) == Path(
        "/x/runs/train-abc/checkpoints/train_log.csv"
    )
