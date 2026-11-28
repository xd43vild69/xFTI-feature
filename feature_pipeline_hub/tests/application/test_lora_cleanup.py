from pathlib import Path

import pytest

from feature_pipeline.application import training_service
from feature_pipeline.infrastructure import training_repository as repo
from feature_pipeline.infrastructure.database import get_connection


@pytest.fixture
def conn(tmp_path):
    connection = get_connection(str(tmp_path / "test.db"))
    yield connection
    connection.close()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(root))
    return root


def _train_run(conn, dataset_run_id: str, output_dir: Path, pid: int = 0) -> str:
    return repo.create_training_run(
        conn,
        dataset_run_id=dataset_run_id,
        kind="train",
        pid=pid,
        log_path=str(output_dir.parent / "log.txt"),
        config={"output_dir": str(output_dir)},
    )


def _populate(output_dir: Path) -> None:
    (output_dir / "resume_checkpoint").mkdir(parents=True)
    (output_dir / "Krea2_LoRA_step_100.safetensors").write_bytes(b"a" * 10)
    (output_dir / "Krea2_FINAL.safetensors").write_bytes(b"b" * 20)
    (output_dir / "resume_checkpoint" / "adapter_model.safetensors").write_bytes(b"c" * 5)
    (output_dir / "train_log.csv").write_text("step\n1\n")


def test_plan_lists_only_safetensors_of_that_dataset(conn, runtime):
    mine = runtime / "runs" / "train-a" / "checkpoints"
    other = runtime / "runs" / "train-b" / "checkpoints"
    _populate(mine)
    _populate(other)
    _train_run(conn, "run-1", mine)
    _train_run(conn, "run-2", other)
    conn.execute("UPDATE training_runs SET status = 'completed'")
    conn.commit()

    plan = training_service.plan_lora_cleanup(conn, "run-1")

    assert len(plan.files) == 3
    assert all(mine in p.parents for p in plan.files)
    assert plan.total_bytes == 35


def test_delete_removes_weights_and_keeps_logs(conn, runtime):
    out = runtime / "runs" / "train-a" / "checkpoints"
    _populate(out)
    run_id = _train_run(conn, "run-1", out)
    repo.update_training_run_status(conn, run_id, "completed")

    deleted, freed, errors = training_service.delete_lora_files(
        training_service.plan_lora_cleanup(conn, "run-1")
    )

    assert (deleted, freed, errors) == (3, 35, [])
    assert not list(out.rglob("*.safetensors"))
    assert (out / "train_log.csv").exists()


def test_output_dir_outside_runs_folder_is_never_touched(conn, runtime, tmp_path):
    outside = tmp_path / "somewhere_else"
    _populate(outside)
    run_id = _train_run(conn, "run-1", outside)
    repo.update_training_run_status(conn, run_id, "completed")

    plan = training_service.plan_lora_cleanup(conn, "run-1")

    assert plan.files == ()
    assert (outside / "Krea2_FINAL.safetensors").exists()


def test_running_training_blocks_cleanup(conn, runtime):
    import os

    out = runtime / "runs" / "train-a" / "checkpoints"
    _populate(out)
    _train_run(conn, "run-1", out, pid=os.getpid())  # alive, status 'running'

    plan = training_service.plan_lora_cleanup(conn, "run-1")

    assert plan.blocked_reason
    assert training_service.delete_lora_files(plan)[0] == 0
    assert (out / "Krea2_FINAL.safetensors").exists()
