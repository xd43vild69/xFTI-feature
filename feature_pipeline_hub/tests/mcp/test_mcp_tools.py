import sys
import time
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_server import server
from feature_pipeline.application import training_service
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure.database import get_connection

SUCCESSFUL_PRECACHE = """
import json, os
settings = json.load(open(os.environ["PRECACHE_SETTINGS_PATH"]))
print("Pre-caching finished! 1 encoded, 0 reused, VRAM freed", flush=True)
"""

STUB_TRAIN = """
import json, os
settings = json.load(open(os.environ["TRAIN_SETTINGS_PATH"]))
print(f"training with total_steps={settings['total_steps']}", flush=True)
"""


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    monkeypatch.setenv("FTI_DB_PATH", str(tmp_path / "test.db"))
    return tmp_path / "test.db"


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


def _make_dataset_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "raw"
    folder.mkdir()
    Image.new("RGB", (512, 512), color="blue").save(folder / "a.png")
    (folder / "a.txt").write_text("a photo")
    return folder


def test_list_dataset_runs_reflects_an_import(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    runs = server.list_dataset_runs()

    assert [r["run_id"] for r in runs] == [imported["run_id"]]
    assert runs[0]["sample_count"] == 1
    assert runs[0]["concept_name"] == "my_concept"


def test_get_run_detail_includes_samples(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    detail = server.get_run_detail(imported["run_id"])

    assert len(detail["concept"]["samples"]) == 1
    assert "sks_concept" in detail["concept"]["samples"][0]["caption"]


def test_get_run_detail_raises_for_unknown_run():
    with pytest.raises(ValueError, match="No such dataset run"):
        server.get_run_detail("does-not-exist")


def test_get_dataset_health_reports_totals(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    server.import_dataset(str(folder), "my_concept", "sks_concept")

    health = server.get_dataset_health()

    assert len(health) == 1
    assert health[0]["total_samples"] == 1
    assert health[0]["active_samples"] == 1


def test_quality_summary_matches_stored_flags(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    summary = server.quality_summary(imported["run_id"])

    assert summary["total"] == 1
    assert summary["active"] == 1
    assert summary["invalid"] == 0


def test_export_dataset_writes_a_flat_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path / "runtime"))
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    result = server.export_dataset(imported["run_id"], "my_concept")

    assert result["exported_count"] == 1
    assert (tmp_path / "runtime" / "datasets" / "my_concept" / "a.png").is_file()


def test_revalidate_run_scoped_to_one_run(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    result = server.revalidate_run(imported["run_id"])

    assert result == {"run_id": imported["run_id"], "changed": 0}


def test_revalidate_run_sweeps_every_run_when_no_id_given(tmp_path):
    folder = _make_dataset_folder(tmp_path)
    server.import_dataset(str(folder), "concept_a", "sks_a")
    server.import_dataset(str(folder), "concept_b", "sks_b")

    result = server.revalidate_run()

    assert result == {"run_id": None, "changed": 0}


def test_start_lora_training_launches_precache_non_blocking(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE))
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")

    result = server.start_lora_training(imported["run_id"], total_steps=10)

    assert result["phase"] == "precache"
    assert result["status"] == "running"


def test_start_lora_training_refuses_when_a_job_is_already_active(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE))
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")
    server.start_lora_training(imported["run_id"])

    with pytest.raises(RuntimeError, match="already running"):
        server.start_lora_training(imported["run_id"])


def test_get_training_status_reports_precache_then_completes(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE))
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")
    started = server.start_lora_training(imported["run_id"])

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        run = training_repo.get_training_run(conn, started["training_run_id"])
        while training_service.precache_status(conn, started["training_run_id"]) == "running":
            time.sleep(0.05)
    finally:
        conn.close()

    status = server.get_training_status(started["training_run_id"])
    assert status == {
        "training_run_id": started["training_run_id"],
        "phase": "precache",
        "status": "completed",
    }


def test_continue_lora_training_requires_precache_completion(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(
        training_service,
        "PRECACHE_SCRIPT",
        _write_stub(tmp_path, "import time\ntime.sleep(30)\n"),
    )
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")
    started = server.start_lora_training(imported["run_id"])

    with pytest.raises(RuntimeError, match="not complete yet"):
        server.continue_lora_training(imported["run_id"], started["training_run_id"])

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        run = training_repo.get_training_run(conn, started["training_run_id"])
        from feature_pipeline.infrastructure import training_runner

        training_runner.stop_process(run.pid, grace_period_seconds=1)
    finally:
        conn.close()


def test_continue_lora_training_launches_the_train_phase(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(training_service, "PRECACHE_SCRIPT", _write_stub(tmp_path, SUCCESSFUL_PRECACHE))
    monkeypatch.setattr(training_service, "TRAIN_SCRIPT", _write_stub(tmp_path, STUB_TRAIN))
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")
    started = server.start_lora_training(imported["run_id"], total_steps=15)

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        while training_service.precache_status(conn, started["training_run_id"]) == "running":
            time.sleep(0.05)
    finally:
        conn.close()

    result = server.continue_lora_training(imported["run_id"], started["training_run_id"], total_steps=15)

    assert result["phase"] == "train"
    assert result["status"] == "running"

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        train_run = training_repo.get_training_run(conn, result["training_run_id"])
        assert train_run.config["total_steps"] == 15
        from feature_pipeline.infrastructure import training_runner

        training_runner.stop_process(train_run.pid, grace_period_seconds=1)
    finally:
        conn.close()


def test_stop_training_marks_the_run_stopped(tmp_path, fake_model_dir, monkeypatch):
    monkeypatch.setattr(
        training_service,
        "PRECACHE_SCRIPT",
        _write_stub(tmp_path, "import time\ntime.sleep(30)\n"),
    )
    folder = _make_dataset_folder(tmp_path)
    imported = server.import_dataset(str(folder), "my_concept", "sks_concept")
    started = server.start_lora_training(imported["run_id"])

    result = server.stop_training(started["training_run_id"])

    assert result == {"training_run_id": started["training_run_id"], "status": "stopped"}


def test_get_training_status_raises_for_unknown_run():
    with pytest.raises(ValueError, match="No such training run"):
        server.get_training_status("does-not-exist")
