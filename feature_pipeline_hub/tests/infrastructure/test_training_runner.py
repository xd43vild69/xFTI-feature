import json
import os
import sys
import time
from pathlib import Path

import pytest

from feature_pipeline.infrastructure import training_runner
from feature_pipeline.infrastructure.training_runner import (
    TrainingEnvironment,
    TrainingUnavailable,
    is_process_alive,
    launch,
    read_lifecycle_event,
    read_log_tail,
    resolve_environment,
    stop_process,
)


def _fake_runtime(tmp_path: Path) -> Path:
    """A training_runtime/ shaped like scripts/setup_training_runtime.sh would produce."""
    runtime = tmp_path / "runtime"
    model = runtime / "model"
    for part in ("transformer", "text_encoder", "vae"):
        (model / part).mkdir(parents=True)
    (runtime / "venv" / "bin").mkdir(parents=True)
    (runtime / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    return runtime


# --- resolve_environment -----------------------------------------------------


def test_missing_interpreter_is_reported(monkeypatch, tmp_path):
    runtime = _fake_runtime(tmp_path)
    (runtime / "venv" / "bin" / "python").unlink()
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime))

    with pytest.raises(TrainingUnavailable, match="No training interpreter"):
        resolve_environment()


def test_incomplete_model_copy_is_reported(monkeypatch, tmp_path):
    runtime = _fake_runtime(tmp_path)
    import shutil

    shutil.rmtree(runtime / "model" / "vae")
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime))

    with pytest.raises(TrainingUnavailable, match="vae"):
        resolve_environment()


def test_a_complete_runtime_resolves(monkeypatch, tmp_path):
    runtime = _fake_runtime(tmp_path)
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime))

    env = resolve_environment()

    assert env.runtime_dir == runtime
    assert env.python == runtime / "venv" / "bin" / "python"


def test_is_available_is_false_without_a_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path / "nope"))

    assert training_runner.is_available() is False


# --- launch / log tail / alive / stop ----------------------------------------
# Real subprocess plumbing, driven with a stub script instead of the real model.

STUB_SCRIPT = """
import json, os, signal, sys, time

settings = json.load(open(os.environ["FTI_TEST_SETTINGS_PATH"]))
print(f"loaded total_steps={settings['total_steps']}", flush=True)

stopping = {"flag": False}
def handle_sigint(signum, frame):
    stopping["flag"] = True
signal.signal(signal.SIGINT, handle_sigint)

for step in range(100):
    if stopping["flag"]:
        print("stopped gracefully", flush=True)
        sys.exit(0)
    print(f"step {step}", flush=True)
    time.sleep(0.05)
"""


@pytest.fixture
def stub_environment(tmp_path):
    script_dir = tmp_path / "script_dir"
    script_dir.mkdir()
    script = script_dir / "worker.py"
    script.write_text(STUB_SCRIPT)
    return TrainingEnvironment(runtime_dir=tmp_path, python=Path(sys.executable)), script


def test_launch_writes_settings_and_returns_immediately(stub_environment, tmp_path):
    env, script = stub_environment
    run_dir = tmp_path / "runs" / "r1"

    started = time.monotonic()
    pid, log_path = launch(script, {"total_steps": 42}, run_dir, "FTI_TEST_SETTINGS_PATH", environment=env)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0  # detached: must not block on the subprocess
    assert json.loads((run_dir / "settings.json").read_text()) == {"total_steps": 42}
    assert is_process_alive(pid)

    stop_process(pid, grace_period_seconds=2)


def test_log_tail_reads_only_new_content(stub_environment, tmp_path):
    env, script = stub_environment
    run_dir = tmp_path / "runs" / "r1"
    pid, log_path = launch(script, {"total_steps": 5}, run_dir, "FTI_TEST_SETTINGS_PATH", environment=env)

    try:
        deadline = time.monotonic() + 5
        first_text = ""
        while time.monotonic() < deadline and not first_text:
            first_text, offset = read_log_tail(log_path)
            time.sleep(0.05)
        assert "loaded total_steps=5" in first_text

        time.sleep(0.2)
        second_text, offset2 = read_log_tail(log_path, since_offset=offset)
        assert second_text != ""
        assert offset2 > offset
        assert "loaded total_steps=5" not in second_text  # already consumed
    finally:
        stop_process(pid, grace_period_seconds=2)


def test_stop_process_sends_sigint_and_the_worker_exits_gracefully(stub_environment, tmp_path):
    env, script = stub_environment
    run_dir = tmp_path / "runs" / "r1"
    pid, log_path = launch(script, {"total_steps": 5}, run_dir, "FTI_TEST_SETTINGS_PATH", environment=env)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and "step 0" not in read_log_tail(log_path)[0]:
        time.sleep(0.05)

    stop_process(pid, grace_period_seconds=3)

    assert not is_process_alive(pid)
    text, _ = read_log_tail(log_path)
    assert "stopped gracefully" in text


def test_is_process_alive_is_false_for_a_nonexistent_pid():
    assert is_process_alive(2**30) is False


def test_stop_process_is_a_no_op_for_an_already_dead_pid():
    stop_process(2**30, grace_period_seconds=0.1)  # must not raise


# --- read_lifecycle_event ------------------------------------------------------
# workers/_telemetry.py prints exactly one of these lines as a worker's last
# output; this reads it back out of an otherwise free-text log.


def test_missing_log_has_no_lifecycle_event(tmp_path):
    assert read_lifecycle_event(str(tmp_path / "nope.txt")) is None


def test_a_log_with_only_free_text_has_no_lifecycle_event(tmp_path):
    log_path = tmp_path / "log.txt"
    log_path.write_text("Model ID: foo\nProcessing: bar\n")

    assert read_lifecycle_event(str(log_path)) is None


def test_finds_the_worker_finished_line_among_free_text(tmp_path):
    log_path = tmp_path / "log.txt"
    log_path.write_text(
        '{"event": "worker_started", "worker": "precache", "timestamp": 1.0}\n'
        "Processing: a.png\nProcessing: b.png\n"
        '{"event": "worker_finished", "worker": "precache", "run_id": "r1", '
        '"timestamp": 2.0, "duration_seconds": 12.5, "gpu_seconds": 12.5}\n'
    )

    event = read_lifecycle_event(str(log_path))

    assert event == {
        "event": "worker_finished",
        "worker": "precache",
        "run_id": "r1",
        "timestamp": 2.0,
        "duration_seconds": 12.5,
        "gpu_seconds": 12.5,
    }


def test_finds_worker_failed_over_a_large_tail(tmp_path):
    log_path = tmp_path / "log.txt"
    filler = "x" * 200 + "\n"
    log_path.write_text(filler * 100 + '{"event": "worker_failed", "worker": "train", "error": "boom"}\n')

    event = read_lifecycle_event(str(log_path))

    assert event["event"] == "worker_failed"
    assert event["error"] == "boom"


def test_ignores_a_log_too_large_to_contain_the_event_in_the_tail(tmp_path):
    log_path = tmp_path / "log.txt"
    filler = "x" * 200 + "\n"
    log_path.write_text('{"event": "worker_finished", "worker": "train"}\n' + filler * 100)

    assert read_lifecycle_event(str(log_path)) is None
