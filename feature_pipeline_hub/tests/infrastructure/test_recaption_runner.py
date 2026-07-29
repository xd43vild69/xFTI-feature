import sys
from pathlib import Path

import pytest

from feature_pipeline.infrastructure import recaption_runner, training_runner
from feature_pipeline.infrastructure.recaption_runner import (
    RecaptionEnvironment,
    RecaptionUnavailable,
    resolve_environment,
    run_recaption,
)


def _fake_runtime(tmp_path: Path) -> Path:
    """A training_runtime/ shaped like scripts/setup_training_runtime.sh produces,
    without the tens of GB of real weights."""
    runtime_dir = tmp_path / "training_runtime"
    for name in ("transformer", "text_encoder", "vae"):
        (runtime_dir / "model" / name).mkdir(parents=True)
    (runtime_dir / "model" / "text_encoder" / "model.safetensors").write_bytes(b"not really weights")
    (runtime_dir / "venv" / "bin").mkdir(parents=True)
    (runtime_dir / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    return runtime_dir


def test_missing_training_runtime_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path / "nope"))

    with pytest.raises(RecaptionUnavailable, match="training runtime"):
        resolve_environment()


def test_missing_weights_are_reported(monkeypatch, tmp_path):
    runtime_dir = _fake_runtime(tmp_path)
    (runtime_dir / "model" / "text_encoder" / "model.safetensors").unlink()
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime_dir))

    with pytest.raises(RecaptionUnavailable, match="No Qwen3-VL weights"):
        resolve_environment()


def test_missing_interpreter_is_reported(monkeypatch, tmp_path):
    runtime_dir = _fake_runtime(tmp_path)
    (runtime_dir / "venv" / "bin" / "python").unlink()
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime_dir))

    with pytest.raises(RecaptionUnavailable, match="training runtime"):
        resolve_environment()


def test_a_complete_runtime_resolves(monkeypatch, tmp_path):
    runtime_dir = _fake_runtime(tmp_path)
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime_dir))

    env = resolve_environment()

    assert env.runtime_dir == runtime_dir
    assert env.python == runtime_dir / "venv" / "bin" / "python"
    assert env.model_dir == runtime_dir / "model" / "text_encoder"


def test_the_interpreter_can_be_overridden(monkeypatch, tmp_path):
    runtime_dir = _fake_runtime(tmp_path)
    other = tmp_path / "other-python"
    other.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv(training_runner.PYTHON_ENV, str(other))

    assert resolve_environment().python == other


def test_is_available_is_false_without_a_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path / "nope"))

    assert recaption_runner.is_available() is False


# --- streaming ---------------------------------------------------------------
# The worker is replaced with a stub interpreter script, so these exercise the
# real subprocess plumbing (stdin job, JSONL stdout, exit codes) without a model.

STUB_WORKER = """
import json, sys
job = json.load(sys.stdin)
print("a warning from some dependency", flush=True)
print(json.dumps({"event": "loaded", "device": "cpu", "seconds": 0.1}), flush=True)
for path in job["images"]:
    print(json.dumps({"event": "caption", "path": path,
                      "caption": f"caption for {path}", "seconds": 0.1}), flush=True)
print(json.dumps({"event": "done", "captioned": len(job["images"]), "failed": 0}), flush=True)
"""

CRASHING_WORKER = """
import sys
sys.stderr.write("Traceback: torch.OutOfMemoryError\\n")
sys.exit(1)
"""


@pytest.fixture
def stub_environment(tmp_path, monkeypatch):
    def _install(script: str) -> RecaptionEnvironment:
        worker = tmp_path / "worker.py"
        worker.write_text(script)
        monkeypatch.setattr(recaption_runner, "WORKER", worker)
        return RecaptionEnvironment(runtime_dir=tmp_path, python=Path(sys.executable))

    return _install


def test_events_stream_back_and_noise_is_dropped(stub_environment):
    env = stub_environment(STUB_WORKER)

    events = list(run_recaption(["/data/a.png", "/data/b.png"], detailed=False, environment=env))

    kinds = [e["event"] for e in events]
    assert kinds == ["loaded", "caption", "caption", "done"]
    assert events[1]["path"] == "/data/a.png"
    assert events[1]["caption"] == "caption for /data/a.png"


ECHO_WORKER = """
import json, sys
job = json.load(sys.stdin)
print(json.dumps({"event": "job", "images": job["images"],
                  "detailed": job["detailed"], "text_encoder_dir": job["text_encoder_dir"]}), flush=True)
"""


def test_the_job_reaches_the_worker_on_stdin(stub_environment):
    env = stub_environment(ECHO_WORKER)

    events = list(run_recaption(["/data/a.png"], detailed=True, environment=env))

    assert events[0]["images"] == ["/data/a.png"]
    assert events[0]["detailed"] is True
    assert events[0]["text_encoder_dir"] == str(env.model_dir)


def test_a_crashing_worker_yields_a_failed_event_with_stderr(stub_environment):
    env = stub_environment(CRASHING_WORKER)

    events = list(run_recaption(["/data/a.png"], detailed=False, environment=env))

    assert events[-1]["event"] == "failed"
    assert "OutOfMemoryError" in events[-1]["message"]


def test_the_real_worker_script_is_shipped():
    """resolve_environment() checks this path, so a rename must not go unnoticed."""
    assert recaption_runner.WORKER.is_file()
