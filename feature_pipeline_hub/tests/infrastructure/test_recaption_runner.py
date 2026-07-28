import sys
from pathlib import Path

import pytest

from feature_pipeline.infrastructure import recaption_runner
from feature_pipeline.infrastructure.recaption_runner import (
    RecaptionEnvironment,
    RecaptionUnavailable,
    resolve_environment,
    run_recaption,
)


def _fake_loralab(tmp_path: Path) -> Path:
    """A checkout shaped like AcademiaSD_LoRAlab-Krea2, without the 8 GB of weights."""
    root = tmp_path / "loralab"
    (root / "Krea-2-NF4" / "text_encoder").mkdir(parents=True)
    (root / "Krea-2-NF4" / "text_encoder" / "model.safetensors").write_bytes(b"not really weights")
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
    return root


def test_missing_env_var_explains_what_to_set(monkeypatch):
    monkeypatch.delenv(recaption_runner.LORALAB_ROOT_ENV, raising=False)

    with pytest.raises(RecaptionUnavailable, match=recaption_runner.LORALAB_ROOT_ENV):
        resolve_environment()


def test_a_nonexistent_root_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(recaption_runner.LORALAB_ROOT_ENV, str(tmp_path / "nope"))

    with pytest.raises(RecaptionUnavailable, match="missing folder"):
        resolve_environment()


def test_missing_weights_are_reported(monkeypatch, tmp_path):
    root = _fake_loralab(tmp_path)
    (root / "Krea-2-NF4" / "text_encoder" / "model.safetensors").unlink()
    monkeypatch.setenv(recaption_runner.LORALAB_ROOT_ENV, str(root))

    with pytest.raises(RecaptionUnavailable, match="No Qwen3-VL weights"):
        resolve_environment()


def test_missing_interpreter_is_reported(monkeypatch, tmp_path):
    root = _fake_loralab(tmp_path)
    (root / "venv" / "bin" / "python").unlink()
    monkeypatch.setenv(recaption_runner.LORALAB_ROOT_ENV, str(root))

    with pytest.raises(RecaptionUnavailable, match="No Python interpreter"):
        resolve_environment()


def test_a_complete_checkout_resolves(monkeypatch, tmp_path):
    root = _fake_loralab(tmp_path)
    monkeypatch.setenv(recaption_runner.LORALAB_ROOT_ENV, str(root))

    env = resolve_environment()

    assert env.loralab_root == root
    assert env.python == root / "venv" / "bin" / "python"
    assert env.model_dir.name == "text_encoder"


def test_the_interpreter_can_be_overridden(monkeypatch, tmp_path):
    root = _fake_loralab(tmp_path)
    other = tmp_path / "other-python"
    other.write_text("#!/bin/sh\n")
    monkeypatch.setenv(recaption_runner.LORALAB_ROOT_ENV, str(root))
    monkeypatch.setenv(recaption_runner.PYTHON_ENV, str(other))

    assert resolve_environment().python == other


def test_is_available_is_false_without_configuration(monkeypatch):
    monkeypatch.delenv(recaption_runner.LORALAB_ROOT_ENV, raising=False)

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
        return RecaptionEnvironment(loralab_root=tmp_path, python=Path(sys.executable))

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
                  "detailed": job["detailed"], "root": job["loralab_root"]}), flush=True)
"""


def test_the_job_reaches_the_worker_on_stdin(stub_environment):
    env = stub_environment(ECHO_WORKER)

    events = list(run_recaption(["/data/a.png"], detailed=True, environment=env))

    assert events[0]["images"] == ["/data/a.png"]
    assert events[0]["detailed"] is True
    assert events[0]["root"] == str(env.loralab_root)


def test_a_crashing_worker_yields_a_failed_event_with_stderr(stub_environment):
    env = stub_environment(CRASHING_WORKER)

    events = list(run_recaption(["/data/a.png"], detailed=False, environment=env))

    assert events[-1]["event"] == "failed"
    assert "OutOfMemoryError" in events[-1]["message"]


def test_the_real_worker_script_is_shipped():
    """resolve_environment() checks this path, so a rename must not go unnoticed."""
    assert recaption_runner.WORKER.is_file()
