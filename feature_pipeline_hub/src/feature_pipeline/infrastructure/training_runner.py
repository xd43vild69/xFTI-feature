"""Process boundary for the self-contained training runtime.

Training (and pre-cache, curation scoring, progressive phases) run in
`training_runtime/venv/`, a dedicated interpreter with its own copy of the model
and heavy dependencies (torch, diffusers, insightface, ...) — see
`scripts/setup_training_runtime.sh`. Unlike `recaption_runner.py`, these jobs run
for minutes to hours, so this module launches them **detached** (their own
process group, output redirected to a log file) and returns immediately instead
of blocking on a streaming pipe. Progress is read back later — from anywhere,
any rerun, any browser session — by tailing that log file and the process's own
`training_run_id` row in SQLite.
"""

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from feature_pipeline.infrastructure.storage import training_runtime_dir

PYTHON_ENV = "FTI_TRAINING_PYTHON"


@dataclass(frozen=True)
class TrainingEnvironment:
    """Everything needed to launch a training-runtime worker, checked to exist."""

    runtime_dir: Path
    python: Path

    @property
    def model_dir(self) -> Path:
        return self.runtime_dir / "model"


class TrainingUnavailable(RuntimeError):
    """The training runtime hasn't been set up yet."""


def resolve_environment() -> TrainingEnvironment:
    """Locate the training runtime's own interpreter and model copy, or explain what's missing."""
    runtime_dir = training_runtime_dir()

    python_value = os.environ.get(PYTHON_ENV, "").strip()
    python = (
        Path(python_value).expanduser() if python_value else runtime_dir / "venv" / "bin" / "python"
    )
    if not python.is_file():
        raise TrainingUnavailable(
            f"No training interpreter at {python}. Run scripts/setup_training_runtime.sh "
            f"first, or set {PYTHON_ENV} if it lives elsewhere."
        )

    model_dir = runtime_dir / "model"
    required = ["transformer", "text_encoder", "vae"]
    missing = [name for name in required if not (model_dir / name).is_dir()]
    if missing:
        raise TrainingUnavailable(
            f"Incomplete model copy at {model_dir} (missing {', '.join(missing)}). "
            "Run scripts/setup_training_runtime.sh first."
        )

    return TrainingEnvironment(runtime_dir=runtime_dir, python=python)


def is_available() -> bool:
    try:
        resolve_environment()
    except TrainingUnavailable:
        return False
    return True


def launch(
    script: Path,
    settings: dict,
    run_dir: Path,
    settings_path_env: str,
    environment: TrainingEnvironment | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Write `settings` to `run_dir/settings.json` and launch `script` against it, detached.

    `settings_path_env` is the env var name the *ported script itself* reads its
    config path from — e.g. "PRECACHE_SETTINGS_PATH" or "TRAIN_SETTINGS_PATH",
    LoRAlab's own names, left completely untouched in workers/*.py so the ported
    scripts stay byte-for-byte identical to their originals apart from
    PROJECT_ROOT's directory depth.

    Returns (pid, log_path) immediately — the caller persists these (see
    training_repository.create_training_run) rather than waiting on the process.
    """
    env = environment or resolve_environment()
    run_dir.mkdir(parents=True, exist_ok=True)

    settings_path = run_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2))

    log_path = run_dir / "log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    process_env = {
        **os.environ,
        settings_path_env: str(settings_path),
        **(extra_env or {}),
    }

    process = subprocess.Popen(
        [str(env.python), "-u", str(script)],
        cwd=str(script.parent),
        env=process_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # own process group: survives the parent (Streamlit) exiting
    )
    log_file.close()

    return process.pid, str(log_path)


def read_log_tail(log_path: str, since_offset: int = 0) -> tuple[str, int]:
    """New text appended to the log since `since_offset`, and the offset to pass next time.

    Reading only the new bytes keeps this cheap to call on every UI refresh even
    once a log has grown to hundreds of thousands of lines over a multi-hour run.
    """
    path = Path(log_path)
    if not path.is_file():
        return "", since_offset

    with path.open("rb") as f:
        f.seek(since_offset)
        chunk = f.read()
        new_offset = f.tell()

    return chunk.decode("utf-8", errors="replace"), new_offset


def is_process_alive(pid: int) -> bool:
    """Whether `pid` is still doing work — false for gone *and* for zombies.

    A detached child that has exited but was never wait()-ed on stays visible to
    os.kill(pid, 0) as a zombie until reaped, which would otherwise make the
    GPU lock (state.is_training_active()) think a finished run is still running
    forever. Opportunistically reaps it if we happen to be its parent; either
    way, a zombie counts as not-alive.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else — can't happen for our own children, but
        # treat it as alive rather than silently declaring a live job dead.
        return True

    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return True  # not on Linux, or the process just vanished — assume alive

    if state != "Z":
        return True

    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child (e.g. app restarted since launch) — can't reap it
    return False


def stop_process(pid: int, grace_period_seconds: float = 10.0) -> None:
    """Ask a training process to stop gracefully, escalating if it doesn't.

    SIGINT first — the training loop's own KeyboardInterrupt handling is what
    writes a resumable checkpoint before exiting. SIGTERM only if it ignores that.
    """
    if not is_process_alive(pid):
        return

    os.kill(pid, signal.SIGINT)

    deadline = time.monotonic() + grace_period_seconds
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return
        time.sleep(0.2)

    if is_process_alive(pid):
        os.kill(pid, signal.SIGTERM)
