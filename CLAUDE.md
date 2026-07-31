# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `feature_pipeline_hub/` (the actual project root; the repo root only holds the worktree/README wrapper):

```bash
cd feature_pipeline_hub

# Run the app
uv run streamlit run ui/app.py     # or: uv run python main.py

# Run the MCP server (stdio transport, for agent tool-calling — see mcp_server/server.py)
uv run python -m mcp_server

# Tests (pytest, no markers/config beyond testpaths=["tests"])
uv run pytest
uv run pytest tests/infrastructure/test_ingestion_repository.py
uv run pytest tests/infrastructure/test_ingestion_repository.py::test_saving_a_run_registers_its_concept -v
```

There is no configured linter/formatter/type-checker (no ruff/black/mypy in `pyproject.toml`) — don't invent lint commands.

One-time provisioning of the training runtime (large model copy + dedicated venv), only needed before Step 5 (Train) or AI recaptioning in Step 2 work:

```bash
FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2 ./scripts/setup_training_runtime.sh
```

## Architecture

Streamlit app for curating LoRA training datasets (Krea 2 / Qwen3-VL), structured as Clean Architecture layers under `src/feature_pipeline/`:

- **`domain/`** — Pydantic models (`models.py`: `DatasetSample`, `ConceptGroup`, `IngestionRun`, `DatasetManifest`) and pure validation (`validators.py`, `worker_contracts.py`, `cost.py`). No I/O.
- **`application/`** — business logic, one service module per pipeline step (`dataset_service`, `caption_service`, `image_service`, `quality_service`, `recaption_service`, `export_service`, `training_service`). Depends on domain + infrastructure interfaces, not on Streamlit.
- **`infrastructure/`** — SQLite persistence (`database.py`, `ingestion_repository.py`, `training_repository.py`, `version_repository.py`), filesystem (`storage.py`, `hf_exporter.py`), and subprocess launchers (`recaption_runner.py`, `training_runner.py`).
- **`ui/`** — Streamlit-only code: `app.py` wires top-level navigation, `state.py` is the sole bridge between UI session state and the SQLite-backed application layer (every DB read/write from the UI goes through `state.py`), `steps/` are thin page wrappers, `components/` hold the actual panel logic.
- **`workers/`** — standalone scripts run as **separate processes in a different Python environment** (`training_runtime/venv`, not the hub's own env): `precache_worker.py` and `train_worker.py` are ported byte-for-byte from the upstream LoRAlab project and must stay diffable against it — don't add instrumentation inside their bodies. `_telemetry.py` wraps their `__main__` entrypoints from the outside to emit structured JSON-lines lifecycle events without touching the vendored code. `recaption_worker.py` / `caption_qwen3vl.py` run Qwen3-VL captioning the same way.
- **`mcp_server/`** (hub root, not under `src/`) — a `FastMCP` server exposing the pipeline as tools for autonomous agents (LangGraph etc.), stdio transport, no auth (the app has none anywhere). `server.py` tools are thin wrappers over `application/`/`infrastructure/` — never reimplement business logic here. Runs as its own process against the same SQLite DB, same per-operation-connection convention as `ui/state.py`.

### Two process boundaries, two purposes

- **`recaption_runner.py`**: short-lived, streaming. Launches the recaption worker with `subprocess.Popen` + pipes, yields parsed JSON-line events as they arrive, blocks until exit. Used for interactive per-batch captioning (seconds to ~1-2s/image).
- **`training_runner.py`**: long-lived, detached. Launches pre-cache/train with `start_new_session=True`, output redirected to a log file, returns `(pid, log_path)` immediately without waiting. Designed to survive Streamlit restarting or the browser closing. Progress is recovered later by tailing the log file and reading a `training_run_id` row from SQLite — never by holding anything in memory. `read_lifecycle_event()` reads only the last ~4KB of the log to find the final `worker_finished`/`worker_failed` JSON line cheaply, even for multi-hour logs.

Pre-cache runs blocking (`training_service._run_precache_blocking`, minutes, has a timeout) before training is launched detached (`_launch_train`) — see `training_service.start_training`, used by the UI. The MCP server can't block a tool call for that long, so it uses the non-blocking split instead: `launch_precache` (fire-and-forget) + `precache_status` (poll) + `launch_train` (the public wrapper around `_launch_train`) — same underlying subprocess launch, just not chained together in one call.

### Runs, concepts, and versions (SQLite: `feature_pipeline.db`)

- A `concept` is a named dataset (concept_name + trigger_word). An `ingestion_run` is one import of that concept — re-scanning the same concept creates a **new** run rather than overwriting the old one, so runs stay independently selectable in the UI. `run_id`, not `concept_id`, is what the UI selects on.
- `samples` belong to a run and carry validation state, perceptual hashes (phash/dhash/colorhash), sharpness, and duplicate/exclude/flag flags.
- `dataset_versions` are export snapshots (materialized flat training folders) with a `manifest_json` used to diff "did this export actually change anything" (`dataset_service.compute_content_hash` hashes sorted (phash, caption) pairs of non-excluded samples).
- `training_runs` track every launched subprocess (precache/train/progressive/curate-scoring): pid, log path, status, and telemetry (duration, GPU-seconds, cost estimate) backfilled by `training_service.finalize_dead_run` once the process exits.
- Schema lives entirely in `database.py`; new columns are added via the `*_COLUMN_MIGRATIONS` dicts (`ALTER TABLE ... ADD COLUMN`), never by editing `CREATE TABLE IF NOT EXISTS` in place, since that leaves pre-existing DBs unchanged.
- Per-step telemetry (`ingestion_runs.import_duration_seconds` etc., read by the observability panel) is written by `ui/step_telemetry.py record_step()`. It does an `UPDATE ... WHERE run_id = ?`, so the corresponding `ingestion_runs` row **must already exist** (i.e. `state.save_run()` must have run first) or the update silently matches zero rows.

### SQLite connection pattern

Connections are **opened per operation, never cached**, because Streamlit reruns can land on different threads and a `sqlite3.Connection` isn't safe to share across them. `ui/state.py` wraps every call in its own `_db()` context manager; the same pattern (`get_connection()` → use → `close()`) is used anywhere else in the codebase that touches the DB directly (e.g. `step_telemetry.py`).

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `FTI_TRAINING_RUNTIME_DIR` | Model weights, dataset, cache, and venv shared by recaptioning + training | `feature_pipeline_hub/training_runtime` |
| `FTI_TRAINING_PYTHON` | Interpreter for training/recaption workers | `<FTI_TRAINING_RUNTIME_DIR>/venv/bin/python` |
| `FTI_DB_PATH` | SQLite metadata DB | `feature_pipeline_hub/data/feature_pipeline.db` |
| `FTI_DATA_DIR` | Base dir for raw dataset uploads | `feature_pipeline_hub/data` |
| `FTI_RUN_ID` | Set by the launcher (not the user) so a worker's telemetry events carry its own run id | — |

### UI conventions worth knowing before touching `ui/`

- The 5 pipeline steps (Import → Curate → Quality → Export → Train) plus a Metrics page are `st.Page`s registered in `ui/app.py`; `state.py` exposes their paths as constants (`IMPORT_STEP`, etc.) rather than hardcoding path strings elsewhere.
- `state.require_active_run()` is the shared "no dataset selected" guard used by steps 2-4.
- Caption editor widgets are **versioned** (`caption_widget_key` / `CAPTION_VERSIONS_KEY`): a keyed Streamlit widget ignores new `value=` on rerun, so any code path that edits a caption from outside its own widget (batch replace, AI recaption, quality panel quick-edit) must bump that sample's version counter or the UI will show stale text.
- Training's live monitoring dashboard uses `@st.fragment(run_every="5s")` to auto-refresh without rerunning the whole page.
