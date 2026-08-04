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

# Type-check (mypy, strict, Pydantic-aware — [tool.mypy] in pyproject.toml)
uv run mypy
```

There is no configured linter/formatter (no ruff/black in `pyproject.toml`) — don't invent lint commands. `mypy` **is** configured (`[tool.mypy]`, with the `pydantic.mypy` plugin) and scoped to `src/feature_pipeline` and `mcp_server` only — not `ui/` (Streamlit typing is noisy) and and the torch-free modules under `workers/krea2/` (listed individually in `files`), but not the rest of `workers/`. CI (`.github/workflows/python-app.yml`) runs both `mypy` and `pytest` on every push/PR to `main`; a red mypy run blocks the same as a red test run.

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
- **`workers/`** — standalone scripts run as **separate processes in a different Python environment** (`training_runtime/venv`, not the hub's own env): `precache_worker.py` is byte-for-byte upstream and must stay that way — don't add instrumentation inside its body. `train_worker.py` has diverged deliberately: it is being split into the `krea2/` package (see below), so it no longer diffs against upstream at all. `_telemetry.py` wraps their `__main__` entrypoints from the outside to emit structured JSON-lines lifecycle events without touching the vendored code. `recaption_worker.py` / `caption_qwen3vl.py` run Qwen3-VL captioning the same way.
- **`workers/krea2/`** — the trainer, split out of `train_worker.py` (2264 → 935 lines). `train_worker.py` stays put as the entrypoint: `training_service.py` launches it by path, and running it as a *script* is what puts `workers/` on `sys.path` so `krea2` and `_telemetry` resolve — don't turn it into a module or move it.

  The package splits by **whether a module imports torch**, not by theme. Torch-free (`config`, `cache_index`, `curation`, `sampling`, `schedule`, `metrics`, `checkpoints`) are listed individually in mypy's `files` and tested by `tests/workers/`; the rest (`dataset`, `math_ops`, `quantization`, `lora_io`, `ema`, `preview`, `state`) need the training runtime. Keep that line — it is the only reason any of this is reachable from CI.

  `config.load_config()` returns a **frozen** `TrainConfig`; pass it explicitly rather than reintroducing module globals. The original reassigned 21 of them after first definition, so any `from ... import COMPACT_TEXT` captured whichever value existed at import time with nothing failing loudly — the dataclass makes that unrepresentable. `train_worker.py` still keeps read-only aliases onto it for the loop's benefit.

### Verifying a change to the trainer

Nothing under `workers/` runs in CI beyond the torch-free modules, so these are run by hand — all three, after any change to `krea2/` or the loop:

```bash
training_runtime/venv/bin/python tests/workers/golden/capture_config.py --check
```
```bash
training_runtime/venv/bin/python tests/workers/golden/capture_behavior.py --check
```
```bash
./tests/workers/golden/smoke_run.sh check
```

The first two compare ~100 recorded values (config resolution, tensor math, samplers, EMA, curation, checkpointing) against baselines captured before the refactor. The third runs 12 real training steps on the GPU and compares `train_log.csv` — it is the only thing covering the loop itself. Drop `--check` / use `record` to re-baseline, and only when a behavior change is intended and understood.

The smoke run compares by tolerance, not diff: the loop is **not bit-reproducible against itself** (`cudnn.benchmark` picks kernels by timing, TF32 is on, reductions are unordered). Measured drift on unchanged code is ~3e-4 on loss and ~2% on grad_norm. See `tests/workers/golden/compare_smoke.py`.
- **`mcp_server/`** (hub root, not under `src/`) — a `FastMCP` server exposing the pipeline as tools for autonomous agents (LangGraph etc.), stdio transport, no auth (the app has none anywhere). `server.py` tools are thin wrappers over `application/`/`infrastructure/` — never reimplement business logic here. Runs as its own process against the same SQLite DB, same per-operation-connection convention as `ui/state.py`.

### Two process boundaries, two purposes

- **`recaption_runner.py`**: short-lived, streaming. Launches the recaption worker with `subprocess.Popen` + pipes, yields parsed JSON-line events as they arrive, blocks until exit. Used for interactive per-batch captioning (seconds to ~1-2s/image).
- **`training_runner.py`**: long-lived, detached. Launches pre-cache/train with `start_new_session=True`, output redirected to a log file, returns `(pid, log_path)` immediately without waiting. Designed to survive Streamlit restarting or the browser closing. Progress is recovered later by tailing the log file and reading a `training_run_id` row from SQLite — never by holding anything in memory. `read_lifecycle_event()` reads only the last ~4KB of the log to find the final `worker_finished`/`worker_failed` JSON line cheaply, even for multi-hour logs.

Pre-cache runs blocking (`training_service._run_precache_blocking`, minutes, has a timeout) before training is launched detached (`_launch_train`) — see `training_service.start_training`, used by the UI. The MCP server can't block a tool call for that long, so it uses the non-blocking split instead: `launch_precache` (fire-and-forget) + `precache_status` (poll) + `launch_train` (the public wrapper around `_launch_train`) — same underlying subprocess launch, just not chained together in one call.

### Structured captioning (AI recaption)

Recaptioning does not ask the VLM for prose. It asks for a **JSON object of fixed slots** and assembles the caption from them in Python, because of how a LoRA divides labour: what the caption *describes* stays conditioned on that text and steerable at inference, and what it *omits* is absorbed into the trigger word. So `subject` mode describes shot/pose/clothing/background/lighting and says nothing about face, hair or build; `location` mode inverts it — transient things (hour, weather, light, passers-by) in, architecture and style out. Asked for free prose, a VLM always ends up narrating the face; the schema plus an explicit prohibition is what holds the line. Technique ported from xLoralizerPro's `backend/normalizer.py`.

The schemas, prompts, parsing and assembly all live in `domain/caption_schema.py`, deliberately: nothing under `workers/` runs in CI, so `recaption_worker.py` is only a relay — the hub sends the resolved instruction down (the prompt travels as text because `feature_pipeline` is not importable from the training runtime's interpreter) and gets the model's raw reply back untouched. `recaption_service` parses it. **Parse before `inject_trigger_word`**: that calls `normalize_caption`, whose whitelist eats `{`, `}`, `:` and `"`.

Qwen3-VL runs through `transformers.generate`, which has no grammar-constrained decoding (Ollama's `format: json`), so a malformed reply is expected occasionally — `parse_slots` raises `CaptionSchemaError` and the service reports that one image as an error, leaving its existing caption alone, rather than salvaging a half-read object. `caption_qwen3vl.py` stays a verbatim port of LoRAlab's apart from two **additive** kwargs on `generate_caption` (`instruction`, `deterministic`); omit both and its behavior is byte-for-byte upstream's.

### Resuming a training run

The trainer already knows how to resume: `krea2.state.CheckpointManager` restores adapter, optimizer, EMA, RNG and sampler position from `output_dir`, and its signal handlers checkpoint on SIGINT/SIGTERM/SIGHUP — so the UI's "Stop training" (which `training_runner.stop_process` sends as SIGINT) leaves a resumable checkpoint behind.

What makes a launch a resume is therefore **only** which `output_dir` it gets. `_launch_train` mints a fresh one per run, so a normal launch always starts at step 0; `resume_training` reuses the original run's `output_dir` while allocating a new run_dir for its own `settings.json`/`log.txt`, so the earlier log isn't truncated and each launch still gets one `training_runs` row. `krea2.metrics` opens the CSVs in append mode for exactly this case, so `train_log.csv` stays continuous across restarts — which is why `training_log_csv_path` reads `output_dir` out of the stored settings instead of deriving it from the log's location.

### What the trainer writes, and where the time comes from

Three CSVs land in `output_dir`. `train_log.csv` and `val_log.csv` are per optimizer update; `checkpoint_log.csv` is one row per `.safetensors` actually written (`krea2.metrics.CheckpointLog`, called from `CheckpointManager.save()` *after* the write succeeded, so a save that no-opped on `step <= 0` or re-entrancy never invents an interval).

The third exists because neither of the first two carries wall-clock time: `train_log.csv`'s `secs` column times a single micro-step and is sampled only on update steps, so summing it undercounts by roughly `grad_accum_steps`. "How long did this run take to reach step N" is only answerable from `checkpoint_log.csv`.

Its clock is **per launch** — `_last_mark` starts when `CheckpointLog` is constructed, so the first span of a resumed run is timed from the moment the process came back up, never from the previous launch's last save. `steps_delta` goes non-positive at the resume seam, and `domain/checkpoint_log.py` drops those spans from its medians while keeping them in the totals. The medians also exclude `interrupt` and `final` spans: both are partial by construction, and averaging them in would report the typical checkpoint as cheaper than any checkpoint ever was.

The `launch_id` column is what separates the two processes' spans in a file a resume appended to. It comes from `FTI_RUN_ID` — the same value `workers/_telemetry.py` stamps on the lifecycle events in `log.txt`, so the two can be cross-referenced — falling back to a local uuid outside the hub. It deliberately does **not** reuse `cfg.run_id`: `checkpoints.belongs_to_run` (called from `krea2/state.py`) discards a checkpoint whose `run_id.txt` does not match, and `resume_training` mints a fresh uuid per relaunch, so writing that field would make every resume silently restart at step 0.

`train_worker.save_checkpoint_now` is the single funnel for every save (cadence, signals, OOM, non-finite gradient, final) and is what tags each row's `reason` — instrument there, not at the individual call sites. The hub reads it through `training_service.read_checkpoint_log_summary` and persists the summary to `training_runs.checkpoint_metrics_json`, mirroring `metrics_json` in every respect; the Metrics page's **Checkpoints** tab renders it.

These figures are per *training*, never aggregated across a dataset: every `start_training` mints its own `output_dir`, so each launch writes its own `checkpoint_log.csv` and gets its own `training_runs` row. `observability_panel._select_lineage` is what lets an operator reach the older ones — without it the page only ever showed `lineages[0]`.

Runs that predate the log are recovered rather than written off. `infrastructure/checkpoint_files.py` reads each `.safetensors`' mtime plus the `step`/`epoch`/`ss_num_train_images` in its header — parsing the 8-byte length + JSON prologue by hand, so the hub needs neither torch nor the safetensors package — and `checkpoint_log.reconstruct` turns those into spans. It is the last resort behind the live CSV and the stored blob, and the summary carries `is_reconstructed` so the UI can say so.

Two things make the reconstruction honest rather than approximate. Passing the launches' `started_at` anchors each launch's first span to when that process started, so a resumed run does not bill the hours it lay dead (12 minutes, on the run this was built against). And `{prefix}_FINAL.safetensors` is merged with the `_step_N` file at the same step — they are the same weights written seconds apart, and two rows would invent a checkpoint that took no time. What it cannot fix: the first span of every launch also contains the model and cache load, worth about 40s against 731s spans.

Adding a column to `train_log.csv` would break `tests/workers/golden/compare_smoke.py`, which diffs it against a recorded baseline — that is why this is a separate file rather than two more columns.

A resume carries every hyperparameter over untouched except `total_steps`: the saved adapter was shaped by the original rank/alpha and `lora_io.load_lora_weights` exits rather than load a mismatch. `total_steps` must be raised above the checkpoint's step or the trainer reports "nothing to do" — `resume_training` rejects that up front. `find_resume_points` decides what is offerable using `checkpoint_step`, which mirrors `CheckpointManager.has_checkpoint()`; keep the two in agreement.

The per-step `Krea2_LoRA_step_N.safetensors` files are **not** resume points — `lora_io.export_lora` rewrites keys to the `transformer.*` layout inference loaders want, which `set_peft_model_state_dict` won't take back. Only `resume_checkpoint/` (written by `save_pretrained`, PEFT layout) plus `optimizer.pt` and `current_step.txt` can be resumed from.

### Runs, concepts, and versions (SQLite: `feature_pipeline.db`)

- A `concept` is a named dataset (concept_name + trigger_word). An `ingestion_run` is one import of that concept — re-scanning the same concept creates a **new** run rather than overwriting the old one, so runs stay independently selectable in the UI. `run_id`, not `concept_id`, is what the UI selects on.
- A run's `source_kind` is `folder` (points at a path the user owns — `delete_managed_folder` refuses to clean it up), `upload` or `clone` (both own a folder under `data/raw/<run_id>/`). `dataset_service.clone_ingestion_run` copies an existing run's bytes into a new folder and takes captions from the DB rather than the `.txt` sidecars, which only get rewritten on export — the clone is fully independent, so it drops the source's duplicate/excluded/flagged verdicts and re-runs validation, but copies `metrics` verbatim since the files are byte-identical.
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
