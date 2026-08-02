# xFTI-feature (Feature Pipeline Hub)

**Feature Pipeline Hub** is an end-to-end **LoRA (Low-Rank Adaptation) model training dataset curation and training execution** pipeline, usable two ways: interactively, through a Streamlit application with top-level page navigation, and programmatically, through an [MCP server](#-mcp-server-agent-integration) that exposes the same pipeline as tools for autonomous agents. It provides a unified pipeline for importing raw images and captions, visually curating and AI-recaptioning samples, performing perceptual deduplication and quality analytics, exporting versioned clean datasets, and launching/monitoring LoRA model training — with cross-dataset health and cost observability throughout.

---

## 🌟 Key Features

The application is organized into **five sequential pipeline steps** plus a standalone **Metrics** page, all accessible via top-level page navigation (Metrics first, then 1 · Import → 5 · Train):

### 📊 Metrics: Cross-Dataset Health & Pipeline Observability
* **Fleet Inventory**: A sortable table across *every* dataset — imported date, image/active/excluded/duplicate counts, missing-description count, invalid count, and median sharpness — plus headline totals (datasets, total images, active images, datasets with issues). Duplicate and sharpness figures are read from each dataset's last stored Quality-step verdict rather than recomputed live.
* **Per-Run Pipeline Telemetry**: Duration, error count, and throughput for each step (Import, Recaption, Quality, Export) of the active run, joined with its matching Training run (status, duration, GPU-seconds).
* **Cost Tiles**: Ingestion cost, training cost, and total cost estimates (`st.metric` tiles) plus a GPU-hours caption, computed from GPU-seconds emitted by the training/pre-cache workers and an hourly rate you configure (see `FTI_GPU_HOURLY_RATE` below).
* Not gated behind an active run — the fleet inventory half is visible with no dataset selected.

### 1. 📥 Step 1: Import & Ingestion
* **Dual Ingestion Sources**:
  * **Local Directory Scan**: Ingest images directly from any local folder path. Files stay in place — nothing is copied or renamed.
  * **Browser File Uploads**: Drag-and-drop images (`.png`, `.jpg`, `.jpeg`, `.webp`) and optional `.txt` caption sidecar files via the UI. Uploaded datasets are persistently cached in `data/raw/<run_id>/`.
* **Concept & Trigger Word Assignment**: Define a dataset **Concept Name** (e.g., `Cyberpunk Style`) and **Trigger Word**. Trigger Word auto-fills as a slugified version of Concept Name (`cyberpunk_style`) as you type, and stops following it the moment you edit it by hand. Trigger words are automatically prefixed to captions during ingestion if not already present.
* **Standardized Image Naming**: Uploaded images (Step 1's "Upload files" and Curate's "Add images", below) are renamed to `<concept_slug>_0001.ext`, `<concept_slug>_0002.ext`, … regardless of their original filename, so a dataset's images are traceable to a single, predictable naming scheme. Folder-scanned imports keep their original filenames, since that source is never copied into the app's own storage.
* **Automated Validation**: Enforces a minimum image resolution (both sides ≥ 512×512px, **or** either side alone ≥ 1024px — a tall or wide crop can carry enough detail on one axis even if the other falls short), validates file extensions, and checks caption character lengths (3 to 500 characters). A validation rule change doesn't retroactively touch datasets imported under the old rule until "Revalidate" (below) is run.
* **Persistent Ingestion Runs**: Stores dataset runs in a local SQLite database (`feature_pipeline.db`), allowing runs to survive app restarts and remain selectable via the global context bar. Re-scanning the same concept creates a new, independently-selectable run rather than overwriting the previous one.

### 2. 🎨 Step 2: Curate & AI Recaptioning
* **Interactive Curation Grid**:
  * Displays square, letterboxed, theme-adaptive thumbnails with an adjustable column layout slider (2 to 6 columns) — thumbnail resolution adapts to column count.
  * Click any thumbnail to open it full-size in a modal, with a "true size" toggle for viewing unscaled pixels with scroll — useful for judging focus and compression artifacts without the resampling a scaled preview would introduce.
  * Filter grid views: `Active`, `All`, `Duplicates`, `Invalid`, `No caption`, and `Excluded`.
* **In-Place Caption Editing**: Edit text captions directly underneath each thumbnail with instant SQLite database persistence.
* **Batch Caption Word Replacement**:
  * Replace specific terms across all captions in the active dataset.
  * Accessible via the top toolbar or the **`F2` keyboard shortcut**.
  * Displays exact match counts and matching sample previews before executing the replacement.
* **Add Images to an Existing Dataset**: Upload images you forgot the first time without starting over — existing exclusions, edited captions, and duplicate flags on the rest of the dataset are left untouched. New images are compared against the dataset already in place and flagged as duplicates if a near-match already exists, and continue the same `<concept_slug>_NNNN` naming sequence Import uses.
* **Sample Inclusion & Exclusion**: Exclude specific images from the final dataset without deleting files from disk.
* **AI Recaptioning Engine (Qwen3-VL-4B)**:
  * Regenerates captions using the **Qwen3-VL-4B** vision-language model (the text encoder model used by Krea 2).
  * Options for **Standard factual captions** or **Detailed multi-sentence descriptions** (covering pose, clothing, lighting, and scene).
  * Runs as an isolated subprocess (`workers/recaption_worker.py`), streaming live progress (model loading, per-image timing, success/error counters) and releasing GPU VRAM on completion.
  * Preserves original captions as `.txt.bak` backups while updating both the database and `.txt` sidecars on disk.

### 3. 🔍 Step 3: Quality Analysis & Deduplication
* **Multi-Hash Perceptual Deduplication**:
  * Combines **pHash** (Perceptual Hash) and **dHash** (Difference Hash) to identify near-duplicate, cropped, or re-encoded images.
  * Features a color-guard check to prevent false positives on flat backdrops or low-detail renders with distinct colors.
  * Adjustable sensitivity slider (max perceptual distance from 0 to 16).
  * **One-Click Action**: Exclude all detected duplicates while keeping one master image per group.
  * Every thumbnail in a duplicate cluster opens the same full-size zoom modal available in Curate, for comparing near-duplicates at true resolution before deciding what to keep.
* **Sharpness (Blur) Ranking**: Surfaces the least-sharp images in the dataset first, by Laplacian variance, with a one-click exclude button per image — the fastest way to weed out out-of-focus shots before they reach training.
* **Caption Completeness Checks**:
  * Identifies images lacking descriptive text (or containing only the trigger word).
  * Provides quick-edit input fields on the quality dashboard to fill missing descriptions.
* **Orphan File Detection**: Flags images with no matching `.txt` caption sidecar and captions with no matching image, for datasets imported by folder path.
* **Dataset Analytics & Distribution Charts** (in a "Statistics" panel):
  * **Resolution Distribution**: Interactive bar charts for image dimensions (W × H).
  * **Aspect Ratio Classification**: Categorizes ratios into standard buckets (`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`, `other`).
  * **Orientation Balance**: Portrait vs. landscape vs. square counts.
  * **Caption Length Distribution**: Histogram of word counts per sample, highlighting captions that risk truncation by the text encoder's token budget during training.

### 4. 📦 Step 4: Export & Versioning
* **Materialize Flat Training Dataset**:
  * Exports all Active (non-excluded) samples of the current run as paired `<name>.png|.jpg` + `<name>.txt` files into `training_runtime/datasets/<destination_name>/`.
  * Formats files into the exact flat layout expected by the training worker scripts in Step 5.
  * Automatically disambiguates filename stem collisions.
  * Includes overwrite warnings and an inline two-step confirmation workflow.
* **Dataset Versioning & Change Detection**:
  * Each export is tagged with a version (auto-suggested as the next `vN`) and recorded with a content-hash manifest — the previous version's tag, timestamp, and sample count are always shown before you export again.
  * The content hash is built from the (perceptual-hash, caption) pair of every active sample, so it changes if an image is added/removed/excluded or a caption is edited, but not if files are simply re-encoded to identical-looking bytes.
  * If nothing changed since the last export, the panel says so plainly ("Identical to `vN`") instead of writing a redundant version; if something did, it shows a field-by-field delta (sample count, duplicates, median sharpness, mean caption words) against the previous version.

### 5. 🏋️ Step 5: Train & Live Monitoring
* **Self-Contained Training Runtime**:
  * Runs training on a dedicated, isolated Python environment (`training_runtime/venv`) equipped with PyTorch, Diffusers, Accelerate, BitsAndBytes, PEFT, and Transformers — none of which are dependencies of the hub app itself (see "Setup & Provisioning" below).
  * One-time setup script (`scripts/setup_training_runtime.sh`) copies the required model weights (~46GB Krea-2-NF4) and provisions dependencies.
* **Hyperparameter Configuration**:
  * Configurable parameters: `Total steps`, `Learning rate`, `LoRA rank`, `LoRA alpha`, `Batch size`, `Grad accumulation steps`, `Save every` checkpoint interval, and `Seed`.
* **Two-Stage Execution**:
  1. **Pre-Cache Stage (Blocking)**: Runs `workers/precache_worker.py` to pre-compute VAE latents and text embeddings into `training_runtime/cache/<dataset_name>/`.
  2. **Training Stage (Detached Process)**: Runs `workers/train_worker.py` in a detached process group (`start_new_session=True`). Continues running in the background even if Streamlit is closed or restarted.
* **Real-Time Monitoring Dashboard**:
  * Auto-refreshing UI fragment (`@st.fragment(run_every="5s")`) displaying live training loss curves (`train_log.csv`), stdout log tailing, elapsed duration, and a running cost estimate once GPU-seconds are available.
  * **Stop Training**: Send SIGINT for graceful checkpoint saving before escalating to process termination.
  * Full run execution history tracked in SQLite (`training_runs` table), including GPU-seconds and cost estimate backfilled once the process exits.

---

## 🔭 Observability & Telemetry

Two independent mechanisms feed the Metrics page and keep it honest without the UI having to poll a live process:

* **Step telemetry** (`ui/step_telemetry.py`): each UI step (Import, Recaption, Quality, Export) times itself and records duration + error count into that run's `ingestion_runs` row.
* **Worker lifecycle events**: `workers/precache_worker.py` and `workers/train_worker.py` are ported from the upstream LoRAlab project and are never instrumented directly — `workers/_telemetry.py` wraps only their `__main__` entrypoints from the outside, emitting one structured JSON line per lifecycle transition (`worker_started` / `worker_finished` / `worker_failed`, with duration and GPU-seconds) to the same stdout the training log already streams to. `training_runner.py` reads the last ~4KB of a (possibly multi-hour) log file to find that final line cheaply, and `training_service.finalize_dead_run()` uses it to backfill `training_runs.gpu_seconds` / `cost_estimate`.

Recaptioning uses a related but separate JSON-lines protocol (`LoadedEvent` / `CaptionEvent` / `ErrorEvent` / `DoneEvent`) defined in `domain/worker_contracts.py`, streamed live rather than tailed after the fact, since that worker runs blocking in the foreground.

---

## 🔌 MCP Server: Agent Integration

Alongside the Streamlit UI, `mcp_server/` exposes the same pipeline as [Model Context
Protocol](https://modelcontextprotocol.io/) tools, so an autonomous agent (e.g. a
LangGraph graph) can inspect dataset inventory and drive LoRA training end-to-end
without a human clicking through the 5 steps. It's a separate process talking to the
same SQLite DB (`feature_pipeline.db`) and `training_runtime/` — the same "separate
process, shared state" pattern already used by `workers/`.

### Running it

```bash
cd feature_pipeline_hub
uv run python -m mcp_server
```

This starts the server on **stdio** — the transport MCP clients (LangGraph's
`langchain-mcp-adapters`, Claude Desktop, the `mcp` CLI inspector, etc.) launch as a
subprocess and talk to over stdin/stdout. There is no network listener and no
authentication layer (this app has none anywhere — it was built as a local,
single-user tool), so **do not** put this server behind an HTTP/SSE transport or
expose it beyond a trusted local agent without adding an auth layer first.

### Available tools

| Tool | What it does |
|---|---|
| `list_dataset_runs` | Every stored ingestion run, newest first (no samples loaded — cheap). |
| `get_dataset_health` | Fleet-wide health: counts, duplicates, median sharpness per dataset. |
| `get_run_detail` | Full detail of one run, including every sample's caption/metrics/flags. |
| `import_dataset` | Ingest a new dataset from a local folder of images (+ optional `.txt` captions). |
| `revalidate_run` | Re-check samples against current validation rules (one run, or every run). |
| `export_dataset` | Materialize a run's active samples as a flat training folder (Step 4's job). |
| `quality_summary` | Headline quality counts (active/excluded/duplicate/missing-caption/invalid). |
| `start_lora_training` | Launch pre-cache for a run; returns immediately (non-blocking). |
| `get_training_status` | Poll a pre-cache or train job by `training_run_id`. |
| `continue_lora_training` | Once pre-cache is `completed`, launch the training phase (detached). |
| `stop_training` | Send a graceful stop (SIGINT) to a running job. |

Training is intentionally split into `start_lora_training` → poll
`get_training_status` → `continue_lora_training`, rather than one call: the UI's
`start_training` blocks the caller for up to 20 minutes while pre-cache runs
(`PRECACHE_TIMEOUT_SECONDS`), which is fine for a Streamlit button but not for an MCP
tool call. Every tool call opens and closes its own SQLite connection — same
convention `ui/state.py` uses — and `start_lora_training` refuses to launch if a
training-runtime job is already active, whether it was started from the UI or another
agent call.

---

## 🛠️ Setup & Provisioning

### Training Runtime Provisioning

Both Step 2 (AI Recaptioning) and Step 5 (Train) run against the same self-contained
`training_runtime/` — a local copy of the Krea-2-NF4 weights (including the Qwen3-VL
text encoder used for captioning) plus a dedicated venv. Run the one-time provisioning
script, pointed at an existing [AcademiaSD_LoRAlab-Krea2](https://github.com/xd43vild69/AcademiaSD_LoRAlab-Krea2)
checkout, to copy the weights (~46GB) and set up the environment:

```bash
FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2 ./scripts/setup_training_runtime.sh
```

That checkout is only needed for this one-time copy — once `training_runtime/` exists,
neither recaptioning nor training reach back out to it, and the app runs standalone
against its own local model + venv. The hub app's own dependencies (`pyproject.toml`)
are deliberately lightweight — Streamlit, Pydantic, Pillow, imagehash, numpy, the `mcp`
SDK — none of the PyTorch/Diffusers/training stack is installed into the app's own
environment.

---

## ⚙️ Environment Variables

| Variable | Description | Default Value |
|---|---|---|
| `FTI_TRAINING_RUNTIME_DIR` | Directory holding model weights, datasets, cache, and the shared venv used by both recaptioning and training | `feature_pipeline_hub/training_runtime` |
| `FTI_TRAINING_PYTHON` | Python interpreter for training and recaption workers | `<FTI_TRAINING_RUNTIME_DIR>/venv/bin/python` |
| `FTI_DB_PATH` | Path to the SQLite metadata database file | `feature_pipeline_hub/data/feature_pipeline.db` |
| `FTI_DATA_DIR` | Base directory for raw dataset uploads | `feature_pipeline_hub/data` |
| `FTI_GPU_HOURLY_RATE` | Your GPU's hourly rate, used to turn recorded GPU-seconds into a cost estimate on the Metrics page and Train dashboard | unset — cost is left unestimated rather than guessed at |
| `FTI_RUN_ID` | Set by the launcher (not the user) so a worker's telemetry events carry its own run id | — |

---

## 📐 Architecture & Project Structure

The project follows Clean Architecture principles: `domain` has no I/O and no framework dependencies; `application` holds business logic and depends on `domain` + infrastructure interfaces, not on Streamlit; `infrastructure` handles persistence, the filesystem, and subprocess launching; `ui` is Streamlit-only and is the sole place that talks to both `application` and the database (through `ui/state.py`).

```
feature_pipeline_hub/
├── main.py                        # Launcher script
├── pyproject.toml                 # Project configuration & dependencies
├── CONTRIBUTING.md                # Git workflow & commit conventions
├── scripts/
│   └── setup_training_runtime.sh  # Provisioning script for training venv & model weights
├── src/
│   └── feature_pipeline/
│       ├── domain/                # Core domain models & pure logic — no I/O
│       │   ├── models.py          # DatasetSample, ConceptGroup, IngestionRun, DatasetManifest
│       │   ├── validators.py      # Extension, resolution & caption validation rules
│       │   ├── naming.py          # Concept-name slugging & standardized image filenames
│       │   ├── worker_contracts.py# Pydantic schemas for worker settings-in / events-out
│       │   └── cost.py            # GPU-seconds → estimated cost
│       ├── application/           # Business logic & services
│       │   ├── dataset_service.py # Ingestion, append, revalidation & dataset assembly
│       │   ├── caption_service.py # Text normalization & trigger word injection
│       │   ├── image_service.py   # pHash, dHash, colorhash, thumbnail generator
│       │   ├── quality_service.py # Perceptual deduplication, sharpness & quality metrics
│       │   ├── inventory_service.py # Cross-dataset health inventory for the Metrics page
│       │   ├── recaption_service.py # AI recaptioning orchestrator
│       │   ├── export_service.py  # Flat dataset exporter for training
│       │   └── training_service.py# Pre-cache & training process orchestrator
│       └── infrastructure/        # Data access & process runners
│           ├── database.py        # SQLite schema, tables & ALTER-TABLE column migrations
│           ├── ingestion_repository.py # Run & sample CRUD persistence
│           ├── version_repository.py   # Dataset export version (manifest) persistence
│           ├── training_repository.py  # Training run status & log persistence
│           ├── storage.py         # File storage, standardized naming & directory helpers
│           ├── recaption_runner.py# Streaming subprocess launcher for the Qwen3-VL worker
│           ├── training_runner.py # Detached process launcher & log reader for training
│           └── hf_exporter.py     # Placeholder for a future HF/Parquet/Kohya export layout
├── ui/                             # Streamlit UI Layer
│   ├── app.py                      # Entry point & page navigation (Metrics + 5 steps)
│   ├── state.py                    # Session state & UI persistence helpers — the only bridge to SQLite
│   ├── step_telemetry.py           # Per-step duration/error timing, feeds the Metrics page
│   ├── steps/                      # Thin page wrappers: Import, Curate, Quality, Export, Train, Metrics
│   └── components/                 # UI panels: Import, Gallery, Image Zoom, Recaption, Quality,
│                                    #   Export, Train, Context Bar, Inventory, Observability
├── mcp_server/                     # MCP server: pipeline tools for autonomous agents (stdio transport)
│   ├── server.py                   # FastMCP tool definitions, thin wrappers over application/
│   └── __main__.py                 # `uv run python -m mcp_server` entrypoint
├── workers/                        # Worker scripts, run as separate processes in a different venv
│   ├── recaption_worker.py         # Qwen3-VL recaption worker
│   ├── caption_qwen3vl.py          # Vendored Qwen3-VL loading/captioning (ported from LoRAlab)
│   ├── precache_worker.py          # VAE & text embedding pre-cache worker (vendored, byte-for-byte)
│   ├── train_worker.py             # LoRA training execution worker (vendored logic; docstrings rewritten locally)
│   └── _telemetry.py               # Wraps precache/train's __main__ with lifecycle JSON events
```

---

## 🚀 How to Run

### Prerequisites
* Python 3.11+
* [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Running the App

```bash
cd feature_pipeline_hub
uv run streamlit run ui/app.py
```
Or via the launcher script:
```bash
cd feature_pipeline_hub
uv run python main.py
```

---

## ⌨️ Shortcuts & UI Helpers

* **`F2`**: Opens the **Rename Word in Captions** dialog in Curate to replace terms across all dataset captions.
* **Click any thumbnail**: Opens it full-size in a modal (Curate and Quality), with a toggle for viewing true, unscaled pixels.
* **Global Context Bar**: Persistent top header bar to switch active datasets, monitor real-time health badges (active count, duplicate count, missing descriptions, excluded count), view concept details, re-run validation against the current rules ("Revalidate"), or delete dataset runs.

---

## 🧪 Running Tests

Run the pytest test suite:

```bash
cd feature_pipeline_hub
uv run pytest
```

---

## ⚙️ Continuous Integration

`.github/workflows/python-app.yml` runs on every push and pull request to `main`:

1. **`uv sync --locked`** — installs the exact dependency set from `uv.lock`.
2. **`mypy`** — strict, Pydantic-aware type-checking (`pydantic.mypy` plugin) over
   `src/feature_pipeline/` and `mcp_server/`, where the domain models and the tools an
   agent calls actually live. `ui/` (Streamlit) and `workers/` (ported
   from upstream LoRAlab) are deliberately out of scope.
3. **`pytest`** — the full test suite (caption normalization, validation rules,
   deduplication, export integrity, MCP tools, and more), the gate against a change
   silently breaking caption quality or a dataset export.

Both steps must pass before a change can be considered safe to merge.

---

## 🤝 Contributing

See [`feature_pipeline_hub/CONTRIBUTING.md`](feature_pipeline_hub/CONTRIBUTING.md) for git branch strategy, commit conventions, and workflow. `.claudecodingrc` at the repo root encodes the same conventions for AI coding agents working in this repo.
