# xFTI-feature (Feature Pipeline Hub)

**Feature Pipeline Hub** is an end-to-end, interactive Streamlit application designed for **LoRA (Low-Rank Adaptation) model training dataset curation and training execution**. It provides a unified pipeline for importing raw images and captions, visually curating and AI-recaptioning samples, performing perceptual deduplication and quality analytics, exporting clean datasets, and launching/monitoring LoRA model training.

---

## 🌟 Key Features

The application is organized into **five sequential workflow steps** accessible via top-level page navigation:

### 1. 📥 Step 1: Import & Ingestion
* **Dual Ingestion Sources**:
  * **Local Directory Scan**: Ingest images directly from any local folder path.
  * **Browser File Uploads**: Drag-and-drop images (`.png`, `.jpg`, `.jpeg`, `.webp`) and optional `.txt` caption sidecar files via the UI. Uploaded datasets are persistently cached in `data/raw/<run_id>/`.
* **Concept & Trigger Word Assignment**: Define a dataset **Concept Name** (e.g., `cyberpunk_style`) and **Trigger Word** (e.g., `sks_style`). Trigger words are automatically prefixed to captions during ingestion if not already present.
* **Automated Validation**: Enforces minimum image resolution requirements (≥ 512×512 px), validates file extensions, and checks caption character lengths (3 to 500 characters).
* **Persistent Ingestion Runs**: Stores dataset runs in a local SQLite database (`feature_pipeline.db`), allowing runs to survive app restarts and remain selectable via the global context bar.

### 2. 🎨 Step 2: Curate & AI Recaptioning
* **Interactive Curation Grid**:
  * Displays square, letterboxed, theme-adaptive thumbnails with an adjustable column layout slider (2 to 6 columns).
  * Filter grid views: `Active`, `All`, `Duplicates`, `Invalid`, `No caption`, and `Excluded`.
* **In-Place Caption Editing**: Edit text captions directly underneath each thumbnail with instant SQLite database persistence.
* **Batch Caption Word Replacement**:
  * Replace specific terms across all captions in the active dataset.
  * Accessible via the top toolbar or the **`F2` keyboard shortcut**.
  * Displays exact match counts and matching sample previews before executing the replacement.
* **Sample Inclusion & Exclusion**: Exclude specific images from the final dataset without deleting files from disk.
* **AI Recaptioning Engine (Qwen3-VL-4B)**:
  * Regenerates captions using the **Qwen3-VL-4B** vision-language model (the text encoder model used by Krea 2).
  * Options for **Standard factual captions** or **Detailed multi-sentence descriptions** (covering pose, clothing, lighting, and scene).
  * Runs as an isolated subprocess (`workers/recaption_worker.py`), streaming live progress (model loading, per-image timing, success/error counters) and releasing GPU VRAM on completion.
  * Preserves original captions as `.txt.bak` backups while updating both the database and `.txt` sidecars on disk.

### 3. 🔍 Step 3: Quality Analysis & Deduplication
* **Multi-Hash Perceptual Deduplication**:
  * Combines **pHash** (Perceptual Hash) and **dHash** (Difference Hash) to identify near-duplicate, cropped, or re-encoded images.
  * Features a 42-bit **Color Guard** check to prevent false positives on flat backdrops or low-detail renders with distinct colors.
  * Adjustable sensitivity slider (max perceptual distance from 0 to 16).
  * **One-Click Action**: Exclude all detected duplicates while keeping one master image per group.
* **Caption Completeness Checks**:
  * Identifies images lacking descriptive text (or containing only the trigger word).
  * Provides quick-edit input fields on the quality dashboard to fill missing descriptions.
* **Dataset Analytics & Distribution Charts**:
  * **Resolution Distribution**: Interactive bar charts for image dimensions ($W \times H$).
  * **Aspect Ratio Classification**: Categorizes ratios into standard buckets (`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`, `other`).
  * **Caption Length Distribution**: Histogram of word counts per sample, highlighting captions over 55 words that risk truncation by CLIP's 77-token limit during training.

### 4. 📦 Step 4: Export & Materialization
* **Materialize Flat Training Dataset**:
  * Exports all Active (non-excluded) samples of the current run as paired `<name>.png|.jpg` + `<name>.txt` files into `training_runtime/datasets/<destination_name>/`.
  * Formats files into the exact flat layout expected by the training worker scripts in Step 5.
  * Automatically disambiguates filename stem collisions.
  * Includes overwrite warnings and an inline two-step confirmation workflow.

### 5. 🏋️ Step 5: Train & Live Monitoring
* **Self-Contained Training Runtime**:
  * Runs training on a dedicated, isolated Python environment (`training_runtime/venv`) equipped with PyTorch, Diffusers, Accelerate, BitsAndBytes, PEFT, and Transformers.
  * One-time setup script (`scripts/setup_training_runtime.sh`) copies the required model weights (~46GB Krea-2-NF4) and provisions dependencies.
* **Hyperparameter Configuration**:
  * Configurable parameters: `Total steps`, `Learning rate`, `LoRA rank`, `LoRA alpha`, `Batch size`, `Grad accumulation steps`, `Save every` checkpoint interval, and `Seed`.
* **Two-Stage Execution**:
  1. **Pre-Cache Stage (Blocking)**: Runs `workers/precache_worker.py` to pre-compute VAE latents and text embeddings into `training_runtime/cache/<dataset_name>/`.
  2. **Training Stage (Detached Process)**: Runs `workers/train_worker.py` in a detached process group (`start_new_session=True`). Continues running in the background even if Streamlit is closed or restarted.
* **Real-Time Monitoring Dashboard**:
  * Auto-refreshing UI fragment (`@st.fragment(run_every="5s")`) displaying live training loss curves (`train_log.csv`) and stdout log tailing.
  * **Stop Training**: Send SIGINT for graceful checkpoint saving before escalating to process termination.
  * Full run execution history tracked in SQLite (`training_runs` table).

---

## 🛠️ Setup & Provisioning

### 1. AI Recaptioning Setup (Qwen3-VL Integration)

Recaptioning uses a subprocess runner (`workers/recaption_worker.py`) pointing to an external checkout of [AcademiaSD_LoRAlab-Krea2](https://github.com/xd43vild69/AcademiaSD_LoRAlab-Krea2):

```bash
export FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2
```
*(Optional)* Set the Python environment if not located at `<root>/venv/bin/python`:
```bash
export FTI_RECAPTION_PYTHON=/path/to/virtualenv/bin/python
```

### 2. Training Runtime Provisioning

To enable Step 5 (Train), run the one-time provisioning script to copy model weights (~46GB) and set up the training environment:

```bash
FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2 ./scripts/setup_training_runtime.sh
```

---

## ⚙️ Environment Variables

| Variable | Description | Default Value |
|---|---|---|
| `FTI_LORALAB_ROOT` | Path to `AcademiaSD_LoRAlab-Krea2` checkout for AI recaptioning | *Unset* |
| `FTI_RECAPTION_PYTHON` | Python interpreter containing Qwen3-VL dependencies | `<FTI_LORALAB_ROOT>/venv/bin/python` |
| `FTI_TRAINING_RUNTIME_DIR` | Directory holding model weights, datasets, cache, and training venv | `feature_pipeline_hub/training_runtime` |
| `FTI_TRAINING_PYTHON` | Python interpreter for training workers | `<FTI_TRAINING_RUNTIME_DIR>/venv/bin/python` |
| `FTI_DB_PATH` | Path to the SQLite metadata database file | `feature_pipeline_hub/data/feature_pipeline.db` |
| `FTI_DATA_DIR` | Base directory for raw dataset uploads | `feature_pipeline_hub/data` |

---

## 📐 Architecture & Project Structure

The project follows Clean Architecture principles:

```
feature_pipeline_hub/
├── main.py                        # Launcher script
├── pyproject.toml                 # Project configuration & dependencies
├── scripts/
│   └── setup_training_runtime.sh  # Provisioning script for training venv & model weights
├── src/
│   └── feature_pipeline/
│       ├── domain/                # Core domain models & validation
│       │   ├── models.py          # DatasetSample, ConceptGroup, IngestionRun, Manifest
│       │   └── validators.py      # Extension, resolution & caption validation
│       ├── application/           # Business logic & services
│       │   ├── dataset_service.py # Ingestion & dataset assembly
│       │   ├── caption_service.py # Text normalization & trigger word injection
│       │   ├── image_service.py   # pHash, dHash, colorhash, thumbnail generator
│       │   ├── quality_service.py # Perceptual deduplication & quality metrics
│       │   ├── recaption_service.py # AI recaptioning orchestrator
│       │   ├── export_service.py  # Flat dataset exporter for training
│       │   └── training_service.py# Pre-cache & training process orchestrator
│       └── infrastructure/        # Data access & process runners
│           ├── database.py        # SQLite schema & migration management
│           ├── ingestion_repository.py # Run & sample CRUD persistence
│           ├── training_repository.py  # Training run status & log persistence
│           ├── storage.py         # File storage & directory helpers
│           ├── recaption_runner.py# Subprocess launcher for Qwen3-VL worker
│           ├── training_runner.py # Detached process launcher & log reader for training
│           └── hf_exporter.py     # Hugging Face export helpers
├── ui/                            # Streamlit UI Layer
│   ├── app.py                     # Entry point & 5-step navigation routing
│   ├── state.py                   # Session state & UI persistence helpers
│   ├── steps/                     # Step page definitions (Import, Curate, Quality, Export, Train)
│   └── components/                # UI panels (Import, Gallery, Recaption, Quality, Export, Train, Context Bar)
├── workers/                       # Worker scripts executed in dedicated environments
│   ├── recaption_worker.py        # Qwen3-VL recaption worker
│   ├── precache_worker.py         # VAE & text embedding pre-cache worker
│   └── train_worker.py            # LoRA training execution worker
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
* **Global Context Bar**: Persistent top header bar to switch active datasets, monitor real-time health badges (active count, duplicate count, missing descriptions, excluded count), view concept details, or delete dataset runs.

---

## 🧪 Running Tests

Run the pytest test suite:

```bash
cd feature_pipeline_hub
uv run pytest
```
