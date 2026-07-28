# xFTI-feature (Feature Pipeline Hub)

**Feature Pipeline Hub** is an interactive, Streamlit-based dataset curation and quality-assurance platform designed for **LoRA (Low-Rank Adaptation) model training**. It streamlines the entire pipeline: importing raw images and captions, performing interactive visual curation and AI-driven recaptioning, running perceptual duplicate detection and text length checks, and exporting clean, versioned datasets.

---

## 🌟 Key Features

The application is structured into four sequential workflow steps accessible via a top navigation bar:

### 1. 📥 Step 1: Import & Ingestion
* **Dual Ingestion Sources**:
  * **Local Folder Scanning**: Point directly to any local directory containing images.
  * **File Uploads**: Drag-and-drop images (`.png`, `.jpg`, `.jpeg`, `.webp`) and optional `.txt` caption sidecar files via the browser interface.
* **Concept & Trigger Word Assignment**: Assign a dataset **Concept Name** (e.g., `cyberpunk_style`) and **Trigger Word** (e.g., `sks_style`). The trigger word is automatically prefixed to image captions during ingestion.
* **Automated Sample Validation**: Checks file extensions, enforces minimum image resolution requirements (≥ 512×512 px), and validates caption character lengths (3 to 500 characters).
* **Persistent Ingestion Runs**: Creates standalone, unique `IngestionRun` records stored in a local SQLite database (`feature_pipeline.db`). Sessions survive app restarts and uploaded datasets are safely cached in `data/raw/<run_id>/`.

### 2. 🎨 Step 2: Curate & AI Recaptioning
* **Adaptive Curation Grid**:
  * Displays square, letterboxed, theme-adaptive image thumbnails with adjustable grid column sizing (2 to 6 columns).
  * Filter dataset views instantly: `Active`, `All`, `Duplicates`, `Invalid`, `No caption`, and `Excluded`.
* **In-Place Caption Editing**: Edit captions directly below each thumbnail with instant SQLite database persistence.
* **Batch Caption Word Replacement**:
  * Replace specific terms across all captions in the active dataset.
  * Accessible via the top toolbar or the **`F2` keyboard shortcut**.
  * Shows exact match counts and affected sample previews before applying changes.
* **Sample Inclusion & Exclusion**: Exclude problematic or unwanted images from the dataset output without deleting the source files from disk.
* **AI Recaptioning Engine (Qwen3-VL-4B)**:
  * Regenerates captions using the **Qwen3-VL-4B** vision-language model (the text encoder used by Krea 2).
  * Choice between **Standard factual captions** or **Detailed multi-sentence descriptions** (covering pose, clothing, lighting, and environment).
  * Displays live progress (model loading, per-image timing, and success/error status).
  * Automatically backs up original text sidecars as `.txt.bak` and updates both the database and `.txt` sidecars on disk.

### 3. 🔍 Step 3: Quality Analysis & Deduplication
* **Multi-Hash Perceptual Deduplication**:
  * Combines **pHash** (Perceptual Hash) and **dHash** (Difference Hash) to identify near-duplicate, cropped, or re-encoded images.
  * Includes a **Color Guard** check (42-bit color hashing) to eliminate false positives on flat backdrops or low-detail renders with distinct colors.
  * Adjustable sensitivity slider (max perceptual distance from 0 to 16).
  * **One-Click Action**: Exclude all detected duplicates while keeping one master image per group.
* **Caption Completeness Checks**:
  * Identifies images lacking descriptive text (or containing only the trigger word).
  * Provides quick-edit input fields to fill missing descriptions directly on the quality dashboard.
* **Dataset Analytics & Statistics**:
  * **Resolution Distribution**: Interactive bar charts for image dimensions ($W \times H$).
  * **Aspect Ratio Classification**: Categorizes aspect ratios into standard buckets (`1:1`, `4:3`, `3:4`, `16:9`, `9:16`, `3:2`, `2:3`, `other`).
  * **Caption Length Distribution**: Histogram of word counts per sample, highlighting captions over 55 words that risk truncation by CLIP's 77-token limit during training.

### 4. 📦 Step 4: Export & Versioning
* Snapshot generation and metadata manifest creation (`DatasetManifest`) for versioning datasets prior to exporting to Hugging Face Datasets or local LoRA trainer folder structures (Kohya / ComfyUI format).

---

## 🤖 AI Recaptioning Setup (Qwen3-VL Integration)

Recaptioning uses a subprocess runner (`workers/recaption_worker.py`) calling into an external checkout of [AcademiaSD_LoRAlab-Krea2](https://github.com/xd43vild69/AcademiaSD_LoRAlab-Krea2).

* **Process Isolation & Memory Optimization**: Heavy dependencies (`torch`, `transformers`, ~9 GB VRAM weights) remain in the LoRAlab virtualenv. The subprocess launches on demand and releases all VRAM immediately upon batch completion (with automatic CPU fallback on GPU OOM).
* **Setup**: Point the application to your LoRAlab checkout:
  ```bash
  export FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2
  ```
  *(Optional)* If the python environment for LoRAlab is not located at `<root>/venv/bin/python`, set:
  ```bash
  export FTI_RECAPTION_PYTHON=/path/to/virtualenv/bin/python
  ```

---

## 🛠️ Configuration & Environment Variables

| Variable | Description | Default Value |
|---|---|---|
| `FTI_LORALAB_ROOT` | Path to the `AcademiaSD_LoRAlab-Krea2` checkout for AI recaptioning | *Unset* (AI recaptioning disabled) |
| `FTI_RECAPTION_PYTHON` | Path to Python interpreter containing Qwen3-VL dependencies | `<FTI_LORALAB_ROOT>/venv/bin/python` |
| `FTI_DB_PATH` | SQLite database file path for metadata storage | `feature_pipeline_hub/data/feature_pipeline.db` |
| `FTI_DATA_DIR` | Directory for raw uploaded image datasets | `feature_pipeline_hub/data` |

---

## 📐 Architecture & Project Structure

The project follows Clean Architecture principles:

```
feature_pipeline_hub/
├── main.py                        # Launcher script
├── pyproject.toml                 # Project metadata & dependencies
├── src/
│   └── feature_pipeline/
│       ├── domain/                # Core domain models, validation rules, schemas
│       │   ├── models.py          # DatasetSample, ConceptGroup, IngestionRun, Manifest
│       │   └── validators.py      # Sample validation functions
│       ├── application/           # Business logic & services
│       │   ├── dataset_service.py # Ingestion & dataset assembly
│       │   ├── caption_service.py # Text normalization & trigger word injection
│       │   ├── image_service.py   # Metrics calculation (pHash, dHash, colorhash, thumbnails)
│       │   ├── quality_service.py # Perceptual deduplication & quality metrics
│       │   └── recaption_service.py # AI recaptioning orchestrator
│       └── infrastructure/        # Data access & external integrations
│           ├── database.py        # SQLite connection & schema migrations
│           ├── ingestion_repository.py # CRUD persistence for runs & samples
│           ├── storage.py         # File storage & sidecar management
│           ├── recaption_runner.py# Subprocess launcher for Qwen3-VL worker
│           └── hf_exporter.py     # Dataset export handlers
├── ui/                            # Streamlit UI Layer
│   ├── app.py                     # Entry point & navigation routing
│   ├── state.py                   # Session state & UI persistence helpers
│   ├── steps/                     # Navigation step views (Import, Curate, Quality, Export)
│   └── components/                # UI components (Gallery, Context Bar, Quality Panel, etc.)
└── workers/
    └── recaption_worker.py        # Standalone worker script executed in LoRAlab environment
```

---

## 🚀 How to Run

### Prerequisites
* Python 3.11+
* [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### Running the Application

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

## ⌨️ Keyboard Shortcuts & Quick Actions

* **`F2`**: Opens the **Rename Word in Captions** dialog to perform batch search-and-replace across all captions in the active dataset.
* **Global Context Bar**: Switch active datasets, view real-time dataset health badges (active, duplicate count, missing descriptions, excluded count), inspect concept details, or safely delete ingestion runs.

---

## 🧪 Running Tests

To run the unit test suite:

```bash
cd feature_pipeline_hub
uv run pytest
```
