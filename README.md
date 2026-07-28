# xFTI-feature

Dataset curation hub for LoRA training: import a folder of images, curate their
captions, check dataset quality, and export a versioned dataset.

## Running

```bash
cd feature_pipeline_hub
uv run streamlit run ui/app.py
```

The app walks through four steps: **Import** a raw folder or uploads, **Curate**
captions in a thumbnail grid, check **Quality** (perceptual duplicates, missing
captions, distributions), and **Export**.

## AI recaptioning (optional)

The Curate step can regenerate captions with Qwen3-VL-4B — the same model Krea 2
uses as its text encoder — by selecting images and running them through the
[AcademiaSD_LoRAlab-Krea2](https://github.com/xd43vild69/AcademiaSD_LoRAlab-Krea2)
checkout as a subprocess. Nothing heavy is installed here: torch, transformers and
the 8 GB of weights stay in that project, and the subprocess exits after each
batch so the VRAM is released.

Point the hub at that checkout before starting it:

```bash
export FTI_LORALAB_ROOT=/path/to/AcademiaSD_LoRAlab-Krea2
```

Set `FTI_RECAPTION_PYTHON` too if its virtualenv is not at `<root>/venv/bin/python`.
Without these the rest of the app works normally and the Curate step just explains
what to set.

Recaptioning writes the new caption both to the local database and to the `.txt`
sidecar next to the image, keeping the pre-AI text as `.txt.bak`, so the same
dataset folder stays usable from LoRAlab.

## Configuration

| Variable | Purpose |
|---|---|
| `FTI_LORALAB_ROOT` | LoRAlab checkout used for AI recaptioning |
| `FTI_RECAPTION_PYTHON` | Interpreter for the recaption worker (defaults to `<root>/venv/bin/python`) |
| `FTI_DB_PATH` | Curation database (defaults to `feature_pipeline_hub/data/feature_pipeline.db`) |
| `FTI_DATA_DIR` | Where uploaded datasets are stored (defaults to `feature_pipeline_hub/data`) |

## Tests

```bash
cd feature_pipeline_hub && uv run pytest
```
