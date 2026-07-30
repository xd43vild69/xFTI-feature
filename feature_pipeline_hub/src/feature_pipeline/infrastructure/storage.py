"""Local filesystem management: raw ingestion scanning and processed-dataset caching.

Writing curated output to `data/processed/` lands in Iteración 4 (Exportador).
"""

import os
import shutil
import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def raw_data_dir() -> Path:
    """Folder where uploaded datasets are kept, overridable via FTI_DATA_DIR."""
    override = os.environ.get("FTI_DATA_DIR")
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "data"
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def scan_raw_folder(folder_path: str) -> list[str]:
    """List image file paths found under a raw ingestion folder (non-recursive)."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a valid directory")

    return sorted(
        str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def scan_caption_files(folder_path: str) -> list[str]:
    """List .txt caption file paths in a raw folder (non-recursive).

    The mirror of `scan_raw_folder`: ingestion only ever looks up captions from an
    image, so a .txt whose image is missing or has an unsupported extension is
    otherwise invisible. Returns an empty list for a folder that is gone, since
    this is used for reporting rather than ingestion.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        return []

    return sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def read_caption_for_image(image_path: str) -> str:
    """Read the .txt caption accompanying an image, or return an empty string if missing."""
    caption_path = Path(image_path).with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return ""


def write_caption_sidecar(image_path: str, caption: str, keep_backup: bool = True) -> str:
    """Write the .txt caption next to its image, backing up any previous one once.

    Written via a temp file and os.replace so an interrupted run never leaves a
    half-written caption. The first overwrite keeps the original as .txt.bak —
    subsequent ones don't, so the backup always holds the pre-AI caption. Mirrors
    what LoRAlab's recaption script does to the same folders.
    """
    caption_path = Path(image_path).with_suffix(".txt")
    backup_path = caption_path.with_suffix(".txt.bak")

    if keep_backup and caption_path.exists() and not backup_path.exists():
        shutil.copyfile(caption_path, backup_path)

    temp_path = caption_path.with_suffix(".txt.tmp")
    temp_path.write_text(caption, encoding="utf-8")
    os.replace(temp_path, caption_path)

    return str(caption_path)


def save_uploaded_files(uploaded_files: list, destination: str | None = None) -> str:
    """Save Streamlit UploadedFile objects to a folder and return that folder's path.

    Args:
        uploaded_files: List of streamlit.UploadedFile objects from st.file_uploader
        destination: Target folder; defaults to a temporary one. Runs that must
            outlive the session pass a folder under `data/raw/`, since /tmp gets
            swept and would break previews of older ingestions.

    Returns:
        Path to the folder containing the uploaded files
    """
    if destination is None:
        folder = Path(tempfile.mkdtemp(prefix="fti_upload_"))
    else:
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = folder / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

    return str(folder)


def _available_stem(taken: set[str], stem: str) -> str:
    """A stem no file in the folder uses yet, suffixed `-1`, `-2`… on collision."""
    if stem not in taken:
        return stem

    index = 1
    while f"{stem}-{index}" in taken:
        index += 1
    return f"{stem}-{index}"


def append_uploaded_files(uploaded_files: list, folder: str) -> list[str]:
    """Save uploads into a folder that already holds files, and return what was written.

    `save_uploaded_files` owns the folder it writes to, so a name collision there
    cannot happen. Adding to a curated run is the opposite case: overwriting would
    swap the bytes under a sample that is already captioned, measured and validated,
    while its stored metrics went on describing the old image. Colliding names take a
    suffix instead, so nothing already in the dataset is ever replaced.

    The suffix is resolved per stem, not per file, so an image and the .txt uploaded
    beside it keep matching names — `read_caption_for_image` pairs them by stem.
    """
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)

    taken = {path.stem for path in destination.iterdir() if path.is_file()}
    stems: dict[str, str] = {}
    written: list[str] = []

    for uploaded_file in uploaded_files:
        source = Path(uploaded_file.name)
        if source.stem not in stems:
            stems[source.stem] = _available_stem(taken, source.stem)
            taken.add(stems[source.stem])

        target = destination / f"{stems[source.stem]}{source.suffix}"
        target.write_bytes(uploaded_file.getbuffer())
        written.append(str(target))

    return written


def run_upload_dir(run_id: str) -> str:
    """Per-run folder under `data/raw/` for files uploaded through the UI."""
    return str(raw_data_dir() / run_id)


def delete_managed_folder(folder_path: str) -> bool:
    """Delete a folder only if it lives under `data/raw/`.

    Guards against wiping a user's own source folder when deleting a run that was
    ingested by path rather than uploaded.
    """
    folder = Path(folder_path).resolve()
    raw_dir = raw_data_dir().resolve()

    if folder == raw_dir or raw_dir not in folder.parents or not folder.is_dir():
        return False

    shutil.rmtree(folder)
    return True


def training_runtime_dir() -> Path:
    """Root for the self-contained training environment (model, venv, datasets).

    Lives inside the project by default, gitignored — overridable via
    FTI_TRAINING_RUNTIME_DIR for machines that keep it on another disk.
    """
    override = os.environ.get("FTI_TRAINING_RUNTIME_DIR")
    base = Path(override) if override else Path(__file__).resolve().parents[3] / "training_runtime"
    base.mkdir(parents=True, exist_ok=True)
    return base


def training_dataset_dir(name: str) -> Path:
    """Where an exported dataset lives, ready for the training scripts to consume."""
    return training_runtime_dir() / "datasets" / name


def clear_training_dataset_dir(name: str) -> None:
    """Wipe a training dataset folder before a fresh export, if it exists.

    Guarded to only ever touch paths under training_runtime/datasets/ — this
    folder is entirely ours (unlike data/raw/, nothing external reads it), so a
    fresh export is free to remove whatever was there before, including a stale
    curation_report.json from a previous export of the same concept.
    """
    target = training_dataset_dir(name).resolve()
    datasets_root = (training_runtime_dir() / "datasets").resolve()

    if datasets_root not in target.parents:
        raise ValueError(f"Refusing to clear a path outside training_runtime/datasets/: {target}")

    if target.is_dir():
        shutil.rmtree(target)
