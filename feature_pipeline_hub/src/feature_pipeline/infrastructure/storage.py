"""Local filesystem management: raw ingestion scanning and processed-dataset caching.

Writing curated output to `data/processed/` lands in Iteración 4 (Exportador).
"""

import os
import shutil
import tempfile
from pathlib import Path

from feature_pipeline.domain.naming import standardized_stem

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


def _available_stem(taken: set[str], stem: str) -> str:
    """A stem no file in the folder uses yet, suffixed `-1`, `-2`… on collision.

    A safety net rather than the normal path: `standardized_stem` combined with
    `next_standard_index` should already hand out a free name, but this still
    guards against a stray file — one dropped in by hand, or left over from before
    the standardized scheme — happening to occupy it.
    """
    if stem not in taken:
        return stem

    index = 1
    while f"{stem}-{index}" in taken:
        index += 1
    return f"{stem}-{index}"


def _pair_uploads(uploaded_files: list) -> tuple[list[tuple], list]:
    """Split uploads into (image, same-stem caption or None) pairs, plus orphan captions.

    Mirrors how ingestion looks up a caption: by stem, next to the image. A .txt
    with no image of that stem in the same batch has nothing to attach to, so it's
    kept aside and written under its own name rather than silently dropped.
    """
    images: dict[str, object] = {}
    captions: dict[str, object] = {}

    for uploaded_file in uploaded_files:
        path = Path(uploaded_file.name)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images[path.stem] = uploaded_file
        elif path.suffix.lower() == ".txt":
            captions[path.stem] = uploaded_file

    pairs = [(image, captions.pop(stem, None)) for stem, image in images.items()]
    return pairs, list(captions.values())


def _write_standardized(
    uploaded_files: list, folder: Path, concept_name: str, start_index: int
) -> list[str]:
    """Write uploads as `<concept_slug>_NNNN.ext`, images numbered in upload order.

    A same-stem `.txt` uploaded alongside an image takes that image's new stem, so
    the pair still matches under `read_caption_for_image`'s by-stem lookup.
    """
    pairs, orphan_captions = _pair_uploads(uploaded_files)
    taken = {path.stem for path in folder.iterdir() if path.is_file()}
    written: list[str] = []
    index = start_index

    for image, caption in pairs:
        stem = _available_stem(taken, standardized_stem(concept_name, index))
        taken.add(stem)
        index += 1

        image_path = folder / f"{stem}{Path(image.name).suffix}"
        image_path.write_bytes(image.getbuffer())
        written.append(str(image_path))

        if caption is not None:
            (folder / f"{stem}.txt").write_bytes(caption.getbuffer())

    for caption in orphan_captions:
        stem = _available_stem(taken, Path(caption.name).stem)
        taken.add(stem)
        (folder / f"{stem}.txt").write_bytes(caption.getbuffer())

    return written


def save_uploaded_files(uploaded_files: list, concept_name: str, destination: str | None = None) -> str:
    """Save Streamlit UploadedFile objects to a folder, named `<concept_slug>_NNNN.ext`.

    Args:
        uploaded_files: List of streamlit.UploadedFile objects from st.file_uploader
        concept_name: The dataset's concept name, source of the standardized stem —
            every image lands under one naming scheme regardless of what it was
            called on the user's machine, which is what later lets a raw filename
            be traced back to its position in the dataset.
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

    _write_standardized(uploaded_files, folder, concept_name, start_index=1)
    return str(folder)


def append_uploaded_files(
    uploaded_files: list, folder: str, concept_name: str, start_index: int
) -> list[str]:
    """Save uploads into a folder that already holds files, and return what was written.

    Continues the standardized numbering from `start_index` — the caller works out
    what that is from the samples the run already has, since a dataset started
    before this naming scheme, or one mixed with files added outside the app,
    won't have a count that matches what's on disk.

    `save_uploaded_files` owns the folder it writes to, so a name collision there
    cannot happen. Adding to a curated run is the opposite case: overwriting would
    swap the bytes under a sample that is already captioned, measured and
    validated, while its stored metrics went on describing the old image — so a
    collision here still falls back to a `-1`, `-2`… suffix instead of replacing
    anything.
    """
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    return _write_standardized(uploaded_files, destination, concept_name, start_index)


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
