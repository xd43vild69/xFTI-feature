"""Local filesystem management: raw ingestion scanning and processed-dataset caching.

Writing curated output to `data/processed/` lands in Iteración 4 (Exportador).
"""

import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def scan_raw_folder(folder_path: str) -> list[str]:
    """List image file paths found under a raw ingestion folder (non-recursive)."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a valid directory")

    return sorted(
        str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_caption_for_image(image_path: str) -> str:
    """Read the .txt caption accompanying an image, or return an empty string if missing."""
    caption_path = Path(image_path).with_suffix(".txt")
    if caption_path.exists():
        return caption_path.read_text(encoding="utf-8").strip()
    return ""


def save_uploaded_files(uploaded_files: list) -> str:
    """Save Streamlit UploadedFile objects to a temporary folder and return the folder path.

    Args:
        uploaded_files: List of streamlit.UploadedFile objects from st.file_uploader

    Returns:
        Path to the temporary folder containing the uploaded files
    """
    temp_dir = tempfile.mkdtemp(prefix="fti_upload_")
    temp_path = Path(temp_dir)

    for uploaded_file in uploaded_files:
        file_path = temp_path / uploaded_file.name
        file_path.write_bytes(uploaded_file.getbuffer())

    return temp_dir
