from pathlib import Path

import pytest
from PIL import Image

from feature_pipeline.infrastructure.storage import (
    read_caption_for_image,
    save_uploaded_files,
    scan_raw_folder,
)


def _make_image(path: Path) -> None:
    Image.new("RGB", (64, 64), color="red").save(path)


def test_scan_raw_folder_finds_supported_images(tmp_path: Path):
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("not an image")

    found = scan_raw_folder(str(tmp_path))

    names = sorted(Path(p).name for p in found)
    assert names == ["a.png", "b.jpg"]


def test_scan_raw_folder_ignores_subdirectories(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    _make_image(tmp_path / "a.png")

    found = scan_raw_folder(str(tmp_path))

    assert len(found) == 1


def test_scan_raw_folder_raises_on_missing_directory(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError):
        scan_raw_folder(str(missing))


def test_read_caption_for_image_returns_text_when_present(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)
    (tmp_path / "a.txt").write_text("a red square  \n")

    assert read_caption_for_image(str(image_path)) == "a red square"


def test_read_caption_for_image_returns_empty_when_missing(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)

    assert read_caption_for_image(str(image_path)) == ""


class _FakeUpload:
    """Stands in for streamlit's UploadedFile: has .name and .getbuffer()."""

    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        return self._payload


def test_save_uploaded_files_writes_files_to_a_scannable_folder(tmp_path: Path):
    source_image = tmp_path / "src.png"
    _make_image(source_image)
    uploads = [
        _FakeUpload("a.png", source_image.read_bytes()),
        _FakeUpload("a.txt", b"a red square"),
    ]

    folder = save_uploaded_files(uploads)

    assert sorted(Path(p).name for p in scan_raw_folder(folder)) == ["a.png"]
    assert read_caption_for_image(str(Path(folder) / "a.png")) == "a red square"


def test_save_uploaded_files_returns_distinct_folders_per_call():
    first = save_uploaded_files([_FakeUpload("a.txt", b"one")])
    second = save_uploaded_files([_FakeUpload("a.txt", b"two")])

    assert first != second
