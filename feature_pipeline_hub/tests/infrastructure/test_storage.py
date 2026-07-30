from pathlib import Path

import pytest
from PIL import Image

from feature_pipeline.infrastructure.storage import (
    append_uploaded_files,
    clear_training_dataset_dir,
    delete_managed_folder,
    raw_data_dir,
    read_caption_for_image,
    scan_caption_files,
    run_upload_dir,
    save_uploaded_files,
    scan_raw_folder,
    training_dataset_dir,
    write_caption_sidecar,
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


def test_save_uploaded_files_honours_an_explicit_destination(tmp_path: Path):
    source_image = tmp_path / "src.png"
    _make_image(source_image)
    destination = tmp_path / "runs" / "run-1"

    folder = save_uploaded_files(
        [_FakeUpload("a.png", source_image.read_bytes())], destination=str(destination)
    )

    assert Path(folder) == destination
    assert (destination / "a.png").exists()


def test_delete_managed_folder_removes_folders_under_data_raw(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_DATA_DIR", str(tmp_path))
    run_folder = Path(run_upload_dir("run-1"))
    run_folder.mkdir(parents=True)
    (run_folder / "a.png").write_bytes(b"x")

    assert delete_managed_folder(str(run_folder)) is True
    assert not run_folder.exists()


def test_delete_managed_folder_refuses_folders_outside_data_raw(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_DATA_DIR", str(tmp_path / "managed"))
    user_folder = tmp_path / "my_own_images"
    user_folder.mkdir()
    _make_image(user_folder / "a.png")

    assert delete_managed_folder(str(user_folder)) is False
    assert (user_folder / "a.png").exists()


def test_delete_managed_folder_refuses_the_raw_root_itself(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_DATA_DIR", str(tmp_path))
    raw_root = raw_data_dir()

    assert delete_managed_folder(str(raw_root)) is False
    assert raw_root.exists()


def test_write_caption_sidecar_creates_the_txt_next_to_the_image(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)

    written = write_caption_sidecar(str(image_path), "sks_style, a red square")

    assert Path(written) == tmp_path / "a.txt"
    assert read_caption_for_image(str(image_path)) == "sks_style, a red square"


def test_write_caption_sidecar_backs_up_the_previous_caption_once(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)
    (tmp_path / "a.txt").write_text("the original caption")

    write_caption_sidecar(str(image_path), "first ai caption")
    write_caption_sidecar(str(image_path), "second ai caption")

    # The backup keeps the pre-AI text, not the previous AI attempt.
    assert (tmp_path / "a.txt.bak").read_text() == "the original caption"
    assert read_caption_for_image(str(image_path)) == "second ai caption"


def test_write_caption_sidecar_can_skip_the_backup(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)
    (tmp_path / "a.txt").write_text("the original caption")

    write_caption_sidecar(str(image_path), "ai caption", keep_backup=False)

    assert not (tmp_path / "a.txt.bak").exists()


def test_write_caption_sidecar_leaves_no_temp_file(tmp_path: Path):
    image_path = tmp_path / "a.png"
    _make_image(image_path)

    write_caption_sidecar(str(image_path), "ai caption")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "a.txt"]


def test_training_dataset_dir_is_scoped_under_the_runtime_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))

    assert training_dataset_dir("my_concept") == tmp_path / "datasets" / "my_concept"


def test_clear_training_dataset_dir_removes_existing_content(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))
    target = training_dataset_dir("my_concept")
    target.mkdir(parents=True)
    (target / "a.png").write_bytes(b"stale")
    (target / "curation_report.json").write_text("{}")

    clear_training_dataset_dir("my_concept")

    assert not target.exists()


def test_clear_training_dataset_dir_is_a_no_op_when_nothing_exists(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FTI_TRAINING_RUNTIME_DIR", str(tmp_path))

    clear_training_dataset_dir("never_exported")  # must not raise


def test_scan_caption_files_finds_txt_sidecars(tmp_path):
    (tmp_path / "a.png").write_bytes(b"pixels")
    (tmp_path / "a.txt").write_text("a cat")
    (tmp_path / "orphan.txt").write_text("no image for this one")
    (tmp_path / "a.txt.bak").write_text("older caption")

    found = [Path(p).name for p in scan_caption_files(str(tmp_path))]

    assert found == ["a.txt", "orphan.txt"]


def test_scan_caption_files_of_a_missing_folder_is_empty(tmp_path):
    """Used for reporting on a run whose source folder may since have been moved."""
    assert scan_caption_files(str(tmp_path / "gone")) == []


def test_append_uploaded_files_keeps_existing_files_intact(tmp_path: Path):
    source_image = tmp_path / "src.png"
    _make_image(source_image)
    folder = tmp_path / "run-1"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"original bytes")

    written = append_uploaded_files(
        [_FakeUpload("a.png", source_image.read_bytes())], str(folder)
    )

    assert (folder / "a.png").read_bytes() == b"original bytes"
    assert [Path(p).name for p in written] == ["a-1.png"]


def test_append_uploaded_files_keeps_an_image_paired_with_its_caption(tmp_path: Path):
    source_image = tmp_path / "src.png"
    _make_image(source_image)
    folder = tmp_path / "run-1"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"original bytes")
    (folder / "a.txt").write_text("the original caption")

    append_uploaded_files(
        [
            _FakeUpload("a.png", source_image.read_bytes()),
            _FakeUpload("a.txt", b"the added caption"),
        ],
        str(folder),
    )

    # Both took the same suffix, so the sidecar still resolves from the image.
    assert read_caption_for_image(str(folder / "a-1.png")) == "the added caption"
    assert read_caption_for_image(str(folder / "a.png")) == "the original caption"


def test_append_uploaded_files_creates_the_folder_when_missing(tmp_path: Path):
    written = append_uploaded_files([_FakeUpload("a.txt", b"one")], str(tmp_path / "new"))

    assert Path(written[0]) == tmp_path / "new" / "a.txt"
