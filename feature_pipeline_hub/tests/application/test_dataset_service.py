from pathlib import Path

from PIL import Image

from feature_pipeline.application.dataset_service import (
    append_images_to_run,
    build_manifest,
    compute_content_hash,
    create_ingestion_run,
    ingest_concept_from_folder,
)
from feature_pipeline.domain.models import ConceptGroup, DatasetSample, ImageMetrics


def _make_image(path: Path, size=(512, 512)) -> None:
    Image.new("RGB", size, color="blue").save(path)


def test_ingest_concept_from_folder_builds_samples_with_trigger_injected(tmp_path: Path):
    _make_image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("a cat sitting")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    assert concept.concept_id == "c1"
    assert len(concept.samples) == 1
    sample = concept.samples[0]
    assert sample.caption == "sks_style, a cat sitting"
    assert sample.original_caption == "a cat sitting"
    assert sample.is_valid is True
    assert sample.validation_errors == []


def test_ingest_concept_from_folder_falls_back_to_trigger_word_when_caption_missing(
    tmp_path: Path,
):
    _make_image(tmp_path / "a.png")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    sample = concept.samples[0]
    # No .txt file next to the image: original_caption stays empty so the UI
    # can flag it, but the trigger word alone becomes a valid caption.
    assert sample.original_caption == ""
    assert sample.caption == "sks_style"
    assert sample.is_valid is True


def test_ingest_concept_from_folder_flags_low_resolution(tmp_path: Path):
    _make_image(tmp_path / "a.png", size=(128, 128))
    (tmp_path / "a.txt").write_text("a cat sitting")

    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    sample = concept.samples[0]
    assert sample.is_valid is False
    assert any("Resolution" in error for error in sample.validation_errors)


def test_ingest_concept_from_folder_empty_folder_returns_no_samples(tmp_path: Path):
    concept = ingest_concept_from_folder(
        folder_path=str(tmp_path),
        concept_id="c1",
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
    )

    assert concept.samples == []


def test_create_ingestion_run_mints_a_new_run_id_per_call(tmp_path: Path):
    _make_image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("a cat sitting")

    first = create_ingestion_run(
        folder_path=str(tmp_path),
        concept_name="cats",
        trigger_word="sks_cat",
        source_kind="folder",
    )
    second = create_ingestion_run(
        folder_path=str(tmp_path),
        concept_name="cats",
        trigger_word="sks_cat",
        source_kind="folder",
    )

    assert first.run_id != second.run_id
    assert first.source_path == str(tmp_path)
    assert first.source_kind == "folder"
    assert len(first.concept.samples) == 1


def test_create_ingestion_run_accepts_a_caller_supplied_run_id(tmp_path: Path):
    _make_image(tmp_path / "a.png")

    run = create_ingestion_run(
        folder_path=str(tmp_path),
        concept_name="cats",
        trigger_word="sks_cat",
        source_kind="upload",
        run_id="run-42",
    )

    assert run.run_id == "run-42"
    assert run.concept.trigger_word == "sks_cat"


# --- manifest -----------------------------------------------------------------


def _concept_with(samples: list[DatasetSample]) -> ConceptGroup:
    return ConceptGroup(
        concept_id="c1",
        concept_name="cyberpunk",
        trigger_word="sks_style",
        samples=samples,
    )


def _sample(
    sample_id: str,
    phash: str = "0000000000000000",
    caption: str = "sks_style, a cat",
    sharpness: float = 100.0,
    is_excluded: bool = False,
    is_duplicate: bool = False,
) -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        image_path=f"/data/{sample_id}.png",
        caption=caption,
        original_caption=caption,
        metrics=ImageMetrics(
            width=512,
            height=512,
            aspect_ratio=1.0,
            format="PNG",
            phash=phash,
            sharpness=sharpness,
        ),
        is_excluded=is_excluded,
        is_duplicate=is_duplicate,
    )


def test_content_hash_ignores_the_order_samples_arrive_in():
    a, b = _sample("a", phash="1111111111111111"), _sample("b", phash="2222222222222222")

    assert compute_content_hash([a, b]) == compute_content_hash([b, a])


def test_content_hash_changes_when_a_caption_is_edited():
    before = compute_content_hash([_sample("a", caption="sks_style, a cat")])
    after = compute_content_hash([_sample("a", caption="sks_style, a dog")])

    assert before != after


def test_content_hash_changes_when_a_sample_is_excluded():
    samples = [_sample("a"), _sample("b", phash="2222222222222222")]
    fewer = [_sample("a"), _sample("b", phash="2222222222222222", is_excluded=True)]

    assert compute_content_hash(samples) != compute_content_hash(fewer)


def test_content_hash_ignores_samples_that_were_already_excluded():
    kept = [_sample("a")]
    kept_plus_dropped = [_sample("a"), _sample("b", phash="9999999999999999", is_excluded=True)]

    assert compute_content_hash(kept) == compute_content_hash(kept_plus_dropped)


def test_build_manifest_snapshots_the_active_samples():
    concept = _concept_with(
        [
            _sample("a", phash="1111111111111111", sharpness=100.0),
            _sample("b", phash="2222222222222222", sharpness=300.0, is_duplicate=True),
            _sample("c", phash="3333333333333333", is_excluded=True),
        ]
    )

    manifest = build_manifest(concept, version="v1", dataset_name="cyberpunk_export")

    assert manifest.dataset_name == "cyberpunk_export"
    assert manifest.version == "v1"
    assert manifest.concept_name == "cyberpunk"
    assert manifest.total_samples == 2
    assert manifest.duplicate_count == 1
    assert manifest.orientation_distribution == {"square": 2}
    assert manifest.median_sharpness == 200.0
    assert manifest.aspect_ratio_distribution == {"1:1": 2}
    assert len(manifest.content_hash) == 64


def test_two_manifests_of_the_same_content_share_a_hash():
    concept = _concept_with([_sample("a")])

    first = build_manifest(concept, version="v1", dataset_name="d")
    second = build_manifest(concept, version="v2", dataset_name="d")

    assert first.content_hash == second.content_hash
    assert first.version != second.version


def _run_with(tmp_path: Path, filenames: list[str], color: str = "blue"):
    for name in filenames:
        Image.new("RGB", (512, 512), color=color).save(tmp_path / name)
    return create_ingestion_run(
        folder_path=str(tmp_path),
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
        source_kind="folder",
    )


def test_append_images_keeps_the_existing_samples_and_their_curation(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run = _run_with(source, ["a.png"])
    kept = run.concept.samples[0]
    kept.caption = "sks_style, a caption edited by hand"
    kept.is_excluded = True

    extra = tmp_path / "extra.png"
    Image.new("RGB", (512, 512), color="red").save(extra)
    added = append_images_to_run(run, [str(extra)])

    assert len(added) == 1
    assert len(run.concept.samples) == 2
    # The curated sample survives untouched: same id, same edits.
    assert run.concept.samples[0].sample_id == kept.sample_id
    assert run.concept.samples[0].caption == "sks_style, a caption edited by hand"
    assert run.concept.samples[0].is_excluded is True


def test_append_images_injects_the_runs_trigger_word(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run = _run_with(source, ["a.png"])

    extra = tmp_path / "extra.png"
    Image.new("RGB", (512, 512), color="red").save(extra)
    (tmp_path / "extra.txt").write_text("a red square")
    added = append_images_to_run(run, [str(extra)])

    assert added[0].caption == "sks_style, a red square"
    assert added[0].is_valid is True


def test_append_images_flags_an_image_already_in_the_dataset(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run = _run_with(source, ["a.png"])

    # A copy of the same picture under a different name: a re-upload of what is there.
    extra = tmp_path / "copy.png"
    Image.new("RGB", (512, 512), color="blue").save(extra)
    added = append_images_to_run(run, [str(extra)])

    assert added[0].is_duplicate is True


def test_append_images_skips_paths_the_run_already_holds(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run = _run_with(source, ["a.png"])
    existing_path = run.concept.samples[0].image_path

    added = append_images_to_run(run, [existing_path])

    assert added == []
    assert len(run.concept.samples) == 1


def test_append_images_changes_the_content_hash(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    run = _run_with(source, ["a.png"])
    before = compute_content_hash(run.concept.samples)

    extra = tmp_path / "extra.png"
    Image.new("RGB", (512, 512), color="red").save(extra)
    append_images_to_run(run, [str(extra)])

    assert compute_content_hash(run.concept.samples) != before
