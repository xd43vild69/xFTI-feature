from pathlib import Path

from PIL import Image

from feature_pipeline.application.dataset_service import (
    append_images_to_run,
    build_manifest,
    clone_ingestion_run,
    compute_content_hash,
    create_ingestion_run,
    ingest_concept_from_folder,
    normalize_samples,
    oversized_samples,
    revalidate_samples,
    rotate_sample,
)
from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetSample,
    ImageMetrics,
    IngestionRun,
)
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.ingestion_repository import (
    load_ingestion_run,
    save_ingestion_run,
)


def _make_image(path: Path, size=(512, 512)) -> None:
    """A gradient rather than a flat fill: a solid colour has no detail to resample."""
    width, height = size
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (x % 256, y % 256, (x * y) % 256)
    img.save(path)


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


def test_revalidate_samples_clears_a_verdict_stale_from_an_older_rule():
    # Simulates a sample imported under an older, stricter resolution rule: stored
    # as invalid even though it now passes (one side >= 1024).
    sample = DatasetSample(
        sample_id="s1",
        image_path="a.png",
        caption="sks_style, a cat",
        original_caption="a cat",
        metrics=ImageMetrics(
            width=406, height=1024, aspect_ratio=406 / 1024, format="PNG", phash="a" * 16
        ),
        is_valid=False,
        validation_errors=["Resolution 406x1024 is below the minimum 512x512"],
    )

    changed = revalidate_samples([sample])

    assert changed == 1
    assert sample.is_valid is True
    assert sample.validation_errors == []


def test_revalidate_samples_leaves_an_already_correct_verdict_untouched():
    sample = DatasetSample(
        sample_id="s1",
        image_path="a.png",
        caption="sks_style, a cat",
        original_caption="a cat",
        metrics=ImageMetrics(
            width=512, height=512, aspect_ratio=1.0, format="PNG", phash="a" * 16
        ),
    )

    assert revalidate_samples([sample]) == 0


# --- clone --------------------------------------------------------------------


def _run_to_clone(tmp_path: Path) -> IngestionRun:
    source_folder = tmp_path / "source"
    source_folder.mkdir()
    for name in ("a", "b"):
        _make_image(source_folder / f"{name}.png")
        (source_folder / f"{name}.txt").write_text("a cat sitting")

    return create_ingestion_run(
        folder_path=str(source_folder),
        concept_name="cats",
        trigger_word="sks_cat",
        source_kind="folder",
    )


def test_clone_copies_images_and_captions_into_a_folder_of_its_own(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    destination = tmp_path / "clone"

    clone = clone_ingestion_run(
        source=source,
        destination=str(destination),
        concept_name="cats_v2",
        trigger_word="sks_cat2",
    )

    assert clone.run_id != source.run_id
    assert clone.concept.concept_id != source.concept.concept_id
    assert clone.source_kind == "clone"
    assert len(clone.concept.samples) == 2

    for sample in clone.concept.samples:
        # Renamed under the new concept's slug, in the new run's own folder.
        assert Path(sample.image_path).parent == destination
        assert Path(sample.image_path).stem.startswith("cats_v2_")
        assert Path(sample.image_path).exists()
        # The sidecar is written too, so the folder stands on its own for export.
        assert Path(sample.image_path).with_suffix(".txt").read_text() == sample.caption

    # Source is untouched.
    assert all(Path(s.image_path).exists() for s in source.concept.samples)


def test_clone_retriggers_captions_without_leaving_the_old_trigger_behind(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    assert source.concept.samples[0].caption == "sks_cat, a cat sitting"

    clone = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone"),
        concept_name="cats_v2",
        trigger_word="sks_cat2",
    )

    sample = clone.concept.samples[0]
    assert sample.caption == "sks_cat2, a cat sitting"
    # A clone is a fresh import, so what it was cloned with is its baseline.
    assert sample.original_caption == "sks_cat2, a cat sitting"


def test_clone_starts_from_the_edited_caption_not_the_sidecar(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    source.concept.samples[0].caption = "sks_cat, an edited description"

    clone = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone"),
        concept_name="cats_v2",
        trigger_word="sks_cat",
    )

    captions = sorted(s.caption for s in clone.concept.samples)
    assert captions == ["sks_cat, a cat sitting", "sks_cat, an edited description"]


def test_clone_drops_curation_verdicts_so_the_copy_starts_clean(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    source.concept.samples[0].is_duplicate = True
    source.concept.samples[0].is_flagged = True

    clone = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone"),
        concept_name="cats_v2",
        trigger_word="sks_cat2",
    )

    assert all(not s.is_duplicate and not s.is_flagged for s in clone.concept.samples)
    assert {s.sample_id for s in clone.concept.samples}.isdisjoint(
        {s.sample_id for s in source.concept.samples}
    )
    # Byte-identical files: metrics carry over rather than being recomputed.
    assert clone.concept.samples[0].metrics.phash == source.concept.samples[0].metrics.phash


def test_clone_leaves_excluded_samples_behind_unless_asked(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    source.concept.samples[0].is_excluded = True

    without = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone_a"),
        concept_name="cats_v2",
        trigger_word="sks_cat2",
    )
    with_excluded = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone_b"),
        concept_name="cats_v3",
        trigger_word="sks_cat3",
        include_excluded=True,
    )

    assert len(without.concept.samples) == 1
    assert len(with_excluded.concept.samples) == 2
    # Brought along, but no longer excluded — the clone is its own dataset.
    assert all(not s.is_excluded for s in with_excluded.concept.samples)


def test_clone_skips_source_images_that_are_gone_from_disk(tmp_path: Path):
    source = _run_to_clone(tmp_path)
    Path(source.concept.samples[0].image_path).unlink()

    clone = clone_ingestion_run(
        source=source,
        destination=str(tmp_path / "clone"),
        concept_name="cats_v2",
        trigger_word="sks_cat2",
    )

    assert len(clone.concept.samples) == 1


def _oversized_run(tmp_path: Path, size=(2400, 1600)) -> IngestionRun:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_image(source_dir / "big.png", size)
    (source_dir / "big.txt").write_text("a wide shot")
    return create_ingestion_run(
        folder_path=str(source_dir),
        concept_name="cyberpunk_style",
        trigger_word="sks_style",
        source_kind="folder",
    )


def test_oversized_samples_reads_stored_metrics(tmp_path: Path):
    run = _oversized_run(tmp_path)

    assert [s.sample_id for s in oversized_samples(run.concept.samples)] == [
        run.concept.samples[0].sample_id
    ]
    assert oversized_samples(run.concept.samples, max_side=4000) == []


def test_normalize_samples_repoints_the_sample_and_recomputes_its_metrics(tmp_path: Path):
    run = _oversized_run(tmp_path)
    sample = run.concept.samples[0]
    original_path = sample.image_path
    original_phash = sample.metrics.phash
    destination = tmp_path / "normalized"

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        outcomes = normalize_samples(conn, run.concept.samples, str(destination))
        reloaded = load_ingestion_run(conn, run.run_id)
    finally:
        conn.close()

    assert len(outcomes) == 1
    assert outcomes[0].source_size == (2400, 1600)
    assert outcomes[0].new_size == (1024, 683)
    assert outcomes[0].error == ""

    assert sample.image_path == str(destination / "big.png")
    assert (sample.metrics.width, sample.metrics.height) == (1024, 683)
    # The hashes describe the new pixels, not the ones that are no longer in use.
    assert sample.metrics.phash != original_phash or sample.metrics.sharpness > 0

    assert reloaded is not None
    stored = reloaded.concept.samples[0]
    assert stored.image_path == str(destination / "big.png")
    assert (stored.metrics.width, stored.metrics.height) == (1024, 683)
    assert stored.metrics.phash == sample.metrics.phash
    # Curation and captions survive the resize untouched.
    assert stored.caption == sample.caption
    assert Path(original_path).exists()


def test_normalize_samples_skips_images_already_within_the_limit(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(800, 600))

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        outcomes = normalize_samples(conn, run.concept.samples, str(tmp_path / "normalized"))
    finally:
        conn.close()

    assert outcomes == []
    assert not (tmp_path / "normalized").exists()


def test_normalize_samples_reports_a_broken_file_without_touching_the_sample(tmp_path: Path):
    run = _oversized_run(tmp_path)
    sample = run.concept.samples[0]
    Path(sample.image_path).write_bytes(b"not an image")

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        outcomes = normalize_samples(conn, run.concept.samples, str(tmp_path / "normalized"))
    finally:
        conn.close()

    assert len(outcomes) == 1
    assert outcomes[0].error
    assert sample.metrics.width == 2400


def test_rotate_sample_swaps_dimensions_and_records_provenance(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(800, 400))
    sample = run.concept.samples[0]
    original_path = sample.image_path
    derived = tmp_path / "derived"

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        outcome = rotate_sample(conn, sample, str(derived), quarter_turns=1)
        reloaded = load_ingestion_run(conn, run.run_id)
    finally:
        conn.close()

    assert outcome.error == ""
    assert outcome.new_size == (400, 800)
    assert sample.rotation_degrees == 90
    assert sample.source_image_path == original_path
    assert sample.image_path == str(derived / "big.png")
    assert (sample.metrics.width, sample.metrics.height) == (400, 800)
    assert Path(original_path).exists()

    assert reloaded is not None
    stored = reloaded.concept.samples[0]
    assert stored.rotation_degrees == 90
    assert stored.source_image_path == original_path
    assert stored.derived_max_side is None


def test_rotate_sample_accumulates_the_angle_from_the_original(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(800, 400))
    sample = run.concept.samples[0]
    derived = tmp_path / "derived"

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        for _ in range(3):
            outcome = rotate_sample(conn, sample, str(derived), quarter_turns=1)
    finally:
        conn.close()

    assert sample.rotation_degrees == 270
    # Every turn re-derives from the untouched 800x400 original, never from the
    # previous derivation — so the reported source size is still the original's.
    assert outcome.source_size == (800, 400)
    assert (sample.metrics.width, sample.metrics.height) == (400, 800)


def test_four_quarter_turns_come_back_to_the_original_pixels(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(800, 400))
    sample = run.concept.samples[0]
    original_phash = sample.metrics.phash

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        for _ in range(4):
            rotate_sample(conn, sample, str(tmp_path / "derived"), quarter_turns=1)
    finally:
        conn.close()

    assert sample.rotation_degrees == 0
    assert (sample.metrics.width, sample.metrics.height) == (800, 400)
    assert sample.metrics.phash == original_phash


def test_rotate_sample_keeps_the_downscale_it_already_had(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(2400, 1600))
    sample = run.concept.samples[0]
    derived = tmp_path / "derived"

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        normalize_samples(conn, [sample], str(derived))
        assert (sample.metrics.width, sample.metrics.height) == (1024, 683)

        rotate_sample(conn, sample, str(derived), quarter_turns=1)
    finally:
        conn.close()

    # Rotated *and* still within the limit: the 1600x2400 turn is re-fitted to 1024
    # on its long axis rather than restored to full resolution.
    assert (sample.metrics.width, sample.metrics.height) == (683, 1024)
    assert sample.derived_max_side == 1024
    assert sample.rotation_degrees == 90


def test_rotate_sample_preserves_caption_and_curation(tmp_path: Path):
    run = _oversized_run(tmp_path, size=(800, 400))
    sample = run.concept.samples[0]
    sample.is_excluded = True
    sample.is_flagged = True

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        rotate_sample(conn, sample, str(tmp_path / "derived"), quarter_turns=-1)
        reloaded = load_ingestion_run(conn, run.run_id)
    finally:
        conn.close()

    assert sample.rotation_degrees == 270
    assert reloaded is not None
    stored = reloaded.concept.samples[0]
    assert stored.caption == "sks_style, a wide shot"
    assert stored.is_excluded is True
    assert stored.is_flagged is True


def test_normalize_samples_records_the_original_as_the_source(tmp_path: Path):
    run = _oversized_run(tmp_path)
    sample = run.concept.samples[0]
    original_path = sample.image_path

    conn = get_connection(str(tmp_path / "test.db"))
    try:
        save_ingestion_run(conn, run)
        normalize_samples(conn, [sample], str(tmp_path / "derived"))
    finally:
        conn.close()

    assert sample.source_image_path == original_path
    assert sample.derived_max_side == 1024
    assert sample.rotation_degrees == 0
