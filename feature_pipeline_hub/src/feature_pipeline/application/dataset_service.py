"""Feature Store assembly, versioning, and snapshotting."""

import hashlib
import sqlite3
import uuid
from pathlib import Path

from feature_pipeline.application import quality_service
from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.application.image_service import compute_image_metrics
from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetManifest,
    DatasetSample,
    IngestionRun,
    SourceKind,
)
from feature_pipeline.domain.validators import validate_sample
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure.storage import read_caption_for_image, scan_raw_folder


def build_sample(image_path: str, trigger_word: str) -> DatasetSample:
    """Turn one image on disk into a captioned, measured, validated sample."""
    original_caption = read_caption_for_image(image_path)
    caption = inject_trigger_word(original_caption, trigger_word)
    metrics = compute_image_metrics(image_path)

    sample = DatasetSample(
        sample_id=str(uuid.uuid4()),
        image_path=image_path,
        caption=caption,
        original_caption=original_caption,
        metrics=metrics,
    )
    errors = validate_sample(sample)
    sample.is_valid = not errors
    sample.validation_errors = errors
    return sample


def ingest_concept_from_folder(
    folder_path: str, concept_id: str, concept_name: str, trigger_word: str
) -> ConceptGroup:
    """Scan a raw folder, build a validated DatasetSample per image, and group them."""
    return ConceptGroup(
        concept_id=concept_id,
        concept_name=concept_name,
        trigger_word=trigger_word,
        samples=[build_sample(path, trigger_word) for path in scan_raw_folder(folder_path)],
    )


def create_ingestion_run(
    folder_path: str,
    concept_name: str,
    trigger_word: str,
    source_kind: SourceKind,
    run_id: str | None = None,
    concept_id: str | None = None,
) -> IngestionRun:
    """Ingest a folder into a standalone, identifiable run.

    Each call mints a fresh `run_id` so re-scanning a concept adds a run instead of
    overwriting the previous one.
    """
    run_id = run_id or str(uuid.uuid4())
    concept = ingest_concept_from_folder(
        folder_path=folder_path,
        concept_id=concept_id or str(uuid.uuid4()),
        concept_name=concept_name,
        trigger_word=trigger_word,
    )

    return IngestionRun(
        run_id=run_id,
        source_path=folder_path,
        source_kind=source_kind,
        concept=concept,
    )


def append_images_to_run(
    run: IngestionRun,
    image_paths: list[str],
    duplicate_threshold: int = quality_service.DEFAULT_PHASH_THRESHOLD,
) -> list[DatasetSample]:
    """Add images to an already-curated run, in place, and return the ones added.

    The counterpart to `create_ingestion_run` for the "I forgot one" case. Re-scanning
    the concept would mint a whole new run, and with it new sample ids, discarding the
    exclusions, edited captions and duplicate verdicts the run already carries — so a
    single missing image would cost re-curating the dataset.

    Paths the run already holds are skipped rather than added twice, which keeps the
    operation safe to repeat with the same files.

    Each new image is compared against what the run already holds, including earlier
    images from this same batch, so an accidental re-upload arrives already flagged.
    That is O(new x existing); the quality step's O(n^2) sweep stays the authority on
    the dataset as a whole, and re-running it can still cluster these differently.
    """
    already_present = {_resolved(s.image_path) for s in run.concept.samples}
    added: list[DatasetSample] = []

    for image_path in image_paths:
        if _resolved(image_path) in already_present:
            continue

        sample = build_sample(image_path, run.concept.trigger_word)
        sample.is_duplicate = any(
            quality_service.perceptual_distance(sample, other) <= duplicate_threshold
            for other in run.concept.samples
            if not other.is_excluded
        )

        run.concept.samples.append(sample)
        already_present.add(_resolved(image_path))
        added.append(sample)

    return added


def revalidate_samples(samples: list[DatasetSample]) -> int:
    """Recompute is_valid/validation_errors in place, and return how many changed.

    Validation runs once, at ingestion, and its verdict is what's stored — so a
    sample imported before a validation rule changed keeps the old rule's verdict
    until this runs. Meant as a one-off correction after a rule change, not
    something called on every render.
    """
    changed = 0
    for sample in samples:
        errors = validate_sample(sample)
        is_valid = not errors
        if is_valid != sample.is_valid or errors != sample.validation_errors:
            sample.is_valid = is_valid
            sample.validation_errors = errors
            changed += 1
    return changed


def revalidate_all_runs(conn: sqlite3.Connection) -> int:
    """Re-check every sample of every stored run against the current validation rules.

    A one-off correction for a validation rule change: samples keep the verdict
    computed when they were imported, so a rule made stricter or looser afterward
    otherwise never reaches datasets that already existed. Returns how many
    samples' verdicts actually changed.
    """
    total_changed = 0
    for summary in repo.list_ingestion_runs(conn):
        run = repo.load_ingestion_run(conn, summary.run_id)
        if run is None:
            continue
        changed = revalidate_samples(run.concept.samples)
        if changed:
            repo.save_ingestion_run(conn, run)
            total_changed += changed
    return total_changed


def append_images(
    conn: sqlite3.Connection,
    run: IngestionRun,
    image_paths: list[str],
    duplicate_threshold: int = quality_service.DEFAULT_PHASH_THRESHOLD,
) -> list[DatasetSample]:
    """`append_images_to_run`, persisted: adds images to `run` in place and saves it.

    Image files must already exist on disk at `image_paths` (a caller that receives
    uploads or raw bytes is responsible for writing them first — this only assembles
    samples and persists the run).
    """
    added = append_images_to_run(run, image_paths, duplicate_threshold)
    if added:
        repo.save_ingestion_run(conn, run)
    return added


def _resolved(image_path: str) -> str:
    """Absolute, symlink-free form of a path, for comparing two references to a file."""
    return str(Path(image_path).resolve())


def compute_content_hash(samples: list[DatasetSample]) -> str:
    """Fingerprint of what a training run would actually see.

    Built from the (pHash, caption) pair of every active sample, sorted so the hash
    does not depend on scan order. That covers both ways a dataset stops being the
    same dataset — an image entering or leaving, and a caption being edited — while
    reading nothing back off disk. Two exports of visually identical but re-encoded
    files hash the same, which is the intended tolerance.
    """
    active = sorted(
        (s.metrics.phash, s.caption) for s in samples if not s.is_excluded
    )
    digest = hashlib.sha256()
    for phash, caption in active:
        digest.update(phash.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(caption.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def build_manifest(concept: ConceptGroup, version: str, dataset_name: str) -> DatasetManifest:
    """Assemble a comparable snapshot of a curated ConceptGroup.

    `dataset_name` is the export destination rather than the concept name: the same
    concept can be exported under different folder names, and the manifest records
    what was written where.
    """
    samples = concept.samples
    active = [s for s in samples if not s.is_excluded]

    return DatasetManifest(
        dataset_name=dataset_name,
        version=version,
        concept_name=concept.concept_name,
        trigger_word=concept.trigger_word,
        total_samples=len(active),
        # The stored flag, like the context bar uses — not a fresh O(n²) clustering.
        duplicate_count=sum(1 for s in active if s.is_duplicate),
        aspect_ratio_distribution=quality_service.aspect_ratio_distribution(samples),
        orientation_distribution=quality_service.orientation_distribution(samples),
        median_sharpness=quality_service.median_sharpness(samples),
        caption_word_stats=quality_service.caption_length_stats(samples),
        content_hash=compute_content_hash(samples),
    )
