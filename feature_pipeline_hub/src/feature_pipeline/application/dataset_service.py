"""Feature Store assembly, versioning, and snapshotting."""

import hashlib
import uuid

from feature_pipeline.application import quality_service
from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.application.image_service import compute_image_metrics
from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetManifest,
    DatasetSample,
    IngestionRun,
)
from feature_pipeline.domain.validators import validate_sample
from feature_pipeline.infrastructure.storage import read_caption_for_image, scan_raw_folder


def ingest_concept_from_folder(
    folder_path: str, concept_id: str, concept_name: str, trigger_word: str
) -> ConceptGroup:
    """Scan a raw folder, build a validated DatasetSample per image, and group them."""
    samples: list[DatasetSample] = []

    for image_path in scan_raw_folder(folder_path):
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
        samples.append(sample)

    return ConceptGroup(
        concept_id=concept_id,
        concept_name=concept_name,
        trigger_word=trigger_word,
        samples=samples,
    )


def create_ingestion_run(
    folder_path: str,
    concept_name: str,
    trigger_word: str,
    source_kind: str,
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
