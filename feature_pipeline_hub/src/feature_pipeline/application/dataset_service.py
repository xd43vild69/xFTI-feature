"""Feature Store assembly, versioning, and snapshotting."""

import uuid

from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.application.image_service import compute_image_metrics
from feature_pipeline.domain.models import ConceptGroup, DatasetManifest, DatasetSample
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


def build_manifest(concept: ConceptGroup, version: str) -> DatasetManifest:
    """Assemble a DatasetManifest snapshot for a curated ConceptGroup.

    Placeholder for Iteración 4 (Versionado y Publicación): will compute
    content_hash and aspect_ratio_distribution, then hand off to hf_exporter.
    """
    raise NotImplementedError("Implemented in Iteración 4")
