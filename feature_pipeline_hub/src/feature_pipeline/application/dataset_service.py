"""Feature Store assembly, versioning, and snapshotting.

Placeholder for Iteración 4 (Versionado y Publicación). Will coordinate
domain validators + image_service + caption_service to assemble a
DatasetManifest, compute its content_hash, and hand off to hf_exporter.
"""

from feature_pipeline.domain.models import ConceptGroup, DatasetManifest


def build_manifest(concept: ConceptGroup, version: str) -> DatasetManifest:
    """Assemble a DatasetManifest snapshot for a curated ConceptGroup."""
    raise NotImplementedError("Implemented in Iteración 4")
