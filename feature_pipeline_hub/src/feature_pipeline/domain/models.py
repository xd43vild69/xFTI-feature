"""Pydantic schemas for the Feature Pipeline domain: samples, concepts, and manifests."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ImageMetrics(BaseModel):
    width: int
    height: int
    aspect_ratio: float
    format: str
    phash: str


class DatasetSample(BaseModel):
    sample_id: str
    image_path: str
    caption: str
    original_caption: str
    metrics: ImageMetrics
    is_duplicate: bool = False
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)


class ConceptGroup(BaseModel):
    concept_id: str
    concept_name: str
    trigger_word: str
    samples: list[DatasetSample] = Field(default_factory=list)


class IngestionRun(BaseModel):
    """One ingestion of a raw folder: what was imported, from where, and when.

    Several runs can coexist (the same concept re-scanned, or different concepts),
    so `run_id` is what the UI selects on, not `concept_id`.
    """

    run_id: str
    source_path: str
    source_kind: str  # "folder" | "upload"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    concept: ConceptGroup


class IngestionRunSummary(BaseModel):
    """Listing row for the run selector: enough to label a run without loading its samples."""

    run_id: str
    concept_name: str
    trigger_word: str
    source_kind: str
    created_at: datetime
    sample_count: int


class DatasetManifest(BaseModel):
    dataset_name: str
    version: str
    concept_name: str
    trigger_word: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_samples: int
    duplicate_count: int
    aspect_ratio_distribution: dict[str, int]
    content_hash: str
