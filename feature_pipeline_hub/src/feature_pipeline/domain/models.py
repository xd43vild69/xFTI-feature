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
