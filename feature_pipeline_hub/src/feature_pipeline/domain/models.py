"""Pydantic schemas for the Feature Pipeline domain: samples, concepts, and manifests."""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

SourceKind = Literal["folder", "upload"]

# `concept_name` becomes a directory name under training_runtime/datasets (see
# training_service.dataset_dir_for), so it has to stay a single, harmless path
# segment. Validated rather than sanitised: silently rewriting a name the user
# typed would point the training job at a folder they never asked for.
PathSafeName = Annotated[str, Field(min_length=1)]


class ImageMetrics(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspect_ratio: float = Field(gt=0)
    format: str
    phash: str
    # Empty/zero for runs ingested before these were recorded.
    dhash: str = ""
    colorhash: str = ""
    # Variance of the Laplacian. Comparable only within one run — see
    # image_service.compute_sharpness.
    sharpness: float = Field(default=0.0, ge=0)


class DatasetSample(BaseModel):
    sample_id: str
    image_path: str
    caption: str
    original_caption: str
    metrics: ImageMetrics
    is_duplicate: bool = False
    is_excluded: bool = False  # kept on disk, left out of the curated dataset
    is_flagged: bool = False  # marked by the user for a second look
    is_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)


class DuplicateCluster(BaseModel):
    """A group of near-identical images found by perceptual hashing.

    `kept` is the sample proposed to survive; `duplicates` are the rest, each with
    its perceptual distance to `kept`.
    """

    kept: DatasetSample
    duplicates: list[tuple[DatasetSample, int]] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return 1 + len(self.duplicates)


class ConceptGroup(BaseModel):
    concept_id: str
    concept_name: PathSafeName
    trigger_word: str
    samples: list[DatasetSample] = Field(default_factory=list)

    @field_validator("concept_name")
    @classmethod
    def _reject_path_traversal(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError(
                "concept_name is used as a folder name and cannot contain path separators"
            )
        return value


class IngestionRun(BaseModel):
    """One ingestion of a raw folder: what was imported, from where, and when.

    Several runs can coexist (the same concept re-scanned, or different concepts),
    so `run_id` is what the UI selects on, not `concept_id`.
    """

    run_id: str
    source_path: str
    source_kind: SourceKind
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    concept: ConceptGroup


class IngestionRunSummary(BaseModel):
    """Listing row for the run selector: enough to label a run without loading its samples."""

    run_id: str
    concept_name: str
    trigger_word: str
    source_kind: SourceKind
    created_at: datetime
    sample_count: int


class DatasetManifest(BaseModel):
    """A measurable snapshot of what was exported, so two versions can be compared.

    Without this the quality metrics are recomputed and discarded on every rerun,
    which makes "is v2 better than v1?" an unanswerable question.
    """

    dataset_name: str
    version: str
    concept_name: str
    trigger_word: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_samples: int
    duplicate_count: int
    aspect_ratio_distribution: dict[str, int]
    orientation_distribution: dict[str, int] = Field(default_factory=dict)
    median_sharpness: float = 0.0
    caption_word_stats: dict[str, float] = Field(default_factory=dict)
    content_hash: str
