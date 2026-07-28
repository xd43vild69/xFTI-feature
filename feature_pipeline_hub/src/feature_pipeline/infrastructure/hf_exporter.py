"""Export to Hugging Face `datasets` / Parquet / local LoRA trainer layout (Kohya/ComfyUI).

Placeholder for Iteración 4 (Versionado y Publicación).
"""

from feature_pipeline.domain.models import DatasetManifest


def export_to_parquet(manifest: DatasetManifest, output_path: str) -> None:
    """Write the curated dataset to a Parquet file alongside the manifest."""
    raise NotImplementedError("Implemented in Iteración 4")


def export_to_lora_layout(manifest: DatasetManifest, output_path: str) -> None:
    """Write the curated dataset to a local folder structure consumable by Kohya/ComfyUI."""
    raise NotImplementedError("Implemented in Iteración 4")
