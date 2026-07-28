"""Materialize a curated run as a flat image+caption folder for training.

The training scripts (pre-cache, train) expect exactly this shape — a folder of
`<name>.png|.jpg + <name>.txt` pairs, no manifest, no subfolders — so exporting is
a pure filesystem write, not a new format.
"""

from dataclasses import dataclass, field
from pathlib import Path

from feature_pipeline.domain.models import IngestionRun
from feature_pipeline.infrastructure import storage


@dataclass(frozen=True)
class ExportResult:
    destination: str
    exported_count: int
    skipped_excluded: int
    renamed_for_collision: list[str] = field(default_factory=list)


def export_active_samples(run: IngestionRun, destination_name: str) -> ExportResult:
    """Export every non-excluded sample of `run` to training_runtime/datasets/<destination_name>/.

    Wipes the destination first — it's exclusively ours, so a fresh export is
    free to replace whatever a previous export of the same name left behind.
    """
    storage.clear_training_dataset_dir(destination_name)
    dest_dir = storage.training_dataset_dir(destination_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    active = [s for s in run.concept.samples if not s.is_excluded]
    skipped = len(run.concept.samples) - len(active)

    used_names: set[str] = set()
    renamed: list[str] = []
    exported = 0

    for sample in active:
        source = Path(sample.image_path)
        target_name = source.name
        if target_name in used_names:
            # Two samples in the same run resolving to the same basename — rare
            # (filenames are unique within a single scanned/uploaded folder), but
            # silently overwriting one would lose an image, so disambiguate.
            target_name = f"{source.stem}__{sample.sample_id[:8]}{source.suffix}"
            renamed.append(target_name)
        used_names.add(target_name)

        target_path = dest_dir / target_name
        target_path.write_bytes(source.read_bytes())
        storage.write_caption_sidecar(str(target_path), sample.caption, keep_backup=False)
        exported += 1

    return ExportResult(
        destination=str(dest_dir),
        exported_count=exported,
        skipped_excluded=skipped,
        renamed_for_collision=renamed,
    )
