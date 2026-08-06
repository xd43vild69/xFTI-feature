"""Materialize a curated run — or an experiment branch of one — as a flat
image+caption folder for training.

The training scripts (pre-cache, train) expect exactly this shape — a folder of
`<name>.png|.jpg + <name>.txt` pairs, no manifest, no subfolders — so exporting is
a pure filesystem write, not a new format.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from feature_pipeline.domain.curation_report import (
    CurationReport,
    Tier,
    WeightProfile,
    build_curation_report,
    is_effective,
)
from feature_pipeline.domain.models import DatasetSample, IngestionRun
from feature_pipeline.infrastructure import curation_report_files, storage


@dataclass(frozen=True)
class ExportResult:
    destination: str
    exported_count: int
    skipped_excluded: int
    renamed_for_collision: list[str] = field(default_factory=list)
    # sample_id -> exported filename (with extension). Empty for callers that never
    # asked for it (export_active_samples), populated for export_branch_dataset,
    # which needs it to translate a UI's per-sample tier choices into the
    # per-stem keys curation_report.json actually uses — the disambiguated name on
    # a basename collision is only known here, where the collision was resolved.
    filenames_by_sample_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchExportResult:
    """What `export_branch_dataset` produced, for the caller to validate before launch."""

    export: ExportResult
    stems: list[str]
    report_path: Path | None
    weights_are_effective: bool


def _write_flat_dataset(samples: Sequence[DatasetSample], destination_name: str) -> ExportResult:
    """Write `samples` into training_runtime/datasets/<destination_name>/.

    Wipes the destination first — it's exclusively ours, so a fresh export is
    free to replace whatever a previous export of the same name left behind.
    Shared by both `export_active_samples` and `export_branch_dataset`; the only
    difference between the two is which samples they hand in and what they do
    with the destination afterward.
    """
    storage.clear_training_dataset_dir(destination_name)
    dest_dir = storage.training_dataset_dir(destination_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    renamed: list[str] = []
    filenames_by_sample_id: dict[str, str] = {}
    exported = 0

    for sample in samples:
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
        filenames_by_sample_id[sample.sample_id] = target_name
        exported += 1

    return ExportResult(
        destination=str(dest_dir),
        exported_count=exported,
        skipped_excluded=len(samples) - exported,
        renamed_for_collision=renamed,
        filenames_by_sample_id=filenames_by_sample_id,
    )


def export_active_samples(run: IngestionRun, destination_name: str) -> ExportResult:
    """Export every non-excluded sample of `run` to training_runtime/datasets/<destination_name>/."""
    active = [s for s in run.concept.samples if not s.is_excluded]
    skipped = len(run.concept.samples) - len(active)
    result = _write_flat_dataset(active, destination_name)
    return ExportResult(
        destination=result.destination,
        exported_count=result.exported_count,
        skipped_excluded=skipped,
        renamed_for_collision=result.renamed_for_collision,
        filenames_by_sample_id=result.filenames_by_sample_id,
    )


def export_branch_dataset(
    samples: Sequence[DatasetSample],
    destination_name: str,
    *,
    tiers: Mapping[str, Tier] | None = None,
    profile: WeightProfile | None = None,
) -> BranchExportResult:
    """Export an explicit sample list as an experiment branch's dataset.

    Takes samples directly rather than an `IngestionRun`, so a branch controls
    exactly which images it sees — added, removed, re-included — without touching
    the parent run's `is_excluded` flags on the samples it shares with it.

    `tiers` maps a **sample_id** (not a filename or stem) to a curation tier
    ("priority"/"good"/"bad"); a sample not present defaults to "good". Keyed by
    sample_id rather than by the exported stem because only this function knows
    the disambiguated name a basename collision produced — the caller reasons about
    samples, this translates that into the stems `curation_report.json` needs.

    When `tiers` (or `profile`) is given, a `curation_report.json` is written into
    the branch's dataset folder, which is what `krea2.curation.load_weights` reads
    to scale each image's loss — nothing under `workers/` changes for this to take
    effect, since the trainer already reads that file whenever it exists. Passing
    neither writes no report at all, which is exactly what makes a branch a
    control: same images, same weights, nothing for the trainer to see differently
    from an uncurated run.
    """
    result = _write_flat_dataset(samples, destination_name)
    dest_dir = storage.training_dataset_dir(destination_name)
    curation_report_files.remove_curation_overrides(dest_dir)

    files_by_stem = {
        Path(filename).stem: filename for filename in result.filenames_by_sample_id.values()
    }
    stems = sorted(files_by_stem)

    if tiers is None and profile is None:
        return BranchExportResult(
            export=result, stems=stems, report_path=None, weights_are_effective=False
        )

    tiers_by_stem = {
        Path(result.filenames_by_sample_id[sample_id]).stem: tier
        for sample_id, tier in (tiers or {}).items()
        if sample_id in result.filenames_by_sample_id
    }
    report: CurationReport = build_curation_report(
        files_by_stem, tiers_by_stem, profile or WeightProfile()
    )
    effective = is_effective(report)
    report_path = curation_report_files.write_curation_report(dest_dir, report)

    return BranchExportResult(
        export=result, stems=stems, report_path=report_path, weights_are_effective=effective
    )
