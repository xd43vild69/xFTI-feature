"""Writing curation_report.json (and clearing curation_overrides.json) to disk.

The domain layer (`domain/curation_report.py`) knows the file's shape; this module
only knows how to get it onto disk without a half-written file ever being visible to
a precache or train worker that might start reading it mid-write.
"""

import os
from pathlib import Path

from feature_pipeline.domain.curation_report import CurationReport

REPORT_FILENAME = "curation_report.json"
OVERRIDES_FILENAME = "curation_overrides.json"


def write_curation_report(dataset_dir: Path, report: CurationReport) -> Path:
    """Write curation_report.json into an exported dataset folder.

    Written via a temp file and os.replace, the same pattern
    `storage.write_caption_sidecar` uses, so a worker reading the folder mid-write
    either sees the old file or the new one, never a truncated one.
    """
    target = dataset_dir / REPORT_FILENAME
    temp_path = dataset_dir / f"{REPORT_FILENAME}.tmp"
    temp_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp_path, target)
    return target


def remove_curation_overrides(dataset_dir: Path) -> bool:
    """Delete a stale curation_overrides.json, if present.

    `curation.load_weights` lets this file's threshold/group overrides win over the
    report we just wrote — belt and braces here, since `clear_training_dataset_dir`
    already wipes the folder before every export and would already have removed it.
    Returns whether a file was actually removed, so the caller can log it.
    """
    path = dataset_dir / OVERRIDES_FILENAME
    if not path.is_file():
        return False
    path.unlink()
    return True
