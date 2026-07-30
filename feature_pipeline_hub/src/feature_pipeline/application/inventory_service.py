"""Cross-dataset reporting: health counts for every ingestion run at once.

The other application modules each own one step of the pipeline; this one owns a
read-only view *across* steps, which is what the Metrics page needs and what no
single step can answer. It is the only place that walks every run, so it is also
the only place where the cost of loading every sample is paid — see
`ingestion_repository.inventory_fingerprint` for how callers avoid paying it on
every rerun.
"""

import sqlite3

from feature_pipeline.application import quality_service
from feature_pipeline.domain.models import DatasetHealth
from feature_pipeline.infrastructure import ingestion_repository as repo


def load_dataset_inventory(conn: sqlite3.Connection) -> list[DatasetHealth]:
    """Health counts for every stored run, newest first.

    Deliberately built on `stored_quality_summary` rather than `quality_summary`:
    the latter re-clusters duplicates at O(n²) per run, which across a whole fleet
    is the difference between milliseconds and minutes.
    """
    inventory: list[DatasetHealth] = []

    for summary in repo.list_ingestion_runs(conn):
        run = repo.load_ingestion_run(conn, summary.run_id)
        if run is None:  # deleted between the listing and the load
            continue

        samples = run.concept.samples
        counts = quality_service.stored_quality_summary(samples, run.concept.trigger_word)

        inventory.append(
            DatasetHealth(
                run_id=run.run_id,
                concept_name=run.concept.concept_name,
                trigger_word=run.concept.trigger_word,
                source_kind=run.source_kind,
                created_at=run.created_at,
                total_samples=counts["total"],
                active_samples=counts["active"],
                excluded_samples=counts["excluded"],
                duplicate_count=counts["duplicates"],
                missing_caption_count=counts["missing_caption"],
                invalid_count=counts["invalid"],
                median_sharpness=quality_service.median_sharpness(samples),
            )
        )

    return inventory
