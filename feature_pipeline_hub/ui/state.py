"""Session state and persistence helpers shared by the UI components.

Connections are opened per operation rather than cached: Streamlit reruns can land
on different threads, and a SQLite connection is not shared across them safely.
"""

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from feature_pipeline.application import (
    caption_service,
    dataset_service,
    image_service,
    inventory_service,
    training_metrics_service,
    training_service,
)
from feature_pipeline.domain.models import (
    ConceptGroup,
    DatasetHealth,
    DatasetManifest,
    DatasetSample,
    IngestionRun,
    IngestionRunSummary,
)
from feature_pipeline.domain.naming import next_standard_index
from feature_pipeline.infrastructure import ingestion_repository as repo
from feature_pipeline.infrastructure import training_repository as training_repo
from feature_pipeline.infrastructure import version_repository as version_repo
from feature_pipeline.infrastructure.database import get_connection
from feature_pipeline.infrastructure.storage import (
    IMAGE_EXTENSIONS,
    append_uploaded_files,
    delete_managed_folder,
    run_derived_dir,
    run_upload_dir,
    write_caption_sidecar,
)

ACTIVE_RUN_KEY = "active_run_id"


@contextmanager
def _db():
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def list_runs() -> list[IngestionRunSummary]:
    with _db() as conn:
        return repo.list_ingestion_runs(conn)


def load_run(run_id: str) -> IngestionRun | None:
    with _db() as conn:
        return repo.load_ingestion_run(conn, run_id)


def revalidate_all_runs() -> int:
    """Re-check every sample of every run against the current validation rules.

    A one-off correction for a validation rule change: samples keep the verdict
    computed when they were imported, so a rule made stricter or looser afterward
    otherwise never reaches datasets that already existed. Returns how many
    samples' verdicts actually changed.
    """
    with _db() as conn:
        return dataset_service.revalidate_all_runs(conn)


def save_run(run: IngestionRun) -> None:
    with _db() as conn:
        repo.save_ingestion_run(conn, run)


def update_concept_trigger_word(run_id: str, trigger_word: str) -> None:
    with _db() as conn:
        repo.update_run_trigger_word(conn, run_id, trigger_word)


def rename_dataset_base(run_id: str, new_name: str) -> None:
    new_name = new_name.strip()
    with _db() as conn:
        old_name, new_name = repo.rename_concept_and_run(conn, run_id, new_name)
    if old_name and old_name != new_name:
        dataset_old = training_service.dataset_dir_for(old_name)
        dataset_new = training_service.dataset_dir_for(new_name)
        if dataset_old.is_dir() and not dataset_new.exists():
            dataset_old.rename(dataset_new)
        cache_ltx_old = training_service.cache_dir_for(old_name, "ltx23")
        cache_ltx_new = training_service.cache_dir_for(new_name, "ltx23")
        if cache_ltx_old.is_dir() and not cache_ltx_new.exists():
            cache_ltx_old.rename(cache_ltx_new)
        cache_krea_old = training_service.cache_dir_for(old_name, "krea2")
        cache_krea_new = training_service.cache_dir_for(new_name, "krea2")
        if cache_krea_old.is_dir() and not cache_krea_new.exists():
            cache_krea_old.rename(cache_krea_new)


def append_images(run: IngestionRun, uploaded_files: list) -> list[DatasetSample]:
    """Add uploaded images to an existing dataset, keeping its curation intact.

    Files land in `data/raw/<run_id>/` whatever the run was originally imported from.
    A run ingested by path points at a folder the user owns, which `delete_managed_folder`
    deliberately refuses to clean up — writing there would both touch a directory that
    is not ours and leave those files behind when the dataset is deleted.

    Non-image uploads are written too, since a .txt beside an image is how a caption
    arrives, but only the images become samples.

    Filenames continue the standardized `<concept_slug>_NNNN` numbering: the next
    index is read off the run's existing sample stems rather than assumed from
    their count, so it still lands on a free number for a run started before this
    naming scheme, or one that mixes standardized and non-standardized names.
    """
    existing_stems = [Path(s.image_path).stem for s in run.concept.samples]
    start_index = next_standard_index(existing_stems, run.concept.concept_name)

    written = append_uploaded_files(
        uploaded_files, run_upload_dir(run.run_id), run.concept.concept_name, start_index
    )
    images = [path for path in written if Path(path).suffix.lower() in IMAGE_EXTENSIONS]

    with _db() as conn:
        return dataset_service.append_images(conn, run, images)


def _inventory_fingerprint() -> tuple:
    with _db() as conn:
        return repo.inventory_fingerprint(conn)


@st.cache_data(max_entries=4, show_spinner=False)
def _cached_inventory(fingerprint: tuple) -> list[DatasetHealth]:
    with _db() as conn:
        return inventory_service.load_dataset_inventory(conn)


def dataset_inventory() -> list[DatasetHealth]:
    """Health counts for every dataset, rebuilt only when something actually changed.

    Same shape as `_cached_thumbnail` below: a cheap key guarding an expensive value.
    Building the inventory loads every sample of every run, and Streamlit reruns the
    whole script on each widget interaction, so without the fingerprint that load
    would repeat on every click of the Metrics page.
    """
    return _cached_inventory(_inventory_fingerprint())


def step_telemetry(run_id: str) -> repo.StepTelemetry | None:
    with _db() as conn:
        return repo.get_step_telemetry(conn, run_id)


def _training_fingerprint(run_id: str) -> tuple:
    """Cheap key for the training metrics cache: changes exactly when they would.

    The log's size and mtime cover a run still in flight (the file grows as it
    trains), and the row count plus latest status cover a relaunch or a run being
    finalized. Everything else about a lineage is immutable once written.
    """
    with _db() as conn:
        runs = training_repo.list_training_runs(conn, dataset_run_id=run_id)
    trains = [run for run in runs if run.kind == "train"]
    if not trains:
        return (run_id, 0, "", 0.0, 0, 0.0, 0)
    train_stat = _stat_or_zero(training_service.training_log_csv_path(trains[0]))
    # checkpoint_log.csv grows once per save_every, far more slowly than train_log.csv
    # — but on its own schedule, so a Checkpoints tab watching only the other file's
    # mtime would sit on a stale table for as long as the cache survived.
    ckpt_stat = _stat_or_zero(training_service.checkpoint_log_csv_path(trains[0]))
    return (run_id, len(trains), trains[0].status, *train_stat, *ckpt_stat)


def _stat_or_zero(path: Path) -> tuple[float, int]:
    """(mtime, size) for a file, or zeros if it is not there yet."""
    try:
        stat = path.stat()
    except OSError:
        return (0.0, 0)
    return (stat.st_mtime, stat.st_size)


@st.cache_data(max_entries=4, show_spinner=False)
def _cached_training_lineages(
    fingerprint: tuple,
) -> list[training_metrics_service.TrainingLineage]:
    with _db() as conn:
        return training_metrics_service.load_training_lineages(conn, fingerprint[0])


def training_lineages(run_id: str) -> list[training_metrics_service.TrainingLineage]:
    """Every training this dataset has been through, newest first.

    Cached on the same principle as `dataset_inventory`: this parses train_log.csv,
    which for a long run is six figures of rows, and Streamlit reruns the whole
    script on every widget interaction.
    """
    return _cached_training_lineages(_training_fingerprint(run_id))


def _branch_fingerprint(
    dataset_run_id: str, parent_training_run_id: str, fork_step: int
) -> tuple:
    """Cheap key for the branch-comparison cache: stats *every* sibling's files.

    `_training_fingerprint` above only stats the dataset's single newest launch,
    which is correct for one lineage but wrong here: three sibling branches forked
    from the same checkpoint are three independent lineages, and an older one still
    training would never invalidate a cache keyed only on the newest. So every
    branch's train_log.csv and val_log.csv gets its own (mtime, size) pair in the key.
    """
    with _db() as conn:
        branches = training_repo.list_experiment_branches(
            conn, parent_training_run_id, fork_step
        )
    stats: list = [dataset_run_id, parent_training_run_id, fork_step]
    for run in branches:
        stats.append(run.training_run_id)
        stats.append(run.status)
        stats.extend(_stat_or_zero(training_service.training_log_csv_path(run)))
        stats.extend(_stat_or_zero(training_service.validation_log_csv_path(run)))
    return tuple(stats)


@st.cache_data(max_entries=4, show_spinner=False)
def _cached_experiment_branches(
    fingerprint: tuple, dataset_run_id: str, parent_training_run_id: str, fork_step: int
) -> list[training_metrics_service.TrainingLineage]:
    with _db() as conn:
        lineages = training_metrics_service.load_training_lineages(conn, dataset_run_id)
    return [
        lineage
        for lineage in lineages
        if lineage.latest.fork_parent_run_id == parent_training_run_id
        and lineage.latest.fork_step == fork_step
    ]


def experiment_branches(
    dataset_run_id: str, parent_training_run_id: str, fork_step: int
) -> list[training_metrics_service.TrainingLineage]:
    """Every branch forked from the same (parent, step), each its own lineage.

    A forked branch always gets a fresh output_dir (see
    training_service.fork_training), so group_training_lineages already keeps each
    branch separate — this only filters load_training_lineages down to the siblings
    that share a fork point, using the fingerprint above rather than
    `_training_fingerprint`, which would go stale while an older sibling is still
    training.
    """
    fingerprint = _branch_fingerprint(dataset_run_id, parent_training_run_id, fork_step)
    return _cached_experiment_branches(
        fingerprint, dataset_run_id, parent_training_run_id, fork_step
    )


@st.cache_data(max_entries=4, show_spinner=False)
def _cached_branch_curves(
    fingerprint: tuple, parent_training_run_id: str, fork_step: int
) -> dict[str, dict[str, pd.DataFrame]]:
    """{branch_label: {"train": df, "val": df}} — the raw per-step series a scalar
    summary can't provide, read straight off each branch's own CSVs."""
    with _db() as conn:
        branches = training_repo.list_experiment_branches(
            conn, parent_training_run_id, fork_step
        )
    curves: dict[str, dict[str, pd.DataFrame]] = {}
    for run in branches:
        label = run.branch_label or run.training_run_id[:8]
        series: dict[str, pd.DataFrame] = {}
        for name, path in (
            ("train", training_service.training_log_csv_path(run)),
            ("val", training_service.validation_log_csv_path(run)),
        ):
            try:
                series[name] = pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
        curves[label] = series
    return curves


def branch_curves(
    dataset_run_id: str, parent_training_run_id: str, fork_step: int
) -> dict[str, dict[str, pd.DataFrame]]:
    fingerprint = _branch_fingerprint(dataset_run_id, parent_training_run_id, fork_step)
    return _cached_branch_curves(fingerprint, parent_training_run_id, fork_step)


def save_caption(sample_id: str, caption: str) -> None:
    with _db() as conn:
        repo.update_sample_caption(conn, sample_id, caption)


CAPTION_VERSIONS_KEY = "caption_widget_versions"


def caption_widget_key(run_id: str, sample_id: str) -> str:
    """Key for a caption editor, versioned so external edits are picked up.

    A keyed Streamlit widget ignores its `value` argument on later reruns, so a
    caption written from another tab would otherwise be reverted by the stale text
    still held in the widget's state. Bumping the version mints a new widget that
    reads the stored caption afresh.
    """
    version = st.session_state.get(CAPTION_VERSIONS_KEY, {}).get(sample_id, 0)
    return f"caption_{run_id}_{sample_id}_v{version}"


def persist_caption(sample_id: str, widget_key: str) -> None:
    """on_change callback: store whatever the user typed into a caption editor."""
    save_caption(sample_id, st.session_state[widget_key])


def persist_description(sample_id: str, widget_key: str, trigger_word: str) -> None:
    """on_change callback for the quality panel: turn a description into a caption."""
    description = st.session_state[widget_key].strip()
    if not description:
        return

    save_caption(sample_id, caption_service.inject_trigger_word(description, trigger_word))
    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})
    versions[sample_id] = versions.get(sample_id, 0) + 1


def preview_replace_counts(samples: list[DatasetSample], old_word: str) -> tuple[int, int]:
    """Calculate matching samples count and total exact word occurrences (case-sensitive)."""
    if not old_word:
        return 0, 0
    matching_samples = 0
    total_matches = 0
    for sample in samples:
        _, count = caption_service.replace_exact_word(sample.caption, old_word, "")
        if count > 0:
            matching_samples += 1
            total_matches += count
    return matching_samples, total_matches


def preview_append_counts(samples: list[DatasetSample], word: str) -> tuple[int, int]:
    """(how many would gain `word`, how many already carry it), for the dialog's preview."""
    if not word:
        return 0, 0

    already = sum(1 for s in samples if caption_service.has_trigger(s.caption, word))
    return len(samples) - already, already


def _store_caption(conn, sample: DatasetSample, caption: str) -> None:
    """Persist one edited caption and make the grid pick it up.

    Bumping the widget version is what stops the editor showing the old text: a
    keyed Streamlit widget ignores `value=` once it exists, so an edit made from
    outside it needs a new key. Writes the DB only — the .txt sidecars are
    rewritten wholesale on export.
    """
    repo.update_sample_caption(conn, sample.sample_id, caption)
    sample.caption = caption
    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})
    versions[sample.sample_id] = versions.get(sample.sample_id, 0) + 1


def batch_replace_caption_word(
    samples: list[DatasetSample], old_word: str, new_word: str
) -> tuple[int, int]:
    """Batch replace exact occurrences of old_word with new_word across `samples`.

    Takes the samples rather than the run so the caller decides the scope — the
    whole dataset, the current filter, or just what is selected.

    Returns (samples_updated_count, total_replacements_count).
    """
    if not old_word:
        return 0, 0

    samples_updated = 0
    total_replacements = 0

    with _db() as conn:
        for sample in samples:
            new_caption, count = caption_service.replace_exact_word(
                sample.caption, old_word, new_word
            )
            if count > 0:
                _store_caption(conn, sample, new_caption)
                samples_updated += 1
                total_replacements += count

    return samples_updated, total_replacements


def batch_append_caption_word(samples: list[DatasetSample], word: str) -> int:
    """Add `word` to the end of each caption in `samples`, and return how many changed.

    Samples that already carry the word are left untouched (see
    `caption_service.append_word`), so re-running over an overlapping scope does
    not duplicate terms.
    """
    if not word:
        return 0

    updated = 0
    with _db() as conn:
        for sample in samples:
            new_caption = caption_service.append_word(sample.caption, word)
            if new_caption != sample.caption:
                _store_caption(conn, sample, new_caption)
                updated += 1

    return updated


def selection_key(run_id: str, sample_id: str) -> str:
    return f"select_{run_id}_{sample_id}"


def selected_samples(run: IngestionRun) -> list[DatasetSample]:
    """Samples currently ticked in the curation grid."""
    return [
        sample
        for sample in run.concept.samples
        if st.session_state.get(selection_key(run.run_id, sample.sample_id))
    ]


def set_selection(run_id: str, samples: list[DatasetSample], selected: bool) -> None:
    """Tick or untick a group of samples.

    Only safe to call before the checkboxes are instantiated on this run —
    Streamlit rejects writes to a widget's key once its widget exists.
    """
    for sample in samples:
        st.session_state[selection_key(run_id, sample.sample_id)] = selected


def apply_recaption(sample: DatasetSample, caption: str) -> None:
    """Store an AI caption in the database and alongside the image.

    Writing the .txt sidecar keeps the source folder interoperable with LoRAlab,
    which reads captions from disk; the pre-AI text is kept as .txt.bak. Bumping
    the widget version makes the grid's editor pick the new text up.
    """
    save_caption(sample.sample_id, caption)
    write_caption_sidecar(sample.image_path, caption)

    versions = st.session_state.setdefault(CAPTION_VERSIONS_KEY, {})
    versions[sample.sample_id] = versions.get(sample.sample_id, 0) + 1


def set_excluded(sample_ids: list[str], excluded: bool) -> None:
    with _db() as conn:
        repo.set_samples_excluded(conn, sample_ids, excluded)


MAX_TRAINING_SIDE = image_service.MAX_TRAINING_SIDE


def oversized_samples(samples: list[DatasetSample]) -> list[DatasetSample]:
    """Those of `samples` whose longest side is above the training resolution."""
    return dataset_service.oversized_samples(samples)


def normalize_samples(
    run: IngestionRun, samples: list[DatasetSample]
) -> list[dataset_service.DerivationOutcome]:
    """Downscale the oversized samples in `samples` to 1024px on their longest side.

    The resized copies go to `data/raw/<run_id>/derived/` and the samples are
    repointed at them; the originals stay untouched wherever they came from. Each
    sample's thumbnail cache invalidates for free, since it is keyed on the path.
    """
    with _db() as conn:
        return dataset_service.normalize_samples(
            conn, samples, run_derived_dir(run.run_id), MAX_TRAINING_SIDE
        )


def rotate_sample(
    run_id: str, sample: DatasetSample, quarter_turns: int
) -> dataset_service.DerivationOutcome:
    """Rotate one sample 90° per quarter turn (positive = left, negative = right).

    Re-derived from the sample's original file with the accumulated angle, so the
    fifth rotation costs a JPEG no more than the first did.
    """
    with _db() as conn:
        return dataset_service.rotate_sample(
            conn, sample, run_derived_dir(run_id), quarter_turns
        )


def is_training_active() -> bool:
    """Whether a training-runtime job (pre-cache/train/progressive/curate) is
    running right now — the GPU can only do one heavy job at a time.

    Self-healing: a 'running' row whose process actually died (crash, machine
    restart) is corrected to 'failed' here rather than blocking the GPU forever.
    """
    with _db() as conn:
        return training_service.is_training_active(conn)


def active_training_run() -> training_repo.TrainingRun | None:
    """The job holding the GPU right now, or None — the same question
    `is_training_active` answers, but returning *which* one.

    Goes through `training_service.is_training_active` first rather than reading
    `find_running_training_run` directly: that call is where a 'running' row whose
    process actually died gets corrected to 'failed'. Skipping it would let one crashed
    run keep the launch buttons disabled forever.
    """
    with _db() as conn:
        if not training_service.is_training_active(conn):
            return None
        return training_repo.find_running_training_run(conn)


def mark_duplicates(run_id: str, sample_ids: list[str]) -> None:
    with _db() as conn:
        repo.mark_duplicates(conn, run_id, sample_ids)


def save_dataset_version(
    concept: ConceptGroup, version_tag: str, manifest: DatasetManifest, exported_path: str
) -> str:
    with _db() as conn:
        return version_repo.create_dataset_version(
            conn,
            concept=concept,
            version_tag=version_tag,
            manifest=manifest,
            exported_path=exported_path,
        )


def latest_dataset_version(concept_id: str) -> version_repo.DatasetVersion | None:
    with _db() as conn:
        return version_repo.latest_version_for_concept(conn, concept_id)


def delete_run(run_id: str) -> None:
    """Drop a run from the database, plus its image folder when we own it."""
    with _db() as conn:
        run = repo.load_ingestion_run(conn, run_id)
        repo.delete_ingestion_run(conn, run_id)

    if run is not None and run.source_kind == "upload":
        delete_managed_folder(run.source_path)

    # Images added later live under data/raw/<run_id>/ whatever the run was imported
    # from, so a run ingested by path can own a managed folder too. Deleting it is a
    # no-op when nothing was ever appended.
    delete_managed_folder(run_upload_dir(run_id))

    if st.session_state.get(ACTIVE_RUN_KEY) == run_id:
        st.session_state.pop(ACTIVE_RUN_KEY, None)


def set_active_run(run_id: str) -> None:
    st.session_state[ACTIVE_RUN_KEY] = run_id


def active_run() -> IngestionRun | None:
    """The run currently selected in the sidebar, reloaded from storage."""
    run_id = st.session_state.get(ACTIVE_RUN_KEY)
    if not run_id:
        return None

    run = load_run(run_id)
    if run is None:  # deleted from another session
        st.session_state.pop(ACTIVE_RUN_KEY, None)
    return run


@st.cache_data(max_entries=256)
def _cached_thumbnail(image_path: str, mtime: float, size: int) -> Image.Image:
    return image_service.make_square_thumbnail(image_path, size)


# Grid cards render at `width="stretch"`, so a card's on-screen size grows as
# columns_per_row shrinks. Without a matching bump here, a fixed 512px thumbnail
# gets upscaled by the browser to fill a wider card and looks blurry. Keyed on the
# same options as the gallery's column slider (curate_step.py / gallery.py).
THUMBNAIL_SIZE_BY_COLUMNS: dict[int, int] = {2: 896, 3: 704, 4: 576, 5: 512, 6: 512}


def thumbnail_size_for_columns(columns_per_row: int) -> int:
    """Thumbnail resolution matched to how wide a grid card will render at this column count."""
    return THUMBNAIL_SIZE_BY_COLUMNS.get(columns_per_row, image_service.THUMBNAIL_SIZE)


def render_thumbnail(image_path: str, *, width: int | str = "stretch", size: int | None = None) -> None:
    """Render a square, theme-adaptive thumbnail, or an error if it's missing/unreadable.

    `mtime` is part of the cache key so an image edited or re-ingested on the same
    path invalidates its cached thumbnail automatically. `width` defaults to filling
    the container (right for a fixed-column grid like the curate gallery); pass a
    pixel value where the surrounding column width varies, so the thumbnail stays a
    consistent size regardless of how many items share the row. `size` overrides the
    generated resolution (default `image_service.THUMBNAIL_SIZE`) — use
    `thumbnail_size_for_columns` when the display width varies with a column count.
    """
    path = Path(image_path)
    if not path.exists():
        st.error(f"Missing file: {path.name}")
        return

    resolved_size = size or image_service.THUMBNAIL_SIZE
    try:
        thumbnail = _cached_thumbnail(str(path), path.stat().st_mtime, resolved_size)
    except Exception:
        st.error(f"Could not read image: {path.name}")
        return

    st.image(thumbnail, width=width)


# `st.image` degrades what it serves in two separate ways, and a full-size preview
# hits both: it downscales anything wider than 1460px with BILINEAR resampling, and
# it re-encodes to JPEG quality 90 whenever the source format differs from the one
# it picks itself (which catches every PNG without alpha, and every WEBP). Naming
# the source format here keeps the re-encode from firing; PNG is the fallback
# because it is lossless for formats Streamlit cannot pass through verbatim.
_PASSTHROUGH_OUTPUT_FORMAT = {"JPEG": "JPEG", "JPG": "JPEG", "PNG": "PNG", "GIF": "auto"}


# Streamlit already writes the image's natural width onto the <img> inline (that is
# what passing `width=` above buys), then keeps it from being honoured two ways: a
# max-width cap, and a flex container that lets the image shrink below its own size.
# Undoing both is all that separates the fitted view from a 1:1 one, so the sizing
# itself is left to that inline width rather than restated here. Selectors verified
# against the bundled frontend: `stDialog` is a class, `stImageContainer` a testid.
_ACTUAL_SIZE_CSS = """
<style>
.stDialog [data-testid="stImageContainer"] {
    overflow: auto;
    max-height: 75vh;
}
.stDialog [data-testid="stImageContainer"] img {
    max-width: none !important;
    flex-shrink: 0 !important;
}
</style>
"""


def render_original_image(
    image_path: str, *, actual_size: bool = False, show_facts: bool = True
) -> None:
    """Render an image at its original quality, scaled to fit only by the browser.

    Passing the image's own pixel width as `width` is what avoids the server-side
    resample: Streamlit only resizes when the source is *wider* than the target, and
    it clamps an oversized width to the container in CSS anyway. So the browser
    receives the untouched file and does the fitting itself, which is both a better
    resample than BILINEAR and correct on high-DPI screens.

    `actual_size` swaps the fitted view for a 1:1 one that scrolls, for judging focus
    and compression artefacts at real pixels. The bytes are the same either way — only
    the CSS differs — so toggling it costs nothing beyond a re-layout.
    """
    path = Path(image_path)
    if not path.exists():
        st.error(f"Missing file: {path.name}")
        return

    try:
        facts = image_service.describe_original(str(path))
    except Exception:
        st.error(f"Could not read image: {path.name}")
        return

    if actual_size:
        st.html(_ACTUAL_SIZE_CSS)

    st.image(
        str(path),
        width=facts.width,
        output_format=_PASSTHROUGH_OUTPUT_FORMAT.get(facts.image_format, "PNG"),
    )

    if show_facts:
        megabytes = facts.byte_size / 1_048_576
        st.caption(
            f"{facts.width} × {facts.height} px · {facts.image_format} · {megabytes:.1f} MB"
        )


OBSERVABILITY_STEP = "steps/observability_step.py"
IMPORT_STEP = "steps/import_step.py"
CURATE_STEP = "steps/curate_step.py"
QUALITY_STEP = "steps/quality_step.py"
EXPORT_STEP = "steps/export_step.py"
TRAIN_STEP = "steps/train_step.py"
SETTINGS_STEP = "steps/settings_step.py"


def require_active_run() -> IngestionRun | None:
    """The active run, or None after drawing the shared 'nothing selected' state.

    Steps 2-4 all need a dataset before they can show anything; routing that
    through one helper keeps the empty state identical everywhere.
    """
    run = active_run()
    if run is not None and run.concept.samples:
        return run

    if run is not None:
        st.warning("This dataset has no images.")
        return None

    st.info("No dataset selected yet.")
    if st.button("Go to import", icon=":material/arrow_forward:"):
        st.switch_page(IMPORT_STEP)
    return None


_RUN_ICONS = {"upload": "📤", "clone": "📋", "folder": "📁"}


def format_run_label(summary: IngestionRunSummary) -> str:
    icon = _RUN_ICONS.get(summary.source_kind, "📁")
    when = summary.created_at.astimezone().strftime("%d %b %H:%M")
    active_trigger = (summary.trigger_word or "").strip()
    if not active_trigger:
        dataset_dir = training_service.dataset_dir_for(summary.concept_name)
        active_trigger = training_service.detect_dominant_trigger_word_in_dataset(dataset_dir) or ""

    if active_trigger and active_trigger != summary.concept_name:
        return f"{icon} {summary.concept_name} [{active_trigger}] · {summary.sample_count} imgs · {when}"
    return f"{icon} {summary.concept_name} · {summary.sample_count} imgs · {when}"
