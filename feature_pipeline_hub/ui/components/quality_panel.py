"""Step 3: quality checks — duplicate clusters, caption pairing, statistics.

Duplicate detection runs on the stored perceptual hashes, so changing the
sensitivity is instant and never re-reads the images. Headline counts live in the
context bar; this page is for acting on problems, not for restating them.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import state
from feature_pipeline.application import quality_service as quality
from feature_pipeline.domain.models import DatasetSample, DuplicateCluster, IngestionRun
from feature_pipeline.domain.validators import find_unpaired_files
from feature_pipeline.infrastructure.storage import scan_caption_files

# Fixed display size for every thumbnail on this page. Groups here (a duplicate
# cluster, the excluded shelf, the blur ranking) vary in size from one to many
# images, so `st.columns(n)` produces columns of very different widths; a
# stretch-to-fill thumbnail would balloon in the narrow groups. A constant pixel
# width keeps every image the same size regardless of how many share the row.
THUMBNAIL_WIDTH = 150


@st.dialog(" ", width="large")
def _render_zoom_dialog(image_path: str) -> None:
    st.image(image_path, width="stretch")


def _zoom_button(image_path: str, key: str) -> None:
    """Icon button that opens `image_path` at full size in a modal.

    Streamlit has no double-click event to bind to, so a click on this icon is
    the closest equivalent for "inspect this one image up close".
    """
    if st.button(
        "",
        icon=":material/zoom_in:",
        type="tertiary",
        help="Ver en tamaño grande",
        key=f"zoom_{key}",
    ):
        _render_zoom_dialog(image_path)


def render() -> None:
    run = state.require_active_run()
    if run is None:
        return

    _render_duplicates(run)
    _render_sharpness(run)
    _render_missing_captions(run)
    _render_orphans(run)
    _render_statistics(run.concept.samples)

    if st.button("Continue to export", icon=":material/arrow_forward:"):
        st.switch_page(state.EXPORT_STEP)


def _render_duplicates(run: IngestionRun) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Duplicates")
        with st.popover("", icon=":material/tune:", help="Detection settings"):
            threshold = st.slider(
                "Max perceptual distance",
                min_value=0,
                max_value=16,
                value=quality.DEFAULT_PHASH_THRESHOLD,
                help="0 finds only identical images; higher values also catch crops, "
                "re-encodes, and light edits.",
                key=f"dup_threshold_{run.run_id}",
            )

    clusters = quality.find_duplicate_clusters(run.concept.samples, threshold)
    # Keep the stored flags in step with what is on screen — the context bar
    # reads them rather than recomputing clusters on every page.
    state.mark_duplicates(
        run.run_id,
        [sample.sample_id for cluster in clusters for sample, _ in cluster.duplicates],
    )

    if not clusters:
        st.caption("No duplicates at this sensitivity.")
    else:
        total_dupes = sum(len(c.duplicates) for c in clusters)
        st.caption(
            f"{len(clusters)} group(s) of near-identical images · "
            f"{total_dupes} beyond the first copy."
        )

        if st.button("Exclude duplicates, keep one per group", key=f"exclude_all_{run.run_id}"):
            state.set_excluded(
                [s.sample_id for cluster in clusters for s, _ in cluster.duplicates], True
            )
            st.rerun()

        for index, cluster in enumerate(clusters, start=1):
            _render_cluster(index, cluster, run)

    _render_excluded_samples(run)


def _render_cluster(index: int, cluster: DuplicateCluster, run: IngestionRun) -> None:
    with st.expander(f"Group {index} · {cluster.size} images · {Path(cluster.kept.image_path).name}"):
        entries = [(cluster.kept, None), *cluster.duplicates]
        for column, (sample, distance) in zip(st.columns(len(entries)), entries):
            with column:
                state.render_thumbnail(sample.image_path, width=THUMBNAIL_WIDTH)
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.caption("kept" if distance is None else f"distance {distance}")
                    _zoom_button(sample.image_path, f"cluster_{run.run_id}_{sample.sample_id}")
                    if distance is not None and st.button(
                        "",
                        icon=":material/block:",
                        type="tertiary",
                        help="Exclude from the dataset",
                        key=f"exclude_{run.run_id}_{sample.sample_id}",
                    ):
                        state.set_excluded([sample.sample_id], True)
                        st.rerun()


def _render_excluded_samples(run: IngestionRun) -> None:
    excluded = [s for s in run.concept.samples if s.is_excluded]
    if not excluded:
        return

    with st.expander(f"Excluded ({len(excluded)})"):
        if st.button("Restore all", icon=":material/undo:", key=f"restore_all_{run.run_id}"):
            state.set_excluded([s.sample_id for s in excluded], False)
            st.rerun()

        for column, sample in zip(st.columns(min(len(excluded), 6)), excluded):
            with column:
                state.render_thumbnail(sample.image_path, width=THUMBNAIL_WIDTH)
                with st.container(horizontal=True, vertical_alignment="center"):
                    _zoom_button(sample.image_path, f"excluded_{run.run_id}_{sample.sample_id}")
                    if st.button(
                        "",
                        icon=":material/undo:",
                        type="tertiary",
                        help="Restore into the dataset",
                        key=f"restore_{run.run_id}_{sample.sample_id}",
                    ):
                        state.set_excluded([sample.sample_id], False)
                        st.rerun()


def _render_sharpness(run: IngestionRun) -> None:
    """The softest images of the run, offered for exclusion.

    Presented as a ranking rather than a pass/fail badge on purpose: the Laplacian
    variance moves with texture and subject as much as with focus, so it can point
    at candidates but cannot decide. The call stays with the user.
    """
    softest = quality.blurriest_samples(run.concept.samples)
    if not softest:
        return

    st.subheader("Sharpness")
    st.caption(
        "Least sharp images in this set, softest first. This is a ranking within "
        "the run, not a verdict — a flat or minimal subject scores low while being "
        "perfectly in focus."
    )

    for column, sample in zip(st.columns(len(softest)), softest):
        with column:
            state.render_thumbnail(sample.image_path, width=THUMBNAIL_WIDTH)
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"{sample.metrics.sharpness:.0f}")
                _zoom_button(sample.image_path, f"blur_{run.run_id}_{sample.sample_id}")
                if st.button(
                    "",
                    icon=":material/block:",
                    type="tertiary",
                    help="Exclude from the dataset",
                    key=f"exclude_blur_{run.run_id}_{sample.sample_id}",
                ):
                    state.set_excluded([sample.sample_id], True)
                    st.rerun()


def _render_orphans(run: IngestionRun) -> None:
    """Files in the source folder that never became a usable pair.

    Images without a .txt are read off stored data; orphan .txt files need a listing
    of the source folder, because ingestion only ever looks a caption up *from* an
    image and so never sees them.
    """
    without_caption = quality.samples_without_source_caption(run.concept.samples)
    _, orphan_captions = find_unpaired_files(
        [s.image_path for s in run.concept.samples],
        scan_caption_files(run.source_path),
    )

    if not without_caption and not orphan_captions:
        return

    with st.expander(f"Unpaired files ({len(without_caption) + len(orphan_captions)})"):
        if without_caption:
            st.caption(
                f"{len(without_caption)} image(s) arrived with no .txt file. They carry "
                "only the trigger word until you describe them above."
            )
        if orphan_captions:
            st.caption(
                f"{len(orphan_captions)} .txt file(s) in the source folder have no "
                "matching image, so they were never imported:"
            )
            for path in orphan_captions:
                st.code(Path(path).name, language=None)


def _render_missing_captions(run: IngestionRun) -> None:
    st.subheader("Captions")

    pending = quality.samples_missing_caption(run.concept.samples, run.concept.trigger_word)

    if not pending:
        st.caption("Every active image has a description.")
        return

    st.caption(
        f"{len(pending)} image(s) have no description beyond the trigger word. "
        "Write one to complete the dataset."
    )

    for sample in pending:
        col_image, col_caption = st.columns([1, 6], vertical_alignment="center")
        with col_image:
            state.render_thumbnail(sample.image_path, width=THUMBNAIL_WIDTH)
            _zoom_button(sample.image_path, f"missing_{run.run_id}_{sample.sample_id}")
        with col_caption:
            description_key = f"missing_caption_{run.run_id}_{sample.sample_id}"
            st.text_input(
                Path(sample.image_path).name,
                placeholder="Describe the image (the trigger word is added automatically)",
                key=description_key,
                on_change=state.persist_description,
                args=(sample.sample_id, description_key, run.concept.trigger_word),
            )


def _render_statistics(samples: list[DatasetSample]) -> None:
    with st.expander("Statistics"):
        resolutions = quality.resolution_distribution(samples)
        ratios = quality.aspect_ratio_distribution(samples)
        caption_stats = quality.caption_length_stats(samples)

        orientations = quality.orientation_distribution(samples)

        col_res, col_ratio, col_orient = st.columns(3)
        with col_res:
            st.caption("Resolutions")
            st.bar_chart(
                pd.DataFrame({"samples": list(resolutions.values())}, index=list(resolutions)),
                horizontal=True,
            )
        with col_ratio:
            st.caption("Aspect ratios")
            st.bar_chart(
                pd.DataFrame({"samples": list(ratios.values())}, index=list(ratios)),
                horizontal=True,
            )
        with col_orient:
            st.caption("Orientation balance")
            st.bar_chart(
                pd.DataFrame({"samples": list(orientations.values())}, index=list(orientations)),
                horizontal=True,
            )

        st.caption("Caption length (words per sample)")
        word_counts = [
            quality.caption_word_count(s.caption) for s in samples if not s.is_excluded
        ]
        histogram = pd.Series(word_counts).value_counts().sort_index()
        st.bar_chart(pd.DataFrame({"samples": histogram.values}, index=histogram.index))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Shortest", f"{caption_stats['min']} words")
        col2.metric("Average", f"{caption_stats['mean']} words")
        col3.metric("Longest", f"{caption_stats['max']} words")
        col4.metric(
            "Risk truncation",
            caption_stats["too_long"],
            help=f"Captions over ~{quality.CAPTION_WORD_WARNING} words risk exceeding the "
            f"{quality.TEXT_ENCODER_MAX_TOKENS}-token budget the pre-cache step encodes "
            "with, and would get cut during training.",
        )
