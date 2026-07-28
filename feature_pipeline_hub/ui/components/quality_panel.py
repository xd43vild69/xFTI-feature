"""Quality dashboard: duplicate clusters, pairing checks, and dataset statistics.

Operates on the ingestion run selected in the sidebar. Duplicate detection runs on
the stored perceptual hashes, so changing the threshold is instant and never
re-reads the images.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import state
from feature_pipeline.application import quality_service as quality
from feature_pipeline.domain.models import DatasetSample, DuplicateCluster, IngestionRun


def render() -> None:
    run = state.active_run()

    if run is None:
        st.info(
            "No ingestion selected. Import a dataset in the **Ingestion** tab, "
            "then pick one under **Stored ingestions** in the sidebar."
        )
        return

    samples = run.concept.samples
    if not samples:
        st.warning("This ingestion has no images.")
        return

    threshold = st.slider(
        "Duplicate sensitivity (max perceptual distance)",
        min_value=0,
        max_value=16,
        value=quality.DEFAULT_PHASH_THRESHOLD,
        help="0 finds only identical images; higher values also catch crops, "
        "re-encodes, and light edits.",
        key=f"dup_threshold_{run.run_id}",
    )

    _render_summary(run, threshold)
    st.divider()
    _render_duplicates(run, threshold)
    st.divider()
    _render_missing_captions(run)
    st.divider()
    _render_statistics(samples)


def _render_summary(run: IngestionRun, threshold: int) -> None:
    summary = quality.quality_summary(run.concept.samples, run.concept.trigger_word, threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active samples", summary["active"])
    col2.metric("Duplicates", summary["duplicates"], help="Excludes the kept copy of each group")
    col3.metric("Missing caption", summary["missing_caption"])
    col4.metric("Excluded", summary["excluded"])

    if summary["excluded"]:
        st.caption(
            f"{summary['excluded']} sample(s) excluded from the dataset. "
            "Their files stay on disk and they can be restored below."
        )


def _render_duplicates(run: IngestionRun, threshold: int) -> None:
    st.subheader("Perceptual duplicates")

    clusters = quality.find_duplicate_clusters(run.concept.samples, threshold)
    # Keep the stored flags in step with what is on screen.
    state.mark_duplicates(
        run.run_id,
        [sample.sample_id for cluster in clusters for sample, _ in cluster.duplicates],
    )

    if not clusters:
        st.success("No duplicates found at this sensitivity.")
    else:
        total_dupes = sum(len(c.duplicates) for c in clusters)
        st.warning(
            f"{len(clusters)} group(s) of near-identical images, "
            f"{total_dupes} sample(s) beyond the first copy."
        )

        if st.button(
            "Exclude every duplicate, keep one per group",
            type="primary",
            key=f"exclude_all_{run.run_id}",
        ):
            state.set_excluded(
                [s.sample_id for cluster in clusters for s, _ in cluster.duplicates], True
            )
            st.rerun()

        for index, cluster in enumerate(clusters, start=1):
            _render_cluster(index, cluster, run)

    _render_excluded_samples(run)


def _render_cluster(index: int, cluster: DuplicateCluster, run: IngestionRun) -> None:
    with st.expander(
        f"Group {index} · {cluster.size} images · {Path(cluster.kept.image_path).name}",
        expanded=index == 1,
    ):
        entries = [(cluster.kept, None), *cluster.duplicates]
        for column, (sample, distance) in zip(st.columns(len(entries)), entries):
            with column:
                state.render_thumbnail(sample.image_path)
                st.caption("✅ kept" if distance is None else f"distance {distance}")
                if distance is not None and st.button(
                    "Exclude",
                    key=f"exclude_{run.run_id}_{sample.sample_id}",
                    width="stretch",
                ):
                    state.set_excluded([sample.sample_id], True)
                    st.rerun()


def _render_excluded_samples(run: IngestionRun) -> None:
    excluded = [s for s in run.concept.samples if s.is_excluded]
    if not excluded:
        return

    with st.expander(f"Excluded samples ({len(excluded)})"):
        if st.button("Restore all", key=f"restore_all_{run.run_id}"):
            state.set_excluded([s.sample_id for s in excluded], False)
            st.rerun()

        for column, sample in zip(st.columns(min(len(excluded), 6)), excluded):
            with column:
                state.render_thumbnail(sample.image_path)
                if st.button(
                    "Restore",
                    key=f"restore_{run.run_id}_{sample.sample_id}",
                    width="stretch",
                ):
                    state.set_excluded([sample.sample_id], False)
                    st.rerun()


def _render_missing_captions(run: IngestionRun) -> None:
    st.subheader("Caption pairing")

    pending = quality.samples_missing_caption(run.concept.samples, run.concept.trigger_word)

    if not pending:
        st.success("Every active image has a description.")
        return

    st.warning(
        f"{len(pending)} image(s) have no description beyond the trigger word. "
        "Write one below to complete the dataset."
    )

    for sample in pending:
        col_image, col_caption = st.columns([1, 4])
        with col_image:
            _render_thumbnail(sample)
        with col_caption:
            st.caption(Path(sample.image_path).name)
            description_key = f"missing_caption_{run.run_id}_{sample.sample_id}"
            st.text_input(
                "Description",
                placeholder="Describe the image (the trigger word is added automatically)",
                key=description_key,
                label_visibility="collapsed",
                on_change=state.persist_description,
                args=(sample.sample_id, description_key, run.concept.trigger_word),
            )


def _render_statistics(samples: list[DatasetSample]) -> None:
    st.subheader("Dataset statistics")

    resolutions = quality.resolution_distribution(samples)
    ratios = quality.aspect_ratio_distribution(samples)
    caption_stats = quality.caption_length_stats(samples)

    col_res, col_ratio = st.columns(2)
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

    st.caption("Caption length (words per sample)")
    word_counts = [quality.caption_word_count(s.caption) for s in samples if not s.is_excluded]
    histogram = pd.Series(word_counts).value_counts().sort_index()
    st.bar_chart(pd.DataFrame({"samples": histogram.values}, index=histogram.index))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shortest", f"{caption_stats['min']} words")
    col2.metric("Average", f"{caption_stats['mean']} words")
    col3.metric("Longest", f"{caption_stats['max']} words")
    col4.metric(
        "Risk truncation",
        caption_stats["too_long"],
        help=f"Captions over ~{quality.CAPTION_WORD_WARNING} words tend to exceed "
        "CLIP's 77-token prompt limit and get cut during training.",
    )
