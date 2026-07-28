"""AI recaptioning: turn worker output into captions ready to store.

The model writes a plain factual description; everything this project expects on
top of that — normalisation and the trigger word — is applied here so an AI
caption ends up indistinguishable in shape from a hand-written one.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.domain.models import DatasetSample
from feature_pipeline.infrastructure import recaption_runner


@dataclass(frozen=True)
class RecaptionProgress:
    """One step of a batch: model loaded, an image captioned, or something failed."""

    kind: str  # "loaded" | "caption" | "error" | "failed"
    sample_id: str = ""
    caption: str = ""
    message: str = ""
    device: str = ""
    seconds: float = 0.0


def recaption_samples(
    samples: list[DatasetSample], trigger_word: str, detailed: bool = False
) -> Iterator[RecaptionProgress]:
    """Recaption `samples`, yielding progress as each image comes back.

    Yields rather than returns because a batch is slow enough (a few seconds of
    model load plus roughly two seconds per image) that the UI needs to show
    movement instead of freezing.
    """
    if not samples:
        return

    by_path = {sample.image_path: sample for sample in samples}

    for event in recaption_runner.run_recaption(list(by_path), detailed):
        kind = event.get("event")

        if kind == "loaded":
            yield RecaptionProgress(
                kind="loaded", device=event.get("device", ""), seconds=event.get("seconds", 0.0)
            )

        elif kind == "caption":
            sample = by_path.get(event.get("path", ""))
            if sample is None:
                continue
            yield RecaptionProgress(
                kind="caption",
                sample_id=sample.sample_id,
                caption=inject_trigger_word(event.get("caption", ""), trigger_word),
                seconds=event.get("seconds", 0.0),
            )

        elif kind == "error":
            sample = by_path.get(event.get("path", ""))
            yield RecaptionProgress(
                kind="error",
                sample_id=sample.sample_id if sample else "",
                message=event.get("message", "Unknown error"),
            )

        elif kind == "failed":
            yield RecaptionProgress(kind="failed", message=event.get("message", ""))
