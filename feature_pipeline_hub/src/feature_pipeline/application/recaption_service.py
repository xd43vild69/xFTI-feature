"""AI recaptioning: turn worker output into captions ready to store.

The model writes a plain factual description; everything this project expects on
top of that — normalisation and the trigger word — is applied here so an AI
caption ends up indistinguishable in shape from a hand-written one.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import ValidationError

from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.domain.models import DatasetSample
from feature_pipeline.domain.worker_contracts import (
    CaptionEvent,
    ErrorEvent,
    FailedEvent,
    LoadedEvent,
    recaption_event_adapter,
)
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

    for raw in recaption_runner.run_recaption(list(by_path), detailed):
        # The runner is a dumb transport: it forwards every JSON object the worker
        # prints. Validating here is what turns that stream into the documented
        # protocol; anything that does not match it is worker noise and is dropped.
        try:
            event = recaption_event_adapter.validate_python(raw)
        except ValidationError:
            continue

        if isinstance(event, LoadedEvent):
            yield RecaptionProgress(kind="loaded", device=event.device, seconds=event.seconds)

        elif isinstance(event, CaptionEvent):
            sample = by_path.get(event.path)
            if sample is None:
                continue
            yield RecaptionProgress(
                kind="caption",
                sample_id=sample.sample_id,
                caption=inject_trigger_word(event.caption, trigger_word),
                seconds=event.seconds,
            )

        elif isinstance(event, ErrorEvent):
            sample = by_path.get(event.path)
            yield RecaptionProgress(
                kind="error",
                sample_id=sample.sample_id if sample else "",
                message=event.message,
            )

        elif isinstance(event, FailedEvent):
            yield RecaptionProgress(kind="failed", message=event.message)
