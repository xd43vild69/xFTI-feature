"""Structured captioning: fixed slots the VLM fills, assembled into a caption here.

A LoRA learns by division of labour between the caption and the trigger word:
whatever the caption *describes* stays conditioned on that text, and so remains
steerable at inference; whatever the caption *omits* gets absorbed into the
trigger. So the useful question is not "how good is this caption" but "what is it
deliberately not saying".

That is what the two modes are. `subject` describes framing, pose, clothing,
background and light, and says nothing about the face, hair or build — those are
the trigger's job, which is what a character LoRA is for. `location` inverts it:
it describes the transient stuff (hour, weather, light, people passing through)
and stays quiet about the architecture and style, so those get baked in instead.

Free prose cannot hold that line — asked for a description, a VLM always ends up
narrating the face. What holds it is asking for a JSON object with exact keys,
plus an explicit prohibition, and then assembling the sentence here from the
slots rather than letting the model write it. Ported from xLoralizerPro's
backend/normalizer.py, where the schemas and their wording are already proven.

One thing the subject schema does say out loud is which side a tattoo or scar is
on, and from the subject's own point of view rather than the camera's. A mark the
caption never locates is learned as "somewhere on an arm", which is how a hand
tattoo comes out on the wrong hand at inference; naming the side conditions it on
text, where it stays steerable, instead of leaving it to the trigger. Anatomical
side rather than viewer side because that is the invariant: the same tattoo would
swap labels between a front and a back view otherwise.

That slot assumes the cached dataset is not mirrored. `flip_x` in
precache_worker.py encodes the prompt once and reuses that one embedding for both
an image and its `__flip` copy, so with mirroring on, every side the caption names
would be asserted of both sides at once — worse than saying nothing. It defaults
to off and the hub never sets it; if that ever changes, this slot has to go with
it.

Pure: no I/O, no model, no network. The worker only relays `instruction_for()`
downstream and hands the raw reply back to `parse_slots`.
"""

import json
import re
from typing import Literal

from pydantic import BaseModel, field_validator

CaptionMode = Literal["subject", "location"]

SUBJECT_INSTRUCTION = """Analyze this photo of a person for AI training captioning.
Return ONLY a JSON object with these exact keys:
- "shot": one of "full body shot", "close-up portrait", "medium shot", "cowboy shot"
- "background": short phrase, e.g. "plain gray studio background", "urban street at night"
- "clothing": short phrase describing visible clothing, or "shirtless" / "bare torso"
- "pose": short phrase, e.g. "standing facing camera", "sitting, arms crossed"
- "lighting": short phrase, e.g. "soft studio lighting", "harsh sunlight"
- "tattoos_visible": true or false
- "asymmetric_marks": short phrase naming each tattoo, scar or birthmark that is on one side of the body and the side it is on from the subject's own point of view, not the viewer's, e.g. "tattoo on their left forearm", "scar above their right eyebrow", or "none"
Do NOT describe facial features, hair, body type, or tattoo designs — say where a mark is and on which side, never what it depicts. No extra text."""

LOCATION_INSTRUCTION = """Analyze this photo of a location for AI training captioning.
Return ONLY a JSON object with these exact keys:
- "time_of_day": short phrase, e.g. "daytime", "night", "golden hour", "unknown"
- "weather": short phrase, e.g. "sunny", "raining", "cloudy", "indoors"
- "lighting": short phrase, e.g. "natural light", "dimly lit", "neon lights", "harsh sunlight"
- "transient_objects": short phrase listing temporary elements, e.g. "people walking", "parked cars", "furniture", "litter", or "none"
- "perspective": short phrase, e.g. "wide angle shot", "eye-level", "aerial view", "close-up"
Do NOT describe the permanent architectural features, walls, style, or the core aesthetic of the location itself. No extra text."""

# Values that mean "this slot has nothing to say". The prompts hand the model
# "none" and "unknown" as legitimate answers, and a caption is better off without
# them than carrying "unknown, sunny, ..." into training.
_PLACEHOLDERS = {"none", "null", "n/a", "na", "unknown", "nothing"}

# The reply usually arrives wrapped — a ```json fence, or a line of courtesy prose
# before the object. Greedy so it spans the whole object rather than stopping at
# the first closing brace.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class CaptionSchemaError(ValueError):
    """The model's reply could not be read as the requested slot schema."""


class _Slots(BaseModel):
    """Shared coercion: the model is asked for strings but does not always comply."""

    @field_validator("*", mode="before")
    @classmethod
    def _coerce(cls, value: object) -> object:
        """Flatten whatever arrived into the declared type's neighbourhood.

        A missing slot is already covered by the field defaults; this handles the
        slot that came back as `null`, or as a list because the model decided
        "parked cars, litter" was better expressed as two items. Booleans are left
        alone so Pydantic still applies its own true/false parsing.
        """
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return value


class SubjectSlots(_Slots):
    """What a character LoRA's caption may say — note the absence of face and hair."""

    shot: str = ""
    pose: str = ""
    clothing: str = ""
    background: str = ""
    lighting: str = ""
    tattoos_visible: bool = False
    asymmetric_marks: str = ""

    @field_validator("tattoos_visible", mode="before")
    @classmethod
    def _tolerant_bool(cls, value: object) -> bool:
        """Anything Pydantic would reject reads as False rather than failing the image.

        One slot the model phrased as "n/a" is not worth discarding a caption over —
        the other five are still usable.
        """
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "1", "y"}


class LocationSlots(_Slots):
    """What a place LoRA's caption may say — everything except the place itself."""

    time_of_day: str = ""
    weather: str = ""
    lighting: str = ""
    perspective: str = ""
    transient_objects: str = ""


Slots = SubjectSlots | LocationSlots


def instruction_for(mode: CaptionMode) -> str:
    """The prompt to send the VLM for `mode`."""
    return SUBJECT_INSTRUCTION if mode == "subject" else LOCATION_INSTRUCTION


def parse_slots(raw: str, mode: CaptionMode) -> Slots:
    """Read the model's reply into the slot schema for `mode`.

    Raises `CaptionSchemaError` rather than guessing, since a caption assembled
    from a misread reply would be indistinguishable from a good one once it is in
    the dataset. Qwen3-VL runs through `transformers.generate`, which has no
    grammar-constrained decoding to lean on, so malformed output is expected
    occasionally and is the caller's to report.
    """
    match = _JSON_BLOCK_RE.search(raw)
    if match is None:
        raise CaptionSchemaError(f"No JSON object in the model's reply: {raw[:120]!r}")

    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise CaptionSchemaError(f"Malformed JSON in the model's reply: {exc}") from exc

    if not isinstance(payload, dict):
        raise CaptionSchemaError(f"Expected a JSON object, got {type(payload).__name__}")

    model = SubjectSlots if mode == "subject" else LocationSlots
    return model.model_validate(payload)


def _spoken(value: str) -> str:
    """The slot's text, or empty when it is blank or one of the placeholder answers."""
    cleaned = " ".join(value.split()).strip(" ,")
    return "" if cleaned.lower() in _PLACEHOLDERS else cleaned


def assemble_caption(slots: Slots) -> str:
    """Join the filled slots into a caption, in a fixed order.

    Deterministic on purpose: the model decides *what* is in each slot, this
    decides how the caption reads. That keeps every caption in a dataset built the
    same way, which is the half of the technique that prompt wording alone cannot
    guarantee.
    """
    if isinstance(slots, SubjectSlots):
        parts = [slots.shot, slots.pose, slots.clothing, slots.background, slots.lighting]
        # Last, and the located phrase wins: "tattoo on their left forearm" already
        # says a tattoo is visible, so the flag only speaks when the model named no
        # side — a mark it could see but not place is still worth conditioning on.
        marks = _spoken(slots.asymmetric_marks)
        if marks:
            parts.append(marks)
        elif slots.tattoos_visible:
            parts.append("visible tattoos")
    else:
        parts = [slots.time_of_day, slots.weather, slots.lighting, slots.perspective]
        # Last, because it is the one slot whose content is incidental to the place.
        parts.append(slots.transient_objects)

    return ", ".join(filter(None, (_spoken(part) for part in parts)))
