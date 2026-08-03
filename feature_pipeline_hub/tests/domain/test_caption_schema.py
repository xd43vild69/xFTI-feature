import json

import pytest

from feature_pipeline.application.caption_service import inject_trigger_word
from feature_pipeline.domain.caption_schema import (
    LOCATION_INSTRUCTION,
    SUBJECT_INSTRUCTION,
    CaptionSchemaError,
    LocationSlots,
    SubjectSlots,
    assemble_caption,
    instruction_for,
    parse_slots,
)


def _subject(**overrides) -> str:
    slots = {
        "shot": "medium shot",
        "pose": "standing facing camera",
        "clothing": "black leather jacket",
        "background": "plain gray studio background",
        "lighting": "soft studio lighting",
        "tattoos_visible": False,
    }
    slots.update(overrides)
    return json.dumps(slots)


# --- instructions -------------------------------------------------------------


def test_each_mode_gets_its_own_instruction():
    assert instruction_for("subject") == SUBJECT_INSTRUCTION
    assert instruction_for("location") == LOCATION_INSTRUCTION


def test_the_instructions_forbid_what_the_trigger_word_is_meant_to_learn():
    # The prohibition is the whole technique — a schema without it still gets a
    # face description stuffed into the "pose" slot.
    assert "Do NOT describe facial features, hair, body type" in SUBJECT_INSTRUCTION
    assert "Do NOT describe the permanent architectural features" in LOCATION_INSTRUCTION


# --- assembly -----------------------------------------------------------------


def test_subject_slots_assemble_in_a_fixed_order():
    caption = assemble_caption(parse_slots(_subject(), "subject"))

    assert caption == (
        "medium shot, standing facing camera, black leather jacket, "
        "plain gray studio background, soft studio lighting"
    )


def test_location_slots_assemble_in_a_fixed_order():
    reply = json.dumps(
        {
            "time_of_day": "night",
            "weather": "raining",
            "lighting": "neon lights",
            "perspective": "eye-level",
            "transient_objects": "parked cars",
        }
    )

    assert assemble_caption(parse_slots(reply, "location")) == (
        "night, raining, neon lights, eye-level, parked cars"
    )


def test_visible_tattoos_are_appended_only_when_the_flag_is_set():
    assert "visible tattoos" in assemble_caption(parse_slots(_subject(tattoos_visible=True), "subject"))
    assert "visible tattoos" not in assemble_caption(parse_slots(_subject(), "subject"))


def test_empty_slots_are_skipped_without_leaving_dangling_commas():
    caption = assemble_caption(parse_slots(_subject(pose="", lighting=""), "subject"))

    assert caption == "medium shot, black leather jacket, plain gray studio background"
    assert ", ," not in caption


@pytest.mark.parametrize("placeholder", ["none", "None", "null", "n/a", "unknown", ""])
def test_placeholder_answers_are_dropped_rather_than_captioned(placeholder):
    # Both prompts offer "none"/"unknown" as legitimate answers, and neither
    # belongs in a training caption.
    reply = json.dumps(
        {
            "time_of_day": "daytime",
            "weather": "sunny",
            "lighting": "natural light",
            "perspective": "wide angle shot",
            "transient_objects": placeholder,
        }
    )

    assert assemble_caption(parse_slots(reply, "location")) == (
        "daytime, sunny, natural light, wide angle shot"
    )


def test_slots_missing_from_the_reply_fall_back_to_empty():
    caption = assemble_caption(parse_slots(json.dumps({"shot": "close-up portrait"}), "subject"))

    assert caption == "close-up portrait"


# --- parsing ------------------------------------------------------------------


def test_json_wrapped_in_a_code_fence_is_still_read():
    raw = f"```json\n{_subject()}\n```"

    assert assemble_caption(parse_slots(raw, "subject")).startswith("medium shot")


def test_json_preceded_by_courtesy_prose_is_still_read():
    raw = f"Sure! Here is the JSON object you asked for:\n\n{_subject()}\n\nLet me know."

    assert assemble_caption(parse_slots(raw, "subject")).startswith("medium shot")


def test_a_reply_with_no_json_at_all_is_rejected():
    with pytest.raises(CaptionSchemaError):
        parse_slots("A man in a leather jacket stands in a studio.", "subject")


def test_malformed_json_is_rejected():
    with pytest.raises(CaptionSchemaError):
        parse_slots('{"shot": "medium shot", "pose":}', "subject")


def test_a_json_value_that_is_not_an_object_is_rejected():
    with pytest.raises(CaptionSchemaError):
        parse_slots('["medium shot", "standing"]', "subject")


def test_a_slot_returned_as_a_list_is_flattened():
    reply = json.dumps(
        {
            "time_of_day": "daytime",
            "weather": "sunny",
            "lighting": "natural light",
            "perspective": "aerial view",
            "transient_objects": ["people walking", "parked cars"],
        }
    )

    assert assemble_caption(parse_slots(reply, "location")).endswith(
        "people walking, parked cars"
    )


def test_a_null_slot_reads_as_empty_rather_than_failing():
    slots = parse_slots(json.dumps({"shot": "medium shot", "pose": None}), "subject")

    assert isinstance(slots, SubjectSlots)
    assert slots.pose == ""


def test_a_non_boolean_tattoo_flag_reads_as_false_rather_than_failing():
    # One slot the model phrased oddly is not worth discarding the other five over.
    assert parse_slots(_subject(tattoos_visible="n/a"), "subject").tattoos_visible is False
    assert parse_slots(_subject(tattoos_visible="yes"), "subject").tattoos_visible is True


def test_the_mode_decides_which_schema_is_used():
    assert isinstance(parse_slots(_subject(), "subject"), SubjectSlots)
    assert isinstance(parse_slots("{}", "location"), LocationSlots)


# --- downstream ---------------------------------------------------------------


def test_an_assembled_caption_survives_trigger_word_injection_intact():
    # normalize_caption (called by inject_trigger_word) strips anything outside
    # [\w\s,.\-'], so an assembled caption must not carry punctuation it would eat.
    caption = assemble_caption(parse_slots(_subject(), "subject"))

    assert inject_trigger_word(caption, "sks_person") == f"sks_person, {caption}"
