from feature_pipeline.application.caption_service import (
    inject_trigger_word,
    normalize_caption,
)


def test_normalize_caption_collapses_whitespace():
    assert normalize_caption("a   cat   sitting") == "a cat sitting"


def test_normalize_caption_strips_special_characters():
    assert normalize_caption("a cat!! sitting @home #cute") == "a cat sitting home cute"


def test_normalize_caption_trims_edges():
    assert normalize_caption("   a cat   ") == "a cat"


def test_inject_trigger_word_prepends_when_absent():
    result = inject_trigger_word("a cat sitting", "sks_style")
    assert result == "sks_style, a cat sitting"


def test_inject_trigger_word_is_idempotent():
    once = inject_trigger_word("a cat sitting", "sks_style")
    twice = inject_trigger_word(once, "sks_style")
    assert once == twice


def test_inject_trigger_word_case_insensitive_detection():
    result = inject_trigger_word("SKS_STYLE a cat sitting", "sks_style")
    assert result.lower().count("sks_style") == 1


def test_inject_trigger_word_handles_empty_caption():
    assert inject_trigger_word("", "sks_style") == "sks_style"


def test_inject_trigger_word_no_trigger_returns_normalized():
    assert inject_trigger_word("a cat sitting", "") == "a cat sitting"
