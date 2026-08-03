from feature_pipeline.application.caption_service import (
    has_trigger,
    inject_trigger_word,
    normalize_caption,
    replace_exact_word,
    strip_trigger_word,
    swap_trigger_word,
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


# The trigger is matched on word boundaries anywhere in the caption, not just as a
# prefix: these are the cases where a substring or startswith check gets it wrong.


def test_inject_trigger_word_detects_a_trigger_mid_caption():
    assert inject_trigger_word("a cat in sks_style clothing", "sks_style") == (
        "a cat in sks_style clothing"
    )


def test_inject_trigger_word_prepends_when_the_trigger_is_only_a_substring():
    result = inject_trigger_word("a sks_stylevariant car", "sks_style")
    assert result == "sks_style, a sks_stylevariant car"


def test_has_trigger_matches_exact_terms_only():
    assert has_trigger("sks_style, a cat", "sks_style")
    assert has_trigger("a cat, SKS_STYLE", "sks_style")
    assert not has_trigger("a sks_stylevariant car", "sks_style")
    assert not has_trigger("anything", "")


def test_strip_trigger_word_leaves_unrelated_words_alone():
    assert strip_trigger_word("a cat in the garden", "cat") == "a  in the garden"
    assert strip_trigger_word("a caterpillar", "cat") == "a caterpillar"


def test_replace_exact_word_case_sensitive():
    new_text, count = replace_exact_word("Cat, cat, sitting", "cat", "dog")
    assert new_text == "Cat, dog, sitting"
    assert count == 1


def test_replace_exact_word_delimiters():
    new_text, count = replace_exact_word("1girl, solo, blue hair, cat, cat.", "cat", "dog")
    assert new_text == "1girl, solo, blue hair, dog, dog."
    assert count == 2


def test_replace_exact_word_avoids_substring_matches():
    new_text, count = replace_exact_word("caterpillar, cat, bobcat", "cat", "dog")
    assert new_text == "caterpillar, dog, bobcat"
    assert count == 1




def test_swap_trigger_word_replaces_the_old_trigger_without_leaving_a_dangling_comma():
    assert swap_trigger_word("sks_cat, a cat sitting", "sks_cat", "sks_cat2") == (
        "sks_cat2, a cat sitting"
    )


def test_swap_trigger_word_cleans_up_a_trigger_removed_from_mid_caption():
    assert swap_trigger_word("a photo of sks_cat, standing", "sks_cat", "new_cat") == (
        "new_cat, a photo of, standing"
    )


def test_swap_trigger_word_handles_a_caption_that_was_only_the_trigger():
    assert swap_trigger_word("sks_cat", "sks_cat", "new_cat") == "new_cat"


def test_swap_trigger_word_is_plain_injection_when_the_trigger_does_not_change():
    assert swap_trigger_word("sks_cat, a cat", "sks_cat", "sks_cat") == "sks_cat, a cat"
    assert swap_trigger_word("a cat", "", "sks_cat") == "sks_cat, a cat"
