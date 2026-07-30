from feature_pipeline.domain.naming import next_standard_index, slugify, standardized_stem


def test_slugify_lowercases_and_joins_words_with_underscores():
    assert slugify("Cyberpunk Style") == "cyberpunk_style"


def test_slugify_collapses_punctuation_and_repeated_separators():
    assert slugify("Cyberpunk -- Style!!") == "cyberpunk_style"


def test_slugify_trims_leading_and_trailing_separators():
    assert slugify("  __weird name__  ") == "weird_name"


def test_slugify_of_empty_text_is_empty():
    assert slugify("") == ""
    assert slugify("   ") == ""


def test_standardized_stem_pads_the_counter_to_four_digits():
    assert standardized_stem("Cyberpunk Style", 1) == "cyberpunk_style_0001"
    assert standardized_stem("Cyberpunk Style", 42) == "cyberpunk_style_0042"


def test_standardized_stem_survives_five_digit_counts_without_truncating():
    assert standardized_stem("cyberpunk_style", 12345) == "cyberpunk_style_12345"


def test_next_standard_index_starts_at_one_for_an_empty_dataset():
    assert next_standard_index([], "cyberpunk_style") == 1


def test_next_standard_index_continues_from_the_highest_existing_stem():
    stems = ["cyberpunk_style_0001", "cyberpunk_style_0003", "cyberpunk_style_0002"]
    assert next_standard_index(stems, "cyberpunk_style") == 4


def test_next_standard_index_ignores_stems_from_other_concepts():
    stems = ["other_concept_0099"]
    assert next_standard_index(stems, "cyberpunk_style") == 1


def test_next_standard_index_ignores_non_standard_filenames():
    stems = ["n_1", "IMG_20240101_120000", "cyberpunk_style_0005"]
    assert next_standard_index(stems, "cyberpunk_style") == 6
