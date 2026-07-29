"""Caption normalization and trigger-word injection: pure text transformations."""

import re

_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s,.\-']", flags=re.UNICODE)


def normalize_caption(caption: str) -> str:
    """Strip special characters, collapse repeated whitespace, and trim the result."""
    cleaned = _SPECIAL_CHARS_RE.sub("", caption)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _trigger_pattern(trigger_word: str) -> re.Pattern[str]:
    """Case-insensitive exact-term matcher for a trigger word.

    Same non-word-boundary trick as `replace_exact_word`, so `sks_style` does not
    match inside `sks_stylevariant`.
    """
    return re.compile(r"(?<!\w)" + re.escape(trigger_word) + r"(?!\w)", flags=re.IGNORECASE)


def has_trigger(caption: str, trigger_word: str) -> bool:
    """True when the trigger word appears in the caption as a standalone term.

    Anywhere in the caption counts, not just at the front: a caption already
    carrying the trigger mid-sentence must not get a second copy prepended.
    """
    if not trigger_word:
        return False
    return _trigger_pattern(trigger_word).search(caption) is not None


def strip_trigger_word(caption: str, trigger_word: str) -> str:
    """Remove standalone occurrences of the trigger word, leaving the description."""
    if not trigger_word:
        return caption
    return _trigger_pattern(trigger_word).sub("", caption)


def inject_trigger_word(caption: str, trigger_word: str) -> str:
    """Prefix the (normalized) caption with the trigger word, unless already present."""
    normalized = normalize_caption(caption)
    if not trigger_word:
        return normalized

    if has_trigger(normalized, trigger_word):
        return normalized

    if not normalized:
        return trigger_word

    return f"{trigger_word}, {normalized}"


def replace_exact_word(caption: str, old_word: str, new_word: str) -> tuple[str, int]:
    """Replace exact occurrences of `old_word` with `new_word` in `caption` (case-sensitive).

    Uses non-word boundary checks (`(?<!\\w)` and `(?!\\w)`) to match exact terms
    delimited by commas, spaces, punctuation, or boundaries.

    Returns (new_caption, count_replaced).
    """
    if not old_word or old_word not in caption:
        return caption, 0

    pattern = re.compile(r"(?<!\w)" + re.escape(old_word) + r"(?!\w)")
    new_caption, count = pattern.subn(new_word, caption)
    return new_caption, count

