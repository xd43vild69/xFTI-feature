"""Caption normalization and trigger-word injection: pure text transformations."""

import re

_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s,.\-']", flags=re.UNICODE)


def normalize_caption(caption: str) -> str:
    """Strip special characters, collapse repeated whitespace, and trim the result."""
    cleaned = _SPECIAL_CHARS_RE.sub("", caption)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def inject_trigger_word(caption: str, trigger_word: str) -> str:
    """Prefix the (normalized) caption with the trigger word, unless already present."""
    normalized = normalize_caption(caption)
    if not trigger_word:
        return normalized

    lowered = normalized.lower()
    trigger_lowered = trigger_word.lower()
    already_present = lowered == trigger_lowered or lowered.startswith(
        (f"{trigger_lowered} ", f"{trigger_lowered},")
    )
    if already_present:
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

