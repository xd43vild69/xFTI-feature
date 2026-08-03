"""Caption normalization and trigger-word injection: pure text transformations."""

import re

_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s,.\-']", flags=re.UNICODE)

# Cleanup for the gap a removed trigger word leaves behind — see `swap_trigger_word`.
_ORPHAN_COMMA_RE = re.compile(r"\s+,")
_REPEATED_COMMA_RE = re.compile(r",(\s*,)+")
_LEADING_SEPARATOR_RE = re.compile(r"^[\s,]+")


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


def append_word(caption: str, word: str) -> str:
    """Add `word` at the end of the caption, unless it is already in there.

    "Already in there" is `has_trigger`'s reading — the exact term, anywhere in the
    caption, any casing — so running this twice over a dataset is a no-op the second
    time, and a caption that mentions the word mid-sentence does not get a second
    copy tacked on. Position within the caption is not enforced: a caption the user
    ordered by hand is left as it is rather than rewritten to move the word.

    Trailing commas and periods at the join are dropped, since neither carries
    meaning where the new term is about to go.
    """
    if not word:
        return caption

    if has_trigger(caption, word):
        return caption

    base = caption.strip().rstrip(" ,.")
    return f"{base}, {word}" if base else word


def swap_trigger_word(caption: str, old_trigger: str, new_trigger: str) -> str:
    """Re-trigger a caption: drop `old_trigger`, then prefix `new_trigger`.

    Written for cloning a dataset under a different trigger word. Stripping on its
    own leaves behind the separator the old trigger sat next to — "sks, a cat"
    becomes ", a cat" — so the orphaned commas are collapsed before the new trigger
    goes on, rather than being carried into every caption of the new dataset.
    """
    if not old_trigger or old_trigger == new_trigger:
        return inject_trigger_word(caption, new_trigger)

    remainder = strip_trigger_word(caption, old_trigger)
    remainder = _ORPHAN_COMMA_RE.sub(",", remainder)
    remainder = _REPEATED_COMMA_RE.sub(",", remainder)
    remainder = _LEADING_SEPARATOR_RE.sub("", remainder)
    return inject_trigger_word(remainder, new_trigger)


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

