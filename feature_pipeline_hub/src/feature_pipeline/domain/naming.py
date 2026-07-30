"""Pure name-formatting rules for concepts and the images ingested under them.

No I/O: turns raw text into the trigger-word default and the standardized image
filename. Kept in the domain layer, not application, so infrastructure code that
writes files to disk can depend on it directly rather than reaching upward into
application code.
"""

import re

_SLUG_CHARS_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_SLUG_EDGE_UNDERSCORE_RE = re.compile(r"^_+|_+$")
STANDARD_INDEX_WIDTH = 4


def slugify(text: str) -> str:
    """Lowercase, `_`-joined form of arbitrary text.

    Runs of anything that is not a letter, digit, or underscore collapse to a
    single `_` — so "Cyberpunk Style!" and "cyberpunk-style" both become
    "cyberpunk_style".
    """
    slug = _SLUG_CHARS_RE.sub("_", text.strip().lower())
    return _SLUG_EDGE_UNDERSCORE_RE.sub("", slug)


def standardized_stem(concept_name: str, index: int) -> str:
    """Filename stem every imported image gets: `<slug>_0001`, `<slug>_0002`…"""
    return f"{slugify(concept_name)}_{index:0{STANDARD_INDEX_WIDTH}d}"


def _standard_index_pattern(concept_name: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(slugify(concept_name))}_(\d+)$")


def next_standard_index(existing_stems: list[str], concept_name: str) -> int:
    """The next free counter value for a concept, given filename stems already in use.

    Reads the highest `<slug>_NNNN` stem present and continues from there. That
    covers appending to a dataset that predates this naming scheme, or one that
    mixes standardized and non-standardized names: unmatched stems are simply
    ignored rather than corrupting the count, and a run with none yet starts at 1.
    """
    matches = (_standard_index_pattern(concept_name).match(stem) for stem in existing_stems)
    indices = [int(match.group(1)) for match in matches if match]
    return max(indices, default=0) + 1
