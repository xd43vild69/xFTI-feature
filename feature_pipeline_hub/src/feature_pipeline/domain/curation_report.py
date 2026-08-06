"""Writing curation_report.json: per-image loss weights the trainer already reads.

`workers/krea2/curation.py` has always been able to train each image at its own
weight — scaling one sample's loss *is* a per-image learning rate — and
`curation_weights` defaults to True in the trainer's config. The only missing piece
was that nothing in the hub ever wrote the file it reads, so the feature sat inert.
This module writes it, which is why the feature lands without a single line changing
under `workers/`.

The file's schema is not ours to design: it comes from LoRAlab's curation scanner,
which scores every image and lets a threshold split them into groups. We are
producing that file synthetically, from tiers an operator assigned by hand rather
than from a score, so the encoding here exists to make `curation.load_weights`
resolve the groups we intend:

- `"good"` scores 1.0 and `"bad"` scores 0.0 against a 0.5 threshold, so
  `resolve_group` lands on the right side with no reliance on curation_overrides.json.
- `"priority"` is not a score at all — it is membership in `baselines`, which
  `load_weights` checks by stem *and* by the entry's `file`. Both are set, so the two
  agree no matter which branch of that check fires first.

Pure, in the same split as `checkpoint_log`: this knows the file's format, the
infrastructure layer writes it.
"""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["priority", "good", "bad"]

# The scores that encode a tier. `curation.resolve_group` compares `score >= threshold`,
# so any two values straddling the threshold would do — these are picked to be obvious
# in a file a human may end up reading.
GOOD_SCORE = 1.0
BAD_SCORE = 0.0
THRESHOLD = 0.5

TIERS: tuple[Tier, ...] = ("priority", "good", "bad")


class WeightProfile(BaseModel):
    """The three weight values, which are per-experiment rather than fixed.

    Defaults mirror `curation.load_weights`' own fallbacks, so a report written with
    an untouched profile trains identically to LoRAlab's.
    """

    model_config = ConfigDict(frozen=True)

    priority: float = Field(default=1.5, gt=0)
    good: float = Field(default=1.0, gt=0)
    bad: float = Field(default=0.5, gt=0)

    def weight_for(self, tier: Tier) -> float:
        return {"priority": self.priority, "good": self.good, "bad": self.bad}[tier]


class ImageEntry(BaseModel):
    """One image's row in the report's `images` map, keyed by stem."""

    score: float
    file: str


class CurationReport(BaseModel):
    """The exact shape `krea2.curation.load_weights` parses.

    `mode` is pinned to "face" because that is the only mode whose default group is
    "good": under "location" an unscored image would fall to "bad", and every image
    here is scored by construction, but pinning it keeps the two readings identical
    even if an entry is ever added without a score.
    """

    images: dict[str, ImageEntry]
    auto_threshold: float = THRESHOLD
    weights: WeightProfile = WeightProfile()
    mode: Literal["face"] = "face"
    baselines: list[str] = Field(default_factory=list)


class CurationReportError(ValueError):
    """A report that would not resolve the way the caller intended."""


def build_curation_report(
    files_by_stem: Mapping[str, str],
    tiers: Mapping[str, Tier],
    profile: WeightProfile,
) -> CurationReport:
    """A report assigning every exported image to a tier.

    `files_by_stem` maps each exported stem to its filename *with* extension — the
    report needs both, and the caller has just written the files so it is the only
    thing that knows the pairing.

    Every stem gets an entry, not only the ones the operator moved off "good".
    `load_weights` gives an image missing from the report weight 1.0 *and* prints a
    "re-run curation" warning per image; emitting all of them keeps that path cold and
    makes the report a complete record of the intervention rather than a diff against
    an implied default.

    Mirrored `__flip` variants need no entry: `load_weights` reduces every cache name
    through `source_stem()` before looking it up, so a flip inherits its source's
    weight. tests/workers/test_krea2_curation.py pins that.
    """
    unknown = sorted(set(tiers) - set(files_by_stem))
    if unknown:
        raise CurationReportError(
            "Tier assigned to stems that were not exported: " + ", ".join(unknown[:8])
        )

    images: dict[str, ImageEntry] = {}
    baselines: list[str] = []

    for stem, filename in files_by_stem.items():
        if "." in stem:
            # `load_weights` stems a baseline with `b.split(".")[0]` — the *first* dot,
            # not the last — so a stem containing one would promote some unrelated
            # shorter stem to priority. Exports go through naming.standardized_stem
            # (`<slug>_NNNN`), so this cannot happen in practice, which is exactly why
            # it is an assertion and not a hope.
            raise CurationReportError(
                f"Stem {stem!r} contains a dot; the report format cannot represent it."
            )

        tier = tiers.get(stem, "good")
        images[stem] = ImageEntry(
            score=BAD_SCORE if tier == "bad" else GOOD_SCORE,
            file=filename,
        )
        if tier == "priority":
            baselines.append(filename)

    return CurationReport(
        images=images,
        auto_threshold=THRESHOLD,
        weights=profile,
        baselines=baselines,
    )


def resolved_weights(report: CurationReport) -> dict[str, float]:
    """Stem -> weight, as `krea2.curation.load_weights` will resolve it.

    Duplicated logic rather than imported, for the same reason CHECKPOINT_LOG_COLUMNS
    is duplicated: `workers/` is not on the hub's sys.path and the training runtime is
    a different interpreter. tests/workers/test_krea2_curation.py runs both sides in
    one process and asserts they agree, which is what keeps the copy honest.
    """
    baseline_stems = {name.split(".")[0] for name in report.baselines}
    weights: dict[str, float] = {}
    for stem, entry in report.images.items():
        if stem in baseline_stems or entry.file in report.baselines:
            weights[stem] = report.weights.priority
        elif entry.score >= report.auto_threshold:
            weights[stem] = report.weights.good
        else:
            weights[stem] = report.weights.bad
    return weights


def is_effective(report: CurationReport) -> bool:
    """Whether writing this report changes training at all.

    `load_weights` collapses an all-1.0 result to `(None, None)` so that a curated run
    with nothing actually curated stays bit-identical to an uncurated one. Good
    behaviour, and a trap for an experiment: a branch that requested weights but
    resolved to all-ones would train exactly like its control while the database
    recorded an intervention. The caller checks this before launching.
    """
    return any(abs(w - 1.0) > 1e-9 for w in resolved_weights(report).values())


def tier_counts(tiers: Mapping[str, Tier], total: int) -> dict[Tier, int]:
    """How many images land in each tier, with the unassigned counted as "good"."""
    counts: dict[Tier, int] = {"priority": 0, "good": 0, "bad": 0}
    for tier in tiers.values():
        counts[tier] += 1
    counts["good"] += total - len(tiers)
    return counts
