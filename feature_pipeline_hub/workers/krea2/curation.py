"""Per-image training weights derived from the dataset's curation report.

0_curate_dataset.py scores every image and splits the dataset into groups; each group
then trains at its own weight. Scaling one sample's loss *is* a per-image learning
rate, which is the whole mechanism here.

The report is regenerated on every scan, while the effective threshold and any manual
regroupings live in curation_overrides.json, owned by the UI, and win over the
automatic verdict.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable

Logger = Callable[[str], None]

FLIP_SUFFIX = "__flip"


def source_stem(cache_name: str) -> str:
    """Strip the mirrored-variant suffix to get the name curation actually scored.

    With flip_x the cache holds `img_001` and `img_001__flip` as separate entries, but
    only the source image was scored. Without this, half the dataset would silently
    train at weight 1.0.
    """
    return cache_name[:-len(FLIP_SUFFIX)] if cache_name.endswith(FLIP_SUFFIX) else cache_name


def resolve_group(score: float | None, threshold: float | None,
                  override: str | None, mode: str = "face") -> str:
    """Effective curation group for one image: "good" or "bad".

    A manual override wins over the score/threshold verdict. Exact replica of
    `resolve_curation_group()` in scripts/python/server.py — keep the two in sync.
    """
    if override in ("good", "bad"):
        return override
    default_group = "good" if mode == "face" else "bad"
    if score is None or threshold is None:
        return default_group
    return "good" if score >= threshold else "bad"


def load_weights(
    dataset_path: str,
    cache_names: Iterable[str],
    *,
    enabled: bool = True,
    log: Logger = print,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Per-cache-entry training weights from the curation report.

    Returns `(weights, summary)`, or `(None, None)` when there is nothing to apply — no
    report, the feature disabled, or every weight resolving to 1.0. In those cases
    training must stay bit-for-bit identical to a run with curation absent, which is
    why an all-ones result collapses rather than being passed through.
    """
    if not enabled:
        return None, None

    report_path = os.path.join(dataset_path, "curation_report.json")
    if not os.path.exists(report_path):
        return None, None
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception as exc:
        log(f"[!] curation_report.json ilegible ({exc}) — se entrena sin pesos / "
            f"training unweighted.")
        return None, None

    overrides: dict[str, Any] = {}
    overrides_path = os.path.join(dataset_path, "curation_overrides.json")
    if os.path.exists(overrides_path):
        try:
            with open(overrides_path, "r", encoding="utf-8") as handle:
                overrides = json.load(handle) or {}
        except Exception as exc:
            log(f"[!] curation_overrides.json ilegible ({exc}) — se usa el umbral automático.")

    images = report.get("images") or {}
    if not images:
        return None, None

    manual = overrides.get("threshold")
    threshold = manual if isinstance(manual, (int, float)) else report.get("auto_threshold")
    group_overrides = overrides.get("groups") or {}
    weights_cfg = report.get("weights") or {}
    mode = report.get("mode") or "face"
    w_priority = float(weights_cfg.get("priority", 1.5))
    w_good = float(weights_cfg.get("good", 1.0))
    w_bad = float(weights_cfg.get("bad", 0.5))

    baselines_raw = report.get("baselines") or []
    baseline_stems = {b.split(".")[0] for b in baselines_raw}

    weights: dict[str, float] = {}
    counts = {"priority": 0, "good": 0, "bad": 0}
    unscored: list[str] = []

    for name in cache_names:
        stem = source_stem(name)
        entry = images.get(stem)
        if entry is None:
            # In the cache but not in the report: added after the last scan. Full
            # weight — never penalize for missing data — and a warning.
            unscored.append(stem)
            weights[name] = 1.0
            continue

        if stem in baseline_stems or entry.get("file") in baselines_raw:
            counts["priority"] += 1
            weights[name] = w_priority
        else:
            group = resolve_group(entry.get("score"), threshold,
                                  group_overrides.get(stem), mode=mode)
            counts[group] += 1
            weights[name] = w_good if group == "good" else w_bad

    if unscored:
        unique = sorted(set(unscored))
        log(f"[!] {len(unique)} imagen(es) del caché no están en curation_report.json "
            f"(añadidas tras el último scan) — entrenan a peso ×1.0: "
            + ", ".join(unique[:8]) + ("…" if len(unique) > 8 else ""))
        log("    Vuelve a curar el dataset para incluirlas / re-run curation to include them.")

    if all(abs(w - 1.0) < 1e-9 for w in weights.values()):
        return None, None

    summary = {
        "priority": counts["priority"], "good": counts["good"], "bad": counts["bad"],
        "w_priority": w_priority, "w_good": w_good, "w_bad": w_bad,
        "threshold": threshold,
    }
    return weights, summary


def format_summary(summary: dict[str, Any]) -> str:
    """The curation block as printed at startup, so the operator sees the split applied."""
    threshold = summary["threshold"]
    shown = "auto" if threshold is None else f"{threshold * 100:.0f}%"
    lines = [f"\n[OK] Curaduría activa / curation active — umbral {shown}:"]
    if summary.get("priority"):
        lines.append(f"     ⭐ Alta Prioridad (Baselines) : {summary['priority']:>4} imagen(es) · "
                     f"peso ×{summary.get('w_priority', 1.5)}")
    lines.append(f"     Buena calificación           : {summary['good']:>4} imagen(es) · "
                 f"peso ×{summary['w_good']}")
    lines.append(f"     Baja calificación            : {summary['bad']:>4} imagen(es) · "
                 f"peso ×{summary['w_bad']}")
    return "\n".join(lines)
