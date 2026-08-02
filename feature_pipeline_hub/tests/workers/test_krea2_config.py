"""Config resolution for the Krea 2 trainer.

These run in CI, unlike the golden scripts next door: `krea2.config` imports no torch,
which is the whole reason the package splits along that line. The goldens pin the
resolved output against the pre-refactor code; these pin the *rules*, so a future
change breaks with a readable assertion rather than a hash mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))

from krea2.config import DEFAULTS, PRESETS, TrainConfig, load_config  # noqa: E402


def write(directory: Path, name: str, payload: dict[str, object]) -> str:
    path = directory / name
    path.write_text(json.dumps(payload))
    return str(path)


def build(tmp_path: Path, settings: dict[str, object] | None = None,
          advanced: dict[str, object] | None = None, **kwargs: object) -> TrainConfig:
    """Resolve a config from in-memory settings, with logging muted."""
    return load_config(
        str(tmp_path),
        settings_path=write(tmp_path, "settings.json", settings or {}),
        advanced_path=(write(tmp_path, "advanced.json", advanced)
                       if advanced is not None else str(tmp_path / "absent.json")),
        cache_root=str(tmp_path / "cache"),
        output_root=str(tmp_path / "out"),
        env={},
        log=lambda _msg: None,
        **kwargs,  # type: ignore[arg-type]
    )


# ── precedence ──────────────────────────────────────────────────────────────

def test_defaults_apply_when_nothing_is_set(tmp_path: Path) -> None:
    cfg = build(tmp_path)
    assert cfg.total_steps == DEFAULTS["total_steps"]
    assert cfg.lora_rank == DEFAULTS["lora_rank"]
    assert cfg.sources["total_steps"] == "default"


def test_settings_beat_advanced_beat_preset_beat_defaults(tmp_path: Path) -> None:
    cfg = build(
        tmp_path,
        settings={"preset": "stable_v2", "lora_rank": 32},
        advanced={"lora_alpha": 64},
    )
    assert (cfg.lora_rank, cfg.sources["lora_rank"]) == (32, "json")
    assert (cfg.lora_alpha, cfg.sources["lora_alpha"]) == (64, "advanced")
    # sampler is only set by the preset...
    assert (cfg.sampler, cfg.sources["sampler"]) == ("epoch", "preset")
    # ...and lr by nothing at all.
    assert cfg.sources["lr"] == "default"


def test_preset_never_overrides_an_explicit_key(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"preset": "stable_v2", "sampler": "legacy"})
    assert cfg.sampler == "legacy"
    # Untouched preset keys still land.
    assert cfg.lora_dtype == PRESETS["stable_v2"]["lora_dtype"]


def test_unknown_preset_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"preset": "nope", "lora_rank": 8})
    assert cfg.preset_name == ""
    assert cfg.lora_rank == 8


def test_preset_can_come_from_the_advanced_sidecar(tmp_path: Path) -> None:
    cfg = build(tmp_path, advanced={"preset": "stable_v2"})
    assert cfg.preset_name == "stable_v2"
    assert cfg.sampler == "epoch"


def test_unreadable_files_are_skipped_not_fatal(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    cfg = load_config(str(tmp_path), settings_path=str(broken),
                      advanced_path=str(broken), cache_root=str(tmp_path),
                      output_root=str(tmp_path), env={}, log=lambda _m: None)
    assert cfg.total_steps == DEFAULTS["total_steps"]


# ── enum validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("key,expected", [
    ("lora_target", "all"), ("lora_dtype", "bf16"), ("sampler", "legacy"),
    ("resume_on_corrupt", "abort"), ("warmup_units", "updates"),
    ("optimizer", "adamw8bit_paged"), ("timestep_weighting", "none"),
    ("content_or_style", "balanced"), ("ema_device", "cpu"),
    ("lr_scheduler", "cosine"), ("loss_display", "cumulative"),
    ("preview_source", "caption"),
])
def test_invalid_enum_falls_back_instead_of_raising(
    tmp_path: Path, key: str, expected: str
) -> None:
    cfg = build(tmp_path, settings={key: "definitely-not-valid"})
    assert getattr(cfg, "optimizer" if key == "optimizer" else key) == expected


def test_enum_values_are_normalized(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"sampler": "  EPOCH  "})
    assert cfg.sampler == "epoch"


# ── warmup unit conversion ──────────────────────────────────────────────────

@pytest.mark.parametrize("units,warmup,expected", [
    ("updates", 50, 50.0),        # already in updates
    ("micro_steps", 400, 100.0),  # 400 / grad_accum 4
    ("ratio", 0.15, 45.0),        # 0.15 * 300 total updates
])
def test_warmup_converts_to_optimizer_updates(
    tmp_path: Path, units: str, warmup: float, expected: float
) -> None:
    cfg = build(tmp_path, settings={"warmup_units": units, "warmup_steps": warmup,
                                    "total_steps": 1200, "grad_accum_steps": 4})
    assert cfg.total_updates == 300.0
    assert cfg.warmup_updates == expected


def test_warmup_longer_than_the_run_is_clamped_to_a_tenth(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"warmup_steps": 5000, "total_steps": 1200,
                                    "grad_accum_steps": 4})
    assert cfg.warmup_updates == pytest.approx(30.0)


def test_total_updates_never_drops_below_one(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"total_steps": 1, "grad_accum_steps": 64})
    assert cfg.total_updates == 1.0


# ── derived values ──────────────────────────────────────────────────────────

def test_compact_text_is_disabled_when_batching(tmp_path: Path) -> None:
    """Compaction leaves each sample a different text length, so torch.cat breaks."""
    assert build(tmp_path, settings={"compact_text": True, "batch_size": 1}).compact_text
    assert not build(tmp_path, settings={"compact_text": True, "batch_size": 2}).compact_text


def test_project_name_derives_the_directories(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"project_name": "concepto",
                                    "cache_dir": "./ignored", "output_dir": "./ignored"})
    assert cfg.cache_dir == str(tmp_path / "cache" / "concepto")
    assert cfg.output_dir == str(tmp_path / "out" / "concepto")


def test_explicit_directories_are_honored_without_a_project_name(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"cache_dir": "./c", "output_dir": "./o"})
    assert cfg.cache_dir == str(tmp_path / "c")
    assert cfg.output_dir == str(tmp_path / "o")


def test_relative_paths_anchor_to_the_project_root(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"dataset_path": "./ds", "init_lora_from": "./prev",
                                    "val_cache_dir": "./val"})
    assert cfg.dataset_path == str(tmp_path / "ds")
    assert cfg.init_lora_from == str(tmp_path / "prev")
    assert cfg.val_cache_dir == str(tmp_path / "val")


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"dataset_path": "/data/elsewhere"})
    assert cfg.dataset_path == "/data/elsewhere"


def test_empty_optional_paths_stay_empty(tmp_path: Path) -> None:
    """An empty init_lora_from must not become the project root."""
    cfg = build(tmp_path)
    assert cfg.init_lora_from == ""
    assert cfg.val_cache_dir == ""


def test_checkpoint_paths_hang_off_the_output_dir(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"output_dir": "./o"})
    assert cfg.resume_dir == str(tmp_path / "o" / "resume_checkpoint")
    assert cfg.optimizer_state_file == str(tmp_path / "o" / "optimizer.pt")
    assert cfg.step_file == str(tmp_path / "o" / "current_step.txt")
    assert cfg.run_id_file == str(tmp_path / "o" / "run_id.txt")


def test_lora_scale_is_alpha_over_rank(tmp_path: Path) -> None:
    """A loader without alpha info assumes alpha == rank and runs the LoRA at half strength."""
    assert build(tmp_path, settings={"lora_rank": 16, "lora_alpha": 32}).lora_scale == 2.0


def test_multiphase_follows_phase_count(tmp_path: Path) -> None:
    assert not build(tmp_path).multiphase
    assert build(tmp_path, settings={"phase_count": 3}).multiphase


def test_global_total_steps_ignores_sidecar_and_preset(tmp_path: Path) -> None:
    """Only the progressive orchestrator writes it, so the sidecar has no say."""
    cfg = build(tmp_path, settings={"total_steps": 800},
                advanced={"global_total_steps": 9999})
    assert cfg.global_total_steps == 800

    cfg = build(tmp_path, settings={"total_steps": 800, "global_total_steps": 3600})
    assert cfg.global_total_steps == 3600


# ── invariants ──────────────────────────────────────────────────────────────

def test_config_is_frozen(tmp_path: Path) -> None:
    """Immutability is the point: the original reassigned 21 of these after definition."""
    cfg = build(tmp_path)
    with pytest.raises(Exception):
        cfg.total_steps = 5  # type: ignore[misc]


def test_hf_token_is_kept_out_of_repr_and_sources(tmp_path: Path) -> None:
    cfg = build(tmp_path, settings={"hf_token": "hf_secret_value"})
    assert cfg.hf_token == "hf_secret_value"
    assert "hf_secret_value" not in repr(cfg)
    assert "hf_token" not in cfg.sources


def test_every_default_key_reaches_a_field(tmp_path: Path) -> None:
    """Guards against a DEFAULTS entry that no longer wires to anything.

    Field names deliberately match the JSON keys one-for-one, so a mismatch here means
    a setting is silently being read from the file and then dropped.
    """
    cfg = build(tmp_path)
    assert [key for key in DEFAULTS if not hasattr(cfg, key)] == []
