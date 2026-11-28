"""Pins the Krea 2 "768×768 Turbo" recipe that produced the stable runs of 2026-11-26/27.

Any change to the preset, to a `TrainingConfig` default the preset leaves unset, or to
`TrainSettings`/`PrecacheSettings` makes these fail. If the change is intended, update
the recorded values here in the same commit, deliberately.
"""

import ast
from pathlib import Path

from feature_pipeline.application.training_service import (
    PrecacheSettings,
    TrainingConfig,
    TrainSettings,
)

TRAIN_PANEL = Path(__file__).resolve().parents[2] / "ui" / "components" / "train_panel.py"
PRESET_NAME = "2. Balance Calidad / Tiempo (768×768 Turbo)"
RUN_SPECIFIC = {"model_id", "dataset_path", "cache_dir", "output_dir", "trigger_word", "checkpoint_prefix"}

# config_json of training run 13a3361c (also identical for 38d96529, 7e064371, c6adbfb6).
STABLE_TRAIN_RECIPE = {
    "activation_offload": False,
    "batch_size": 1,
    "caption_dropout_rate": 0.05,
    "compile_transformer": True,
    "content_or_style": "balanced",
    "dop_weight": 0.0,
    "ema_decay": 0.98,
    "ema_device": "cpu",
    "grad_accum_steps": 1,
    "gradient_checkpointing": True,
    "gradient_checkpointing_interval": 1,
    "high_precision_targets": True,
    "lora_alpha": 48,
    "lora_dtype": "fp32",
    "lora_rank": 48,
    "lora_target": "attn+ff",
    "loraplus_lr_ratio": 3.0,
    "lr": 0.00015,
    "lr_num_cycles": 3,
    "lr_scheduler": "cosine",
    "min_lr_ratio": 0.1,
    "noise_offset": 0.0,
    "optimizer": "adamw8bit",
    "optimizer_betas": [0.9, 0.999],
    "optimizer_eps": 1e-08,
    "sampler": "epoch",
    "save_every": 250,
    "seed": 42,
    "timestep_sampling": "sigmoid",
    "timestep_weighting": "none",
    "total_steps": 1250,
    "use_ema": False,
    "warmup_steps": 60,
    "warmup_units": "micro_steps",
}

STABLE_PRECACHE_RECIPE = {
    "target_area": 768 * 768,
    "max_seq_len": 256,
    "prune_orphans": "delete",
}


def _krea2_presets() -> dict[str, dict[str, object]]:
    # Read from source: train_panel imports Streamlit-only modules (`state`, `components`).
    tree = ast.parse(TRAIN_PANEL.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "_KREA2_PRESETS"
    )
    assert node.value is not None
    presets: dict[str, dict[str, object]] = eval(
        compile(ast.Expression(node.value), str(TRAIN_PANEL), "eval"), {"__builtins__": {}}
    )
    return presets


def _config_from_preset() -> TrainingConfig:
    # Same steps the UI takes: defaults, preset on top, then validate the known fields.
    config = TrainingConfig().model_dump()
    config.update(_krea2_presets()[PRESET_NAME])
    return TrainingConfig(**{k: v for k, v in config.items() if k in TrainingConfig.model_fields})


def _without_run_specific(settings: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in settings.items() if k not in RUN_SPECIFIC}


def test_stable_preset_still_exists():
    assert PRESET_NAME in _krea2_presets()


def test_stable_preset_produces_the_recorded_train_settings():
    config = _config_from_preset()
    settings = TrainSettings(
        model_id="m", dataset_path="d", cache_dir="c", output_dir="o",
        trigger_word="t", checkpoint_prefix="p",
        **config.model_dump(exclude={"checkpoint_name", "target_area"}),
    ).model_dump()

    assert _without_run_specific(settings) == STABLE_TRAIN_RECIPE


def test_stable_preset_produces_the_recorded_precache_settings():
    config = _config_from_preset()
    settings = PrecacheSettings(
        model_id="m", dataset_path="d", cache_dir="c", trigger_word="t",
        target_area=config.target_area,
    ).model_dump()

    assert _without_run_specific(settings) == STABLE_PRECACHE_RECIPE
