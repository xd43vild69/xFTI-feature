"""Generate the settings.json fixtures used to pin train_worker's config resolution.

Each fixture targets a specific derivation or reassignment path in the module-level
config block — the 21 globals that get assigned more than once are exactly where a
refactor is most likely to silently change behavior.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
FIXTURES = HERE / "fixtures"

CASES = {
    # Nothing set: every value must come from DEFAULTS.
    "empty": {},

    # Preset changes defaults but must never beat an explicit key.
    "preset_stable_v2": {"preset": "stable_v2"},
    "preset_overridden": {"preset": "stable_v2", "sampler": "legacy", "lora_dtype": "bf16"},

    # WARMUP_UNITS: three different conversions into optimizer updates.
    "warmup_updates": {"warmup_steps": 50, "warmup_units": "updates",
                       "total_steps": 1200, "grad_accum_steps": 4},
    "warmup_micro_steps": {"warmup_steps": 400, "warmup_units": "micro_steps",
                           "total_steps": 1200, "grad_accum_steps": 4},
    "warmup_ratio": {"warmup_steps": 0.15, "warmup_units": "ratio",
                     "total_steps": 1200, "grad_accum_steps": 4},
    # Warmup >= total updates must clamp to 10%.
    "warmup_clamp": {"warmup_steps": 5000, "warmup_units": "updates",
                     "total_steps": 1200, "grad_accum_steps": 4},

    # compact_text is force-disabled when batch_size > 1 (a reassignment).
    "compact_text_batch2": {"compact_text": True, "batch_size": 2},

    # project_name derives CACHE_DIR/OUTPUT_DIR instead of using the explicit ones.
    "project_name": {"project_name": "mi_concepto",
                     "cache_dir": "./ignored_cache", "output_dir": "./ignored_out"},
    "explicit_dirs": {"cache_dir": "./cache_x", "output_dir": "./out_x"},

    # Every _validate_choice fallback fires at once.
    "invalid_enums": {
        "lora_target": "nope", "lora_dtype": "fp8", "sampler": "chaos",
        "resume_on_corrupt": "maybe", "warmup_units": "furlongs",
        "optimizer": "sgd", "timestep_weighting": "gong",
        "content_or_style": "vibes", "ema_device": "tpu",
        "lr_scheduler": "spiral", "loss_display": "interpretive",
        "preview_source": "vibes",
    },
    "unknown_preset": {"preset": "does_not_exist"},

    # Progressive multi-phase context.
    "multiphase": {"run_id": "run-abc", "phase_index": 1, "phase_count": 3,
                   "phase_label": "768", "global_step_offset": 400,
                   "global_total_steps": 3600, "total_steps": 1200},

    # Sigma clamps and timestep options.
    "sigma_clamped": {"sigma_min": 0.2, "sigma_max": 0.8,
                      "timestep_sampling": "logit_normal",
                      "logit_normal_mu": 0.5, "logit_normal_sigma": 1.5},

    # EMA + validation + logging knobs together.
    "full_featured": {
        "use_ema": True, "ema_decay": 0.999, "ema_device": "cuda",
        "val_split": 0.2, "validate_every": 50, "val_seed": 99,
        "csv_log": True, "loss_display": "window", "loss_window": 50,
        "max_checkpoints_to_keep": 3, "caption_dropout_rate": 0.1,
        "curation_weights": True, "noise_offset": 0.05,
        "lr_scheduler": "cosine_with_restarts", "lr_num_cycles": 5,
    },
}

# Sidecar precedence: train_settings.json must beat train_advanced.json.
ADVANCED = {
    "advanced_sidecar": (
        {"lora_rank": 32},                                  # settings.json
        {"lora_rank": 8, "lora_alpha": 64, "sampler": "epoch"},  # train_advanced.json
    ),
}


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, cfg in CASES.items():
        (FIXTURES / f"{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name, (settings, advanced) in ADVANCED.items():
        (FIXTURES / f"{name}.json").write_text(json.dumps(settings, indent=2) + "\n")
        (FIXTURES / f"{name}.advanced.json").write_text(json.dumps(advanced, indent=2) + "\n")
    total = len(CASES) + len(ADVANCED)
    print(f"wrote {total} fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
