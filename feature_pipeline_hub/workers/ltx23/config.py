"""Configuration resolution for the LTX 2.3 trainer: files and presets in, one frozen object out.

Layered precedence: train_settings.json, then an optional preset bundle, then DEFAULTS.
`load_config` resolves every key once, applies derivations, and returns an immutable
`LTX23TrainConfig`.

Deliberately free of torch, so this module (and its tests) run in the hub environment
rather than the training runtime.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

DEFAULTS: dict[str, Any] = {
    "model_id": "./LTX23-NF4",
    "cache_dir": "./cached_data_ltx23",
    "output_dir": "./ltx23_lora_output",
    "total_steps": 3200,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1.5e-4,
    "min_lr_ratio": 0.05,
    "warmup_steps": 150,
    "lora_rank": 32,
    "lora_alpha": 32,
    "weight_decay": 0.001,
    "max_grad_norm": 1.0,
    "save_every": 200,
    "seed": 314159,
    "frame_rate": 24.0,
    "project_name": "",
    "trigger_word": "",
    "max_text_tokens": 256,
    "lora_only_attn": True,
    "cast_frozen_bf16": True,
    "use_audio_loss": False,
    "preview_every": 0,
    "preview_steps": 30,
    "preview_cfg": 3.0,
    "preview_sampler": "euler",
    "preview_lora_scale": 1.0,
    "preview_cfg_max": 7.0,
    "preview_cfg_rescale": 0.0,
    "preview_caption_mode": "first",
    "preview_custom_prompt": "",
    "preview_mode": "gen",  # gen | recon | onestep
    "preview_recon_sigma": 0.55,
    "preview_frame_index": -1,
    "preview_shift": 1.0,
    "preview_vae_fp32": True,
    "preview_audio_cfg": 1.0,
    "preview_sample_name": "",
    "preview_compare_base": False,
    "preview_vae_use_scaling_factor": True,
    "lora_key_prefix": "diffusion_model.",
    "low_vram_12gb": True,
    "activation_offload": True,
    "loss_chunk_elements": 2000000,
    "timestep_sampling": "logit_normal",
    "timestep_shift": 1.0,
    "use_loss_weighting": False,
    "caption_dropout_prob": 0.05,
    "conditioning_mode": "i2v",  # t2v | i2v
    "lr_schedule": "constant_with_warmup",  # constant_with_warmup | cosine
    "cond_noise_prob": 0.15,
    "cond_noise_scale": 0.03,
    "use_ema": True,
    "use_dora": False,
}

CHOICES: dict[str, tuple[tuple[str, ...], str]] = {
    "preview_mode": (("gen", "recon", "onestep"), "gen"),
    "preview_caption_mode": (("first", "random", "custom"), "first"),
    "timestep_sampling": (("logit_normal", "uniform"), "logit_normal"),
    "conditioning_mode": (("t2v", "i2v"), "i2v"),
    "lr_schedule": (("constant_with_warmup", "cosine"), "constant_with_warmup"),
}

Logger = Callable[[str], None]


@dataclass(frozen=True)
class LTX23TrainConfig:
    """Every resolved LTX 2.3 training setting, immutable once built."""

    project_root: str

    model_id: str
    cache_dir: str
    output_dir: str
    total_steps: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    min_lr_ratio: float
    warmup_steps: int
    lora_rank: int
    lora_alpha: int
    weight_decay: float
    max_grad_norm: float
    save_every: int
    seed: int
    frame_rate: float
    project_name: str
    trigger_word: str
    max_text_tokens: int
    lora_only_attn: bool
    cast_frozen_bf16: bool
    use_audio_loss: bool
    preview_every: int
    preview_steps: int
    preview_cfg: float
    preview_sampler: str
    preview_lora_scale: float
    preview_cfg_max: float
    preview_cfg_rescale: float
    preview_caption_mode: str
    preview_custom_prompt: str
    preview_mode: str
    preview_recon_sigma: float
    preview_frame_index: int
    preview_shift: float
    preview_vae_fp32: bool
    preview_audio_cfg: float
    preview_sample_name: str
    preview_compare_base: bool
    preview_vae_use_scaling_factor: bool
    lora_key_prefix: str
    low_vram_12gb: bool
    activation_offload: bool
    loss_chunk_elements: int
    timestep_sampling: str
    timestep_shift: float
    use_loss_weighting: bool
    caption_dropout_prob: float
    conditioning_mode: str
    lr_schedule: str
    cond_noise_prob: float
    cond_noise_scale: float
    use_ema: bool
    use_dora: bool
    preset_name: str | None = None


def _cfg_get(sources: list[Mapping[str, Any]], key: str, default: Any) -> Any:
    for src in sources:
        if key in src:
            return src[key]
        if (key + " ") in src:
            return src[key + " "]
        if (" " + key) in src:
            return src[" " + key]
    return default


def _cfg_bool(sources: list[Mapping[str, Any]], key: str, default: bool) -> bool:
    val = _cfg_get(sources, key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(val)


def _cfg_choice(sources: list[Mapping[str, Any]], key: str, log: Logger = print) -> str:
    allowed, fallback = CHOICES[key]
    raw = _cfg_get(sources, key, fallback)
    val = str(raw).strip().lower() if raw is not None else fallback
    if val not in allowed:
        log(f"   [!] Unknown value {raw!r} for {key}; using {fallback!r}")
        return fallback
    return val


def _anchor_path(path: str, root: str) -> str:
    """Make relative paths absolute against root."""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root, path))


def load_config(
    settings_path: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    project_root: str | None = None,
    log: Logger = print,
) -> LTX23TrainConfig:
    """Resolve and build an immutable LTX23TrainConfig."""
    root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    file_cfg: dict[str, Any] = {}
    path = settings_path or os.environ.get("TRAIN_SETTINGS_PATH", os.path.join(root, "train_settings.json"))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            log(f"✓ Configuration loaded from / Configuración cargada desde: {path}")
        except Exception as err:
            log(f"[!] Warning reading {path}: {err}; using defaults.")
    else:
        log(f"⚠ Settings file not found at {path}, using defaults.")

    sources: list[Mapping[str, Any]] = [overrides or {}, file_cfg, DEFAULTS]

    project_name = str(_cfg_get(sources, "project_name", DEFAULTS["project_name"])).strip()

    explicit_cache = _cfg_get([overrides or {}, file_cfg], "cache_dir", None)
    if explicit_cache is not None and str(explicit_cache).strip():
        cache_dir = str(explicit_cache).strip()
    elif project_name:
        cache_dir = f"./cached_data_ltx23_{project_name}"
    else:
        cache_dir = str(DEFAULTS["cache_dir"]).strip()
    cache_dir = _anchor_path(cache_dir, root)

    explicit_output = _cfg_get([overrides or {}, file_cfg], "output_dir", None)
    if explicit_output is not None and str(explicit_output).strip():
        output_dir = str(explicit_output).strip()
    elif project_name:
        output_dir = f"./ltx23_lora_output_{project_name}"
    else:
        output_dir = str(DEFAULTS["output_dir"]).strip()
    output_dir = _anchor_path(output_dir, root)

    model_id = str(_cfg_get(sources, "model_id", DEFAULTS["model_id"])).strip()
    if not os.path.isabs(model_id) and os.path.isdir(os.path.join(root, model_id)):
        model_id = os.path.normpath(os.path.join(root, model_id))

    return LTX23TrainConfig(
        project_root=root,
        model_id=model_id,
        cache_dir=cache_dir,
        output_dir=output_dir,
        total_steps=int(_cfg_get(sources, "total_steps", DEFAULTS["total_steps"])),
        batch_size=int(_cfg_get(sources, "batch_size", DEFAULTS["batch_size"])),
        grad_accum_steps=int(_cfg_get(sources, "grad_accum_steps", DEFAULTS["grad_accum_steps"])),
        lr=float(_cfg_get(sources, "lr", DEFAULTS["lr"])),
        min_lr_ratio=float(_cfg_get(sources, "min_lr_ratio", DEFAULTS["min_lr_ratio"])),
        warmup_steps=int(_cfg_get(sources, "warmup_steps", DEFAULTS["warmup_steps"])),
        lora_rank=int(_cfg_get(sources, "lora_rank", DEFAULTS["lora_rank"])),
        lora_alpha=int(_cfg_get(sources, "lora_alpha", DEFAULTS["lora_alpha"])),
        weight_decay=float(_cfg_get(sources, "weight_decay", DEFAULTS["weight_decay"])),
        max_grad_norm=float(_cfg_get(sources, "max_grad_norm", DEFAULTS["max_grad_norm"])),
        save_every=int(_cfg_get(sources, "save_every", DEFAULTS["save_every"])),
        seed=int(_cfg_get(sources, "seed", DEFAULTS["seed"])),
        frame_rate=float(_cfg_get(sources, "frame_rate", DEFAULTS["frame_rate"])),
        project_name=project_name,
        trigger_word=str(_cfg_get(sources, "trigger_word", DEFAULTS["trigger_word"])).strip(),
        max_text_tokens=int(_cfg_get(sources, "max_text_tokens", DEFAULTS["max_text_tokens"]) or 0),
        lora_only_attn=_cfg_bool(sources, "lora_only_attn", DEFAULTS["lora_only_attn"]),
        cast_frozen_bf16=_cfg_bool(sources, "cast_frozen_bf16", DEFAULTS["cast_frozen_bf16"]),
        use_audio_loss=_cfg_bool(sources, "use_audio_loss", DEFAULTS["use_audio_loss"]),
        preview_every=int(_cfg_get(sources, "preview_every", DEFAULTS["preview_every"])),
        preview_steps=int(_cfg_get(sources, "preview_steps", DEFAULTS["preview_steps"])),
        preview_cfg=float(_cfg_get(sources, "preview_cfg", DEFAULTS["preview_cfg"])),
        preview_sampler=str(_cfg_get(sources, "preview_sampler", DEFAULTS["preview_sampler"])).strip(),
        preview_lora_scale=float(_cfg_get(sources, "preview_lora_scale", DEFAULTS["preview_lora_scale"])),
        preview_cfg_max=float(_cfg_get(sources, "preview_cfg_max", DEFAULTS["preview_cfg_max"])),
        preview_cfg_rescale=float(_cfg_get(sources, "preview_cfg_rescale", DEFAULTS["preview_cfg_rescale"])),
        preview_caption_mode=_cfg_choice(sources, "preview_caption_mode", log),
        preview_custom_prompt=str(_cfg_get(sources, "preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip(),
        preview_mode=_cfg_choice(sources, "preview_mode", log),
        preview_recon_sigma=float(_cfg_get(sources, "preview_recon_sigma", DEFAULTS["preview_recon_sigma"])),
        preview_frame_index=int(_cfg_get(sources, "preview_frame_index", DEFAULTS["preview_frame_index"])),
        preview_shift=float(_cfg_get(sources, "preview_shift", DEFAULTS["preview_shift"])),
        preview_vae_fp32=_cfg_bool(sources, "preview_vae_fp32", DEFAULTS["preview_vae_fp32"]),
        preview_audio_cfg=float(_cfg_get(sources, "preview_audio_cfg", DEFAULTS["preview_audio_cfg"])),
        preview_sample_name=str(_cfg_get(sources, "preview_sample_name", DEFAULTS["preview_sample_name"])).strip(),
        preview_compare_base=_cfg_bool(sources, "preview_compare_base", DEFAULTS["preview_compare_base"]),
        preview_vae_use_scaling_factor=_cfg_bool(sources, "preview_vae_use_scaling_factor", DEFAULTS["preview_vae_use_scaling_factor"]),
        lora_key_prefix=str(_cfg_get(sources, "lora_key_prefix", DEFAULTS["lora_key_prefix"])),
        low_vram_12gb=_cfg_bool(sources, "low_vram_12gb", DEFAULTS["low_vram_12gb"]),
        activation_offload=_cfg_bool(sources, "activation_offload", DEFAULTS["activation_offload"]),
        loss_chunk_elements=int(_cfg_get(sources, "loss_chunk_elements", DEFAULTS["loss_chunk_elements"])),
        timestep_sampling=_cfg_choice(sources, "timestep_sampling", log),
        timestep_shift=float(_cfg_get(sources, "timestep_shift", DEFAULTS["timestep_shift"])),
        use_loss_weighting=_cfg_bool(sources, "use_loss_weighting", DEFAULTS["use_loss_weighting"]),
        caption_dropout_prob=float(_cfg_get(sources, "caption_dropout_prob", DEFAULTS["caption_dropout_prob"])),
        conditioning_mode=_cfg_choice(sources, "conditioning_mode", log),
        lr_schedule=_cfg_choice(sources, "lr_schedule", log),
        cond_noise_prob=float(_cfg_get(sources, "cond_noise_prob", DEFAULTS["cond_noise_prob"])),
        cond_noise_scale=float(_cfg_get(sources, "cond_noise_scale", DEFAULTS["cond_noise_scale"])),
        use_ema=_cfg_bool(sources, "use_ema", DEFAULTS["use_ema"]),
        use_dora=_cfg_bool(sources, "use_dora", DEFAULTS["use_dora"]),
    )
