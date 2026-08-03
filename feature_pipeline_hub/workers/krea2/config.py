"""Configuration resolution for the Krea 2 trainer: files and presets in, one frozen object out.

Layered precedence, highest first: train_settings.json, then the train_advanced.json
sidecar, then an optional preset bundle, then DEFAULTS. `load_config` resolves every
key once, applies the derivations that used to happen inline at module scope, and
returns an immutable `TrainConfig`.

Immutability is the point. The original resolved into ~93 module-level globals, 21 of
which were reassigned after their first definition — validated enums, path anchoring,
`compact_text` being force-disabled at batch > 1. Any module doing `from ... import
COMPACT_TEXT` would capture whichever value happened to exist at import time, and
nothing would fail loudly. A frozen object passed explicitly makes that unrepresentable.

Deliberately free of torch, so this module (and its tests) run in the hub environment
rather than the training runtime. `lora_dtype` stays a string here; mapping it to a
torch dtype belongs to the layer that has torch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# ── DEFAULTS ────────────────────────────────────────────────────────────────
# Everything added after the original block is additive and opt-in: with an empty
# sidecar and no preset, these values reproduce the historical behavior.
DEFAULTS: dict[str, Any] = {
    "model_id": "Krea-2-NF4",
    "cache_dir": "",          # filled from cache_root by load_config
    "output_dir": "",         # filled from output_root by load_config
    "total_steps": 1200,
    "batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 1e-4,
    "min_lr_ratio": 0.1,
    "warmup_steps": 100,
    "lora_rank": 16,
    "lora_alpha": 32,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "save_every": 25,
    "seed": 42,
    "timestep_sampling": "krea2_shift",
    "preview_every": 0,
    "preview_steps": 28,
    "preview_cfg": 3.5,
    "preview_caption_mode": "first",
    "project_name": "",
    "checkpoint_prefix": "",
    "trigger_word": "",
    "lora_target": "all",
    "compact_text": True,
    "init_lora_from": "",
    "gradient_checkpointing": True,

    # ── A1/A10: precision ────────────────────────────────────────────────────
    "lora_dtype": "bf16",              # "bf16" | "fp32" (fp32 = +235 MB VRAM)
    "high_precision_targets": False,   # generate noise and target in fp32

    # ── A2: dataset sampling ─────────────────────────────────────────────────
    "sampler": "legacy",               # "epoch" | "legacy"

    # ── A3/A8: guards ────────────────────────────────────────────────────────
    "nan_guard": True,
    "nan_abort_after": 20,
    "max_loss": 0.0,                   # 0 = disabled
    "oom_guard": True,
    "oom_abort_after": 3,

    # ── A4: checkpoints ──────────────────────────────────────────────────────
    "resume_on_corrupt": "abort",      # "abort" | "restart"

    # ── A5/A7: schedule and optimizer ────────────────────────────────────────
    "warmup_units": "updates",         # "updates" | "micro_steps" | "ratio"
    "optimizer": "adamw8bit_paged",    # | "adamw8bit" | "adamw"
    "optimizer_eps": 1e-8,
    "optimizer_betas": [0.9, 0.999],

    # ── B1: timesteps ────────────────────────────────────────────────────────
    "timestep_weighting": "none",      # "none" | "bell" | "half_bell"
    "logit_normal_mu": 0.0,
    "logit_normal_sigma": 1.0,
    "sigma_min": 0.0,
    "sigma_max": 1.0,
    "content_or_style": "balanced",    # "balanced" | "content" | "style"
    "noise_offset": 0.0,               # see B3: discouraged under rectified flow

    # ── B2: EMA ──────────────────────────────────────────────────────────────
    "use_ema": False,
    "ema_decay": 0.99,
    "ema_device": "cpu",               # "cpu" | "cuda"

    # ── B4: LR scheduler ─────────────────────────────────────────────────────
    "lr_scheduler": "cosine",          # | "constant" | "linear" | "cosine_with_restarts" | "step"
    "lr_num_cycles": 3,
    "lr_step_gamma": 0.5,
    "lr_step_count": 4,

    # ── C1: validation ───────────────────────────────────────────────────────
    "val_split": 0.0,
    "val_cache_dir": "",
    "validate_every": 0,
    "validation_sigmas": [0.25, 0.5, 0.75, 1.0],
    "val_seed": 1234,

    # ── C2/C3/C4: observability ──────────────────────────────────────────────
    "loss_window": 100,
    "loss_display": "cumulative",      # "window" | "cumulative"
    "csv_log": True,
    "max_checkpoints_to_keep": 0,      # 0 = keep all
    "export_metadata": True,
    "export_alpha_tensors": False,

    # ── C5: previews ─────────────────────────────────────────────────────────
    "preview_source": "caption",       # "caption" | "prompts"
    "preview_walk_seed": False,

    # ── D3: caption dropout ──────────────────────────────────────────────────
    "caption_dropout_rate": 0.0,

    # ── Dataset curation ─────────────────────────────────────────────────────
    # 0_curate_dataset.py splits the dataset into two groups by facial identity;
    # each group then trains at its own weight. With no curation_report.json in the
    # dataset folder this is an exact no-op, so an uncurated run behaves exactly as
    # it did before the option existed.
    "dataset_path": "./dataset",
    "curation_weights": True,

    # ── Progressive / multi-phase ────────────────────────────────────────────
    "run_id": "",
    "phase_index": 0,
    "phase_count": 1,
    "phase_label": "",
    "global_step_offset": 0,
}

# Bundles of recommended values. These change the *defaults* and never override an
# explicit user key. They exist so switching on the sensible set costs one keystroke
# instead of fifteen, while everything else stays opt-in.
PRESETS: dict[str, dict[str, Any]] = {
    "stable_v2": {
        "sampler": "epoch",
        "lora_dtype": "fp32",
        "optimizer_eps": 1e-6,
        "optimizer_betas": [0.9, 0.99],
        "high_precision_targets": True,
        "loss_display": "window",
        "max_checkpoints_to_keep": 5,
        "nan_guard": True,
        "oom_guard": True,
    },
}

# Allowed values per enum key, and the value to fall back to. A bad enum must never
# abort: a multi-hour job should not die over a typo in one optional setting.
CHOICES: dict[str, tuple[tuple[str, ...], str]] = {
    "lora_target": (("all", "attn", "attn+ff"), "all"),
    "lora_dtype": (("bf16", "fp32"), "bf16"),
    "sampler": (("epoch", "legacy"), "legacy"),
    "resume_on_corrupt": (("abort", "restart"), "abort"),
    "warmup_units": (("updates", "micro_steps", "ratio"), "updates"),
    "optimizer": (("adamw8bit_paged", "adamw8bit", "adamw"), "adamw8bit_paged"),
    "timestep_weighting": (("none", "bell", "half_bell"), "none"),
    "content_or_style": (("balanced", "content", "style"), "balanced"),
    "ema_device": (("cpu", "cuda"), "cpu"),
    "lr_scheduler": (("cosine", "constant", "linear", "cosine_with_restarts", "step"), "cosine"),
    "loss_display": (("window", "cumulative"), "cumulative"),
    "preview_source": (("caption", "prompts"), "caption"),
}

Logger = Callable[[str], None]


@dataclass(frozen=True)
class TrainConfig:
    """Every resolved training setting, immutable once built.

    Field names match the JSON keys, except where a derived value has no key of its
    own (`total_updates`, `warmup_updates`, `multiphase`, `preset_name`). Paths are
    already absolute and enums already validated, so consumers never re-derive.
    """

    project_root: str

    model_id: str
    cache_dir: str
    output_dir: str
    total_steps: int
    batch_size: int
    grad_accum_steps: int
    lr: float
    min_lr_ratio: float
    warmup_steps: float
    lora_rank: int
    lora_alpha: int
    weight_decay: float
    max_grad_norm: float
    save_every: int
    seed: int
    timestep_sampling: str
    preview_every: int
    preview_steps: int
    preview_cfg: float
    preview_caption_mode: str
    project_name: str
    checkpoint_prefix: str
    trigger_word: str
    lora_target: str
    compact_text: bool
    init_lora_from: str
    gradient_checkpointing: bool

    lora_dtype: str
    high_precision_targets: bool
    sampler: str

    nan_guard: bool
    nan_abort_after: int
    max_loss: float
    oom_guard: bool
    oom_abort_after: int
    resume_on_corrupt: str

    warmup_units: str
    optimizer: str
    optimizer_eps: float
    optimizer_betas: tuple[float, ...]

    timestep_weighting: str
    logit_normal_mu: float
    logit_normal_sigma: float
    sigma_min: float
    sigma_max: float
    content_or_style: str
    noise_offset: float

    use_ema: bool
    ema_decay: float
    ema_device: str

    lr_scheduler: str
    lr_num_cycles: int
    lr_step_gamma: float
    lr_step_count: int

    val_split: float
    val_cache_dir: str
    validate_every: int
    validation_sigmas: tuple[float, ...]
    val_seed: int

    loss_window: int
    loss_display: str
    csv_log: bool
    max_checkpoints_to_keep: int
    export_metadata: bool
    export_alpha_tensors: bool

    preview_source: str
    preview_walk_seed: bool
    caption_dropout_rate: float

    dataset_path: str
    curation_weights: bool

    run_id: str
    phase_index: int
    phase_count: int
    phase_label: str
    global_step_offset: int
    global_total_steps: int

    # Derived, with no JSON key of their own.
    preset_name: str
    total_updates: float
    warmup_updates: float
    multiphase: bool

    sources: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    # Secret: kept out of repr so it cannot reach a log or a crash dump, and resolved
    # straight from the settings file rather than through the layered resolver, so it
    # never appears in `sources` either.
    hf_token: str = field(default="", repr=False, compare=False)

    # ── Paths derived from output_dir ────────────────────────────────────────
    @property
    def resume_dir(self) -> str:
        return os.path.join(self.output_dir, "resume_checkpoint")

    @property
    def optimizer_state_file(self) -> str:
        return os.path.join(self.output_dir, "optimizer.pt")

    @property
    def step_file(self) -> str:
        return os.path.join(self.output_dir, "current_step.txt")

    @property
    def run_id_file(self) -> str:
        return os.path.join(self.output_dir, "run_id.txt")

    @property
    def lora_scale(self) -> float:
        """alpha / rank — what a loader must apply for the adapter to run at full strength."""
        return self.lora_alpha / max(1, self.lora_rank)


def from_root(project_root: str, path: str) -> str:
    """Resolve a relative path against the project root; absolute paths pass through."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


def _read_json(path: str, label: str, log: Logger) -> tuple[dict[str, Any], bool]:
    """Load a JSON config file, returning `({}, False)` when absent or unreadable."""
    if not os.path.exists(path):
        return {}, False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception as exc:
        log(f"[!] Could not read {path}: {exc} — ignoring / ignorando.")
        return {}, False
    if not isinstance(loaded, dict):
        log(f"[!] {label} at {path} is not a JSON object — ignoring / ignorando.")
        return {}, False
    return loaded, True


class _Resolver:
    """Applies the four-layer precedence and records where each value came from."""

    def __init__(self, settings: dict[str, Any], advanced: dict[str, Any],
                 preset: dict[str, Any], log: Logger) -> None:
        self.settings = settings
        self.advanced = advanced
        self.preset = preset
        self.log = log
        self.sources: dict[str, str] = {}

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.settings:
            source, value = "json", self.settings[key]
        elif key in self.advanced:
            source, value = "advanced", self.advanced[key]
        elif key in self.preset:
            source, value = "preset", self.preset[key]
        elif key in DEFAULTS:
            source, value = "default", DEFAULTS[key]
        else:
            source, value = "default", default
        self.sources[key] = source
        return value

    def choice(self, key: str) -> str:
        """Resolve an enum key, warning and falling back rather than raising."""
        allowed, fallback = CHOICES[key]
        value = str(self.get(key)).strip().lower()
        if value not in allowed:
            self.log(f"[!] Invalid {key} '{value}'. Using '{fallback}' / "
                     f"valor inválido. Usando '{fallback}'.")
            return fallback
        return value


def _resolve_warmup(warmup_steps: float, warmup_units: str, total_updates: float,
                    grad_accum_steps: int, log: Logger) -> float:
    """Convert `warmup_steps` into optimizer updates, then clamp it below the run length.

    The schedule runs in updates but `total_steps` is in micro-steps; mixing the two
    silently is this config's classic trap. With the defaults (100 / 1200 / GA=4) an
    unconverted warmup eats a third of the run.
    """
    if warmup_units == "micro_steps":
        warmup_updates = warmup_steps / max(1, grad_accum_steps)
    elif warmup_units == "ratio":
        warmup_updates = warmup_steps * total_updates
    else:
        warmup_updates = float(warmup_steps)

    if warmup_updates >= total_updates:
        log(f"[!] Warmup ({warmup_updates:.0f} updates) >= total updates "
            f"({total_updates:.0f}); clamping to 10% / recortando al 10%.")
        warmup_updates = 0.1 * total_updates
    return warmup_updates


def load_config(
    project_root: str,
    *,
    settings_path: str | None = None,
    advanced_path: str | None = None,
    cache_root: str | None = None,
    output_root: str | None = None,
    env: Mapping[str, str] | None = None,
    log: Logger = print,
) -> TrainConfig:
    """Resolve every setting into one frozen `TrainConfig`.

    Paths default to the env vars the progressive orchestrator sets per phase
    (`TRAIN_SETTINGS_PATH`, `TRAIN_ADVANCED_PATH`) so it never has to overwrite the
    user's own train_settings.json, falling back to files beside `project_root`.
    """
    environ = os.environ if env is None else env

    settings_path = settings_path or environ.get(
        "TRAIN_SETTINGS_PATH", os.path.join(project_root, "train_settings.json"))
    advanced_path = advanced_path or environ.get(
        "TRAIN_ADVANCED_PATH", os.path.join(project_root, "train_advanced.json"))
    cache_root = cache_root or environ.get(
        "CACHE_DIR", os.path.join(project_root, "cached_data_local"))
    output_root = output_root or environ.get(
        "OUTPUT_DIR", os.path.join(project_root, "output_local"))

    settings, found = _read_json(settings_path, "settings", log)
    if found:
        log(f"[OK] Configuration loaded from {settings_path} / "
            f"Configuración cargada desde {settings_path}")
    else:
        log(f"[!] {settings_path} not found, using default values / "
            f"No se encontró {settings_path}, usando valores por defecto.")

    advanced, found = _read_json(advanced_path, "advanced settings", log)
    if found:
        log(f"[OK] Advanced settings merged from {advanced_path} ({len(advanced)} keys)")

    preset_name = str(settings.get("preset", advanced.get("preset", ""))).strip()
    if preset_name and preset_name not in PRESETS:
        log(f"[!] Unknown preset '{preset_name}'. Valid: {', '.join(PRESETS)} — "
            f"ignoring / ignorando.")
        preset_name = ""

    r = _Resolver(settings, advanced, PRESETS.get(preset_name, {}), log)

    # The local model lives at the project root. It is anchored there only when that
    # folder actually exists, so a Hugging Face repo id still works; without this,
    # running from another directory would trigger a full model download.
    model_id = str(r.get("model_id"))
    if not os.path.isabs(model_id) and os.path.isdir(from_root(project_root, model_id)):
        model_id = from_root(project_root, model_id)

    project_name = str(r.get("project_name")).strip()
    batch_size = int(r.get("batch_size"))

    # Caption compaction leaves each sample a different text length, so torch.cat
    # would stop working; it is only applicable at batch 1.
    compact_text = bool(r.get("compact_text"))
    if compact_text and batch_size > 1:
        log("[!] compact_text requires batch_size 1; disabling / "
            "compact_text requiere batch_size 1; desactivado.")
        compact_text = False

    # Folder layout follows project_name when set. Without it, explicit
    # cache_dir/output_dir are honored — that is how run_progressive.py points each
    # phase at its own resolution subdir.
    if project_name:
        cache_dir = os.path.join(cache_root, project_name)
        output_dir = os.path.join(output_root, project_name)
    else:
        cache_dir = from_root(project_root, str(r.get("cache_dir") or f"{cache_root}/default"))
        output_dir = from_root(project_root, str(r.get("output_dir") or f"{output_root}/default"))

    total_steps = int(r.get("total_steps"))
    grad_accum_steps = int(r.get("grad_accum_steps"))
    total_updates = max(1.0, total_steps / max(1, grad_accum_steps))
    warmup_units = r.choice("warmup_units")
    warmup_steps = float(r.get("warmup_steps"))
    warmup_updates = _resolve_warmup(warmup_steps, warmup_units, total_updates,
                                     grad_accum_steps, log)

    init_lora_from = str(r.get("init_lora_from")).strip()
    if init_lora_from:
        init_lora_from = from_root(project_root, init_lora_from)
    val_cache_dir = str(r.get("val_cache_dir")).strip()
    if val_cache_dir:
        val_cache_dir = from_root(project_root, val_cache_dir)

    phase_count = int(r.get("phase_count"))

    return TrainConfig(
        project_root=project_root,
        model_id=model_id,
        cache_dir=cache_dir,
        output_dir=output_dir,
        total_steps=total_steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=float(r.get("lr")),
        min_lr_ratio=float(r.get("min_lr_ratio")),
        warmup_steps=warmup_steps,
        lora_rank=int(r.get("lora_rank")),
        lora_alpha=int(r.get("lora_alpha")),
        weight_decay=float(r.get("weight_decay")),
        max_grad_norm=float(r.get("max_grad_norm")),
        save_every=int(r.get("save_every")),
        seed=int(r.get("seed")),
        timestep_sampling=str(r.get("timestep_sampling")),
        preview_every=int(r.get("preview_every")),
        preview_steps=int(r.get("preview_steps")),
        preview_cfg=float(r.get("preview_cfg")),
        preview_caption_mode=str(r.get("preview_caption_mode")),
        project_name=project_name,
        checkpoint_prefix=str(r.get("checkpoint_prefix")).strip(),
        trigger_word=str(r.get("trigger_word")),
        lora_target=r.choice("lora_target"),
        compact_text=compact_text,
        init_lora_from=init_lora_from,
        gradient_checkpointing=bool(r.get("gradient_checkpointing")),

        lora_dtype=r.choice("lora_dtype"),
        high_precision_targets=bool(r.get("high_precision_targets")),
        sampler=r.choice("sampler"),

        nan_guard=bool(r.get("nan_guard")),
        nan_abort_after=int(r.get("nan_abort_after")),
        max_loss=float(r.get("max_loss")),
        oom_guard=bool(r.get("oom_guard")),
        oom_abort_after=int(r.get("oom_abort_after")),
        resume_on_corrupt=r.choice("resume_on_corrupt"),

        warmup_units=warmup_units,
        optimizer=r.choice("optimizer"),
        optimizer_eps=float(r.get("optimizer_eps")),
        optimizer_betas=tuple(float(b) for b in r.get("optimizer_betas")),

        timestep_weighting=r.choice("timestep_weighting"),
        logit_normal_mu=float(r.get("logit_normal_mu")),
        logit_normal_sigma=float(r.get("logit_normal_sigma")),
        sigma_min=float(r.get("sigma_min")),
        sigma_max=float(r.get("sigma_max")),
        content_or_style=r.choice("content_or_style"),
        noise_offset=float(r.get("noise_offset")),

        use_ema=bool(r.get("use_ema")),
        ema_decay=float(r.get("ema_decay")),
        ema_device=r.choice("ema_device"),

        lr_scheduler=r.choice("lr_scheduler"),
        lr_num_cycles=int(r.get("lr_num_cycles")),
        lr_step_gamma=float(r.get("lr_step_gamma")),
        lr_step_count=int(r.get("lr_step_count")),

        val_split=float(r.get("val_split")),
        val_cache_dir=val_cache_dir,
        validate_every=int(r.get("validate_every")),
        validation_sigmas=tuple(float(s) for s in r.get("validation_sigmas")),
        val_seed=int(r.get("val_seed")),

        loss_window=int(r.get("loss_window")),
        loss_display=r.choice("loss_display"),
        csv_log=bool(r.get("csv_log")),
        max_checkpoints_to_keep=int(r.get("max_checkpoints_to_keep")),
        export_metadata=bool(r.get("export_metadata")),
        export_alpha_tensors=bool(r.get("export_alpha_tensors")),

        preview_source=r.choice("preview_source"),
        preview_walk_seed=bool(r.get("preview_walk_seed")),
        caption_dropout_rate=float(r.get("caption_dropout_rate")),

        dataset_path=from_root(project_root, str(r.get("dataset_path")).strip()),
        curation_weights=bool(r.get("curation_weights")),

        run_id=str(r.get("run_id")).strip(),
        phase_index=int(r.get("phase_index")),
        phase_count=phase_count,
        phase_label=str(r.get("phase_label")).strip(),
        global_step_offset=int(r.get("global_step_offset")),
        # Read from the raw settings dict rather than through the resolver: the
        # orchestrator is the only writer, so the sidecar and presets have no say.
        global_total_steps=int(settings.get("global_total_steps", total_steps)),

        preset_name=preset_name,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        multiphase=phase_count > 1,
        sources=dict(r.sources),
        hf_token=str(settings.get("hf_token", "") or ""),
    )
