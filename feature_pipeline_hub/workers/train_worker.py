# -*- coding: utf-8 -*-
"""
train_worker.py — LoRA training for Krea 2 (RAW) on an NF4-quantized base.

Runs as a detached process in the training runtime's own virtualenv, reading
VAE latents and text embeddings that the pre-cache stage already wrote to disk,
and emitting LoRA checkpoints plus a final `Krea2_FINAL_LoRA.safetensors`.

Originally ported from AcademiaSD_LoRAlab-Krea2's `2_train_lora_krea2.py` and
still tracks it for *logic* — the training loop, guards, checkpointing, curation
weighting and progressive-phase handling are unchanged, so behavioral drift
against upstream stays reviewable. The documentation has deliberately diverged:
docstrings here are English-only and describe each function at its definition.
Two other local changes: PROJECT_ROOT is one directory shallower (this file
lives in workers/, not scripts/python/), and __main__ is wrapped by _telemetry
to emit lifecycle events. Inline comments and log messages remain as upstream
wrote them, bilingual.

Configuration is read from train_settings.json when present, layered over
train_advanced.json, an optional preset, and DEFAULTS — in that precedence.
"""
import os
import gc
import time
import random
import sys
import zlib
import collections

# Debe fijarse antes de que se inicialice el asignador CUDA. Reduce la
# fragmentación de VRAM, que es el modo de fallo típico en GPUs de 12 GB al
# entrenar a 768x768 o más.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Project root. This worker lives in workers/, one level shallower
# than scripts/python/ in LoRAlab — hence 2 dirname() here, not 3. Every path is
# anchored here rather than to the working directory, so it works invoked from
# anywhere.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The trainer's pieces live in the krea2 package. It is importable because this file is
# launched as a script, which puts workers/ on sys.path — the same reason
# `from _telemetry import ...` works below.
from krea2 import config as _config             # noqa: E402
from krea2 import curation as _curation         # noqa: E402
from krea2 import dataset as _dataset           # noqa: E402
from krea2 import ema as _ema                   # noqa: E402
from krea2 import lora_io as _lora_io           # noqa: E402
from krea2 import math_ops as _math             # noqa: E402
from krea2 import metrics as _metrics           # noqa: E402
from krea2 import preview as _preview           # noqa: E402
from krea2 import quantization as _quant        # noqa: E402
from krea2 import sampling as _sampling         # noqa: E402
from krea2 import schedule as _schedule         # noqa: E402
from krea2 import state as _state               # noqa: E402


def from_root(path):
    """Resolve a relative path against the project root; absolute paths pass through."""
    return _config.from_root(PROJECT_ROOT, path)


# One frozen object, resolved once. The module-level names below are read-only aliases
# onto it, kept so the rest of this file reads as it always has; they are retired as
# each section moves into the package.
CFG = _config.load_config(PROJECT_ROOT)

MODEL_ID           = CFG.model_id
TOTAL_STEPS        = CFG.total_steps
BATCH_SIZE         = CFG.batch_size
GRAD_ACCUM_STEPS   = CFG.grad_accum_steps
LR                 = CFG.lr
MIN_LR_RATIO       = CFG.min_lr_ratio
WARMUP_STEPS       = CFG.warmup_steps
LORA_RANK          = CFG.lora_rank
LORA_ALPHA         = CFG.lora_alpha
WEIGHT_DECAY       = CFG.weight_decay
MAX_GRAD_NORM      = CFG.max_grad_norm
SAVE_EVERY         = CFG.save_every
SEED               = CFG.seed
TIMESTEP_SAMPLING  = CFG.timestep_sampling
PREVIEW_EVERY      = CFG.preview_every
PREVIEW_STEPS      = CFG.preview_steps
PREVIEW_CFG        = CFG.preview_cfg
PREVIEW_CAPTION_MODE = CFG.preview_caption_mode
TRIGGER_WORD       = CFG.trigger_word
PROJECT_NAME       = CFG.project_name
LORA_TARGET        = CFG.lora_target
COMPACT_TEXT       = CFG.compact_text
INIT_LORA_FROM     = CFG.init_lora_from
GRAD_CHECKPOINTING = CFG.gradient_checkpointing
RUN_ID             = CFG.run_id
PHASE_INDEX        = CFG.phase_index
PHASE_COUNT        = CFG.phase_count
PHASE_LABEL        = CFG.phase_label
GLOBAL_STEP_OFFSET = CFG.global_step_offset
GLOBAL_TOTAL_STEPS = CFG.global_total_steps
MULTIPHASE         = CFG.multiphase

LORA_DTYPE_NAME    = CFG.lora_dtype
HIGH_PREC_TARGETS  = CFG.high_precision_targets
SAMPLER_MODE       = CFG.sampler
NAN_GUARD          = CFG.nan_guard
NAN_ABORT_AFTER    = CFG.nan_abort_after
MAX_LOSS           = CFG.max_loss
OOM_GUARD          = CFG.oom_guard
OOM_ABORT_AFTER    = CFG.oom_abort_after
RESUME_ON_CORRUPT  = CFG.resume_on_corrupt
WARMUP_UNITS       = CFG.warmup_units
OPTIMIZER_NAME     = CFG.optimizer
OPTIMIZER_EPS      = CFG.optimizer_eps
OPTIMIZER_BETAS    = CFG.optimizer_betas
TIMESTEP_WEIGHTING = CFG.timestep_weighting
LOGIT_NORMAL_MU    = CFG.logit_normal_mu
LOGIT_NORMAL_SIGMA = CFG.logit_normal_sigma
SIGMA_MIN          = CFG.sigma_min
SIGMA_MAX          = CFG.sigma_max
CONTENT_OR_STYLE   = CFG.content_or_style
NOISE_OFFSET       = CFG.noise_offset
USE_EMA            = CFG.use_ema
EMA_DECAY          = CFG.ema_decay
EMA_DEVICE         = CFG.ema_device
LR_SCHEDULER       = CFG.lr_scheduler
LR_NUM_CYCLES      = CFG.lr_num_cycles
LR_STEP_GAMMA      = CFG.lr_step_gamma
LR_STEP_COUNT      = CFG.lr_step_count
VAL_SPLIT          = CFG.val_split
VAL_CACHE_DIR      = CFG.val_cache_dir
VALIDATE_EVERY     = CFG.validate_every
VALIDATION_SIGMAS  = list(CFG.validation_sigmas)
VAL_SEED           = CFG.val_seed
LOSS_WINDOW        = CFG.loss_window
LOSS_DISPLAY       = CFG.loss_display
CSV_LOG            = CFG.csv_log
MAX_CKPT_KEEP      = CFG.max_checkpoints_to_keep
EXPORT_METADATA    = CFG.export_metadata
EXPORT_ALPHA_TENSORS = CFG.export_alpha_tensors
PREVIEW_SOURCE     = CFG.preview_source
PREVIEW_WALK_SEED  = CFG.preview_walk_seed
CAPTION_DROPOUT    = CFG.caption_dropout_rate
DATASET_PATH       = CFG.dataset_path
CURATION_WEIGHTS   = CFG.curation_weights

CACHE_DIR          = CFG.cache_dir
OUTPUT_DIR         = CFG.output_dir
TOTAL_UPDATES      = CFG.total_updates
WARMUP_UPDATES     = CFG.warmup_updates
PRESET_NAME        = CFG.preset_name
_CFG_SOURCE        = dict(CFG.sources)

# A1: master dtype of the LoRA weights. In bf16 (8-bit mantissa) the relative epsilon
# is ~0.0039, so any update smaller than 0.39% of the weight's magnitude rounds away:
# late in the cosine, with the LR already low, a good share of updates simply never
# land. PEFT casts the input to lora_A's dtype and the result back, so fp32 works over
# the NF4 base without touching it (cost: ~+235 MB VRAM).
LORA_DTYPE  = torch.float32 if LORA_DTYPE_NAME == "fp32" else torch.bfloat16
# A1b: the model's compute dtype is a known constant, not something inferred from
# `next(model.parameters())` — that order depends on PEFT's wrapping and on
# quantization, and would one day hand back a Params4bit (uint8).
MODEL_DTYPE = torch.bfloat16


def _print_effective_config():
    """Dump every resolved setting with the layer it came from.

    With four levels of precedence and every feature opt-in, this block is the
    only thing that makes "why didn't my setting apply?" debuggable. It goes to
    stdout, so the web UI surfaces it as-is.
    """
    rows = [
        ("model_id",             MODEL_ID),
        ("project_name",         PROJECT_NAME or "(default)"),
        ("trigger_word",         TRIGGER_WORD),
        ("cache_dir",            CACHE_DIR),
        ("output_dir",           OUTPUT_DIR),
        ("total_steps",          f"{TOTAL_STEPS} micro-steps = {TOTAL_UPDATES:.0f} updates"),
        ("batch_size",           BATCH_SIZE),
        ("grad_accum_steps",     GRAD_ACCUM_STEPS),
        ("lr",                   LR),
        ("lr_scheduler",         f"{LR_SCHEDULER} (min_lr_ratio {MIN_LR_RATIO})"),
        ("warmup_steps",         f"{WARMUP_STEPS} {WARMUP_UNITS} = {WARMUP_UPDATES:.0f} updates "
                                 f"({100.0 * WARMUP_UPDATES / TOTAL_UPDATES:.1f}% of run)"),
        ("lora_rank",            f"{LORA_RANK} / alpha {LORA_ALPHA} (scale {LORA_ALPHA / max(1, LORA_RANK):.2f})"),
        ("lora_target",          LORA_TARGET),
        ("lora_dtype",           LORA_DTYPE_NAME),
        ("optimizer",            f"{OPTIMIZER_NAME} betas={list(OPTIMIZER_BETAS)} eps={OPTIMIZER_EPS} "
                                 f"wd={WEIGHT_DECAY}"),
        ("max_grad_norm",        MAX_GRAD_NORM),
        ("sampler",              SAMPLER_MODE),
        ("timestep_sampling",    TIMESTEP_SAMPLING),
        ("timestep_weighting",   TIMESTEP_WEIGHTING),
        ("content_or_style",     CONTENT_OR_STYLE),
        ("sigma_min",            f"{SIGMA_MIN} .. {SIGMA_MAX}"),
        ("noise_offset",         NOISE_OFFSET),
        ("high_precision_targets", HIGH_PREC_TARGETS),
        ("caption_dropout_rate", CAPTION_DROPOUT),
        ("curation_weights",     CURATION_WEIGHTS),
        ("use_ema",              f"{USE_EMA} (decay {EMA_DECAY}, {EMA_DEVICE})" if USE_EMA else False),
        ("compact_text",         COMPACT_TEXT),
        ("gradient_checkpointing", GRAD_CHECKPOINTING),
        ("nan_guard",            f"{NAN_GUARD} (abort after {NAN_ABORT_AFTER})" if NAN_GUARD else False),
        ("oom_guard",            f"{OOM_GUARD} (abort after {OOM_ABORT_AFTER})" if OOM_GUARD else False),
        ("max_loss",             MAX_LOSS or "off"),
        ("resume_on_corrupt",    RESUME_ON_CORRUPT),
        ("save_every",           SAVE_EVERY),
        ("max_checkpoints_to_keep", MAX_CKPT_KEEP or "keep all"),
        ("val_split",            f"{VAL_SPLIT} every {VALIDATE_EVERY} steps" if VALIDATE_EVERY else "off"),
        ("loss_display",         f"{LOSS_DISPLAY} (window {LOSS_WINDOW})"),
        ("csv_log",              CSV_LOG),
        ("preview_every",        f"{PREVIEW_EVERY} (source {PREVIEW_SOURCE})" if PREVIEW_EVERY else "off"),
        ("seed",                 SEED),
    ]
    if MULTIPHASE:
        rows.append(("phase", f"{PHASE_INDEX+1}/{PHASE_COUNT} ({PHASE_LABEL}²), global {GLOBAL_STEP_OFFSET}-{GLOBAL_STEP_OFFSET + TOTAL_STEPS}/{GLOBAL_TOTAL_STEPS}"))
    if RUN_ID:
        rows.append(("run_id", RUN_ID))
    width = max(len(k) for k, _ in rows)
    print("\n" + "=" * 78)
    print(f"EFFECTIVE CONFIG" + (f"  [preset: {PRESET_NAME}]" if PRESET_NAME else ""))
    print("=" * 78)
    for key, val in rows:
        print(f"  {key:<{width}}  {val}   [{_CFG_SOURCE.get(key, 'derived')}]")
    if INIT_LORA_FROM:
        print(f"  {'init_lora_from':<{width}}  {INIT_LORA_FROM}   [{_CFG_SOURCE.get('init_lora_from')}]")
    print("=" * 78 + "\n")


_print_effective_config()

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESUME_DIR   = CFG.resume_dir
OPT_FILE     = CFG.optimizer_state_file
STEP_FILE    = CFG.step_file
RUN_ID_FILE  = CFG.run_id_file




torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)


def free_vram():
    gc.collect()
    torch.cuda.empty_cache()


patch_attention_for_low_vram = _quant.patch_attention_for_low_vram


def ensure_model_downloaded(local_path, repo_id):
    """Return a usable model directory, pulling it from Hugging Face if absent.

    A non-empty `local_path` is trusted as-is; otherwise the full repo is
    snapshot-downloaded, which is a multi-GB, multi-minute operation. Returns the
    resolved path.
    """
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"[OK] Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")
    print(f"  This may take several minutes / Esto puede tardar varios minutos...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with / Se requiere 'huggingface_hub':\n"
            "  pip install huggingface_hub"
        )

    hf_token = CFG.hf_token or os.environ.get("HF_TOKEN")
    dl_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_path,
    }
    if hf_token:
        dl_kwargs["token"] = hf_token

    downloaded_path = snapshot_download(**dl_kwargs)

    print(f"[OK] Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


calculate_shift = _math.calculate_shift


def sample_sigma(batch_size, image_seq_len, device, shift_cfg):
    """Sample one flow-matching sigma per batch item, shifted by resolution."""
    return _math.sample_sigma(
        batch_size, image_seq_len, device, shift_cfg,
        sampling=TIMESTEP_SAMPLING, content_or_style=CONTENT_OR_STYLE,
        logit_normal_mu=LOGIT_NORMAL_MU, logit_normal_sigma=LOGIT_NORMAL_SIGMA,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)


def timestep_weight(sigma):
    """ai-toolkit's BSMNTW/HBSMNTW loss weighting over sigma."""
    return _math.timestep_weight(sigma, TIMESTEP_WEIGHTING)


pack_latents = _math.pack_latents


unpack_latents = _math.unpack_latents


prepare_position_ids = _math.prepare_position_ids


quantize_to_nf4_ = _quant.quantize_in_place


load_nf4_cache_ = _quant.load_nf4_cache


_lora_b_norm = _lora_io.lora_b_norm


load_lora_weights = _lora_io.load_lora_weights


def _export_lora(model, path, step=None, epoch=None, num_images=None):
    """Export the adapter as a flat bf16 safetensors file with training metadata."""
    _lora_io.export_lora(model, path, CFG, step=step, epoch=epoch, num_images=num_images)


# ── UTILIDADES DE ESTADO / STATE UTILITIES ──────────────────────────────────







def load_curation_weights(dataset_path, cache_names):
    """Per-cache-entry training weights from the dataset's curation report.

    Returns `(weights, summary)`, or `(None, None)` when there is nothing to apply.
    """
    return _curation.load_weights(dataset_path, cache_names, enabled=CURATION_WEIGHTS)


EpochSampler = _sampling.EpochSampler


LegacySampler = _sampling.LegacySampler


EMA = _ema.EMA


VaeHolder = _preview.VaeHolder


def run_preview(model, scheduler, embed, mask, neg, size, step, shift_cfg,
                steps=None, cfg_scale=None, seed=None):
    """Generate one preview image from the current weights and save it to OUTPUT_DIR."""
    _preview.render(
        model, scheduler, embed, mask, neg, size, step, shift_cfg,
        output_dir=OUTPUT_DIR, model_id=MODEL_ID,
        steps=int(steps or PREVIEW_STEPS),
        cfg_scale=PREVIEW_CFG if cfg_scale is None else float(cfg_scale),
        seed=int(SEED if seed is None else seed))
    free_vram()


# ── ENTRENAMIENTO / TRAINING ─────────────────────────────────────────────────
def train_krea2():
    """Run one full training phase, from model load to final LoRA export.

    In order: load the transformer and get it onto the GPU in NF4, attach the LoRA
    adapter and optimizer, restore a checkpoint or hand off weights from a previous
    phase, load every cached latent into RAM and bucket it by resolution, then run
    the micro-step loop — sample a batch, add noise at a sampled sigma, predict,
    accumulate gradients, and step the optimizer every `grad_accum_steps`.

    Exits non-zero on a missing cache, an unusable checkpoint, or a failed weight
    hand-off, since the progressive orchestrator treats that as an abort signal.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    patch_attention_for_low_vram()

    if not os.path.exists(CACHE_DIR) or not any(f.endswith("_latent.pt") for f in os.listdir(CACHE_DIR)):
        print(f"\n[!] ERROR: Cache directory '{CACHE_DIR}' is empty or does not exist.")
        print(f"[!] Please run Pre-Cache first! / ¡Por favor ejecuta el Pre-Caché primero!")
        # Código de salida ≠ 0: con `return` esto era un fallo invisible para
        # run_batch_cli.sh y run_progressive.py, que lo reportaban como éxito.
        sys.exit(1)

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading Krea-2 Transformer... / Cargando Transformer de Krea-2...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        vae=None,
        text_encoder=None,
        torch_dtype=torch.bfloat16,
    )

    transformer = pipe.transformer
    scheduler   = pipe.scheduler

    del pipe
    free_vram()

    shift_cfg = (
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 6400),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )

    NF4_CACHE_DIR = MODEL_ID

    if _quant.has_nf4_cache(NF4_CACHE_DIR):
        print("\n¡NF4 CACHE DETECTED! / ¡CACHÉ NF4 DETECTADA!")
        print("Skipping NF4 quantization... / No se ejecutará cuantización NF4.")
        t0 = time.time()
        transformer = load_nf4_cache_(transformer, NF4_CACHE_DIR)
        print(f"[NF4] Cache loaded in / Caché cargada en {time.time() - t0:.1f}s", flush=True)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    else:
        print("\nNo NF4 cache found. Performing quantization... / No se encontró caché NF4. Ejecutando cuantización...")
        quantize_to_nf4_(transformer)
        transformer.to("cuda")
        free_vram()
        print(f"Transformer 12B pinned in VRAM. Usage / Uso: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    if GRAD_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()
    else:
        # Sin recompute: más rápido pero más VRAM de activaciones. Sólo apto en las
        # fases de baja resolución donde la VRAM sobra (lo decide el orquestador).
        print("Gradient checkpointing OFF (faster, more VRAM) / Checkpointing desactivado.")

    target_modules, all_linears = _lora_io.target_modules(transformer, LORA_TARGET)
    print(f"Target LoRA Layers / Capas LoRA objetivo: {len(target_modules)}/{len(all_linears)} ({LORA_TARGET})")

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
        target_modules=target_modules, use_dora=False, init_lora_weights=True,
    )

    model = get_peft_model(transformer, lora_config)

    _lora_io.set_lora_dtype(model, LORA_DTYPE)

    model.print_trainable_parameters()
    print(f"LoRA master dtype / dtype de los pesos LoRA: {LORA_DTYPE_NAME}")

    def _make_inputs_require_grad(module, input, output):
        """Re-open the autograd chain at the model's input projection.

        The NF4 base is entirely frozen, so activations would reach the first LoRA
        layer with requires_grad=False and no gradient would ever flow. Required for
        gradient checkpointing to work at all here.
        """
        output.requires_grad_(True)

    transformer.img_in.register_forward_hook(_make_inputs_require_grad)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = _lora_io.build_optimizer(OPTIMIZER_NAME, trainable, LR, OPTIMIZER_BETAS,
                                         OPTIMIZER_EPS, WEIGHT_DECAY)

    ema = None
    if USE_EMA:
        ema = EMA(trainable, decay=EMA_DECAY, device=EMA_DEVICE)
        warning = _ema.horizon_warning(EMA_DECAY, TOTAL_UPDATES)
        if warning:
            print(warning)
        print(f"[OK] EMA enabled / activada: decay {EMA_DECAY} on {EMA_DEVICE}")

    def lr_at(step):
        """Learning rate for a given micro-step, after warmup and the configured decay."""
        return _schedule.lr_at_step(
            step, lr=LR, grad_accum_steps=GRAD_ACCUM_STEPS,
            warmup_updates=WARMUP_UPDATES, total_updates=TOTAL_UPDATES,
            min_lr_ratio=MIN_LR_RATIO, scheduler=LR_SCHEDULER,
            num_cycles=LR_NUM_CYCLES, step_gamma=LR_STEP_GAMMA,
            step_count=LR_STEP_COUNT)

    FINGERPRINT = _lora_io.fingerprint(CFG)

    # ── RESTAURACIÓN DE CHECKPOINT Y HAND-OFF ENTRE FASES ────────────────────
    ckpt = _state.CheckpointManager(CFG, model, optimizer)
    FINGERPRINT = ckpt.fingerprint
    restored = ckpt.restore(ema)
    if restored.already_complete:
        return
    start_step = restored.start_step
    pending_state = restored.pending
    if start_step == 0:
        ckpt.load_previous_phase()

    last_step_executed = start_step
    sampler = None             # lo construye el cargador de caché, más abajo
    saving = {"busy": False, "done_on_exit": False}

    def save_checkpoint_now(current_s):
        """Save the full resumable state atomically, and non-reentrantly."""
        ckpt.save(current_s, sampler=sampler, ema=ema, num_images=len(cache_data))

    ckpt.install_signal_handlers(lambda: last_step_executed, save_checkpoint_now)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data = _dataset.load(
        CACHE_DIR,
        compact=COMPACT_TEXT,
        pin=torch.cuda.is_available(),
        val_split=VAL_SPLIT,
        val_cache_dir=VAL_CACHE_DIR,
    )
    cache_data = data.samples
    buckets = data.buckets
    val_data, val_names = data.val_samples, data.val_names
    neg = data.negative
    sample_prompts = data.sample_prompts

    if not cache_data:
        print("[!] ERROR: no training images left after the validation split.")
        sys.exit(1)

    # ── Curaduría: peso por imagen ───────────────────────────────────────────
    # Se resuelve sobre el set de entrenamiento ya definitivo (post-split), para
    # que el recuento por grupo refleje lo que de verdad recibe gradiente.
    curation_w, curation_summary = load_curation_weights(DATASET_PATH, cache_data.keys())
    if curation_summary:
        print(_curation.format_summary(curation_summary))

    if CAPTION_DROPOUT > 0 and neg is None:
        print("[!] caption_dropout_rate is set but no _neg_embed.pt in the cache; "
              "re-run Pre-Cache / vuelve a ejecutar el Pre-Caché. Disabling.")
        CAPTION_DROPOUT_ACTIVE = 0.0
    else:
        CAPTION_DROPOUT_ACTIVE = CAPTION_DROPOUT

    pos_ids_cache = _dataset.PositionIdCache("cuda")
    get_pos_ids = pos_ids_cache.get

    if PREVIEW_SOURCE == "prompts" and not sample_prompts:
        print("[!] preview_source='prompts' but no _sample*_embed.pt in the cache; "
              "add sample_prompts to pre_cache_settings.json and re-run Pre-Cache. "
              "Falling back to captions / usando captions.")
    elif sample_prompts:
        print(f"[OK] {len(sample_prompts)} sample prompts loaded for previews / "
              f"prompts de preview cargados.")

    all_preview_names = sorted(cache_data.keys())

    def get_preview_sample(step):
        """Pick which cached caption to render a preview from, per `preview_caption_mode`."""
        return _preview.choose_caption(all_preview_names, step,
                                       PREVIEW_CAPTION_MODE, PREVIEW_EVERY)

    # ── A2: sampler ──────────────────────────────────────────────────────────
    sampler = _sampling.build_sampler(SAMPLER_MODE, buckets, BATCH_SIZE, SEED)
    if SAMPLER_MODE != "epoch":
        # El default histórico es un bug medible: avisar con el número real.
        warning = _sampling.bias_warning(buckets)
        if warning:
            print(f"\n{warning}")

    # ── A6: restaurar RNG y estado del sampler tras un resume ────────────────
    ckpt.apply_pending(pending_state, sampler)

    # ── C1: loss de validación ───────────────────────────────────────────────
    @torch.no_grad()
    def validation_loss():
        """Mean holdout loss at fixed sigmas with fixed per-image noise.

        Pinning both is the whole trick: it turns a scalar otherwise dominated by
        sigma variance into a curve readable enough to decide early stopping.
        Evaluated on the EMA weights when EMA is enabled.
        """
        if ema is not None:
            ema.apply()
        was_training = model.training
        model.eval()
        total, count = 0.0, 0
        try:
            for name in val_names:
                lat, emb, msk = val_data[name]
                lat = lat.to("cuda", non_blocking=True)
                emb = emb.to("cuda", non_blocking=True)
                msk_c = msk.to("cuda", non_blocking=True) if (msk is not None and not COMPACT_TEXT) else None
                x = pack_latents(lat).float()
                pos = get_pos_ids(emb.shape[1], lat.shape[2], lat.shape[3])
                gen = torch.Generator(device="cuda").manual_seed(
                    VAL_SEED + (zlib.crc32(name.encode("utf-8")) & 0xFFFF))
                eps = torch.randn(x.shape, generator=gen, device="cuda", dtype=torch.float32)
                tgt = eps - x
                for s in VALIDATION_SIGMAS:
                    sig = torch.full((x.shape[0],), float(s), device="cuda")
                    noisy_v = ((1 - s) * x + s * eps).to(MODEL_DTYPE)
                    pred_v = model(hidden_states=noisy_v, encoder_hidden_states=emb,
                                   timestep=sig, position_ids=pos,
                                   encoder_attention_mask=msk_c, return_dict=False)[0]
                    total += F.mse_loss(pred_v.float(), tgt).item()
                    count += 1
        finally:
            if was_training:
                model.train()
            if ema is not None:
                ema.restore()
            free_vram()
        return total / max(1, count)

    # ── C2: logging estructurado ─────────────────────────────────────────────
    logs = _metrics.CsvLogs(OUTPUT_DIR, enabled=CSV_LOG)

    loss_hist = collections.deque(maxlen=max(1, LOSS_WINDOW))
    nan_count = oom_streak = skipped_outlier = 0
    accum_count = 0

    def on_oom(step, size):
        """Discard the accumulation window, free VRAM, and decide whether to give up.

        Returns True when too many OOMs have hit consecutively and the error should
        propagate; a checkpoint is saved first in that case.
        """
        nonlocal oom_streak, accum_count
        oom_streak += 1
        accum_count = 0
        optimizer.zero_grad(set_to_none=True)
        free_vram()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print(f"\n[!] CUDA OOM at step {step} (bucket {size[0]}x{size[1]}) — batch skipped "
              f"({oom_streak}/{OOM_ABORT_AFTER} consecutive)")
        if oom_streak >= OOM_ABORT_AFTER:
            print("[!] Persistent OOM; saving checkpoint and aborting / guardando y abortando.")
            save_checkpoint_now(last_step_executed)
            return True
        return False
    running_loss, t_step_avg = 0.0, 0.0
    print(f"\nSTARTING TRAINING / ¡ARRANCANDO ENTRENAMIENTO! {len(cache_data)} images in "
          f"{len(buckets)} buckets, sampler '{SAMPLER_MODE}'.")

    try:
        for step in range(start_step + 1, TOTAL_STEPS + 1):
            last_step_executed = step
            t0 = time.time()

            size, names = sampler.next()
            latents = torch.cat([cache_data[n][0] for n in names]).to("cuda", non_blocking=True)
            masks   = None
            # D3: caption dropout reutilizando el embedding de prompt vacío que el
            # pre-caché ya escribe para el CFG de los previews. get_pos_ids está
            # cacheado por (text_len, lh, lw), así que el neg obtiene los suyos.
            if CAPTION_DROPOUT_ACTIVE > 0 and random.random() < CAPTION_DROPOUT_ACTIVE:
                # Los embeds son 4-D [B, T, 12, 2560]: se repite de forma agnóstica
                # al rango, y sólo cuando el batch lo exige (con batch 1 no se copia).
                neg_emb = neg[0].to("cuda", non_blocking=True)
                embeds = (neg_emb if len(names) == 1
                          else neg_emb.repeat(len(names), *([1] * (neg_emb.dim() - 1))))
                if not COMPACT_TEXT and neg[1] is not None:
                    neg_msk = neg[1].to("cuda", non_blocking=True)
                    masks = (neg_msk if len(names) == 1
                             else neg_msk.repeat(len(names), *([1] * (neg_msk.dim() - 1))))
            else:
                embeds = torch.cat([cache_data[n][1] for n in names]).to("cuda", non_blocking=True)
                if not COMPACT_TEXT:
                    masks = torch.cat([cache_data[n][2] for n in names]).to("cuda", non_blocking=True)

            try:
                latent_patched = pack_latents(latents)
                B, seq_img, _ = latent_patched.shape

                sigma  = sample_sigma(B, seq_img, "cuda", shift_cfg)
                # A10: en fp32 el target deja de arrastrar el redondeo de bf16. La
                # entrada al modelo se castea a MODEL_DTYPE igualmente.
                noise_dtype = torch.float32 if HIGH_PREC_TARGETS else latent_patched.dtype
                noise = torch.randn(latent_patched.shape, device=latent_patched.device,
                                    dtype=noise_dtype)
                if NOISE_OFFSET > 0:
                    noise = _math.add_noise_offset(noise, latents.shape[1], NOISE_OFFSET)

                t_exp = sigma.view(-1, 1, 1).to(noise.dtype)
                base  = latent_patched.to(noise.dtype)
                noisy = ((1 - t_exp) * base + t_exp * noise).to(MODEL_DTYPE)
                target = noise - base

                pos_ids = get_pos_ids(embeds.shape[1], size[0], size[1])

                pred = model(
                    hidden_states=noisy,
                    encoder_hidden_states=embeds,
                    timestep=sigma,
                    position_ids=pos_ids,
                    encoder_attention_mask=masks,
                    return_dict=False,
                )[0]

                if TIMESTEP_WEIGHTING == "none" and curation_w is None:
                    raw_loss = F.mse_loss(pred.float(), target.float())
                    loss_value = raw_loss.detach()
                else:
                    per_sample = F.mse_loss(pred.float(), target.float(),
                                            reduction="none").mean(dim=(1, 2))
                    if TIMESTEP_WEIGHTING != "none":
                        per_sample = per_sample * timestep_weight(sigma.float())
                    # El guard de max_loss, el NaN guard y el CSV leen la loss SIN
                    # el peso de curaduría, por dos motivos: atenuar una imagen la
                    # volvería inmune al descarte de outliers, y la curva del log
                    # dejaría de ser comparable con la de un run sin curar.
                    loss_value = per_sample.mean().detach()
                    if curation_w is not None:
                        # Escalar la loss de una muestra ES un LR por imagen. No se
                        # renormaliza a media 1.0 por batch: eso anularía el efecto,
                        # y que un batch atenuado aporte menos a la actualización
                        # acumulada es justamente lo que se busca.
                        w = torch.tensor([curation_w.get(n, 1.0) for n in names],
                                         device=per_sample.device, dtype=per_sample.dtype)
                        per_sample = per_sample * w
                    raw_loss = per_sample.mean()
            except torch.cuda.OutOfMemoryError:
                if not OOM_GUARD or on_oom(step, size):
                    raise
                continue

            # ── A3: guardas sobre la loss ────────────────────────────────────
            if MAX_LOSS > 0 and torch.isfinite(loss_value) and loss_value.item() > MAX_LOSS:
                skipped_outlier += 1
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                print(f"\n[!] Loss {loss_value.item():.3f} > max_loss {MAX_LOSS} at step {step} "
                      f"— window discarded ({skipped_outlier} total)")
                continue
            if NAN_GUARD and not torch.isfinite(loss_value):
                nan_count += 1
                # Descartar la ventana entera: un solo inf haría que clip_grad_norm_
                # calculase una norma NaN y escalase TODOS los gradientes a NaN,
                # y AdamW escribiría NaN en exp_avg de forma permanente.
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                print(f"\n[!] Non-finite loss at step {step} — batch skipped "
                      f"({nan_count}/{NAN_ABORT_AFTER})")
                if nan_count >= NAN_ABORT_AFTER:
                    print("[!] Too many non-finite losses; saving and aborting / abortando.")
                    save_checkpoint_now(last_step_executed)
                    return
                continue

            try:
                # El backward es el pico real de memoria, así que necesita la
                # misma protección que el forward.
                (raw_loss / GRAD_ACCUM_STEPS).backward()
            except torch.cuda.OutOfMemoryError:
                if not OOM_GUARD or on_oom(step, size):
                    raise
                continue

            step_loss = loss_value.item()
            running_loss += step_loss
            loss_hist.append(step_loss)
            accum_count += 1
            oom_streak = 0

            grad_norm = 0.0
            did_update = False
            if accum_count >= GRAD_ACCUM_STEPS:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                if NAN_GUARD and not torch.isfinite(gnorm):
                    nan_count += 1
                    print(f"\n[!] Non-finite grad norm at step {step} — update skipped "
                          f"({nan_count}/{NAN_ABORT_AFTER})")
                    if nan_count >= NAN_ABORT_AFTER:
                        print("[!] Too many non-finite gradients; saving and aborting / abortando.")
                        save_checkpoint_now(last_step_executed)
                        return
                else:
                    grad_norm = gnorm.item()
                    for gparam in optimizer.param_groups:
                        gparam["lr"] = lr_at(step)
                    optimizer.step()
                    did_update = True
                    if ema is not None:
                        # Sólo en updates reales: por micro-batch el decay efectivo
                        # sería decay ** GRAD_ACCUM_STEPS.
                        ema.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0

            t_step     = time.time() - t0
            t_step_avg = _metrics.smooth(t_step_avg, t_step)
            avg_loss = _metrics.average_loss(LOSS_DISPLAY, loss_hist,
                                             running_loss, step - start_step)
            print("\r" + _metrics.format_progress(
                step=step, total_steps=TOTAL_STEPS, avg_loss=avg_loss,
                grad_norm=grad_norm, lr=lr_at(step), epoch=sampler.epoch,
                seconds_per_step=t_step_avg, multiphase=MULTIPHASE,
                phase_index=PHASE_INDEX, phase_count=PHASE_COUNT,
                phase_label=PHASE_LABEL, global_step_offset=GLOBAL_STEP_OFFSET,
                global_total_steps=GLOBAL_TOTAL_STEPS), end="", flush=True)

            if did_update:
                logs.log_step(
                    step=step, update=int(step / max(1, GRAD_ACCUM_STEPS)),
                    epoch=sampler.epoch, loss=step_loss, avg_loss=avg_loss,
                    grad_norm=grad_norm, lr=lr_at(step),
                    sigma=sigma.mean().item(), bucket=size, seconds=t_step,
                    vram_peak_gb=torch.cuda.max_memory_allocated() / 1e9)

            if step % SAVE_EVERY == 0:
                print()
                save_checkpoint_now(step)

            if VALIDATE_EVERY > 0 and val_names and step % VALIDATE_EVERY == 0:
                vloss = validation_loss()
                print(f"\n  [Val] step {step} | val_loss {vloss:.4f} | {len(val_names)} images")
                logs.log_validation(step=step,
                                    update=int(step / max(1, GRAD_ACCUM_STEPS)),
                                    epoch=sampler.epoch, val_loss=vloss)

            if PREVIEW_EVERY > 0 and step % PREVIEW_EVERY == 0:
                if PREVIEW_SOURCE == "prompts" and sample_prompts:
                    # Rota entre los prompts configurados; cada uno puede fijar su
                    # propio tamaño, seed, pasos y CFG.
                    idx = (step // PREVIEW_EVERY) % len(sample_prompts)
                    emb0, msk0, meta = sample_prompts[idx]
                    ref = cache_data[all_preview_names[0]][0]
                    size_px = (int(meta.get("height", ref.shape[2] * 8)),
                               int(meta.get("width",  ref.shape[3] * 8)))
                    p_seed = int(meta.get("seed", SEED)) + (step if PREVIEW_WALK_SEED else 0)
                    print(f"\n  [Preview] Prompt {idx}: {str(meta.get('prompt', ''))[:60]}")
                else:
                    p_name = get_preview_sample(step)
                    lat0, emb0, msk0 = cache_data[p_name]
                    meta, size_px, p_seed = {}, (lat0.shape[2] * 8, lat0.shape[3] * 8), SEED
                    print(f"\n  [Preview] Mode: {PREVIEW_CAPTION_MODE} | Sample: {p_name}")

                if ema is not None:
                    ema.apply()
                try:
                    run_preview(model, scheduler, emb0, msk0, neg, size_px, step, shift_cfg,
                                steps=int(meta.get("steps", PREVIEW_STEPS)),
                                cfg_scale=float(meta.get("cfg", PREVIEW_CFG)),
                                seed=p_seed)
                finally:
                    if ema is not None:
                        ema.restore()

    except torch.cuda.OutOfMemoryError:
        # No debería llegar aquí (el guard interno la captura), pero si el guard
        # está desactivado conviene salvar el trabajo antes de morir.
        print(f"\n[!] CUDA OOM at step {last_step_executed}; saving checkpoint / guardando.")
        save_checkpoint_now(last_step_executed)
        raise
    except (KeyboardInterrupt, SystemExit):
        if not ckpt.saved_on_exit:
            save_checkpoint_now(last_step_executed)
        return

    # A5: la ventana final de acumulación se descartaba si total_steps no era
    # múltiplo de grad_accum_steps.
    if accum_count > 0:
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
        if torch.isfinite(gnorm):
            for gparam in optimizer.param_groups:
                gparam["lr"] = lr_at(TOTAL_STEPS)
            optimizer.step()
            if ema is not None:
                ema.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"\n[i] Flushed final partial accumulation window ({accum_count} micro-steps).")

    print("\n\nTraining completed! / ¡Entrenamiento finalizado!")
    if nan_count or skipped_outlier:
        print(f"[i] Skipped batches: {nan_count} non-finite, {skipped_outlier} over max_loss.")
    if VALIDATE_EVERY > 0 and val_names:
        print(f"[i] Final validation loss: {validation_loss():.4f}")
    # Guardar también el resume_checkpoint al terminar: en el pipeline progresivo
    # es el hand-off de pesos que carga la fase siguiente vía init_lora_from.
    save_checkpoint_now(TOTAL_STEPS)
    final = ckpt.export_final(TOTAL_STEPS, sampler=sampler, ema=ema,
                              num_images=len(cache_data))
    print(f"✓ Final LoRA saved to / Tu LoRA definitivo está en: {final}")


if __name__ == "__main__":
    from _telemetry import emit_lifecycle

    _started = time.time()
    emit_lifecycle("worker_started", "train")
    try:
        train_krea2()
    except Exception as _exc:
        emit_lifecycle(
            "worker_failed",
            "train",
            duration_seconds=round(time.time() - _started, 1),
            gpu_seconds=round(time.time() - _started, 1),
            error=f"{type(_exc).__name__}: {_exc}",
        )
        raise
    emit_lifecycle(
        "worker_finished",
        "train",
        duration_seconds=round(time.time() - _started, 1),
        gpu_seconds=round(time.time() - _started, 1),
    )