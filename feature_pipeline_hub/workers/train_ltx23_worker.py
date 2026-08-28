"""train_ltx23_worker.py — Standalone training worker for LTX 2.3 LoRA.

Launched as a separate process by training_runner in the training_runtime environment.
"""
from __future__ import annotations

import os
import platform
import sys

# Early environment setup before importing torch
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("DIFFUSERS_NO_ADVISORY_WARNINGS", "1")

if platform.system() != "Windows":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

# Add workers directory to sys.path so ltx23 and _telemetry resolve
WORKERS_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKERS_DIR not in sys.path:
    sys.path.insert(0, WORKERS_DIR)

import gc
import math
import random
import time
import bitsandbytes as bnb
import numpy as np
import torch
from diffusers import DiffusionPipeline

from ltx23.config import load_config
from ltx23.dataset import LTX23Dataset
from ltx23.lora_io import inject_lora
from ltx23.math_ops import (
    align_video_latent_to_patch,
    make_video_timestep,
    mse_loss_chunked,
    patch_audio_latent,
    patch_video_latent,
    sample_continuous_sigma,
)
from ltx23.metrics import TrainLog, format_progress, smooth
from ltx23.quantization import (
    activation_offload_context,
    cast_frozen_to_bf16,
    enable_memory_efficient_attention,
    load_nf4_cache_,
)
from ltx23.state import CheckpointManager, register_signal_handlers

try:
    import _telemetry
except ImportError:
    _telemetry = None


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cfg = load_config()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para entrenamiento LTX 2.3.")

    print()
    print("================================================================")
    print("       LTX 2.3 LORA TRAINER (xFTI Worker)")
    print("================================================================")
    print(f"Modelo: {cfg.model_id}")
    print(f"Caché: {cfg.cache_dir}")
    print(f"Salida: {cfg.output_dir}")
    print(f"Pasos totales: {cfg.total_steps} | Rank: {cfg.lora_rank} | LR: {cfg.lr}")
    print("================================================================")
    print()

    # Load Base Diffusion Pipeline
    from ltx23.downloader import ensure_ltx23_model_downloaded
    model_path = str(ensure_ltx23_model_downloaded(cfg.model_id))
    print(f"Cargando LTX-2.3 Pipeline desde {model_path}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        vae=None,
        audio_vae=None,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        vocoder=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    transformer = pipe.transformer
    connectors = getattr(pipe, "connectors", None)
    audio_channels = int(getattr(transformer.config, "audio_in_channels", 128))

    # Dataset & Precomputed Text Conditioning
    print(f"Cargando dataset desde {cfg.cache_dir}...")
    dataset = LTX23Dataset.from_cache(
        cfg.cache_dir,
        audio_channels=audio_channels,
        max_text_tokens=cfg.max_text_tokens,
        connectors=connectors,
    )
    print(f"✓ Dataset cargado: {len(dataset)} muestras.")

    del pipe, connectors
    gc.collect()
    torch.cuda.empty_cache()

    # NF4 Cache Reconstruction
    print("Cargando y verificando NF4 Transformer...")
    transformer = load_nf4_cache_(transformer, cfg.model_id)
    transformer.requires_grad_(False)
    if cfg.cast_frozen_bf16:
        cast_frozen_to_bf16(transformer)
    transformer.to("cuda")

    enable_memory_efficient_attention(transformer)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        try:
            transformer.enable_gradient_checkpointing()
            print("✓ Gradient checkpointing activado.")
        except Exception:
            pass

    # Inject LoRA Adapters
    print("Inyectando adaptadores LoRA...")
    model = inject_lora(transformer, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = bnb.optim.PagedAdamW8bit(
        trainable,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ckpt_mgr = CheckpointManager(cfg)
    start_step = ckpt_mgr.restore(model, optimizer)

    last_step_executed = start_step

    def on_signal(sig: int) -> None:
        print(f"\n[!] Señal de detención recibida ({sig}). Guardando checkpoint de seguridad...")
        ckpt_mgr.save(model, optimizer, last_step_executed, reason="interrupt")

    register_signal_handlers(on_signal)

    # LR schedule function
    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))

    if cfg.seed > 0:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_log = TrainLog(cfg.output_dir)
    running_loss = 0.0
    avg_step_sec = 0.0

    print()
    print(f"ARRANCANDO ENTRENAMIENTO LTX 2.3 ({len(dataset)} entradas cacheadas)...")

    patch_size = int(getattr(transformer.config, "patch_size", 1))
    patch_size_t = int(getattr(transformer.config, "patch_size_t", 1))
    timestep_mult = float(getattr(transformer.config, "timestep_scale_multiplier", 1000.0))

    try:
        for step in range(start_step + 1, cfg.total_steps + 1):
            last_step_executed = step
            t0 = time.time()

            entry = dataset.sample()

            video_clean = entry["video"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
            audio_clean = entry["audio"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            if video_clean.ndim == 4:
                video_clean = video_clean.unsqueeze(0)
            if audio_clean.ndim == 2:
                audio_clean = audio_clean.unsqueeze(0)

            if cfg.caption_dropout_prob > 0.0 and random.random() < cfg.caption_dropout_prob and dataset.neg_conditioning is not None:
                video_text, audio_text = dataset.neg_conditioning
                video_text = video_text.to("cuda", dtype=torch.bfloat16, non_blocking=True)
                audio_text = audio_text.to("cuda", dtype=torch.bfloat16, non_blocking=True)
            else:
                video_text = entry["video_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
                audio_text = entry["audio_text"].to("cuda", dtype=torch.bfloat16, non_blocking=True)

            if video_text.ndim == 2:
                video_text = video_text.unsqueeze(0)
            if audio_text.ndim == 2:
                audio_text = audio_text.unsqueeze(0)

            video_clean = align_video_latent_to_patch(video_clean, patch_size, patch_size_t)
            video_tokens = patch_video_latent(video_clean, patch_size, patch_size_t)
            audio_tokens = patch_audio_latent(audio_clean)

            B = video_tokens.shape[0]
            video_seq_len = video_tokens.shape[1]
            num_frames = video_clean.shape[2]
            height = video_clean.shape[3]
            width = video_clean.shape[4]
            audio_num_frames = audio_clean.shape[-1]

            sigma = sample_continuous_sigma(B, device="cuda", mode=cfg.timestep_sampling)
            noise_video = torch.randn_like(video_tokens)
            noise_audio = torch.randn_like(audio_tokens)

            t_video = sigma.view(B, 1, 1)
            t_audio = sigma.view(B, 1, 1)

            noisy_video = (1.0 - t_video) * video_tokens + t_video * noise_video
            noisy_audio = (1.0 - t_audio) * audio_tokens + t_audio * noise_audio

            target_video = noise_video - video_tokens
            target_audio = (noise_audio - audio_tokens) if cfg.use_audio_loss else None

            timestep = make_video_timestep(sigma, video_seq_len, device="cuda", dtype=torch.bfloat16, multiplier=timestep_mult)
            audio_timestep = (sigma * timestep_mult).to(torch.bfloat16)

            forward_kwargs = {
                "hidden_states": noisy_video,
                "audio_hidden_states": noisy_audio,
                "encoder_hidden_states": video_text,
                "audio_encoder_hidden_states": audio_text,
                "timestep": timestep,
                "audio_timestep": audio_timestep,
                "sigma": sigma,
                "audio_sigma": sigma,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": cfg.frame_rate,
                "audio_num_frames": audio_num_frames,
                "return_dict": False,
            }

            with activation_offload_context(cfg.activation_offload):
                output = model(**forward_kwargs)

            if isinstance(output, tuple):
                pred_video = output[0]
                pred_audio = output[1] if len(output) > 1 and cfg.use_audio_loss else None
            else:
                pred_video = getattr(output, "video", getattr(output, "sample", None))
                pred_audio = getattr(output, "audio", getattr(output, "audio_sample", None)) if cfg.use_audio_loss else None

            if pred_video is None:
                raise RuntimeError("LTX-2.3 forward pass no devolvió predicción de video.")

            if cfg.use_audio_loss and pred_audio is not None and target_audio is not None:
                loss_v = mse_loss_chunked(pred_video, target_video, chunk_elements=cfg.loss_chunk_elements)
                loss_a = mse_loss_chunked(pred_audio, target_audio, chunk_elements=cfg.loss_chunk_elements)
                loss = (loss_v + loss_a) * 0.5
            else:
                loss = mse_loss_chunked(pred_video, target_video, chunk_elements=cfg.loss_chunk_elements)

            scaled_loss = loss / cfg.grad_accum_steps
            scaled_loss.backward()

            step_loss = loss.item()
            running_loss += step_loss

            grad_norm = 0.0
            current_lr = lr_at(step)

            if step % cfg.grad_accum_steps == 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm).item())
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                train_log.write_row(
                    step=step,
                    update=step // cfg.grad_accum_steps,
                    loss=step_loss,
                    loss_avg=running_loss / max(1, step - start_step),
                    grad_norm=grad_norm,
                    lr=current_lr,
                    secs=time.time() - t0,
                    vram_peak_gb=torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
                )

            elapsed = time.time() - t0
            avg_step_sec = smooth(avg_step_sec, elapsed)

            line = format_progress(
                step=step,
                total_steps=cfg.total_steps,
                avg_loss=running_loss / max(1, step - start_step),
                grad_norm=grad_norm,
                lr=current_lr,
                seconds_per_step=avg_step_sec,
            )
            print(f"\r{line}", end="", flush=True)

            if cfg.save_every > 0 and step % cfg.save_every == 0:
                ckpt_mgr.save(model, optimizer, step, reason="periodic")

        # Final Save
        print("\nGuardando checkpoint final...")
        ckpt_mgr.save(model, optimizer, cfg.total_steps, reason="final", is_final=True)
        print("✓ Entrenamiento completado exitosamente.")

    finally:
        train_log.close()
        ckpt_mgr.close()


if __name__ == "__main__":
    _started = time.time()
    if _telemetry is not None:
        _telemetry.emit_lifecycle("worker_started", "train_ltx23")
    try:
        main()
        if _telemetry is not None:
            _telemetry.emit_lifecycle(
                "worker_finished",
                "train_ltx23",
                duration_seconds=round(time.time() - _started, 1),
                gpu_seconds=round(time.time() - _started, 1),
            )
    except Exception as _exc:
        if _telemetry is not None:
            _telemetry.emit_lifecycle(
                "worker_failed",
                "train_ltx23",
                duration_seconds=round(time.time() - _started, 1),
                gpu_seconds=round(time.time() - _started, 1),
                error=f"{type(_exc).__name__}: {_exc}",
            )
        raise
