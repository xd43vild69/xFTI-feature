"""VAE decoding and preview image generation for LTX 2.3.

Imports torch and diffusers: runs in the training_runtime environment.
"""
from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from ltx23.config import LTX23TrainConfig
from ltx23.math_ops import patch_audio_latent, patch_video_latent, unpack_video_latent


class LTXVaeHolder:
    """Singleton holder for LTX-2.3 VAE with optional FP32 execution."""

    vae: Any = None

    @classmethod
    def get(cls, model_id: str, use_fp32: bool = True) -> Any:
        if cls.vae is None:
            dtype = torch.float32 if use_fp32 else torch.bfloat16
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                transformer=None,
                text_encoder=None,
                audio_vae=None,
                tokenizer=None,
                processor=None,
                vocoder=None,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            cls.vae = pipe.vae
            if cls.vae is None:
                raise RuntimeError("No se pudo cargar el VAE para preview de LTX 2.3.")
            cls.vae.requires_grad_(False).eval()
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return cls.vae


def decode_preview_latent(
    vae: Any,
    latent: torch.Tensor,
    use_scaling_factor: bool = True,
    frame_index: int = -1,
) -> Image.Image:
    """Un-normalize statistical VAE latent and decode to PIL Image."""
    dtype = next(vae.parameters()).dtype
    latent = latent.detach().to("cuda", dtype=dtype)

    if getattr(vae, "latents_mean", None) is not None and getattr(vae, "latents_std", None) is not None:
        latents_mean = torch.as_tensor(vae.latents_mean, device=latent.device, dtype=dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.as_tensor(vae.latents_std, device=latent.device, dtype=dtype).view(1, -1, 1, 1, 1)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

        if use_scaling_factor and abs(scaling_factor) > 1e-12:
            latent = latent * latents_std / scaling_factor + latents_mean
        else:
            latent = latent * latents_std + latents_mean

    with torch.no_grad():
        decoded = vae.decode(latent, return_dict=False)[0]

    if decoded.ndim == 5:
        f = decoded.shape[2]
        idx = f // 2 if frame_index < 0 else min(frame_index, f - 1)
        decoded = decoded[:, :, idx]

    decoded = decoded[:, :3].detach().float()
    decoded = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
    image_np = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_np)
