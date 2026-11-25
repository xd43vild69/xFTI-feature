"""Rendering preview images mid-training, so a run can be judged before it finishes.

This is a full inference pipeline living inside a trainer: denoise from pure noise,
optionally with classifier-free guidance, then decode through the VAE. It exists
because loss alone says very little about whether a LoRA has learned the concept.

The VAE is the only piece of the model training never touches — everything else runs on
cached latents — so it is loaded lazily and pushed back to CPU immediately after.
"""
from __future__ import annotations

import os
import random
from typing import Any, Callable

import numpy as np
import torch

from krea2.math_ops import calculate_shift, pack_latents, prepare_position_ids, unpack_latents

Logger = Callable[[str], None]

# Buckets must be multiples of 16: the patch grid is H/16 by W/16, since the VAE
# downsamples by 8 and pack_latents halves each latent dimension again.
BUCKET_MULTIPLE = 16


class VaeHolder:
    """Lazy per-process singleton for the VAE, which only previews need.

    Loading it costs seconds and ~300 MB, so it is created on first use and cached in
    pinned host RAM (64 GB) for fast non-blocking H2D transfers per preview.
    """

    vae: Any = None

    @classmethod
    def _pin_cpu_params(cls, module: torch.nn.Module) -> None:
        if not torch.cuda.is_available():
            return
        try:
            for p in module.parameters():
                if p.device.type == "cpu" and not p.is_pinned():
                    p.data = p.data.pin_memory()
            for b in module.buffers():
                if b.device.type == "cpu" and not b.is_pinned():
                    b.data = b.data.pin_memory()
        except Exception:
            pass

    @classmethod
    def get(cls, model_id: str) -> Any:
        if cls.vae is None:
            from diffusers import AutoencoderKLQwenImage
            cls.vae = AutoencoderKLQwenImage.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.bfloat16)
            cls._pin_cpu_params(cls.vae)
        return cls.vae

    @classmethod
    def offload_to_pinned_cpu(cls) -> None:
        if cls.vae is not None:
            cls.vae.to("cpu")
            cls._pin_cpu_params(cls.vae)

    @classmethod
    def reset(cls) -> None:
        cls.vae = None


def snap_to_bucket(size: tuple[int, int]) -> tuple[int, int]:
    """Round a pixel size down to what the patch grid can represent."""
    height, width = size
    return (max(BUCKET_MULTIPLE, (height // BUCKET_MULTIPLE) * BUCKET_MULTIPLE),
            max(BUCKET_MULTIPLE, (width // BUCKET_MULTIPLE) * BUCKET_MULTIPLE))


def choose_caption(names: list[str], step: int, mode: str, every: int) -> str:
    """Pick which cached caption to render from, per `preview_caption_mode`.

    "random" picks freely, "rotate4" cycles the first four, anything else pins the first
    sample so previews stay visually comparable from step to step.
    """
    if mode == "random":
        return random.choice(names)
    if mode == "rotate4":
        return names[(step // max(1, every)) % min(4, len(names))]
    return names[0]


@torch.no_grad()
def render(
    model: torch.nn.Module,
    scheduler: Any,
    embeds: torch.Tensor,
    mask: torch.Tensor | None,
    negative: tuple[torch.Tensor, torch.Tensor | None] | None,
    size: tuple[int, int],
    step: int,
    shift_cfg: tuple[float, float, float, float],
    *,
    output_dir: str,
    model_id: str,
    steps: int,
    cfg_scale: float,
    seed: int,
    log: Logger = print,
) -> str:
    """Generate one preview image from the current weights and save it to `output_dir`.

    Restores the model's previous training mode on the way out, so calling this from
    inside the loop is safe. Returns the path written.
    """
    height, width = snap_to_bucket(size)
    grid_h, grid_w = height // BUCKET_MULTIPLE, width // BUCKET_MULTIPLE
    device = "cuda"

    was_training = model.training
    model.eval()

    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = torch.randn((1, 16, height // 8, width // 8), generator=generator,
                          device=device, dtype=torch.bfloat16)
    latents = pack_latents(latents)

    pos_ids = prepare_position_ids(embeds.shape[1], grid_h, grid_w, device)
    embeds = embeds.to(device)
    mask = mask.to(device) if mask is not None else None

    neg_pos_ids = None
    if negative is not None:
        negative = (negative[0].to(device),
                    negative[1].to(device) if negative[1] is not None else None)
        # Compacted, the negative prompt has fewer tokens than the positive one, so it
        # needs position ids of its own.
        neg_pos_ids = (pos_ids if negative[0].shape[1] == embeds.shape[1]
                       else prepare_position_ids(negative[0].shape[1], grid_h, grid_w, device))

    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    mu = calculate_shift(latents.shape[1], *shift_cfg)
    scheduler.set_timesteps(steps, device=device, sigmas=sigmas, mu=mu)

    can_batch_cfg = (
        negative is not None
        and negative[0].shape[1] == embeds.shape[1]
        and ((mask is None) == (negative[1] is None))
    )
    cat_embeds = torch.cat([embeds, negative[0]], dim=0) if (can_batch_cfg and negative is not None) else None
    cat_masks = (
        torch.cat([mask, negative[1]], dim=0)
        if (can_batch_cfg and negative is not None and mask is not None and negative[1] is not None)
        else None
    )

    for t in scheduler.timesteps:
        if can_batch_cfg and cat_embeds is not None:
            try:
                t_batch = (t / scheduler.config.num_train_timesteps).expand(2).to(torch.bfloat16)
                lat_batch = torch.cat([latents, latents], dim=0)
                out_batch = model(
                    hidden_states=lat_batch,
                    encoder_hidden_states=cat_embeds,
                    timestep=t_batch,
                    position_ids=pos_ids,
                    encoder_attention_mask=cat_masks,
                    return_dict=False,
                )[0]
                pred, uncond = out_batch.chunk(2, dim=0)
                pred = pred + cfg_scale * (pred - uncond)
                latents = scheduler.step(pred, t, latents, return_dict=False)[0]
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                can_batch_cfg = False

        timestep = (t / scheduler.config.num_train_timesteps).expand(1).to(torch.bfloat16)
        pred = model(hidden_states=latents, encoder_hidden_states=embeds,
                     timestep=timestep, position_ids=pos_ids,
                     encoder_attention_mask=mask, return_dict=False)[0]
        if negative is not None:
            uncond = model(hidden_states=latents, encoder_hidden_states=negative[0],
                           timestep=timestep, position_ids=neg_pos_ids,
                           encoder_attention_mask=negative[1], return_dict=False)[0]
            pred = pred + cfg_scale * (pred - uncond)
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]

    path = _decode_and_save(latents, height, width, step, output_dir, model_id, device)
    log(f"\n  ↳ Preview saved to / Preview guardada: {path}")

    if was_training:
        model.train()
    return path


def _decode_and_save(latents: torch.Tensor, height: int, width: int, step: int,
                     output_dir: str, model_id: str, device: str) -> str:
    """Decode packed latents through the VAE and write the PNG.

    The VAE expects latents in its own normalized space, hence the mean/std rescale, and
    a temporal axis, hence the unsqueeze — it is a video autoencoder used on one frame.
    """
    from PIL import Image

    vae = VaeHolder.get(model_id).to(device, non_blocking=True)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    try:
        h_lat, w_lat = height // 8, width // 8
        lat = unpack_latents(latents, h_lat, w_lat).to(vae.dtype)
        # Guard against diffusers' AutoencoderKLQwenImage.tiled_decode remainder-tile bug
        # on dimensions like 784px (98 latent px = 24*4 + 2px sliver).
        ratio = int(getattr(vae, "spatial_compression_ratio", 8))
        t_min_h = int(getattr(vae, "tile_sample_min_height", 256)) // ratio
        t_min_w = int(getattr(vae, "tile_sample_min_width", 256)) // ratio
        t_str_h = int(getattr(vae, "tile_sample_stride_height", 192)) // ratio
        t_str_w = int(getattr(vae, "tile_sample_stride_width", 192)) // ratio

        def _pad_lat(sz: int, t_min: int, t_str: int) -> int:
            if sz <= t_min:
                return 0
            rem = (sz - t_min) % t_str
            return 0 if rem == 0 else (t_str - rem)

        pad_h = _pad_lat(h_lat, t_min_h, t_str_h)
        pad_w = _pad_lat(w_lat, t_min_w, t_str_w)
        if pad_h > 0 or pad_w > 0:
            lat = torch.nn.functional.pad(lat, (0, pad_w, 0, pad_h), mode="replicate")

        lat = lat.unsqueeze(2)
        mean = torch.tensor(vae.config.latents_mean, device=device,
                            dtype=lat.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device,
                           dtype=lat.dtype).view(1, -1, 1, 1, 1)
        decoded = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0, :height, :width]
        image = ((decoded.float() / 2 + 0.5).clamp(0, 1)[0]
                 .cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
    finally:
        VaeHolder.offload_to_pinned_cpu()

    path = os.path.join(output_dir, f"preview_step_{step}.png")
    Image.fromarray(image).save(path)
    return path
