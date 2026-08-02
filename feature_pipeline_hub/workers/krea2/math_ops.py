"""Pure tensor and schedule math for the Krea 2 flow-matching trainer.

Every function here is deterministic given its arguments and the torch RNG state —
nothing reads configuration, opens a file, or touches a model. In the single-file
version `sample_sigma` and `timestep_weight` read six and one module-level globals
respectively, which is why neither could be tested; they now take those as keyword
arguments, so the wiring is visible at the call site and checkable in isolation.

Imports torch, so this module lives outside mypy's and pytest's reach in the hub
environment. It is pinned instead by tests/workers/golden/capture_behavior.py, run
under the training runtime.
"""
from __future__ import annotations

import math

import torch

# ∫₀¹ exp(-2(s-0.5)²) ds and its half-bell variant. These normalize the mean weight to
# 1 so switching schemes does not silently change the effective learning rate — without
# them, `bell` would inflate it by 17%.
BELL_MEAN = 0.8556243918920983
HALF_BELL_MEAN = 0.9278121959460491


def calculate_shift(image_seq_len: float, base_seq_len: float = 256,
                    max_seq_len: float = 6400, base_shift: float = 0.5,
                    max_shift: float = 1.15) -> float:
    """Interpolate the flow-matching timestep shift for a given latent sequence length.

    Higher resolutions need more of the schedule spent at high sigma, so the shift ramps
    linearly from `base_shift` at `base_seq_len` to `max_shift` at `max_seq_len`. Feeds
    both `sample_sigma` and the preview scheduler.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def sample_sigma(
    batch_size: int,
    image_seq_len: int,
    device: str | torch.device,
    shift_cfg: tuple[float, float, float, float],
    *,
    sampling: str = "krea2_shift",
    content_or_style: str = "balanced",
    logit_normal_mu: float = 0.0,
    logit_normal_sigma: float = 1.0,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
) -> torch.Tensor:
    """Sample one flow-matching sigma per batch item, shifted by resolution.

    `content_or_style` skews the underlying uniform before the shift is applied
    (arXiv 2302.08453 §3.4): `content` (u³) concentrates on low sigma — fine detail and
    likeness — while `style` (1−u³) concentrates on high sigma — composition and color.
    Neither applies under `logit_normal` sampling, which draws its own distribution.
    """
    if sampling == "logit_normal":
        u = torch.sigmoid(logit_normal_mu + logit_normal_sigma
                          * torch.randn(batch_size, device=device))
    else:
        u = torch.rand(batch_size, device=device)
        if content_or_style == "content":
            u = u ** 3
        elif content_or_style == "style":
            u = 1.0 - u ** 3

    mu = calculate_shift(image_seq_len, *shift_cfg)
    e_mu = math.exp(mu)
    sigma = e_mu / (e_mu + (1.0 / u.clamp(1e-6, 1 - 1e-6) - 1.0))
    sigma = sigma.clamp(1e-4, 1.0 - 1e-4)
    if sigma_min > 0.0 or sigma_max < 1.0:
        sigma = sigma.clamp(max(1e-4, sigma_min), min(1.0 - 1e-4, sigma_max))
    return sigma


def timestep_weight(sigma: torch.Tensor, weighting: str = "none") -> torch.Tensor:
    """ai-toolkit's BSMNTW/HBSMNTW loss weighting, re-expressed over sigma ∈ (0,1).

    Normalized, the weight spans 0.709 to 1.169, so at batch 1 it is a ±20% per-step
    modulation of the LR; its real effect comes from re-weighting micro-batches against
    each other inside the gradient-accumulation window.
    """
    if weighting == "bell":
        return torch.exp(-2.0 * (sigma - 0.5) ** 2) / BELL_MEAN
    if weighting == "half_bell":
        bell = torch.exp(-2.0 * (sigma - 0.5) ** 2)
        return torch.where(sigma < 0.5, torch.ones_like(bell), bell) / HALF_BELL_MEAN
    return torch.ones_like(sigma)


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """Patchify [B, C, H, W] latents into the [B, (H/2)*(W/2), C*4] sequence the transformer takes.

    Each 2x2 spatial block becomes one token carrying all four of its values per channel.
    `unpack_latents` is the exact inverse.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Inverse of `pack_latents`: rebuild [B, C, H, W] latents from the token sequence.

    `H` and `W` are latent-space dimensions, not pixels.
    """
    B, _, C = x.shape
    x = x.view(B, H // 2, W // 2, C // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C // 4, H, W)


def prepare_position_ids(text_seq_len: int, grid_h: int, grid_w: int,
                         device: str | torch.device) -> torch.Tensor:
    """Build the concatenated [text; image] RoPE position ids for one forward pass.

    Text tokens all get position (0, 0, 0) — they carry no positional signal, which is
    what makes caption compaction exact — while image tokens get their (row, col)
    coordinate on the patch grid.
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_h, grid_w, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_h * grid_w, 3)], dim=0)


def add_noise_offset(noise: torch.Tensor, channels: int, offset_scale: float) -> torch.Tensor:
    """Add a per-channel constant to packed noise.

    The noise is generated already packed, so channel c occupies indices c*4 … c*4+3 —
    hence the repeat_interleave. Discouraged under rectified flow, where sigma=1 is
    already pure noise in distribution; kept for parity with configurations that use it.
    """
    B = noise.shape[0]
    offset = torch.randn((B, 1, channels), device=noise.device, dtype=noise.dtype)
    return noise + offset_scale * offset.repeat_interleave(4, dim=2)
