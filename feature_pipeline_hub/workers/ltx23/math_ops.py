"""Pure tensor and mathematical operations for LTX 2.3 spatio-temporal trainer.

Imports torch: runs in the training_runtime environment.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def align_video_latent_to_patch(latent: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Trim video latent to exact multiples of spatial and temporal patch sizes."""
    if latent.ndim != 5:
        return latent

    B, C, Fm, H, W = latent.shape
    patch_size = max(1, int(patch_size))
    patch_size_t = max(1, int(patch_size_t))

    Fm = (Fm // patch_size_t) * patch_size_t
    H = (H // patch_size) * patch_size
    W = (W // patch_size) * patch_size

    return latent[:, :, :Fm, :H, :W].contiguous()


def patch_video_latent(latent: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Patchify 5D video latents [B, C, F, H, W] into 3D token sequences [B, S, C_out]."""
    if latent.ndim != 5:
        raise RuntimeError(f"Video latent expected [B, C, F, H, W], got shape {tuple(latent.shape)}")

    B, C, Fm, H, W = latent.shape
    patch_size = max(1, int(patch_size))
    patch_size_t = max(1, int(patch_size_t))

    if Fm % patch_size_t != 0:
        Fm = (Fm // patch_size_t) * patch_size_t
    if H % patch_size != 0:
        H = (H // patch_size) * patch_size
    if W % patch_size != 0:
        W = (W // patch_size) * patch_size

    latent = latent[:, :, :Fm, :H, :W]

    x = latent.view(
        B,
        C,
        Fm // patch_size_t,
        patch_size_t,
        H // patch_size,
        patch_size,
        W // patch_size,
        patch_size,
    )
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(B, -1, C * patch_size_t * patch_size * patch_size)


def patch_audio_latent(latent: torch.Tensor) -> torch.Tensor:
    """Format audio latents [B, C, T] or [C, T] to token sequence [B, T, C]."""
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3:
        raise RuntimeError(f"Audio latent expected [B, C, T] or [C, T], got shape {tuple(latent.shape)}")
    return latent.transpose(1, 2).contiguous()


def unpack_video_latent(
    tokens: torch.Tensor,
    latent_shape: tuple[int, ...],
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> torch.Tensor:
    """Unpatchify token sequence back to 5D video latent [B, C, F, H, W]."""
    B, C, Fm, H, W = tuple(latent_shape)
    pt = max(1, int(patch_size_t))
    p = max(1, int(patch_size))

    Fp = Fm // pt
    Hp = H // p
    Wp = W // p
    expected_seq = Fp * Hp * Wp

    tokens = tokens[:, :expected_seq, :]
    x = tokens.view(B, Fp, Hp, Wp, C, pt, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(B, C, Fp * pt, Hp * p, Wp * p)


def make_video_timestep(
    sigma: torch.Tensor,
    seq_len: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    multiplier: float = 1000.0,
) -> torch.Tensor:
    """Expand sigma into per-token positional timestep tensor."""
    return (
        sigma.view(-1, 1).expand(-1, seq_len) * float(multiplier)
    ).to(device=device, dtype=dtype)


def mse_loss_chunked(
    pred: torch.Tensor,
    target: torch.Tensor,
    chunk_elements: int = 2000000,
) -> torch.Tensor:
    """Compute MSE loss in fixed-size chunks to avoid CUDA allocation spikes on large tensors."""
    chunk_elements = max(64, int(chunk_elements))

    if pred.numel() == 0:
        return pred.new_zeros((), dtype=torch.float32)

    if pred.numel() <= chunk_elements:
        return F.mse_loss(pred.float(), target.float())

    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    n = pred_flat.numel()
    loss_sum = torch.zeros((), device=pred.device, dtype=torch.float32)

    for start in range(0, n, chunk_elements):
        end = min(start + chunk_elements, n)
        p = pred_flat[start:end].float()
        t = target_flat[start:end].float()
        loss_sum = loss_sum + F.mse_loss(p, t, reduction="sum")
        del p, t

    return loss_sum / float(n)


def sample_continuous_sigma(
    batch_size: int,
    device: str | torch.device = "cuda",
    mode: str = "logit_normal",
    mean: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Sample continuous sigma with logit-normal or uniform distribution."""
    mode = str(mode or "logit_normal").strip().lower()
    if mode == "logit_normal":
        u = torch.randn(batch_size, device=device, dtype=torch.float32) * float(std) + float(mean)
        return torch.sigmoid(u).clamp(1e-4, 1.0 - 1e-4)
    return torch.rand(batch_size, device=device, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
