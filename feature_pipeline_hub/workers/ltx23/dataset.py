"""Dataset loading and in-memory tensor caching for LTX 2.3 multimodal training.

Imports torch: runs in the training_runtime environment.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Sequence

import torch

from ltx23.cache_index import CacheEntry, get_text_cache_paths, scan_cache_directory


def pin_cpu_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
    """Pin tensor to CPU memory if CUDA is available for fast asynchronous transfers."""
    if t is None:
        return None
    if torch.cuda.is_available() and not t.is_pinned():
        try:
            return t.pin_memory()
        except Exception:
            return t
    return t


def load_prompt_structure(cache_dir: str, prefix: str) -> Any:
    """Recursively reconstruct nested prompt conditioning structure from JSON and tensor files."""
    path = os.path.join(cache_dir, f"{prefix}_structure.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    def recurse(node: Any) -> Any:
        if isinstance(node, dict) and node.get("type") == "tensor":
            return torch.load(
                os.path.join(cache_dir, node["file"]),
                map_location="cpu",
                weights_only=True,
            )
        if isinstance(node, dict) and node.get("type") == "dict":
            return {k: recurse(v) for k, v in node["items"].items()}
        if isinstance(node, dict) and node.get("type") == "tuple":
            return tuple(recurse(v) for v in node["items"])
        if isinstance(node, dict) and node.get("type") == "list":
            return [recurse(v) for v in node["items"]]
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return node

    return recurse(structure)


def get_prompt_pair(prompt_result: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract (prompt_embeds, prompt_attention_mask) from prompt structure."""
    if isinstance(prompt_result, tuple):
        embeds = prompt_result[0]
        mask = prompt_result[1] if len(prompt_result) > 1 else None
        return embeds, mask
    if isinstance(prompt_result, dict):
        embeds = prompt_result.get("prompt_embeds")
        mask = prompt_result.get("prompt_attention_mask")
        return embeds, mask
    if torch.is_tensor(prompt_result):
        return prompt_result, None
    raise RuntimeError(f"Formato desconocido de prompt_result: {type(prompt_result)}")


def run_text_connectors(prompt_result: Any, connectors: torch.nn.Module, max_text_tokens: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Run text connector projection layers to produce (video_text, audio_text)."""
    embeds, mask = get_prompt_pair(prompt_result)
    embeds = embeds.to("cuda", dtype=torch.bfloat16)
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)

    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.int64, device="cuda")
    else:
        mask = mask.to("cuda")

    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    with torch.no_grad():
        out = connectors(embeds, mask, padding_side="left")

    if isinstance(out, (tuple, list)):
        video_text = out[0]
        audio_text = out[1]
    else:
        video_text = getattr(out, "video_text", getattr(out, "video_embeds", getattr(out, "video", None)))
        audio_text = getattr(out, "audio_text", getattr(out, "audio_embeds", getattr(out, "audio", None)))

    if video_text is None or audio_text is None:
        raise RuntimeError("connectors() no devolvió video_text y audio_text.")

    return video_text, audio_text


class LTX23Dataset:
    """Manages cached multimodal sample tensors in host memory."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        neg_conditioning: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.entries = entries
        self.neg_conditioning = neg_conditioning

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self) -> dict[str, Any]:
        """Draw one random sample."""
        return random.choice(self.entries)

    @classmethod
    def from_cache(
        cls,
        cache_dir: str,
        audio_channels: int = 128,
        max_text_tokens: int = 0,
        connectors: torch.nn.Module | None = None,
    ) -> LTX23Dataset:
        raw_entries = scan_cache_directory(cache_dir, max_text_tokens=max_text_tokens)
        if not raw_entries:
            raise RuntimeError(f"No se encontraron entradas válidas en la caché: {cache_dir}")

        # Check for missing precomputed connector outputs
        missing = [
            e for e in raw_entries
            if not (os.path.exists(e.video_text_path) and os.path.exists(e.audio_text_path))
        ]

        if missing and connectors is not None:
            connectors.to("cuda").eval().requires_grad_(False)
            for entry in missing:
                prompt_res = load_prompt_structure(cache_dir, f"{entry.name}_prompt")
                if prompt_res is None:
                    continue
                v_text, a_text = run_text_connectors(prompt_res, connectors, max_text_tokens=max_text_tokens)
                torch.save(v_text.detach().to("cpu", dtype=torch.bfloat16).contiguous(), entry.video_text_path)
                torch.save(a_text.detach().to("cpu", dtype=torch.bfloat16).contiguous(), entry.audio_text_path)
            connectors.to("cpu")

        # Negative prompt conditioning for caption dropout
        neg_cond = None
        neg_v_path = os.path.join(cache_dir, "_neg_video_text.pt")
        neg_a_path = os.path.join(cache_dir, "_neg_audio_text.pt")
        if os.path.exists(neg_v_path) and os.path.exists(neg_a_path):
            try:
                neg_cond = (
                    pin_cpu_tensor(torch.load(neg_v_path, map_location="cpu", weights_only=True).to(torch.bfloat16)),
                    pin_cpu_tensor(torch.load(neg_a_path, map_location="cpu", weights_only=True).to(torch.bfloat16)),
                )
            except Exception:
                pass
        elif connectors is not None:
            neg_prompt = load_prompt_structure(cache_dir, "_neg")
            if neg_prompt is not None:
                try:
                    connectors.to("cuda").eval().requires_grad_(False)
                    v_text, a_text = run_text_connectors(neg_prompt, connectors, max_text_tokens=max_text_tokens)
                    neg_cond = (
                        pin_cpu_tensor(v_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()),
                        pin_cpu_tensor(a_text.detach().to("cpu", dtype=torch.bfloat16).contiguous()),
                    )
                    torch.save(neg_cond[0], neg_v_path)
                    torch.save(neg_cond[1], neg_a_path)
                    connectors.to("cpu")
                except Exception:
                    pass

        loaded: list[dict[str, Any]] = []
        for entry in raw_entries:
            if not (os.path.exists(entry.video_text_path) and os.path.exists(entry.audio_text_path)):
                continue

            video_latent = torch.load(entry.video_path, map_location="cpu", weights_only=True)
            if video_latent is None:
                continue
            video_latent = pin_cpu_tensor(video_latent.to(torch.bfloat16))

            audio_latent_raw = torch.load(entry.audio_path, map_location="cpu", weights_only=True)
            if audio_latent_raw is None:
                bsz = video_latent.shape[0] if video_latent.ndim == 5 else 1
                audio_latent = torch.zeros((bsz, audio_channels, 1), dtype=torch.bfloat16)
            else:
                audio_latent = audio_latent_raw.to(torch.bfloat16)
            audio_latent = pin_cpu_tensor(audio_latent)

            video_text = pin_cpu_tensor(
                torch.load(entry.video_text_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
            )
            audio_text = pin_cpu_tensor(
                torch.load(entry.audio_text_path, map_location="cpu", weights_only=True).to(torch.bfloat16)
            )

            loaded.append({
                "name": entry.name,
                "video": video_latent,
                "audio": audio_latent,
                "video_text": video_text,
                "audio_text": audio_text,
            })

        if not loaded:
            raise RuntimeError("No se pudieron cargar tensores de entrenamiento completos.")

        return cls(loaded, neg_conditioning=neg_cond)
