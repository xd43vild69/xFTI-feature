"""Loading the pre-cached dataset: latents, captions, buckets, and the holdout split.

Training never touches images. The pre-cache stage has already run every image through
the VAE and every caption through the text encoder, so this module only reads tensors
off disk, groups them by latent resolution, and sets a slice aside for validation.

Everything is loaded into RAM up front and pinned. Datasets here are small — hundreds
of images, not millions — and a training step is far too short to hide disk latency.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from krea2.cache_index import (
    choose_holdout, entry_names, find_orphans, format_orphan_warning,
)

Logger = Callable[[str], None]

Bucket = tuple[int, int]
# (latent, text embedding, attention mask). The mask is None once captions are compacted.
Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]


def compact_caption(embeds: torch.Tensor,
                    mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop a caption's padding tokens and its mask, returning `(embeds, None)`.

    Exact, not an approximation: text tokens carry no RoPE — `prepare_position_ids`
    assigns them all position 0 — so attention is permutation-equivariant over them, and
    their outputs are discarded at `hidden_states[:, text_seq_len:]`. The mask was only
    key-padding, so removing the padded tokens is equivalent to masking them, and with
    no mask SDPA can use flash attention instead of falling back to the math backend.

    A caption with no valid tokens is returned untouched rather than emptied.
    """
    idx = mask[0].nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return embeds, mask
    return embeds[:, idx].contiguous(), None


def load_sample(directory: str, name: str, *, compact: bool, pin: bool) -> Sample:
    """Load one sample's `(latent, embedding, mask)` triple from the pre-cache.

    Pinning lets the training loop's host-to-device copies run async against compute.
    """
    latent = torch.load(f"{directory}/{name}_latent.pt", weights_only=True)
    embeds = torch.load(f"{directory}/{name}_embed.pt", weights_only=True)
    mask = torch.load(f"{directory}/{name}_mask.pt", weights_only=True).bool()

    latent, embeds = latent.to(torch.bfloat16), embeds.to(torch.bfloat16)
    optional_mask: torch.Tensor | None = mask
    if compact:
        embeds, optional_mask = compact_caption(embeds, mask)
    if pin:
        latent, embeds = latent.pin_memory(), embeds.pin_memory()
        if optional_mask is not None:
            optional_mask = optional_mask.pin_memory()
    return latent, embeds, optional_mask


@dataclass
class CachedDataset:
    """Everything the training loop reads from disk, loaded once."""

    samples: dict[str, Sample] = field(default_factory=dict)
    buckets: dict[Bucket, list[str]] = field(default_factory=dict)
    val_samples: dict[str, Sample] = field(default_factory=dict)
    val_names: list[str] = field(default_factory=list)
    # Embedding of the empty prompt, for CFG previews and caption dropout.
    negative: tuple[torch.Tensor, torch.Tensor | None] | None = None
    # Preview prompts encoded by the pre-cache, since the trainer has no text encoder.
    sample_prompts: list[tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]] = \
        field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def names(self) -> list[str]:
        return sorted(self.samples)


def bucket_of(sample: Sample) -> Bucket:
    """Latent (height, width) of a sample — batches may only mix within one."""
    return sample[0].shape[2], sample[0].shape[3]


def load(
    cache_dir: str,
    *,
    compact: bool,
    pin: bool,
    val_split: float = 0.0,
    val_cache_dir: str = "",
    log: Logger = print,
) -> CachedDataset:
    """Load the whole cache into memory, split off validation, and bucket by resolution.

    An explicit `val_cache_dir` wins over `val_split`; the split is applied before the
    sampler is built so the holdout never receives a gradient.
    """
    dataset = CachedDataset()
    buckets: dict[Bucket, list[str]] = defaultdict(list)

    for name in entry_names(cache_dir):
        sample = load_sample(cache_dir, name, compact=compact, pin=pin)
        dataset.samples[name] = sample
        buckets[bucket_of(sample)].append(name)

    orphans = find_orphans(list(dataset.samples), cache_dir)
    if orphans:
        log(format_orphan_warning(orphans))

    if val_cache_dir and os.path.isdir(val_cache_dir):
        for name in entry_names(val_cache_dir):
            dataset.val_samples[name] = load_sample(val_cache_dir, name,
                                                    compact=compact, pin=pin)
            dataset.val_names.append(name)
        log(f"[OK] Validation set from {val_cache_dir}: {len(dataset.val_names)} images")
    elif val_split > 0:
        holdout = choose_holdout(list(dataset.samples), val_split)
        if not holdout and dataset.samples:
            log("[!] val_split would hold out the whole dataset; disabling / desactivado.")
        for name in holdout:
            sample = dataset.samples.pop(name)
            dataset.val_samples[name] = sample
            dataset.val_names.append(name)
            size = bucket_of(sample)
            buckets[size].remove(name)
            if not buckets[size]:
                del buckets[size]
        if holdout:
            log(f"[OK] Holdout split: {len(holdout)} validation / "
                f"{len(dataset.samples)} training images")

    dataset.buckets = dict(buckets)
    dataset.negative = load_negative(cache_dir, compact=compact)
    dataset.sample_prompts = load_sample_prompts(cache_dir, compact=compact)
    return dataset


def load_negative(cache_dir: str, *,
                  compact: bool) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Load the cached empty-prompt embedding, or None when the pre-cache did not write one."""
    embed_path = os.path.join(cache_dir, "_neg_embed.pt")
    if not os.path.exists(embed_path):
        return None
    embeds = torch.load(embed_path, weights_only=True)
    mask = torch.load(os.path.join(cache_dir, "_neg_mask.pt"), weights_only=True).bool()
    optional_mask: torch.Tensor | None = mask
    if compact:
        embeds, optional_mask = compact_caption(embeds, mask)
    return embeds, optional_mask


def load_sample_prompts(
    cache_dir: str, *, compact: bool, limit: int = 64,
) -> list[tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]]:
    """Load preview prompts the pre-cache encoded, stopping at the first gap.

    The trainer holds no text encoder, so preview prompts are encoded during pre-cache
    and read back here the same way the negative embedding is. Each may carry its own
    size, seed, step count and CFG scale in a sidecar meta file.
    """
    prompts = []
    for i in range(limit):
        embed_path = os.path.join(cache_dir, f"_sample{i}_embed.pt")
        if not os.path.exists(embed_path):
            break
        embeds = torch.load(embed_path, weights_only=True).to(torch.bfloat16)
        mask = torch.load(os.path.join(cache_dir, f"_sample{i}_mask.pt"),
                          weights_only=True).bool()
        optional_mask: torch.Tensor | None = mask
        if compact:
            embeds, optional_mask = compact_caption(embeds, mask)

        meta: dict[str, Any] = {}
        meta_path = os.path.join(cache_dir, f"_sample{i}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        prompts.append((embeds, optional_mask, meta))
    return prompts


class PositionIdCache:
    """Memoizes RoPE position ids per (text length, latent height, latent width).

    Every step in a bucket rebuilds the same ids otherwise, and there are only a handful
    of distinct shapes in a run.
    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = device
        self._cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def get(self, text_len: int, latent_h: int, latent_w: int) -> torch.Tensor:
        from krea2.math_ops import prepare_position_ids

        key = (text_len, latent_h, latent_w)
        if key not in self._cache:
            self._cache[key] = prepare_position_ids(
                text_len, latent_h // 2, latent_w // 2, self.device)
        return self._cache[key]
