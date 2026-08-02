"""Loading LoRA weights into an adapter and exporting them back out.

Both directions are failure-prone in the same quiet way: PEFT's loader is
non-strict, and safetensors metadata is free-form, so a mismatch produces a file that
loads fine everywhere and trains — or renders — to something subtly wrong. Both
functions here check rather than assume.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load, save_file

from krea2.config import TrainConfig

Logger = Callable[[str], None]


def lora_b_norm(model: torch.nn.Module) -> float:
    """Global ‖lora_B‖ across the adapter.

    PEFT initializes lora_B to exactly zero, so any value > 0 proves a weight load
    actually landed on the adapter rather than silently matching nothing.
    """
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def load_lora_weights(model: torch.nn.Module, blob: bytes, src: str,
                      log: Logger = print) -> float:
    """Load LoRA weights into the adapter and verify the load actually took effect.

    `set_peft_model_state_dict` uses load_state_dict(strict=False): if the keys did not
    match — a different rank, alpha or lora_target than the checkpoint — nothing would
    raise and the phase would train from random init in silence.

    Exits the process with code 1 rather than raising, which is what aborts the
    progressive orchestrator's pipeline. Deliberate: a traceback here would bury the
    one line that says which setting to fix.
    """
    result = set_peft_model_state_dict(model, load(blob))
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        log(f"[!] {len(unexpected)} unexpected keys loading / claves inesperadas al cargar {src}")
        for key in unexpected[:5]:
            log(f"    - {key}")
        sys.exit(1)

    b_norm = lora_b_norm(model)
    if b_norm == 0.0:
        log(f"[!] LoRA load was a no-op (‖lora_B‖ = 0) / la carga no surtió efecto: {src}")
        log("[!] ¿Coinciden lora_rank/lora_alpha/lora_target con los del checkpoint?")
        sys.exit(1)
    log(f"    ‖lora_B‖ = {b_norm:.4f}")
    return b_norm


def build_metadata(cfg: TrainConfig, step: int | None = None, epoch: int | None = None,
                   num_images: int | None = None) -> dict[str, str]:
    """Assemble the safetensors metadata map for an exported adapter.

    `ss_network_alpha` is the one that matters: at rank 16 and alpha 32 the correct
    scale is 2.0, but a loader with no alpha information assumes alpha == rank and
    applies 1.0 — the LoRA then runs at half strength with nothing to indicate it.
    """
    meta = {"format": "pt"}
    if not cfg.export_metadata:
        return meta

    name = cfg.project_name or "krea2_lora"
    meta.update({
        "ss_network_dim":           str(cfg.lora_rank),
        "ss_network_alpha":         str(cfg.lora_alpha),
        "ss_network_module":        "peft.LoraConfig",
        "ss_base_model_version":    "krea2",
        "ss_output_name":           name,
        "ss_learning_rate":         str(cfg.lr),
        "ss_optimizer":             cfg.optimizer,
        "ss_lr_scheduler":          cfg.lr_scheduler,
        "ss_seed":                  str(cfg.seed),
        "ss_training_comment":      f"AcademiaSD LoRAlab-Krea2 target={cfg.lora_target} "
                                    f"dtype={cfg.lora_dtype} ema={cfg.use_ema}",
        "modelspec.architecture":   "krea2/lora",
        "modelspec.implementation": "AcademiaSD_LoRAlab-Krea2",
        "modelspec.title":          name,
    })
    if num_images is not None:
        meta["ss_num_train_images"] = str(num_images)
    # The fake `1_<trigger>` entry is what makes the activation word visible in
    # ComfyUI's and A1111's metadata panels.
    if cfg.trigger_word:
        meta["ss_tag_frequency"] = json.dumps(
            {f"1_{cfg.trigger_word}": {cfg.trigger_word: 1}})
    if step is not None:
        meta["training_info"] = json.dumps({"step": int(step), "epoch": int(epoch or 0)})
    return meta


def export_lora(model: torch.nn.Module, path: str, cfg: TrainConfig,
                step: int | None = None, epoch: int | None = None,
                num_images: int | None = None) -> None:
    """Export the adapter as a flat bf16 safetensors file with training metadata.

    Keys are rewritten from PEFT's `base_model.model.*` layout to the `transformer.*`
    one that inference loaders expect. Per-module `.alpha` tensors stay opt-in, since
    adding keys that are not `lora_*` can confuse loaders expecting only A/B pairs.
    """
    clean: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if "lora_" not in key:
            continue
        new_key = "transformer." + key.replace("base_model.model.", "")
        clean[new_key] = value.to(torch.bfloat16).cpu().contiguous()

    if cfg.export_alpha_tensors:
        alpha = torch.tensor(float(cfg.lora_alpha), dtype=torch.bfloat16)
        for key in [k for k in clean if k.endswith("lora_A.default.weight")]:
            clean[key.replace("lora_A.default.weight", "alpha")] = alpha.clone()

    save_file(clean, path, metadata=build_metadata(cfg, step, epoch, num_images))


def set_lora_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast every LoRA adapter tensor to `dtype`, leaving the quantized base untouched.

    In bf16 (8-bit mantissa) the relative epsilon is ~0.0039, so any update smaller than
    0.39% of a weight's magnitude rounds away entirely: late in the cosine, with the LR
    already low, a good share of updates simply never land. PEFT casts the input to
    lora_A's dtype and the result back, so fp32 adapters work over the NF4 base without
    touching it — the cost is ~235 MB of VRAM.
    """
    for module in model.modules():
        for attr in ("lora_A", "lora_B"):
            container = getattr(module, attr, None)
            if container is not None:
                for adapter in container.values():
                    adapter.to(dtype=dtype)
        for attr in ("lora_embedding_A", "lora_embedding_B"):
            container = getattr(module, attr, None)
            if container is not None:
                for adapter in container.values():
                    adapter.data = adapter.data.to(dtype)


def target_modules(transformer: torch.nn.Module, lora_target: str,
                   log: Logger = print) -> tuple[list[str], list[str]]:
    """Pick which linear layers get an adapter. Returns (targets, all linear names).

    'all' includes the text_fusion blocks; the narrower presets stay inside the image
    `transformer_blocks`, where a LoRA does most of its work. An empty match falls back
    to everything rather than training an adapter attached to nothing.
    """
    import bitsandbytes as bnb

    all_linears = [name for name, m in transformer.named_modules()
                   if isinstance(m, (torch.nn.Linear, bnb.nn.Linear4bit))]

    def keep(name: str) -> bool:
        if lora_target == "all":
            return True
        if not name.startswith("transformer_blocks."):
            return False
        if ".attn." in name:
            return True
        return lora_target == "attn+ff" and ".ff." in name

    targets = [n for n in all_linears if keep(n)]
    if not targets:
        log(f"[!] lora_target '{lora_target}' matched no layers; falling back to 'all' / "
            f"no coincidió con ninguna capa.")
        targets = all_linears
    return targets, all_linears


def fingerprint(cfg: TrainConfig) -> str:
    """Identity a checkpoint must match for resuming to be meaningful.

    A change of rank or target means the saved weights no longer fit; a change of cache
    or dtype invalidates the optimizer state that was accumulated against them.
    """
    return json.dumps({
        "rank": cfg.lora_rank, "alpha": cfg.lora_alpha, "target": cfg.lora_target,
        "cache_dir": cfg.cache_dir, "lora_dtype": cfg.lora_dtype,
        "batch_size": cfg.batch_size, "optimizer": cfg.optimizer,
    }, sort_keys=True)


def build_optimizer(name: str, params: list[torch.nn.Parameter], lr: float,
                    betas: tuple[float, ...], eps: float,
                    weight_decay: float) -> Any:
    """Construct the optimizer named by `name`.

    The paged variant spills optimizer state to RAM under VRAM pressure instead of
    dying with an OOM at the peaks of the high-resolution phases, which is why it is
    the default.
    """
    import bitsandbytes as bnb

    kwargs = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if name == "adamw8bit":
        return bnb.optim.AdamW8bit(params, **kwargs)
    return bnb.optim.PagedAdamW8bit(params, **kwargs)
