"""LoRA discovery, PEFT injection, weight loading, and safetensors export for LTX 2.3.

Imports torch and PEFT: runs in the training_runtime environment.
"""
from __future__ import annotations

import sys
from typing import Any, Callable

import bitsandbytes as bnb
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load, save_file

from ltx23.config import LTX23TrainConfig

Logger = Callable[[str], None]


def discover_lora_targets(transformer: torch.nn.Module, only_attn: bool = True) -> list[str]:
    """Find Linear4bit target layers in transformer_blocks, excluding audio modules."""
    targets: list[str] = []

    excluded_parts = {
        "audio_attn1",
        "audio_attn2",
        "audio_ff",
        "audio_to_video_attn",
        "video_to_audio_attn",
    }

    attn_markers = (
        "attn1",
        "attn2",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    )

    for name, module in transformer.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit) or not name:
            continue

        parts = name.split(".")
        if any(part in excluded_parts for part in parts) or any(part.startswith("audio_") for part in parts):
            continue

        if "transformer_blocks" not in parts:
            continue

        if only_attn and not any(marker in name for marker in attn_markers):
            continue

        targets.append(name)

    targets = list(dict.fromkeys(targets))
    if not targets:
        raise RuntimeError("No se encontraron módulos Linear4bit visuales para LoRA.")

    return targets


def inject_lora(transformer: torch.nn.Module, cfg: LTX23TrainConfig) -> torch.nn.Module:
    """Inject PEFT LoRA adapters into visual Linear4bit layers."""
    targets = discover_lora_targets(transformer, only_attn=cfg.lora_only_attn)

    dora_val = getattr(cfg, "use_dora", False)
    try:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type=None,
            init_lora_weights=True,
            use_dora=dora_val,
        )
        model = get_peft_model(transformer, lora_config)
    except Exception as exc:
        if dora_val:
            print(f"[!] DoRA error ({exc}); revirtiendo a LoRA estándar.")
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=0.0,
                target_modules=targets,
                bias="none",
                task_type=None,
                init_lora_weights=True,
                use_dora=False,
            )
            model = get_peft_model(transformer, lora_config)
        else:
            raise exc

    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter in module.lora_A.values():
                adapter.to(dtype=torch.bfloat16)
        if hasattr(module, "lora_B"):
            for adapter in module.lora_B.values():
                adapter.to(dtype=torch.bfloat16)

    # Register gradient requirement hook on input projection layers
    def make_inputs_require_grad(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        if not torch.is_grad_enabled():
            return
        if torch.is_tensor(output):
            output.requires_grad_(True)

    for name in ("proj_in", "video_in", "audio_in", "x_embedder"):
        if hasattr(transformer, name):
            try:
                getattr(transformer, name).register_forward_hook(make_inputs_require_grad)
                break
            except Exception:
                pass

    return model


def lora_b_norm(model: torch.nn.Module) -> float:
    """Compute Frobenius norm of all lora_B parameters."""
    total = 0.0
    for name, p in model.named_parameters():
        if "lora_B" in name:
            total += p.detach().float().pow(2).sum().item()
    return total**0.5


def load_lora_weights(model: torch.nn.Module, blob: bytes, src: str, log: Logger = print) -> float:
    """Load LoRA weights into adapter and verify."""
    result = set_peft_model_state_dict(model, load(blob))
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        log(f"[!] {len(unexpected)} unexpected keys loading {src}")
        for key in unexpected[:5]:
            log(f"    - {key}")
        sys.exit(1)

    b_norm = lora_b_norm(model)
    if b_norm == 0.0:
        log(f"[!] LoRA load was a no-op (‖lora_B‖ = 0): {src}")
        sys.exit(1)
    log(f"    ‖lora_B‖ = {b_norm:.4f}")
    return b_norm


def save_lora(
    model: torch.nn.Module,
    path: str,
    cfg: LTX23TrainConfig,
    prefix: str | None = None,
    step: int | None = None,
) -> None:
    """Export standalone safetensors LoRA with baked alpha scaling and diffusion_model. prefix."""
    pref = cfg.lora_key_prefix if prefix is None else prefix
    scaling = float(cfg.lora_alpha) / float(max(1, cfg.lora_rank))

    state: dict[str, torch.Tensor] = {}

    for name, tensor in model.state_dict().items():
        if "lora_" not in name:
            continue

        clean = name.replace("base_model.model.", "").replace(".default.", ".")
        t = tensor.detach().to(torch.float32).cpu()

        if ".lora_B." in clean:
            t = t * scaling

        state[pref + clean] = t.to(torch.bfloat16).contiguous()

    metadata = {
        "format": "ltx23_lora",
        "lora_key_prefix": pref,
        "baked_scaling": f"{scaling:.6f}",
        "lora_rank": str(cfg.lora_rank),
        "lora_alpha": str(cfg.lora_alpha),
    }
    if step is not None:
        metadata["step"] = str(step)

    save_file(state, path, metadata=metadata)
