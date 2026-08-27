"""NF4 loading, quantization reconstruction, and memory optimization for LTX 2.3.

Imports torch and bitsandbytes: runs in the training_runtime environment.
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Iterator

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open

try:
    from torch.autograd.graph import save_on_cpu as _save_on_cpu_ctx
    SAVE_ON_CPU_AVAILABLE = True
except Exception:
    _save_on_cpu_ctx = None
    SAVE_ON_CPU_AVAILABLE = False


def get_parent_module(root: torch.nn.Module, target_name: str) -> tuple[torch.nn.Module, str]:
    """Retrieve parent module and attribute name for a dot-separated module path."""
    parts = target_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def resolve_weight_path(cache_dir: str, filename: str) -> str:
    """Find weight safetensors across standard candidate subpaths."""
    candidates = [
        os.path.join(cache_dir, "weights", filename),
        os.path.join(cache_dir, filename),
        os.path.join(cache_dir, "weights", os.path.basename(filename)),
        os.path.join(cache_dir, os.path.basename(filename)),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return os.path.join(cache_dir, "weights", filename)


def load_nf4_cache_(transformer: torch.nn.Module, cache_dir: str) -> torch.nn.Module:
    """Load NF4 quantized weights and reconstruct Linear4bit modules into transformer."""
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No existe index.json: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    unquantized = index.get("unquantized", {})

    replaced = 0

    for name, info in quantized.items():
        filepath = resolve_weight_path(cache_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No existe peso NF4: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight_data = f.get_tensor("weight")

            bias_data = None
            if info.get("bias", False):
                bias_data = f.get_tensor("bias")

            qs_dict = {}
            for key in f.keys():
                if key.startswith("quant_state."):
                    qs_dict[key[len("quant_state."):]] = f.get_tensor(key)

            packed_qs = {k: v for k, v in qs_dict.items()}

        new_layer = Linear4bit(
            int(info["in_features"]),
            int(info["out_features"]),
            bias=info.get("bias", False),
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )

        new_weight = Params4bit.from_prequantized(
            data=weight_data,
            quantized_stats=packed_qs,
            requires_grad=False,
            device="cuda",
            module=new_layer,
        )

        new_layer.weight = new_weight

        if bias_data is not None:
            new_layer.bias = torch.nn.Parameter(
                bias_data.to("cuda", dtype=torch.bfloat16),
                requires_grad=False,
            )

        setattr(parent, child_name, new_layer)
        replaced += 1

    for name, info in unquantized.items():
        filepath = resolve_weight_path(cache_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No existe peso unquantized: {filepath}")

        parent, child_name = get_parent_module(transformer, name)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            weight = f.get_tensor("weight")
            bias = None
            if info.get("bias", False):
                bias = f.get_tensor("bias")

        layer = torch.nn.Linear(
            int(info["in_features"]),
            int(info["out_features"]),
            bias=info.get("bias", False),
        )

        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)

        setattr(parent, child_name, layer)

    verified = 0
    for _, module in transformer.named_modules():
        if isinstance(module, Linear4bit):
            if getattr(module.weight, "bnb_quantized", False) and getattr(module.weight, "quant_state", None) is not None:
                verified += 1

    if verified != replaced:
        raise RuntimeError(f"La verificación NF4 no coincide: {verified} vs {replaced}")

    return transformer


def cast_frozen_to_bf16(module: torch.nn.Module) -> None:
    """Cast all non-quantized parameters in frozen module to bfloat16."""
    for param in module.parameters():
        if not isinstance(param, Params4bit) and param.dtype in (torch.float32, torch.float16):
            param.data = param.data.to(torch.bfloat16)


def enable_memory_efficient_attention(transformer: torch.nn.Module) -> None:
    """Enable xformers or PyTorch Flash / SDP attention kernels."""
    try:
        transformer.enable_xformers_memory_efficient_attention()
        return
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass


@contextlib.contextmanager
def activation_offload_context(active: bool = True) -> Iterator[None]:
    """Context manager for offloading activations to pinned CPU memory during forward pass."""
    if active and SAVE_ON_CPU_AVAILABLE and _save_on_cpu_ctx is not None:
        with _save_on_cpu_ctx(pin_memory=True):
            yield
    else:
        yield
