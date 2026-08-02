"""Getting the 12B transformer into 12 GB of VRAM: NF4 quantization and an attention patch.

Two ways to reach a quantized model — quantize the bf16 weights in place, or rebuild
the layers from a pre-quantized on-disk cache. The cache path is the one that runs in
practice; quantizing takes minutes and produces the same result every time.

Imports torch and bitsandbytes, so this module only loads under the training runtime.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import torch
from bitsandbytes.functional import QuantState
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open

Logger = Callable[[str], None]

# Small embedding and projection layers at the model's edges. Quantizing them costs
# accuracy and saves almost no VRAM, so they stay in bf16.
SKIP_QUANT = ("img_in", "time_embed", "time_mod_proj", "txt_in", "final_layer")


def quantize_in_place(module: torch.nn.Module, prefix: str = "") -> None:
    """Recursively swap `nn.Linear` layers for 4-bit NF4 ones, in place.

    Layers whose qualified name contains any `SKIP_QUANT` fragment are left alone.
    Slow — prefer `load_nf4_cache` whenever a pre-quantized cache exists.
    """
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear) and not any(s in full for s in SKIP_QUANT):
            w = child.weight.data.float().contiguous()
            new_layer = Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None, quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )
            new_layer.weight = Params4bit(w, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                new_layer.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
            setattr(module, name, new_layer)
            del child, w
        else:
            quantize_in_place(child, full)


def _parent_of(root: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    """Walk a dotted module path, returning (owning module, final attribute name)."""
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _rebuild_layer(filepath: str, info: dict[str, Any]) -> Linear4bit:
    """Reassemble one `Linear4bit` from its packed weight and serialized QuantState."""
    with safe_open(filepath, framework="pt", device="cpu") as f:
        weight_data = f.get_tensor("weight")
        bias_data = f.get_tensor("bias") if info.get("bias", False) else None

        qs_dict = {key[len("quant_state."):]: f.get_tensor(key)
                   for key in f.keys() if key.startswith("quant_state.")}
        # The nested_* entries are the second-level quantization of absmax itself;
        # QuantState.from_dict rejects the blob if any of the five is missing.
        packed_qs = {
            "absmax": qs_dict["absmax"],
            "nested_absmax": qs_dict["nested_absmax"],
            "nested_quant_map": qs_dict["nested_quant_map"],
            "quant_map": qs_dict["quant_map"],
            "quant_state.bitsandbytes__nf4": qs_dict["quant_state.bitsandbytes__nf4"],
        }
        quant_state = QuantState.from_dict(packed_qs, device="cpu")

    weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4",
                        quant_storage=torch.uint8)
    weight.quant_state = quant_state
    weight.bnb_quantized = True

    layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"],
                       quant_type="nf4", compute_dtype=torch.bfloat16)
    layer.weight = weight
    if bias_data is not None:
        layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)
    return layer


def has_nf4_cache(cache_dir: str) -> bool:
    """True when `cache_dir` holds a pre-quantized model rather than plain weights."""
    return os.path.exists(os.path.join(cache_dir, "index.json"))


def load_nf4_cache(transformer: torch.nn.Module, cache_dir: str,
                   log: Logger = print) -> torch.nn.Module:
    """Rebuild NF4 layers from a pre-quantized on-disk cache instead of quantizing.

    `index.json` names every quantized layer and the safetensors file holding its packed
    weight and QuantState. Each rebuilt layer is re-checked for a live quant_state
    afterwards, and a count mismatch raises — training on half-dequantized weights would
    otherwise run to completion and simply produce garbage.
    """
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"index.json not found in NF4 cache / No existe index.json en caché NF4: {cache_dir}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    quantized = index.get("quantized", {})
    weights_dir = os.path.join(cache_dir, "weights")
    replaced = 0

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"NF4 weight file not found / No existe archivo NF4: {filepath}")
        parent, child_name = _parent_of(transformer, name)
        setattr(parent, child_name, _rebuild_layer(filepath, info))
        replaced += 1

    log(f"Reconstructed NF4 layers / Capas NF4 reconstruidas: {replaced}")

    verified = sum(
        1 for _, layer in transformer.named_modules()
        if isinstance(layer, Linear4bit)
        and getattr(layer.weight, "bnb_quantized", False)
        and layer.weight.quant_state is not None
    )
    log(f"Verified NF4 layers / Capas NF4 verificadas: {verified}")
    if verified != replaced:
        raise RuntimeError("NF4 Verification mismatch / La verificación NF4 no coincide")

    log("[OK] NF4 cache loaded successfully / Caché NF4 cargada correctamente.")
    return transformer


def patch_attention_for_low_vram(log: Logger = print) -> None:
    """Monkey-patch Krea2 attention to avoid SDPA's `math` backend when a mask is used.

    PyTorch rejects `attn_mask` together with `enable_gqa=True` on both the flash and
    mem-efficient kernels, falling back to `math`, which materializes the full
    [B, heads, S, S] score matrix (~568 MB at 768x768). Expanding K/V to Q's head count
    lets us pass `enable_gqa=False` and get mem-efficient back — tens of MB of K/V
    instead of hundreds of MB of scores.

    Inert when captions are compacted, since flash already handles GQA with no mask.
    """
    from diffusers.models.transformers import transformer_krea2

    original = transformer_krea2.dispatch_attention_fn

    def dispatch(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 *args: Any, attn_mask: torch.Tensor | None = None,
                 enable_gqa: bool = False, **kwargs: Any) -> Any:
        if attn_mask is not None and enable_gqa and key.shape[2] != query.shape[2]:
            repeats = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
            enable_gqa = False
        return original(query, key, value, *args, attn_mask=attn_mask,
                        enable_gqa=enable_gqa, **kwargs)

    transformer_krea2.dispatch_attention_fn = dispatch
    log("[OK] Attention patched to avoid the SDPA math backend / "
        "Atención parcheada para evitar el backend math.")
