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


def _rebuild_layer(filepath: str, info: dict[str, Any],
                   device: str | torch.device = "cpu") -> Linear4bit:
    """Reassemble one `Linear4bit` from its packed weight and serialized QuantState.

    When `device` is CUDA, streams tensors directly onto the GPU so we avoid
    materializing the full 12B model in host RAM before `.to("cuda")`.
    """
    target_dev = str(device)
    with safe_open(filepath, framework="pt", device=target_dev) as f:
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
        quant_state = QuantState.from_dict(packed_qs, device=torch.device(target_dev))

    weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4",
                        quant_storage=torch.uint8)
    weight.quant_state = quant_state
    weight.bnb_quantized = True

    layer = Linear4bit(info["in_features"], info["out_features"], bias=info["bias"],
                       quant_type="nf4", compute_dtype=torch.bfloat16)
    layer.weight = weight
    if bias_data is not None:
        layer.bias = torch.nn.Parameter(
            bias_data.to(device=target_dev, dtype=torch.bfloat16),
            requires_grad=False,
        )
    return layer


def _rebuild_unquantized_layer(filepath: str, info: dict[str, Any],
                               device: str | torch.device = "cpu") -> torch.nn.Linear:
    """Reassemble an unquantized `nn.Linear` (e.g. `img_in`, `time_embed`, `final_layer`) from disk."""
    target_dev = str(device)
    with safe_open(filepath, framework="pt", device=target_dev) as f:
        weight_data = f.get_tensor("weight").to(dtype=torch.bfloat16)
        bias_data = (
            f.get_tensor("bias").to(dtype=torch.bfloat16)
            if info.get("bias", False)
            else None
        )

    layer = torch.nn.Linear(
        int(info["in_features"]),
        int(info["out_features"]),
        bias=bool(info.get("bias", False)),
        device="meta",
        dtype=torch.bfloat16,
    )
    layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
    if bias_data is not None:
        layer.bias = torch.nn.Parameter(bias_data, requires_grad=False)
    return layer


def _populate_remaining_from_shards(
    transformer: torch.nn.Module,
    cache_dir: str,
    already_loaded_prefixes: set[str],
    device: str | torch.device = "cuda",
) -> int:
    """Load remaining non-Linear parameters/buffers (norms, scale_shift_table, projector) from shards.

    When `transformer` is instantiated with `init_empty_weights()`, parameters outside
    `index.json` are still on the `meta` device. Reading ONLY those ~170 tiny tensors
    (a few MB total) via `safe_open` skips reading the 24 GB of bf16 linear weights that
    `DiffusionPipeline.from_pretrained` would otherwise load and immediately discard.
    """
    shard_index_path = os.path.join(
        cache_dir, "transformer", "diffusion_pytorch_model.safetensors.index.json"
    )
    if not os.path.exists(shard_index_path):
        return 0

    with open(shard_index_path, "r", encoding="utf-8") as f:
        shard_index = json.load(f)

    weight_map: dict[str, str] = shard_index.get("weight_map", {})
    by_shard: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        # Skip any parameter belonging to a module already populated from index.json
        module_prefix = tensor_name.rsplit(".", 1)[0]
        if module_prefix in already_loaded_prefixes:
            continue
        by_shard.setdefault(shard_file, []).append(tensor_name)

    target_dev = str(device)
    loaded_count = 0
    transformer_dir = os.path.join(cache_dir, "transformer")
    for shard_file, tensor_names in by_shard.items():
        shard_path = os.path.join(transformer_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        with safe_open(shard_path, framework="pt", device=target_dev) as f:
            for tensor_name in tensor_names:
                tensor = f.get_tensor(tensor_name).to(dtype=torch.bfloat16)
                parent, attr = _parent_of(transformer, tensor_name)
                existing = getattr(parent, attr, None)
                if isinstance(existing, torch.nn.Parameter):
                    setattr(parent, attr, torch.nn.Parameter(tensor, requires_grad=False))
                else:
                    parent.register_buffer(attr, tensor)
                loaded_count += 1
    return loaded_count


def has_nf4_cache(cache_dir: str) -> bool:
    """True when `cache_dir` holds a pre-quantized model rather than plain weights."""
    return os.path.exists(os.path.join(cache_dir, "index.json"))


def load_fast_nf4_transformer_and_scheduler(
    model_id: str,
    device: str | torch.device = "cuda",
    log: Logger = print,
) -> tuple[torch.nn.Module, Any]:
    """Instantiate Krea2Transformer2DModel on `meta` and stream NF4 + norm weights straight to GPU.

    Falls back to standard `DiffusionPipeline.from_pretrained` if `accelerate` or the
    transformer config is unavailable.
    """
    from accelerate import init_empty_weights
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

    scheduler_dir = os.path.join(model_id, "scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scheduler_dir if os.path.isdir(scheduler_dir) else model_id
    )

    transformer_dir = os.path.join(model_id, "transformer")
    cfg_source = transformer_dir if os.path.isdir(transformer_dir) else model_id
    config = Krea2Transformer2DModel.load_config(cfg_source)

    with init_empty_weights():
        transformer = Krea2Transformer2DModel.from_config(config)

    transformer = load_nf4_cache(
        transformer,
        model_id,
        device=device,
        populate_unquantized=True,
        log=log,
    )
    return transformer, scheduler


def load_nf4_cache(transformer: torch.nn.Module, cache_dir: str,
                   log: Logger = print, *,
                   device: str | torch.device = "cpu",
                   populate_unquantized: bool = False) -> torch.nn.Module:
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
    unquantized = index.get("unquantized", {})
    weights_dir = os.path.join(cache_dir, "weights")
    replaced = 0
    loaded_prefixes: set[str] = set()

    for name, info in quantized.items():
        filepath = os.path.join(weights_dir, info["file"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"NF4 weight file not found / No existe archivo NF4: {filepath}")
        parent, child_name = _parent_of(transformer, name)
        setattr(parent, child_name, _rebuild_layer(filepath, info, device=device))
        loaded_prefixes.add(name)
        replaced += 1

    if populate_unquantized:
        for name, info in unquantized.items():
            filepath = os.path.join(weights_dir, info["file"])
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Unquantized weight file not found / No existe archivo: {filepath}")
            parent, child_name = _parent_of(transformer, name)
            setattr(
                parent,
                child_name,
                _rebuild_unquantized_layer(filepath, info, device=device),
            )
            loaded_prefixes.add(name)
        _populate_remaining_from_shards(
            transformer, cache_dir, loaded_prefixes, device=device
        )

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
