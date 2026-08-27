import os
import platform

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("DIFFUSERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

if platform.system() != "Windows":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import gc
import hashlib
import importlib
import json
import math
import shutil
import sys
import time
import traceback
from typing import Any

import torch
import torchvision.transforms.functional as F_vision
from diffusers import DiffusionPipeline
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def from_root(path: str) -> str:
    """Resolve relative path against project root."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# ── RAM Detection ───────────────────────────────────────────────────────────
def _detect_system_ram_gb() -> float | None:
    try:
        import psutil
        return float(psutil.virtual_memory().total / (1024**3))
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys / (1024**3))
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return float(kb / (1024**2))
        except Exception:
            pass

    return None


SYSTEM_RAM_GB = _detect_system_ram_gb()

# ── DEFAULTS ────────────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    "model_id": "./LTX23-NF4",
    "dataset_path": "./dataset",
    "cache_dir": "./cached_data_ltx23",
    "target_area": 512 * 512,
    "max_side": 1280,
    "multiple": 32,
    "max_seq_len": 1024,
    "frame_rate": 24.0,
    "num_frames": 1,
    "project_name": "",
    "trigger_word": "",
    "preview_custom_prompt": "",
    "precache_offload": "sequential",  # none | model | sequential | cpu
    "text_encoder_4bit": True,
    "low_ram_threshold_gb": 48.0,
    "low_ram_allow_cpu_fallback": False,
}

CONFIG_PATH = os.environ.get("PRECACHE_SETTINGS_PATH", os.path.join(PROJECT_ROOT, "pre_cache_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from / Configuración cargada desde: {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ Settings file not found at {CONFIG_PATH}, using defaults.")


def _cfg_get(key: str, default: Any) -> Any:
    for candidate in (key, key + " ", " " + key):
        if candidate in cfg:
            return cfg[candidate]
    return default


def _cfg_bool(key: str, default: bool) -> bool:
    val = _cfg_get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(val)


from ltx23.downloader import ensure_ltx23_model_downloaded

MODEL_ID_RAW = str(_cfg_get("model_id", DEFAULTS["model_id"])).strip()
if not os.path.isabs(MODEL_ID_RAW):
    MODEL_ID_RAW = from_root(MODEL_ID_RAW)

MODEL_ID = str(ensure_ltx23_model_downloaded(MODEL_ID_RAW))

DATASET_PATH = from_root(str(_cfg_get("dataset_path", DEFAULTS["dataset_path"])).strip())
TARGET_AREA = int(_cfg_get("target_area", DEFAULTS["target_area"]))
MAX_SIDE = int(_cfg_get("max_side", DEFAULTS["max_side"]))
MULTIPLE = max(32, int(_cfg_get("multiple", DEFAULTS["multiple"])))
MAX_SEQ_LEN = int(_cfg_get("max_seq_len", DEFAULTS["max_seq_len"]))
FRAME_RATE = float(_cfg_get("frame_rate", DEFAULTS["frame_rate"]))
NUM_FRAMES = int(_cfg_get("num_frames", DEFAULTS["num_frames"]))
TRIGGER_WORD = str(_cfg_get("trigger_word", DEFAULTS["trigger_word"])).strip()
PROJECT_NAME = str(_cfg_get("project_name", DEFAULTS["project_name"])).strip()
PREVIEW_CUSTOM_PROMPT = str(_cfg_get("preview_custom_prompt", DEFAULTS["preview_custom_prompt"])).strip()

PRECACHE_OFFLOAD = str(_cfg_get("precache_offload", DEFAULTS["precache_offload"])).strip().lower()
TEXT_ENCODER_4BIT = _cfg_bool("text_encoder_4bit", DEFAULTS["text_encoder_4bit"])
LOW_RAM_THRESHOLD_GB = float(_cfg_get("low_ram_threshold_gb", DEFAULTS["low_ram_threshold_gb"]))
LOW_RAM_ALLOW_CPU_FALLBACK = _cfg_bool("low_ram_allow_cpu_fallback", DEFAULTS["low_ram_allow_cpu_fallback"])

LOW_RAM_MODE = bool(
    LOW_RAM_THRESHOLD_GB > 0 and SYSTEM_RAM_GB is not None and SYSTEM_RAM_GB <= LOW_RAM_THRESHOLD_GB
)

raw_cache_dir = str(_cfg_get("cache_dir", "")).strip()
if raw_cache_dir:
    CACHE_DIR = from_root(raw_cache_dir)
elif PROJECT_NAME:
    CACHE_DIR = from_root(f"./cached_data_ltx23_{PROJECT_NAME}")
else:
    CACHE_DIR = from_root(str(DEFAULTS["cache_dir"]).strip())

if LOW_RAM_MODE:
    PRECACHE_OFFLOAD = "sequential"
    TEXT_ENCODER_4BIT = True
    MAX_SEQ_LEN = min(MAX_SEQ_LEN, 512)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


# ── Helpers ─────────────────────────────────────────────────────────────────
def vram_gb() -> float:
    return float(torch.cuda.memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0


def vram_peak_gb() -> float:
    return float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0


def free_vram(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bucket_size(w: int, h: int, area: int | None = None) -> tuple[int, int]:
    if area is None:
        area = TARGET_AREA
    ar = w / h
    bh = math.sqrt(area / ar)
    bw = ar * bh
    bw = max(MULTIPLE, round(bw / MULTIPLE) * MULTIPLE)
    bh = max(MULTIPLE, round(bh / MULTIPLE) * MULTIPLE)
    if max(bw, bh) > MAX_SIDE:
        s = MAX_SIDE / max(bw, bh)
        bw = max(MULTIPLE, int(bw * s) // MULTIPLE * MULTIPLE)
        bh = max(MULTIPLE, int(bh * s) // MULTIPLE * MULTIPLE)
    return bw, bh


def read_audio_channels(model_id: str, default: int = 128) -> int:
    for rel in ("transformer/config.json", os.path.join("transformer", "config.json")):
        p = os.path.join(model_id, rel)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    c = json.load(f)
                return int(c.get("audio_in_channels", default))
            except Exception:
                pass
    return default


def read_model_index_components(model_id: str) -> dict[str, Any]:
    path = os.path.join(model_id, "model_index.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if not k.startswith("_") and isinstance(v, (list, tuple)) and len(v) == 2:
                    out[k] = v
        return out
    except Exception:
        return {}


def resolve_component_class(model_id: str, component_name: str, current_object: Any = None) -> Any:
    info = read_model_index_components(model_id).get(component_name)
    if info is not None:
        module_name, class_name = info
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, class_name)
        except Exception:
            pass
    return type(current_object) if current_object is not None else None


def list_text_encoder_components(pipe: Any) -> list[str]:
    names: list[str] = []
    for key in read_model_index_components(MODEL_ID).keys():
        if key.lower().startswith("text_encoder") and key not in names:
            names.append(key)
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if hasattr(pipe, attr) and attr not in names:
            names.append(attr)
    return names


def unload_pipeline_component(pipe: Any, name: str) -> None:
    obj = getattr(pipe, name, None)
    if obj is not None:
        try:
            setattr(pipe, name, None)
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_precache_pipeline_light(model_id: str, skip_text_encoders: bool = True) -> Any:
    overrides: dict[str, Any] = {"transformer": None}
    if skip_text_encoders:
        for name in read_model_index_components(model_id).keys():
            if name.lower().startswith("text_encoder"):
                overrides[name] = None
    try:
        return DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **overrides,
        )
    except Exception:
        return DiffusionPipeline.from_pretrained(
            model_id,
            transformer=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )


def _patch_module_to_noop_device(module: Any) -> None:
    if module is None or getattr(module, "_ltx_to_patched", False):
        return
    orig_to = module.to

    def _to(*args: Any, **kwargs: Any) -> Any:
        dtype = kwargs.get("dtype", None)
        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a
        if dtype is not None:
            try:
                return orig_to(dtype=dtype)
            except Exception:
                return module
        return module

    try:
        module.to = _to
        module._ltx_to_patched = True
    except Exception:
        pass


def quantize_one_text_encoder_4bit(pipe: Any, component_name: str) -> tuple[bool, str | None]:
    current_obj = getattr(pipe, component_name, None)
    comp_class = resolve_component_class(MODEL_ID, component_name, current_obj)
    if comp_class is None:
        return False, None

    unload_pipeline_component(pipe, component_name)
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        return False, None

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        new_obj = comp_class.from_pretrained(
            MODEL_ID,
            subfolder=component_name,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cuda:0" if not LOW_RAM_MODE else "auto",
        )
        setattr(pipe, component_name, new_obj)
        return True, "4bit"
    except Exception as exc:
        print(f"[4bit] Error cuantizando {component_name}: {exc}")
        # Fallback to BF16 CPU
        try:
            fallback = comp_class.from_pretrained(
                MODEL_ID,
                subfolder=component_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            setattr(pipe, component_name, fallback)
            return True, "bf16 CPU fallback"
        except Exception:
            return False, None


def try_quantize_text_encoders_4bit(pipe: Any) -> tuple[bool, str]:
    names = list_text_encoder_components(pipe)
    if not names:
        return False, "cpu"

    all_ok = True
    for name in names:
        ok, _ = quantize_one_text_encoder_4bit(pipe, name)
        if not ok:
            all_ok = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (True, "cuda") if all_ok else (False, "cpu")


def setup_offload(pipe: Any, mode: str) -> str:
    vae = getattr(pipe, "vae", None)
    if mode == "none":
        pipe.to("cuda")
        return "cuda"
    elif mode == "model":
        pipe.enable_model_cpu_offload()
        return "cuda"
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))
        return "cuda"
    elif mode == "cpu":
        if vae is not None:
            vae.to("cuda")
        return "cpu"
    return "cuda"


def encode_video_latent(vae: Any, image: Image.Image) -> torch.Tensor:
    image_tensor = F_vision.pil_to_tensor(image).float() / 127.5 - 1.0
    image_tensor = (
        image_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1)
    ).to("cuda", dtype=torch.bfloat16)

    encoded = vae.encode(image_tensor)
    if hasattr(encoded, "latent_dist"):
        latent = encoded.latent_dist.sample()
    elif torch.is_tensor(encoded):
        latent = encoded
    elif isinstance(encoded, tuple):
        latent = encoded[0]
    else:
        raise RuntimeError(f"Salida inesperada de VAE.encode: {type(encoded)}")

    latent = latent.detach()

    # VAE mean and standard deviation statistical normalization
    latents_mean = getattr(vae, "latents_mean", None)
    latents_std = getattr(vae, "latents_std", None)

    if latents_mean is not None and latents_std is not None:
        latents_mean = latents_mean.to(device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        latents_std = latents_std.to(device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
        latent = (latent - latents_mean) * scaling_factor / latents_std

    return latent.detach().to(torch.bfloat16).cpu().contiguous()


def make_audio_latent(video_latent: torch.Tensor, audio_channels: int) -> torch.Tensor:
    return torch.zeros((video_latent.shape[0], audio_channels, 1), dtype=torch.bfloat16)


def read_prompt(base: str) -> str:
    path = os.path.join(DATASET_PATH, f"{base}.txt")
    prompt = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
    if TRIGGER_WORD and TRIGGER_WORD.lower() not in prompt.lower():
        prompt = f"{TRIGGER_WORD}, {prompt}".strip(", ")
    return prompt


def save_prompt_result(result: Any, prefix: str) -> Any:
    def recurse(obj: Any, path: str) -> Any:
        if torch.is_tensor(obj):
            safe_path = path.replace(".", "_").replace("/", "_").replace("\\", "_")
            filename = f"{prefix}_{safe_path}.pt"
            torch.save(obj.detach().cpu(), os.path.join(CACHE_DIR, filename))
            return {
                "type": "tensor",
                "file": filename,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, dict):
            return {"type": "dict", "items": {str(k): recurse(v, f"{path}_{k}") for k, v in obj.items()}}
        if isinstance(obj, (tuple, list)):
            return {
                "type": "tuple" if isinstance(obj, tuple) else "list",
                "items": [recurse(v, f"{path}_{i}") for i, v in enumerate(obj)],
            }
        return {"type": "value", "value": obj}

    structure = recurse(result, "root")
    atomic_json(structure, os.path.join(CACHE_DIR, f"{prefix}_structure.json"))
    return structure


def encode_prompt(pipe: Any, prompt: str, text_device: str) -> Any:
    return pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        max_sequence_length=MAX_SEQ_LEN,
        device=torch.device(text_device),
        dtype=torch.bfloat16,
    )


def atomic_json(data: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Main Pre-Cache Process ──────────────────────────────────────────────────
def preprocess_ltx23() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible para pre-cache LTX 2.3.")

    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    images = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not images:
        raise RuntimeError(f"No hay imágenes en {DATASET_PATH}")

    print("================================================================")
    print("       LTX 2.3 PRE-CACHE (xFTI Worker)")
    print("================================================================")
    print(f"Modelo: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH} ({len(images)} imágenes)")
    print(f"Caché: {CACHE_DIR}")
    print(f"Target Area: {TARGET_AREA} | Multiple: {MULTIPLE}")
    print(f"Offload: {PRECACHE_OFFLOAD} | TextEnc 4-bit: {TEXT_ENCODER_4BIT}")
    print("================================================================")

    audio_channels = read_audio_channels(MODEL_ID, 128)
    pipe = load_precache_pipeline_light(MODEL_ID, skip_text_encoders=TEXT_ENCODER_4BIT)

    used_4bit = False
    text_device = "cpu"
    if TEXT_ENCODER_4BIT:
        used_4bit, text_device = try_quantize_text_encoders_4bit(pipe)

    if used_4bit:
        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "connectors"):
            _patch_module_to_noop_device(getattr(pipe, attr, None))
        text_device = "cuda"
    else:
        text_device = setup_offload(pipe, PRECACHE_OFFLOAD)

    torch.cuda.reset_peak_memory_stats()

    # ── Phase 1: Text Prompt Pre-encoding ─────────────────────────────────────
    # Text encoders are executed while VAE remains on CPU, saving ~1.5 GB VRAM.
    print("\n--- Fase 1/2: Pre-codificación de Prompts de Texto ---")
    with torch.inference_mode():
        neg_result = encode_prompt(pipe, "", text_device)
    save_prompt_result(neg_result, "_neg")
    free_vram(neg_result)

    # Pre-encode custom prompt if provided
    if PREVIEW_CUSTOM_PROMPT:
        cp = PREVIEW_CUSTOM_PROMPT
        if TRIGGER_WORD and TRIGGER_WORD.lower() not in cp.lower():
            cp = f"{TRIGGER_WORD}, {cp}".strip(", ")
        with torch.inference_mode():
            custom_res = encode_prompt(pipe, cp, text_device)
        save_prompt_result(custom_res, "_custom")
        free_vram(custom_res)

    prompt_structures: dict[str, Any] = {}
    for idx, filename in enumerate(images, start=1):
        base = os.path.splitext(filename)[0]
        prompt_struct_path = os.path.join(CACHE_DIR, f"{base}_prompt_structure.json")
        if os.path.exists(prompt_struct_path):
            try:
                with open(prompt_struct_path, "r", encoding="utf-8") as f:
                    prompt_structures[base] = json.load(f)
                continue
            except Exception:
                pass

        prompt = read_prompt(base)
        with torch.inference_mode():
            prompt_res = encode_prompt(pipe, prompt, text_device)
        struct = save_prompt_result(prompt_res, f"{base}_prompt")
        prompt_structures[base] = struct
        free_vram(prompt_res)

    # ── Free Text Encoders before Phase 2 ─────────────────────────────────────
    # Unload text encoders to free GPU memory for VAE processing
    for te_name in list_text_encoder_components(pipe):
        unload_pipeline_component(pipe, te_name)
    free_vram()

    # ── Phase 2: VAE Video Latent Encoding ────────────────────────────────────
    print("\n--- Fase 2/2: Pre-codificación de Latentes VAE de Video ---")
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        try:
            vae.to("cuda", dtype=torch.bfloat16)
        except Exception:
            pass

    for idx, filename in enumerate(images, start=1):
        base = os.path.splitext(filename)[0]
        video_path = os.path.join(CACHE_DIR, f"{base}_video_latent.pt")
        audio_path = os.path.join(CACHE_DIR, f"{base}_audio_latent.pt")
        info_path = os.path.join(CACHE_DIR, f"{base}_info.json")

        if os.path.exists(video_path) and os.path.exists(audio_path) and os.path.exists(info_path):
            print(f"[{idx}/{len(images)}] Up to date / Omitido: {filename}")
            continue

        print(f"[{idx}/{len(images)}] Procesando VAE: {filename}")
        img = Image.open(os.path.join(DATASET_PATH, filename)).convert("RGB")
        bw, bh = bucket_size(img.width, img.height)

        scale = max(bw / img.width, bh / img.height)
        img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)), Image.LANCZOS)
        left = (img.width - bw) // 2
        top = (img.height - bh) // 2
        img = img.crop((left, top, left + bw, top + bh))

        with torch.inference_mode():
            video_latent = encode_video_latent(pipe.vae, img)
        torch.save(video_latent, video_path)

        audio_latent = make_audio_latent(video_latent, audio_channels)
        torch.save(audio_latent, audio_path)

        prompt = read_prompt(base)
        struct = prompt_structures.get(base)
        if struct is None:
            struct_path = os.path.join(CACHE_DIR, f"{base}_prompt_structure.json")
            if os.path.exists(struct_path):
                with open(struct_path, "r", encoding="utf-8") as f:
                    struct = json.load(f)

        atomic_json(
            {
                "filename": filename,
                "width": bw,
                "height": bh,
                "num_frames": NUM_FRAMES,
                "frame_rate": FRAME_RATE,
                "prompt": prompt,
                "video_latent": os.path.basename(video_path),
                "audio_latent": os.path.basename(audio_path),
                "prompt_structure": struct,
            },
            info_path,
        )
        free_vram(audio_latent, video_latent)

    # Final manifest
    atomic_json(
        {
            "format": "LTX23-LoRA-Precache",
            "version": 1,
            "model_id": MODEL_ID,
            "dataset_path": DATASET_PATH,
            "cache_dir": CACHE_DIR,
            "target_area": TARGET_AREA,
            "multiple": MULTIPLE,
            "frame_rate": FRAME_RATE,
            "num_frames": NUM_FRAMES,
            "max_sequence_length": MAX_SEQ_LEN,
            "trigger_word": TRIGGER_WORD,
            "audio_latent_channels": audio_channels,
        },
        os.path.join(CACHE_DIR, "cache_info.json"),
    )

    free_vram(pipe)
    print("✓ Pre-cache LTX 2.3 completado exitosamente.")


if __name__ == "__main__":
    try:
        t_start = time.time()
        try:
            import _telemetry
            _telemetry.emit_lifecycle("worker_started", "precache_ltx23")
        except Exception:
            _telemetry = None

        preprocess_ltx23()

        if _telemetry is not None:
            _telemetry.emit_lifecycle("worker_finished", "precache_ltx23", duration_seconds=time.time() - t_start)
    except Exception as err:
        traceback.print_exc()
        if _telemetry is not None:
            _telemetry.emit_lifecycle("worker_failed", "precache_ltx23", error=str(err))
        sys.exit(1)
