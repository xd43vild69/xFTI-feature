# -*- coding: utf-8 -*-
"""
precache_worker.py — Pre-caché de latentes + embeddings para Krea 2 (RAW)
Pre-cache of latents + embeddings for Krea 2 (RAW)

Ported verbatim from AcademiaSD_LoRAlab-Krea2's 1_pre_cache_krea2.py, run here
against feature_pipeline_hub's own training_runtime/ instead of that checkout.
The only change from the original is PROJECT_ROOT's directory depth below —
this file lives one level shallower (workers/ vs scripts/python/) than the
original. Everything else, including the bilingual log messages, is untouched
so the two stay easy to diff against each other as LoRAlab evolves.

Lee configuración desde pre_cache_settings.json si existe.
Reads configuration from pre_cache_settings.json if present.
"""
import os
import gc
import math
import json
import re
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DiffusionPipeline
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Raíz del proyecto (este worker vive en workers/, un nivel más superficial que
# scripts/python/ en LoRAlab — de ahí que aquí sean 2 dirname(), no 3). Todas
# las rutas se anclan aquí en vez de al directorio de trabajo, para que
# funcione invocado desde cualquier sitio.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def from_root(path):
    """Resuelve una ruta relativa contra la raíz del proyecto (absolutas intactas)."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(PROJECT_ROOT, path))


# Carpeta contenedora de todo lo que genera el pre-caché (una sola en la raíz).
CACHE_ROOT = os.path.join(PROJECT_ROOT, "cached_data_local")

# ── DEFAULTS / VALORES POR DEFECTO ──────────────────────────────────────────
DEFAULTS = {
    "model_id": "Krea-2-NF4",
    "dataset_path": "./dataset",
    "cache_dir": f"{CACHE_ROOT}/default",
    "target_area": 1024 * 1024,
    "max_side": 1280,
    "multiple": 16,
    "max_seq_len": 256,
    "project_name": "",
    "trigger_word": "",
    "resolutions": [],

    # ── D1: manifiesto e invalidación de caché ───────────────────────────────
    "cache_key": "mtime",        # "mtime" | "sha1"
    "prune_orphans": "delete",   # "warn" | "delete" | "off"
    "force_recache": False,
    # ── D2: aumento por espejado ─────────────────────────────────────────────
    "flip_x": False,
    # ── C5: prompts de preview ───────────────────────────────────────────────
    "sample_prompts": [],
}

# Se incrementa cuando cambia la normalización o el empaquetado de latentes, para
# invalidar automáticamente todas las cachés existentes.
LATENT_VERSION = 1

# ── CARGAR CONFIGURACIÓN / LOAD CONFIG ──────────────────────────────────────
CONFIG_PATH = os.environ.get("PRECACHE_SETTINGS_PATH",
                             os.path.join(PROJECT_ROOT, "pre_cache_settings.json"))

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"✓ Configuration loaded from {CONFIG_PATH} / Configuración cargada desde {CONFIG_PATH}")
else:
    cfg = {}
    print(f"⚠ {CONFIG_PATH} not found, using defaults / No se encontró {CONFIG_PATH}, usando valores por defecto.")

MODEL_ID     = cfg.get("model_id",     DEFAULTS["model_id"])
# El modelo local vive en la raíz del proyecto. Se ancla sólo si la carpeta
# existe ahí, para no romper el caso de un repo-id de Hugging Face; sin esto,
# ejecutar desde otro directorio dispararía una descarga completa del modelo.
if not os.path.isabs(MODEL_ID) and os.path.isdir(from_root(MODEL_ID)):
    MODEL_ID = from_root(MODEL_ID)
DATASET_PATH = from_root(cfg.get("dataset_path", DEFAULTS["dataset_path"]))
TARGET_AREA  = cfg.get("target_area",  DEFAULTS["target_area"])
MAX_SIDE     = cfg.get("max_side",     DEFAULTS["max_side"])
MULTIPLE     = cfg.get("multiple",     DEFAULTS["multiple"])
MAX_SEQ_LEN  = cfg.get("max_seq_len",  DEFAULTS["max_seq_len"])
TRIGGER_WORD = cfg.get("trigger_word", "")
PROJECT_NAME = cfg.get("project_name", "").strip()

# Resolución progresiva: si 'resolutions' trae una lista de áreas, se pre-cachea
# el dataset a cada una en un subdirectorio {label}/. Si está vacía, se usa la
# resolución única de 'target_area' (comportamiento clásico, retrocompatible).
RESOLUTIONS = cfg.get("resolutions", DEFAULTS["resolutions"]) or []

CACHE_KEY     = str(cfg.get("cache_key", DEFAULTS["cache_key"])).strip().lower()
PRUNE_ORPHANS = str(cfg.get("prune_orphans", DEFAULTS["prune_orphans"])).strip().lower()
FORCE_RECACHE = bool(cfg.get("force_recache", DEFAULTS["force_recache"]))
FLIP_X        = bool(cfg.get("flip_x", DEFAULTS["flip_x"]))
SAMPLE_PROMPTS = list(cfg.get("sample_prompts", DEFAULTS["sample_prompts"]) or [])

if CACHE_KEY not in ("mtime", "sha1"):
    print(f"⚠ Invalid cache_key '{CACHE_KEY}'. Using 'mtime' / usando 'mtime'.")
    CACHE_KEY = "mtime"
if PRUNE_ORPHANS not in ("warn", "delete", "off"):
    print(f"⚠ Invalid prune_orphans '{PRUNE_ORPHANS}'. Using 'warn' / usando 'warn'.")
    PRUNE_ORPHANS = "warn"


def apply_trigger_word(text):
    """Antepone TRIGGER_WORD salvo que ya esté presente como término suelto.

    Duplica a propósito `caption_service.inject_trigger_word` (fuente de verdad):
    este worker corre en su propio intérprete y no puede importar el paquete
    `feature_pipeline`. El criterio de "ya presente" debe seguir siendo el mismo:
    límites de palabra e insensible a mayúsculas, de modo que `sks_style` no se dé
    por presente dentro de `sks_stylevariant`.
    """
    if not TRIGGER_WORD:
        return text
    pattern = re.compile(r"(?<!\w)" + re.escape(TRIGGER_WORD) + r"(?!\w)", flags=re.IGNORECASE)
    if pattern.search(text):
        return text
    return f"{TRIGGER_WORD}, {text}".strip(", ")


def res_label(area):
    """Etiqueta corta del subdir a partir del área (ej. 589824 -> '768')."""
    return str(int(round(math.sqrt(area))))

# Formato automático de carpeta de caché según nombre del proyecto.
# Todo lo generado se agrupa bajo CACHE_ROOT para no ensuciar la raíz.
# Sin project_name se respeta un cache_dir explícito (lo usa run_progressive.py
# para apuntar al subdirectorio de resolución de cada fase).
if PROJECT_NAME:
    CACHE_DIR = os.path.join(CACHE_ROOT, PROJECT_NAME)
else:
    CACHE_DIR = from_root(cfg.get("cache_dir", DEFAULTS["cache_dir"]))

# Validar que Múltiplo sea únicamente 16, 32 o 64 (Default: 16).
# 8 no vale: pack_latents() del entrenador divide las dimensiones latentes (px/8)
# entre 2, así que los píxeles deben ser múltiplo de 16 o el empaquetado se
# corrompe en silencio.
if MULTIPLE not in (16, 32, 64):
    print(f"⚠ Invalid Multiple {MULTIPLE}. Defaulting to 16 / Múltiplo inválido {MULTIPLE}. Usando 16 por defecto.")
    MULTIPLE = 16

print(f"  Model ID / ID Modelo        : {MODEL_ID}")
print(f"  Project Name / Proyecto     : {PROJECT_NAME if PROJECT_NAME else '(Default)'}")
print(f"  Trigger Word / Palabra      : {TRIGGER_WORD}")
print(f"  Dataset Path / Ruta Dataset : {DATASET_PATH}")
print(f"  Cache Dir / Carpeta Caché   : {CACHE_DIR}")
if RESOLUTIONS:
    print(f"  Resolutions / Progresiva    : {[res_label(a)+'²' for a in RESOLUTIONS]}")
else:
    print(f"  Target Area / Área Objetivo : {TARGET_AREA} px²")
print(f"  Max Side / Lado Máximo      : {MAX_SIDE}")
print(f"  Multiple / Múltiplo         : {MULTIPLE}")
print(f"  Max Seq Len / Long. Sec.    : {MAX_SEQ_LEN}")
print(f"  Cache key / Invalidación    : {CACHE_KEY} (prune_orphans={PRUNE_ORPHANS}"
      f"{', force_recache' if FORCE_RECACHE else ''})")
print(f"  Flip augmentation / Espejado: {'ON (duplica el dataset)' if FLIP_X else 'OFF'}")
if SAMPLE_PROMPTS:
    print(f"  Sample prompts / Previews   : {len(SAMPLE_PROMPTS)}")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def free_vram(*tensors):
    """Libera VRAM. Sólo para transiciones de fase, nunca por imagen.

    Ojo con la firma: `del t` borra el nombre local del bucle, no la referencia
    del llamante, así que pasar tensores aquí no los libera —lo único que hace
    es el gc + empty_cache. Se llamaba 12 veces por imagen (~18 ms cada una)
    creyendo que servía para algo, y encima empty_cache destruye el caching
    allocator y obliga a cudaMalloc de nuevo en la iteración siguiente.
    """
    for t in tensors:
        del t
    gc.collect()
    torch.cuda.empty_cache()


def save_tensor(tensor, path):
    """torch.save borrando antes el destino.

    El borrado importa desde que se usan hardlinks: sin él, `torch.save` abriría
    en modo 'wb' el inodo compartido y una escritura a medias corrompería todas
    las resoluciones a la vez, en vez de sólo ésta.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    torch.save(tensor, path)


def link_or_copy(source, dest, tensor):
    """Enlaza `dest` a `source` (mismo contenido) o, si no se puede, lo escribe.

    El fallback cubre Windows sin privilegios, FAT y cachés repartidas entre
    discos distintos; en esos casos se vuelve al comportamiento de siempre.
    """
    try:
        os.remove(dest)
    except FileNotFoundError:
        pass
    try:
        os.link(source, dest)
    except OSError:
        torch.save(tensor, dest)


def bucket_size(w: int, h: int, area=None):
    """(ancho, alto) múltiplos de MULTIPLE, área ≈ `area`, conservando el aspecto."""
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


# ── D1: MANIFIESTO DE CACHÉ / CACHE MANIFEST ────────────────────────────────

def manifest_config():
    """Ajustes que, si cambian, invalidan todos los latentes ya cacheados."""
    return {
        "target_area": TARGET_AREA,
        "multiple": MULTIPLE,
        "max_side": MAX_SIDE,
        "max_seq_len": MAX_SEQ_LEN,
        "trigger_word": TRIGGER_WORD,
        "flip_x": FLIP_X,
        "latent_version": LATENT_VERSION,
    }


def load_manifest(out_dir):
    path = os.path.join(out_dir, "cache_manifest.json")
    if not os.path.exists(path):
        return {"version": 1, "config": {}, "entries": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("config") != manifest_config():
            print(f"   [i] Cache config changed for {os.path.basename(out_dir)}; "
                  f"re-encoding all / la configuración cambió; se reencoda todo.")
            return {"version": 1, "config": {}, "entries": {}}
        return data
    except Exception:
        return {"version": 1, "config": {}, "entries": {}}


def save_manifest(out_dir, entries):
    data = {"version": 1, "config": manifest_config(), "entries": entries}
    path = os.path.join(out_dir, "cache_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def file_fingerprint(path):
    """Huella de la imagen fuente: mtime+size (rápido) o sha1 (robusto ante copias)."""
    stat = os.stat(path)
    if CACHE_KEY == "sha1":
        digest = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return {"sha1": digest.hexdigest(), "src_size": stat.st_size}
    return {"src_mtime": int(stat.st_mtime), "src_size": stat.st_size}


def caption_fingerprint(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def prune_orphans(out_dir, valid_names):
    """Latentes de imágenes borradas o renombradas.

    Sin esto se quedaban en la caché para siempre y el entrenador los seguía
    entrenando en silencio; con el sampler por épocas además tendrían un turno
    garantizado cada época.
    """
    present = {f[:-len("_latent.pt")] for f in os.listdir(out_dir)
               if f.endswith("_latent.pt") and not f.startswith(".")}
    orphans = sorted(present - set(valid_names))
    if not orphans:
        return
    label = os.path.basename(out_dir)
    if PRUNE_ORPHANS == "delete":
        for name in orphans:
            for suffix in ("_latent.pt", "_embed.pt", "_mask.pt"):
                try:
                    os.remove(os.path.join(out_dir, f"{name}{suffix}"))
                except OSError:
                    pass
        print(f"   [OK] Pruned {len(orphans)} orphan entries from {label} / eliminados.")
    elif PRUNE_ORPHANS == "warn":
        print(f"\n[!] {len(orphans)} orphan cache entries in {label} "
              f"(source image deleted or renamed) / huérfanos:")
        for name in orphans[:10]:
            print(f"      {name}")
        if len(orphans) > 10:
            print(f"      ... and {len(orphans) - 10} more")
        print("[!] They will still be trained on. Set prune_orphans='delete' to remove them "
              "/ se seguirán entrenando.")


def ensure_model_downloaded(local_path, repo_id):
    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_content = any(
            os.path.exists(os.path.join(local_path, f))
            for f in ["index.json", "model_index.json", "config.json"]
        ) or len(os.listdir(local_path)) > 0
        if has_content:
            print(f"✓ Local model found at / Modelo local encontrado en: {local_path}")
            return local_path

    print(f"⚠ Local model not found at / No se encontró modelo local en: {local_path}")
    print(f"  Downloading from Hugging Face / Descargando desde Hugging Face: {repo_id}")
    print(f"  This may take several minutes / Esto puede tardar varios minutos...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with / Se requiere 'huggingface_hub'. Instálalo con:\n"
            "  pip install huggingface_hub"
        )

    hf_token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
    dl_kwargs = {
        "repo_id": repo_id,
        "local_dir": local_path,
    }
    if hf_token:
        dl_kwargs["token"] = hf_token

    downloaded_path = snapshot_download(**dl_kwargs)
    print(f"✓ Model downloaded to / Modelo descargado en: {downloaded_path}")
    return downloaded_path


def preprocess_krea2():
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Dataset folder does not exist / La carpeta del dataset no existe: {DATASET_PATH}")
        return

    # Auto-limpieza de archivos fantasma ocultos de macOS (._* y .DS_Store)
    for f in os.listdir(DATASET_PATH):
        if f.startswith("._") or f == ".DS_Store":
            try:
                os.remove(os.path.join(DATASET_PATH, f))
            except Exception:
                pass

    archivos_img = sorted(f for f in os.listdir(DATASET_PATH)
                          if not f.startswith(".") and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    if not archivos_img:
        print(f"[!] No images found in '{DATASET_PATH}'. Please add images and optional .txt files. / No hay imágenes en '{DATASET_PATH}'. Añade imágenes y sus .txt opcionales.")
        return

    ensure_model_downloaded(
        local_path=MODEL_ID,
        repo_id="AcademiaSD/Krea-2-NF4-for-LoRA-Training"
    )

    print("Loading Text Encoder (Qwen3-VL-4B) & VAE (Qwen-Image) on CPU... / Cargando en CPU...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        transformer=None,
        torch_dtype=torch.bfloat16,
    )

    if not hasattr(pipe.vae.config, "latents_mean") or not hasattr(pipe.vae.config, "latents_std"):
        raise RuntimeError("This VAE lacks latents_mean/latents_std configuration / Este VAE no tiene latents_mean/latents_std")

    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    latents_std  = torch.tensor(pipe.vae.config.latents_std,  device="cuda", dtype=torch.float32).view(1, z_dim, 1, 1, 1)
    print(f"VAE z_dim={z_dim} | Channel normalization OK / Normalización por canal OK")

    # Objetivos de resolución: cada uno es una caché autocontenida. En modo
    # progresivo van a subdirs {label}/; en modo clásico, a CACHE_DIR directamente.
    if RESOLUTIONS:
        targets = [(os.path.join(CACHE_DIR, res_label(a)), a) for a in RESOLUTIONS]
    else:
        targets = [(CACHE_DIR, TARGET_AREA)]
    for out_dir, _ in targets:
        os.makedirs(out_dir, exist_ok=True)

    def prepare_image_tensor_cpu(img_pil, area):
        """Resize/crop on CPU and return `(bw, bh, tensor_bf16_cpu)` ready for batched VAE encode."""
        bw, bh = bucket_size(img_pil.width, img_pil.height, area)
        ratio = max(bw / img_pil.width, bh / img_pil.height)
        im = img_pil.resize(
            (math.ceil(img_pil.width * ratio), math.ceil(img_pil.height * ratio)),
            Image.LANCZOS,
        )
        left = (im.width - bw) // 2
        top = (im.height - bh) // 2
        im = im.crop((left, top, left + bw, top + bh))

        img_tensor = F_vision.pil_to_tensor(im).unsqueeze(1)  # [3,1,H,W]
        img_tensor = ((img_tensor.float() / 127.5) - 1.0).to(torch.bfloat16).contiguous()
        if torch.cuda.is_available():
            try:
                img_tensor = img_tensor.pin_memory()
            except Exception:
                pass
        return bw, bh, img_tensor

    def _safe_tiled_encode_single(inp: torch.Tensor) -> torch.Tensor:
        """Fallback OOM path: pad spatial dims so diffusers' tiled_encode never slices a <256px remainder tile."""
        _, _, _, h_px, w_px = inp.shape
        t_min_h = int(getattr(pipe.vae, "tile_sample_min_height", 256))
        t_min_w = int(getattr(pipe.vae, "tile_sample_min_width", 256))
        t_str_h = int(getattr(pipe.vae, "tile_sample_stride_height", 192))
        t_str_w = int(getattr(pipe.vae, "tile_sample_stride_width", 192))

        def _padded_dim(size: int, t_min: int, t_str: int) -> int:
            if size <= t_min:
                return t_min
            steps = math.ceil((size - t_min) / t_str)
            return t_min + steps * t_str

        pad_h = _padded_dim(h_px, t_min_h, t_str_h) - h_px
        pad_w = _padded_dim(w_px, t_min_w, t_str_w) - w_px
        if pad_h > 0 or pad_w > 0:
            inp_2d = torch.nn.functional.pad(inp[:, :, 0], (0, pad_w, 0, pad_h), mode="replicate")
            inp = inp_2d.unsqueeze(2)

        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        try:
            z = pipe.vae.encode(inp).latent_dist.mode().float()
        finally:
            if hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()
        return z[:, :, :, : h_px // 8, : w_px // 8]

    def encode_vae_batch(batch_tensors_cpu):
        """Encode a same-resolution batch `[B,3,1,H,W]` on CUDA without tiling seams/bugs, falling back safely on OOM."""
        if len(batch_tensors_cpu) > 1:
            try:
                stacked = torch.stack(batch_tensors_cpu, dim=0).to("cuda", dtype=torch.bfloat16, non_blocking=True)
                z = pipe.vae.encode(stacked).latent_dist.mode().float()
                lat = ((z - latents_mean) / latents_std)[:, :, 0].to(torch.bfloat16).cpu()
                return [lat[i : i + 1].contiguous() for i in range(lat.shape[0])]
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()

        results = []
        for single_cpu in batch_tensors_cpu:
            inp = single_cpu.unsqueeze(0).to("cuda", dtype=torch.bfloat16, non_blocking=True)
            try:
                z = pipe.vae.encode(inp).latent_dist.mode().float()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                z = _safe_tiled_encode_single(inp)
            lat = ((z - latents_mean) / latents_std)[:, :, 0].to(torch.bfloat16).cpu()
            results.append(lat)
        return results

    def encode_prompt_batch(prompts_list):
        """Encode a list of prompts in one batched call, falling back per-item if needed."""
        if len(prompts_list) == 1:
            e, m = pipe.encode_prompt(prompt=prompts_list[0], max_sequence_length=MAX_SEQ_LEN)
            return [(e.cpu(), m.cpu())]
        try:
            e_b, m_b = pipe.encode_prompt(prompt=prompts_list, max_sequence_length=MAX_SEQ_LEN)
            return [(e_b[i:i + 1].cpu(), m_b[i:i + 1].cpu()) for i in range(len(prompts_list))]
        except Exception:
            out = []
            for p in prompts_list:
                e, m = pipe.encode_prompt(prompt=p, max_sequence_length=MAX_SEQ_LEN)
                out.append((e.cpu(), m.cpu()))
            return out

    manifests = {out_dir: ({} if FORCE_RECACHE else load_manifest(out_dir))
                 for out_dir, _ in targets}
    reused = encoded = 0

    # Scan dataset once to separate cached entries from pending ones before touching GPU.
    pending_items = []
    for idx, archivo in enumerate(archivos_img, 1):
        nombre_base = os.path.splitext(archivo)[0]
        ruta_texto  = os.path.join(DATASET_PATH, f"{nombre_base}.txt")
        ruta_img    = os.path.join(DATASET_PATH, archivo)

        prompt = ""
        if os.path.exists(ruta_texto):
            with open(ruta_texto, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        prompt = apply_trigger_word(prompt)

        entry = dict(file_fingerprint(ruta_img))
        entry["caption_sha1"] = caption_fingerprint(prompt)

        def is_cached(out_dir):
            previous = manifests[out_dir].get("entries", {}).get(nombre_base)
            if previous is None:
                return False
            if {k: previous.get(k) for k in entry} != entry:
                return False
            names = [nombre_base] + ([f"{nombre_base}__flip"] if FLIP_X else [])
            return all(os.path.exists(os.path.join(out_dir, f"{n}{suffix}"))
                       for n in names for suffix in ("_latent.pt", "_embed.pt", "_mask.pt"))

        pending = [(out_dir, area) for out_dir, area in targets if not is_cached(out_dir)]
        if not pending:
            reused += 1
            print(f"[{idx}/{len(archivos_img)}] Up to date / Sin cambios: {archivo}")
            for out_dir, _ in targets:
                manifests[out_dir].setdefault("entries", {})[nombre_base] = entry
            continue
        pending_items.append((idx, archivo, nombre_base, ruta_img, prompt, entry, pending))

    with torch.inference_mode():
        # ── Phase 1: Text Encoder on CUDA (VAE stays on CPU to fit comfortably in 12 GB VRAM) ──
        print("\n--- Phase 1/2: Encoding prompts (Qwen3-VL-4B on CUDA) ---")
        pipe.text_encoder.to("cuda")

        neg_embed, neg_mask = pipe.encode_prompt(prompt="", max_sequence_length=MAX_SEQ_LEN)
        for out_dir, _ in targets:
            torch.save(neg_embed.cpu(), os.path.join(out_dir, "_neg_embed.pt"))
            torch.save(neg_mask.cpu(),  os.path.join(out_dir, "_neg_mask.pt"))
        del neg_embed, neg_mask

        for out_dir, _ in targets:
            for old in os.listdir(out_dir):
                if old.startswith("_sample") and old.endswith((".pt", ".json")):
                    os.remove(os.path.join(out_dir, old))
        for i, item in enumerate(SAMPLE_PROMPTS):
            spec = {"prompt": item} if isinstance(item, str) else dict(item)
            text = str(spec.get("prompt", "")).strip()
            if not text:
                continue
            text = apply_trigger_word(text)
            s_emb, s_msk = pipe.encode_prompt(prompt=text, max_sequence_length=MAX_SEQ_LEN)
            spec["prompt"] = text
            for out_dir, _ in targets:
                torch.save(s_emb.cpu(), os.path.join(out_dir, f"_sample{i}_embed.pt"))
                torch.save(s_msk.cpu(), os.path.join(out_dir, f"_sample{i}_mask.pt"))
                with open(os.path.join(out_dir, f"_sample{i}_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(spec, f, indent=2, ensure_ascii=False)
            del s_emb, s_msk
            print(f"   [OK] Sample prompt {i} cached / cacheado: {text[:60]}")

        encoded_texts = {}
        TEXT_BATCH_SIZE = 16
        for start in range(0, len(pending_items), TEXT_BATCH_SIZE):
            batch = pending_items[start:start + TEXT_BATCH_SIZE]
            results = encode_prompt_batch([item[4] for item in batch])
            for item, pair in zip(batch, results):
                encoded_texts[item[2]] = pair

        # Release Text Encoder VRAM before moving VAE to CUDA
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
        free_vram()

        # ── Phase 2: Multithreaded CPU PIL/Lanczos prep (64 GB RAM) + Batched VAE on CUDA ──
        # Tiling stays OFF by default here: Qwen3-VL has already been freed so all 12 GB VRAM
        # are available (~1.5 GB needed for full 1024² encode), avoiding both tile-blending
        # seams and diffusers' AutoencoderKLQwenImage.tiled_encode remainder-tile crash on
        # dimensions like 784 or 1168 (192*k + 16 -> 16px sliver -> 2px < 3x3 conv kernel).
        print("\n--- Phase 2/2: Encoding image latents (Parallel RAM Decode + Batched VAE on CUDA) ---")
        pipe.vae.to("cuda")
        if hasattr(pipe.vae, "disable_tiling"):
            pipe.vae.disable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

        def _decode_item_cpu(item_tuple):
            idx, archivo, nombre_base, ruta_img, _prompt, entry, pending = item_tuple
            try:
                with Image.open(ruta_img) as raw_im:
                    img = raw_im.convert("RGB")
            except Exception as err:
                return (idx, archivo, nombre_base, entry, None, err)

            variants = [("", img)]
            if FLIP_X:
                variants.append(("__flip", img.transpose(Image.FLIP_LEFT_RIGHT)))

            prepared = []
            sizes_seen = []
            for out_dir, area in pending:
                bw_last = bh_last = 0
                for suffix, variant in variants:
                    bw_last, bh_last, tensor_cpu = prepare_image_tensor_cpu(variant, area)
                    prepared.append((out_dir, suffix, bw_last, bh_last, tensor_cpu))
                sizes_seen.append(f"{bw_last}x{bh_last}")
            return (idx, archivo, nombre_base, entry, (prepared, sizes_seen), None)

        cpu_workers = min(8, max(1, os.cpu_count() or 4))
        io_workers = min(4, max(1, (os.cpu_count() or 4) // 2))
        PREFETCH_WINDOW = 16

        with ThreadPoolExecutor(max_workers=cpu_workers) as decode_pool, \
             ThreadPoolExecutor(max_workers=io_workers) as io_pool:
            io_futures = []
            for win_start in range(0, len(pending_items), PREFETCH_WINDOW):
                window = pending_items[win_start : win_start + PREFETCH_WINDOW]
                decoded_window = list(decode_pool.map(_decode_item_cpu, window))

                # Group prepared tensors by (bw, bh) across the window for batched GPU VAE execution
                by_size = defaultdict(list)
                valid_records = []
                for idx, archivo, nombre_base, entry, payload, err in decoded_window:
                    if err is not None or payload is None:
                        print(f"[!] Warning: Cannot open image / No se pudo abrir la imagen '{archivo}': {err}")
                        encoded_texts.pop(nombre_base, None)
                        continue
                    prepared, sizes_seen = payload
                    embeds_cpu, mask_cpu = encoded_texts.pop(nombre_base)
                    rec_idx = len(valid_records)
                    valid_records.append((idx, archivo, nombre_base, entry, sizes_seen, embeds_cpu, mask_cpu, []))
                    for out_dir, suffix, bw, bh, tensor_cpu in prepared:
                        by_size[(bw, bh)].append((rec_idx, out_dir, suffix, tensor_cpu))

                for (bw, bh), bucket_tasks in by_size.items():
                    vae_bs = 4 if (bw * bh) <= (768 * 768) else 2
                    for b_start in range(0, len(bucket_tasks), vae_bs):
                        sub = bucket_tasks[b_start : b_start + vae_bs]
                        latents_out = encode_vae_batch([t[3] for t in sub])
                        for (rec_idx, out_dir, suffix, _t_cpu), lat_cpu in zip(sub, latents_out):
                            valid_records[rec_idx][7].append((out_dir, suffix, lat_cpu))

                def _write_record_files(nombre_base_w, embeds_w, mask_w, encoded_list):
                    first_written = {}
                    for out_dir_w, suffix_w, latent_w in encoded_list:
                        name_w = f"{nombre_base_w}{suffix_w}"
                        save_tensor(latent_w, os.path.join(out_dir_w, f"{name_w}_latent.pt"))
                        for kind, tensor in (("embed", embeds_w), ("mask", mask_w)):
                            dest = os.path.join(out_dir_w, f"{name_w}_{kind}.pt")
                            source = first_written.get((suffix_w, kind))
                            if source is None:
                                save_tensor(tensor, dest)
                                first_written[(suffix_w, kind)] = dest
                            else:
                                link_or_copy(source, dest, tensor)

                for idx, archivo, nombre_base, entry, sizes_seen, embeds_cpu, mask_cpu, enc_list in valid_records:
                    print(f"[{idx}/{len(archivos_img)}] Processing / Procesando: {archivo}")
                    io_futures.append(
                        io_pool.submit(_write_record_files, nombre_base, embeds_cpu, mask_cpu, enc_list)
                    )
                    for out_dir, _ in targets:
                        manifests[out_dir].setdefault("entries", {})[nombre_base] = entry
                    encoded += 1
                    flip_note = " (+flip)" if FLIP_X else ""
                    print(f"   [OK] {' · '.join(sizes_seen)}{flip_note} | Latents & Embeddings cached "
                          f"/ Latentes y embeddings cacheados.")

            for fut in io_futures:
                fut.result()

    # El manifiesto se reconstruye a partir de las imágenes realmente presentes:
    # arrastrar las entradas antiguas haría que una imagen borrada o renombrada
    # siguiera contando como válida y su latente nunca se reportase como huérfano.
    present_names = {os.path.splitext(f)[0] for f in archivos_img}
    for out_dir, _ in targets:
        entries = {name: data
                   for name, data in manifests[out_dir].get("entries", {}).items()
                   if name in present_names}
        save_manifest(out_dir, entries)
        valid = list(entries) + ([f"{n}__flip" for n in entries] if FLIP_X else [])
        prune_orphans(out_dir, valid)

    free_vram(pipe)
    print(f"\n✓ Pre-caching finished! {encoded} encoded, {reused} reused, VRAM freed "
          f"/ ¡Pre-caché finalizado! {encoded} codificadas, {reused} reutilizadas.")


if __name__ == "__main__":
    import time as _time

    from _telemetry import emit_lifecycle

    _started = _time.time()
    emit_lifecycle("worker_started", "precache")
    try:
        preprocess_krea2()
    except Exception as _exc:
        emit_lifecycle(
            "worker_failed",
            "precache",
            duration_seconds=round(_time.time() - _started, 1),
            gpu_seconds=round(_time.time() - _started, 1),
            error=f"{type(_exc).__name__}: {_exc}",
        )
        raise
    emit_lifecycle(
        "worker_finished",
        "precache",
        duration_seconds=round(_time.time() - _started, 1),
        gpu_seconds=round(_time.time() - _started, 1),
    )