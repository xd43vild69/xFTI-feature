"""Standalone recaption worker — runs under the training runtime's virtualenv, not this project's.

Loading Qwen3-VL needs torch/transformers/accelerate (~6 GB of dependencies) and
~9 GB of VRAM, so instead of pulling that into the hub this script is launched as
a subprocess against training_runtime/venv's interpreter (the same one training
and pre-cache use). It loads the model once, captions the images it is given, and
exits — which is what releases the VRAM.

The captioning itself is not reimplemented: `caption_qwen3vl` is vendored
alongside this script (see that file's docstring) so both projects stay on
exactly the same model and loading technique.

This worker is a relay, deliberately: the hub sends the prompt down and gets the
model's raw reply back untouched. The prompt is a request for a JSON object of
fixed slots, and reading that object back into a caption happens in the hub
(domain/caption_schema.py), not here — nothing under workers/ runs in CI, so the
part that can silently misparse belongs on the other side of this boundary.

Protocol: a JSON job on stdin, JSON Lines events on stdout. Every event carries
`timestamp` (absolute, unix seconds) and `run_id` (echoed back from the job,
empty if the caller did not set one) so events from concurrent or successive
runs can be told apart downstream.

    job    {"text_encoder_dir": str, "images": [str], "instruction": str, "run_id": str}
    events {"event": "loaded",  "device": str, "seconds": float, "timestamp": float, "run_id": str}
           {"event": "caption", "path": str, "caption": str, "seconds": float, "timestamp": float, "run_id": str}
           {"event": "error",   "path": str, "message": str, "timestamp": float, "run_id": str}
           {"event": "done",    "captioned": int, "failed": int, "timestamp": float, "run_id": str}
"""

import json
import sys
import time

_run_id = ""


def _emit(**event) -> None:
    print(json.dumps({"timestamp": time.time(), "run_id": _run_id, **event}), flush=True)


def main() -> int:
    global _run_id
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    job = json.load(sys.stdin)
    text_encoder_dir = job["text_encoder_dir"]
    images = job["images"]
    instruction = job.get("instruction") or None
    _run_id = str(job.get("run_id", ""))

    import torch
    from caption_qwen3vl import generate_caption, generate_captions_batch, load_captioner

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Free VRAM if ComfyUI or similar service is holding memory on the same GPU
    try:
        import urllib.request
        _req = urllib.request.Request(
            "http://127.0.0.1:8188/free",
            data=b'{"unload_models": true, "free_memory": true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(_req, timeout=0.5):
            pass
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model, processor = load_captioner(text_encoder_dir, device=device, dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        # The GPU may be shared with ComfyUI or a training run. Falling back to CPU
        # is minutes per image rather than seconds, but it finishes instead of
        # failing the whole batch.
        if device == "cpu":
            raise
        torch.cuda.empty_cache()
        device = "cpu"
        model, processor = load_captioner(text_encoder_dir, device=device, dtype=torch.bfloat16)

    # A weight that arrives non-finite makes every logit NaN, and greedy decoding
    # turns NaN logits into token 0 over and over — the model "replies" with a wall
    # of "!!!!" and the only visible symptom is a caption that will not parse. The
    # bytes on disk are usually fine (checksum them before believing otherwise); a
    # bad read is what puts NaN in RAM. Cheap enough to check on every load, and it
    # turns a baffling symptom into a stated cause.
    try:
        with torch.no_grad():
            params = list(model.named_parameters())
            # Vectorize finite check into a single GPU->CPU sync instead of ~400 .item() stalls
            all_finite = (
                torch.stack([torch.isfinite(p).all() for _, p in params]).all().item()
                if params
                else True
            )
            if not all_finite:
                corrupt = [name for name, p in params if not torch.isfinite(p).all().item()]
                raise RuntimeError(
                    f"Model weights loaded with non-finite values in {len(corrupt)} tensor(s), "
                    f"first: {corrupt[0]}. Generation would return garbage. Retry — this is "
                    "usually a bad read rather than a bad file. If it keeps happening, verify "
                    "the checkpoint against its source and test system RAM (memtest86+)."
                )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()

    _emit(event="loaded", device=device, seconds=round(time.time() - started, 1))

    captioned = failed = 0
    batch_size = int(job.get("batch_size", 4 if device == "cuda" else 1))
    batch_size = max(1, batch_size)

    for b_start in range(0, len(images), batch_size):
        chunk = images[b_start : b_start + batch_size]
        if len(chunk) > 1:
            chunk_started = time.time()
            try:
                replies = generate_captions_batch(
                    model, processor, chunk, instruction=instruction, deterministic=True
                )
                per_img_sec = round((time.time() - chunk_started) / len(chunk), 2)
                for path, caption in zip(chunk, replies):
                    captioned += 1
                    _emit(event="caption", path=path, caption=caption, seconds=per_img_sec)
                continue
            except Exception:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for path in chunk:
            image_started = time.time()
            try:
                caption = generate_caption(
                    model, processor, path, instruction=instruction, deterministic=True
                )
            except Exception as exc:  # one bad image must not abort the batch
                failed += 1
                _emit(event="error", path=path, message=f"{type(exc).__name__}: {exc}")
                continue

            captioned += 1
            _emit(
                event="caption",
                path=path,
                caption=caption,
                seconds=round(time.time() - image_started, 1),
            )

    _emit(event="done", captioned=captioned, failed=failed)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
