"""Standalone recaption worker — runs under the training runtime's virtualenv, not this project's.

Loading Qwen3-VL needs torch/transformers/accelerate (~6 GB of dependencies) and
~9 GB of VRAM, so instead of pulling that into the hub this script is launched as
a subprocess against training_runtime/venv's interpreter (the same one training
and pre-cache use). It loads the model once, captions the images it is given, and
exits — which is what releases the VRAM.

The captioning itself is not reimplemented: `caption_qwen3vl` is vendored
alongside this script (see that file's docstring) so both projects stay on
exactly the same model, prompts and decoding parameters.

Protocol: a JSON job on stdin, JSON Lines events on stdout. Every event carries
`timestamp` (absolute, unix seconds) and `run_id` (echoed back from the job,
empty if the caller did not set one) so events from concurrent or successive
runs can be told apart downstream.

    job    {"text_encoder_dir": str, "images": [str], "detailed": bool, "run_id": str}
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
    detailed = bool(job.get("detailed", False))
    _run_id = str(job.get("run_id", ""))

    import torch
    from caption_qwen3vl import generate_caption, load_captioner

    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    _emit(event="loaded", device=device, seconds=round(time.time() - started, 1))

    captioned = failed = 0
    for path in images:
        image_started = time.time()
        try:
            caption = generate_caption(model, processor, path, detailed=detailed)
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
