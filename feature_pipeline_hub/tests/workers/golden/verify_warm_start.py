"""Checks krea2.state's warm-start path against the hub that stages it.

The two halves of a warm start live in different interpreters — the hub writes
`warm_start.json` and the rewritten adapter with no torch installed, the trainer reads
them with torch and PEFT — so nothing in `uv run pytest` can cover the seam. This runs
under the training runtime's interpreter and does, which is why it sits beside the
golden captures rather than under `tests/`:

    training_runtime/venv/bin/python tests/workers/golden/verify_warm_start.py

Needs no GPU. Reuses capture_behavior.py's fake model and optimizer so the objects are
the same shape the golden captures exercise. Exits non-zero on any failure.
"""
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # feature_pipeline_hub/
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from capture_behavior import _FakeModel, _FakeOptimizer  # noqa: E402
from krea2 import config as kconfig  # noqa: E402
from krea2 import state as kstate  # noqa: E402

from feature_pipeline.infrastructure.checkpoint_files import (  # noqa: E402
    materialize_warm_start,
)

FAILURES = []

# _FakeModel is not PEFT-wrapped, so the real set_peft_model_state_dict would raise and
# the abort path would swallow every case below. capture_behavior.py stubs it for the
# same reason; sections that care about the bytes install their own spy.
_REAL_LOAD = kstate.lora_io.load_lora_weights
kstate.lora_io.load_lora_weights = lambda model, blob, src, log=print: 1.0


def check(name, actual, expected):
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {actual!r}" + ("" if ok else f" != {expected!r}"))
    if not ok:
        FAILURES.append(name)


def make_cfg(root, **settings):
    path = pathlib.Path(root) / "settings.json"
    path.write_text(json.dumps(settings))
    return kconfig.load_config(
        root, settings_path=str(path),
        advanced_path=str(pathlib.Path(root) / "none.json"),
        cache_root=os.path.join(root, "c"), output_root=os.path.join(root, "o"),
        env={}, log=lambda _m: None)


def write_export(path, step):
    """What lora_io.export_lora writes: transformer.* keys, adapter-name suffix intact."""
    from safetensors.torch import save_file
    model = _FakeModel(scale=0.75)
    clean = {
        "transformer." + k[len("base_model.model."):]: v.to(torch.bfloat16).contiguous()
        for k, v in model.state_dict().items() if "lora_" in k
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(clean, path, metadata={
        "format": "pt", "training_info": json.dumps({"step": step, "epoch": 0})})
    return model


# ── 1. no marker: byte-identical to the old behaviour ───────────────────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=100)
    os.makedirs(cfg.output_dir, exist_ok=True)
    result = kstate.CheckpointManager(
        cfg, _FakeModel(), _FakeOptimizer(), log=lambda _m: None).restore()
    check("no marker -> start_step", result.start_step, 0)
    check("no marker -> pending", result.pending, None)
    check("no marker -> already_complete", result.already_complete, False)

# ── 2. no marker and nothing logged: the path must stay silent ──────────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=100)
    os.makedirs(cfg.output_dir, exist_ok=True)
    lines = []
    kstate.CheckpointManager(
        cfg, _FakeModel(), _FakeOptimizer(), log=lines.append).restore()
    check("no marker -> log lines", lines, [])

# ── 3. a staged warm start restores the weights and the step ────────────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=3000)
    os.makedirs(cfg.output_dir, exist_ok=True)
    export = os.path.join(tmp, "parent", "p_step_900.safetensors")
    source = write_export(export, 900)
    materialize_warm_start(step_export=pathlib.Path(export), adapter_config=None,
                           destination_dir=pathlib.Path(cfg.output_dir), step=900,
                           source_label="p_step_900.safetensors")

    model = _FakeModel(scale=0.0)
    lines = []
    result = kstate.CheckpointManager(
        cfg, model, _FakeOptimizer(), log=lines.append).restore()
    check("warm -> start_step", result.start_step, 900)
    check("warm -> pending is cold", result.pending, None)
    check("warm -> already_complete", result.already_complete, False)
    check("warm -> has_checkpoint stayed false",
          kstate.CheckpointManager(cfg, model, _FakeOptimizer(),
                                   log=lambda _m: None).has_checkpoint(), False)
    check("warm -> logged the cold-start warning",
          any("cold" in line for line in lines), True)
    check("warm -> logged the source", any("p_step_900" in line for line in lines), True)

# ── 4. the weights actually landed in the model ─────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=3000)
    os.makedirs(cfg.output_dir, exist_ok=True)
    export = os.path.join(tmp, "parent", "p_step_900.safetensors")
    write_export(export, 900)
    materialize_warm_start(step_export=pathlib.Path(export), adapter_config=None,
                           destination_dir=pathlib.Path(cfg.output_dir), step=900)

    loaded = {}

    def spy(model, blob, src, log=print):
        from safetensors.torch import load
        loaded.update(load(blob))
        return 1.0

    previous = kstate.lora_io.load_lora_weights
    kstate.lora_io.load_lora_weights = spy
    try:
        kstate.CheckpointManager(cfg, _FakeModel(), _FakeOptimizer(),
                                 log=lambda _m: None).restore()
    finally:
        kstate.lora_io.load_lora_weights = previous
    check("warm -> keys handed to PEFT", sorted(loaded), [
        "base_model.model.blk.lora_A.weight", "base_model.model.blk.lora_B.weight"])
    check("warm -> values survived the rewrite",
          torch.equal(loaded["base_model.model.blk.lora_A.weight"],
                      torch.full((4, 3), 0.75, dtype=torch.bfloat16)), True)

# ── 5. total_steps at or below the warm step is flagged, not looped ─────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=900)
    os.makedirs(cfg.output_dir, exist_ok=True)
    export = os.path.join(tmp, "parent", "p_step_900.safetensors")
    write_export(export, 900)
    materialize_warm_start(step_export=pathlib.Path(export), adapter_config=None,
                           destination_dir=pathlib.Path(cfg.output_dir), step=900)
    result = kstate.CheckpointManager(
        cfg, _FakeModel(), _FakeOptimizer(), log=lambda _m: None).restore()
    check("warm -> already_complete at the target", result.already_complete, True)

# ── 6. a corrupt marker aborts rather than silently training from 0 ─────────
with tempfile.TemporaryDirectory() as tmp:
    cfg = make_cfg(tmp, output_dir="./out", total_steps=3000)
    os.makedirs(cfg.output_dir, exist_ok=True)
    export = os.path.join(tmp, "parent", "p_step_900.safetensors")
    write_export(export, 900)
    materialize_warm_start(step_export=pathlib.Path(export), adapter_config=None,
                           destination_dir=pathlib.Path(cfg.output_dir), step=900)
    with open(os.path.join(cfg.output_dir, "warm_start.json"), "w") as handle:
        handle.write('{"format_version": 99, "step": 900}')
    code = None
    try:
        kstate.CheckpointManager(cfg, _FakeModel(), _FakeOptimizer(),
                                 log=lambda _m: None).restore()
    except SystemExit as exc:
        code = exc.code
    check("corrupt marker -> exits 2", code, 2)

print()
print("FAILURES:", FAILURES or "none")
sys.exit(1 if FAILURES else 0)
