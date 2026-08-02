"""Capture numeric and sequencing behavior of train_worker's pure helpers.

Pins the functions that Stage 3 turns from global-reading into parameterized ones —
if a keyword argument gets wired to the wrong config field, these catch it, which a
config-only test cannot.

Each tensor is recorded as a sha256 over its exact bytes (the strict gate) plus a
human-readable summary (what you actually read when one fails).

Run with the training runtime's interpreter:

    training_runtime/venv/bin/python tests/workers/golden/capture_behavior.py
"""
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
EXPECTED = HERE / "expected"
WORKERS = HERE.parents[2] / "workers"

os.environ.setdefault("TRAIN_SETTINGS_PATH", str(HERE / "fixtures" / "empty.json"))
os.environ.setdefault("TRAIN_ADVANCED_PATH", str(HERE / "does-not-exist.json"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(WORKERS))

import torch  # noqa: E402
import train_worker as tw  # noqa: E402


def fingerprint(t):
    """sha256 of a tensor's exact bytes, plus a summary for reading failures.

    Hashed as float64: lossless for the float32 tensors these helpers produce, and
    it keeps full precision for anything already float64 rather than truncating it.
    The rounded summary fields are for reading a failure, not for detecting one —
    the hash is the gate.
    """
    t = t.detach().cpu()
    digest = hashlib.sha256(t.double().numpy().tobytes()).hexdigest()[:32]
    flat = t.double().flatten()
    return {
        "sha256": digest,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sum": round(float(flat.sum()), 6),
        "min": round(float(flat.min()), 6),
        "max": round(float(flat.max()), 6),
        "head": [round(float(x), 6) for x in flat[:6]],
    }


def capture_latent_packing():
    """pack_latents/unpack_latents over several shapes, including the round-trip."""
    out = {}
    for B, C, H, W in [(1, 16, 32, 32), (1, 16, 64, 96), (2, 16, 48, 48), (1, 16, 128, 64)]:
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(B, C, H, W, generator=g)
        packed = tw.pack_latents(x)
        restored = tw.unpack_latents(packed, H, W)
        out[f"{B}x{C}x{H}x{W}"] = {
            "packed": fingerprint(packed),
            "roundtrip_exact": bool(torch.equal(x, restored)),
        }
    return out


def capture_shift():
    """calculate_shift across sequence lengths and both scheduler configs."""
    out = {}
    for seq in [256, 1024, 4096, 6400, 9216]:
        out[str(seq)] = {
            "default": round(tw.calculate_shift(seq), 10),
            "custom": round(tw.calculate_shift(seq, 128, 8192, 0.3, 1.5), 10),
        }
    return out


def capture_position_ids():
    """prepare_position_ids for several text lengths and image grids."""
    out = {}
    for text_len, gh, gw in [(1, 16, 16), (77, 32, 48), (256, 8, 8)]:
        ids = tw.prepare_position_ids(text_len, gh, gw, "cpu")
        out[f"t{text_len}_g{gh}x{gw}"] = fingerprint(ids)
    return out


def capture_timestep_weight():
    """timestep_weight over a sigma grid, for all three weighting modes."""
    sigma = torch.linspace(0.01, 0.99, 25)
    out = {}
    original = tw.TIMESTEP_WEIGHTING
    try:
        for mode in ("none", "bell", "half_bell"):
            tw.TIMESTEP_WEIGHTING = mode
            out[mode] = fingerprint(tw.timestep_weight(sigma))
    finally:
        tw.TIMESTEP_WEIGHTING = original
    return out


def capture_sample_sigma():
    """sample_sigma across every branch of its six config globals, at a fixed seed.

    Stage 3 replaces those globals with keyword arguments; this is what proves the
    wiring is right rather than merely plausible.
    """
    shift_cfg = (256, 6400, 0.5, 1.15)
    saved = {k: getattr(tw, k) for k in
             ("TIMESTEP_SAMPLING", "CONTENT_OR_STYLE", "LOGIT_NORMAL_MU",
              "LOGIT_NORMAL_SIGMA", "SIGMA_MIN", "SIGMA_MAX")}
    combos = [
        ("uniform_balanced",   "krea2_shift",  "balanced", 0.0, 1.0, 0.0, 1.0),
        ("uniform_content",    "krea2_shift",  "content",  0.0, 1.0, 0.0, 1.0),
        ("uniform_style",      "krea2_shift",  "style",    0.0, 1.0, 0.0, 1.0),
        ("clamped",            "krea2_shift",  "balanced", 0.0, 1.0, 0.25, 0.75),
        ("logit_normal",       "logit_normal", "balanced", 0.0, 1.0, 0.0, 1.0),
        ("logit_normal_muneg", "logit_normal", "balanced", -0.5, 1.5, 0.0, 1.0),
    ]
    out = {}
    try:
        for name, sampling, cos, mu, sig, smin, smax in combos:
            tw.TIMESTEP_SAMPLING, tw.CONTENT_OR_STYLE = sampling, cos
            tw.LOGIT_NORMAL_MU, tw.LOGIT_NORMAL_SIGMA = mu, sig
            tw.SIGMA_MIN, tw.SIGMA_MAX = smin, smax
            for seq_len in (1024, 4096):
                torch.manual_seed(4242)
                s = tw.sample_sigma(8, seq_len, "cpu", shift_cfg)
                out[f"{name}_seq{seq_len}"] = fingerprint(s)
    finally:
        for k, v in saved.items():
            setattr(tw, k, v)
    return out


def capture_samplers():
    """Batch sequences from both samplers at a fixed seed, plus resume-state fidelity."""
    buckets = {(64, 64): [f"img_{i:03d}" for i in range(16)],
               (48, 80): [f"wide_{i:03d}" for i in range(5)],
               (80, 48): ["tall_000"]}
    out = {}

    for label, make in [
        ("epoch_bs1", lambda: tw.EpochSampler(buckets, 1, 777)),
        ("epoch_bs2", lambda: tw.EpochSampler(buckets, 2, 777)),
        ("legacy_bs1", lambda: tw.LegacySampler(buckets, 1, 777)),
    ]:
        sampler = make()
        seq = [[list(sz), names] for sz, names in (sampler.next() for _ in range(60))]
        out[label] = {"sequence": seq, "epoch_after_60": sampler.epoch}

        # Save/restore mid-stream must reproduce the tail exactly — this is what a
        # resumed run depends on.
        resumed = make()
        for _ in range(25):
            resumed.next()
        state = resumed.state_dict()
        tail_before = [[list(sz), names] for sz, names in
                       (resumed.next() for _ in range(20))]
        restored = make()
        restored.load_state_dict(state)
        tail_after = [[list(sz), names] for sz, names in
                      (restored.next() for _ in range(20))]
        out[label]["resume_reproduces_tail"] = tail_before == tail_after

    # Per-image repeats only exist on EpochSampler.
    rep = tw.EpochSampler({(64, 64): ["a", "b", "c"]}, 1, 5, repeats={"a": 3, "b": 2})
    out["epoch_repeats"] = {"sequence": [[list(sz), n] for sz, n in
                                         (rep.next() for _ in range(12))]}
    return out


def capture_ema():
    """EMA shadow after a fixed update sequence, including the decay warmup."""
    torch.manual_seed(11)
    params = [torch.nn.Parameter(torch.randn(4, 3)) for _ in range(2)]
    ema = tw.EMA(params, decay=0.99, device="cpu")
    steps = []
    for i in range(10):
        with torch.no_grad():
            for p in params:
                p.add_(0.05 * (i + 1))
        ema.update()
        steps.append(round(float(ema.shadow[0].sum()), 6))

    ema.apply()
    applied = round(float(params[0].sum()), 6)
    ema.restore()
    restored = round(float(params[0].sum()), 6)
    return {"shadow_sum_per_update": steps, "updates": ema.updates,
            "shadow0": fingerprint(ema.shadow[0]),
            "sum_while_applied": applied, "sum_after_restore": restored}


def capture_curation():
    """load_curation_weights over the report/override shapes it must handle."""
    import tempfile
    report = {
        "mode": "face",
        "auto_threshold": 0.6,
        "weights": {"priority": 1.5, "good": 1.0, "bad": 0.5},
        "baselines": ["hero.png"],
        "images": {
            "hero":  {"file": "hero.png",  "score": 0.9},
            "good1": {"file": "good1.png", "score": 0.8},
            "bad1":  {"file": "bad1.png",  "score": 0.2},
            "edge":  {"file": "edge.png",  "score": 0.6},
            "nulls": {"file": "nulls.png", "score": None},
        },
    }
    names = ["hero", "good1", "good1__flip", "bad1", "edge", "nulls", "not_in_report"]
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "curation_report.json").write_text(json.dumps(report))

        w, s = tw.load_curation_weights(str(d), names)
        out["auto_threshold"] = {"weights": w, "summary": s}

        (d / "curation_overrides.json").write_text(json.dumps(
            {"threshold": 0.85, "groups": {"bad1": "good", "hero": "bad"}}))
        w, s = tw.load_curation_weights(str(d), names)
        out["with_overrides"] = {"weights": w, "summary": s}

    with tempfile.TemporaryDirectory() as tmp:
        w, s = tw.load_curation_weights(tmp, names)
        out["no_report"] = {"weights": w, "summary": s}

    # All-1.0 weights must collapse to (None, None) so training stays untouched.
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "curation_report.json").write_text(json.dumps({
            "mode": "face", "auto_threshold": 0.5,
            "weights": {"priority": 1.0, "good": 1.0, "bad": 1.0},
            "images": {"a": {"file": "a.png", "score": 0.9}}}))
        w, s = tw.load_curation_weights(str(d), ["a"])
        out["all_ones_collapses"] = {"weights": w, "summary": s}

    return out


def capture_rotation():
    """rotate_checkpoints must prune by parsed step number and never touch FINAL."""
    import tempfile
    out = {}
    for keep in (0, 1, 3, 10):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            for step in [25, 50, 100, 200, 1000, 75]:
                (d / f"Krea2_LoRA_step_{step}.safetensors").write_text("x")
            (d / "Krea2_FINAL_LoRA.safetensors").write_text("x")
            (d / "unrelated.txt").write_text("x")
            tw.rotate_checkpoints(str(d), keep)
            out[f"keep_{keep}"] = sorted(p.name for p in d.iterdir())
    return out


def capture_curation_group():
    """_curation_group's decision table, including the None and override paths."""
    out = {}
    for mode in ("face", "style"):
        for score in (None, 0.2, 0.5, 0.8):
            for thr in (None, 0.5):
                for ovr in (None, "good", "bad", "garbage"):
                    key = f"{mode}_s{score}_t{thr}_o{ovr}"
                    out[key] = tw._curation_group(score, thr, ovr, mode=mode)
    return out


def capture_lora_metadata():
    """The safetensors metadata map, across the config fields that feed it.

    Went from reading 13 module globals to taking a TrainConfig, so this pins the
    wiring. `ss_network_alpha` is the field that matters most: get it wrong and every
    loader silently runs the adapter at the wrong strength.
    """
    import tempfile

    from krea2 import config as kconfig
    from krea2 import lora_io

    cases = {
        "defaults": {},
        "named_with_trigger": {"project_name": "mi_lora", "trigger_word": "sks_person"},
        "rank32_alpha64": {"lora_rank": 32, "lora_alpha": 64},
        "fp32_ema_attn": {"lora_dtype": "fp32", "use_ema": True, "lora_target": "attn"},
        "metadata_off": {"export_metadata": False},
        "alt_optimizer": {"optimizer": "adamw", "lr_scheduler": "linear",
                          "lr": 5e-5, "seed": 7},
    }
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name, settings in cases.items():
            path = root / f"{name}.json"
            path.write_text(json.dumps(settings))
            cfg = kconfig.load_config(str(root), settings_path=str(path),
                                      advanced_path=str(root / "absent.json"),
                                      cache_root=str(root / "c"),
                                      output_root=str(root / "o"),
                                      env={}, log=lambda _m: None)
            out[name] = {
                "metadata": lora_io.build_metadata(cfg, step=250, epoch=3, num_images=42),
                "fingerprint": lora_io.fingerprint(cfg).replace(str(root), "<ROOT>"),
                "lora_scale": cfg.lora_scale,
            }
        # No step means no training_info key at all, not an empty one.
        cfg = kconfig.load_config(str(root), settings_path=str(root / "defaults.json"),
                                  advanced_path=str(root / "absent.json"),
                                  cache_root=str(root / "c"), output_root=str(root / "o"),
                                  env={}, log=lambda _m: None)
        out["no_step_no_images"] = {"metadata": lora_io.build_metadata(cfg)}
    return out


def capture_ema_horizon():
    """When the EMA decay implies a window longer than the run itself."""
    from krea2 import ema as kema

    out = {}
    for decay in (0.9, 0.99, 0.999, 0.9999):
        for total_updates in (100.0, 300.0, 5000.0):
            warning = kema.horizon_warning(decay, total_updates)
            out[f"d{decay}_u{total_updates:.0f}"] = warning
    return out


def capture_noise_offset():
    """Per-channel noise offset over packed latents, at a fixed seed."""
    out = {}
    for channels, scale in [(16, 0.05), (16, 0.1), (4, 0.05)]:
        torch.manual_seed(31337)
        noise = torch.randn(2, 64, channels * 4)
        torch.manual_seed(555)
        out[f"c{channels}_s{scale}"] = fingerprint(
            tw._math.add_noise_offset(noise, channels, scale))
    return out


CAPTURES = {
    "behavior_latent_packing": capture_latent_packing,
    "behavior_shift": capture_shift,
    "behavior_position_ids": capture_position_ids,
    "behavior_timestep_weight": capture_timestep_weight,
    "behavior_sample_sigma": capture_sample_sigma,
    "behavior_samplers": capture_samplers,
    "behavior_ema": capture_ema,
    "behavior_curation": capture_curation,
    "behavior_curation_group": capture_curation_group,
    "behavior_rotation": capture_rotation,
    "behavior_lora_metadata": capture_lora_metadata,
    "behavior_ema_horizon": capture_ema_horizon,
    "behavior_noise_offset": capture_noise_offset,
}


def walk_diff(expected, actual, path=""):
    """Recursive value-level differences, reported as dotted paths."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            sub = f"{path}.{key}" if path else key
            if key not in actual:
                out.append(f"      {sub}: MISSING")
            elif key not in expected:
                out.append(f"      {sub}: UNEXPECTED")
            else:
                out += walk_diff(expected[key], actual[key], sub)
        return out
    if expected != actual:
        return [f"      {path}: {expected!r} -> {actual!r}"]
    return []


def main():
    check = "--check" in sys.argv
    EXPECTED.mkdir(parents=True, exist_ok=True)
    failures = 0

    for name, fn in CAPTURES.items():
        data = fn()
        target = EXPECTED / f"{name}.json"

        if not check:
            target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            print(f"  {name:<28} {len(data)} cases")
            continue

        if not target.is_file():
            print(f"  MISSING GOLDEN  {name}")
            failures += 1
            continue
        # Round-trip through JSON so tuple/list and int/float compare as they serialize.
        fresh = json.loads(json.dumps(data, sort_keys=True))
        problems = walk_diff(json.loads(target.read_text()), fresh)
        if problems:
            failures += 1
            print(f"  FAIL  {name}")
            print("\n".join(problems[:25]))
            if len(problems) > 25:
                print(f"      … and {len(problems) - 25} more")
        else:
            print(f"  ok    {name}")

    if not check:
        print(f"\ncaptured {len(CAPTURES)} behavior goldens into {EXPECTED}")
        return
    print(f"\n{len(CAPTURES) - failures}/{len(CAPTURES)} behavior goldens match")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
