# Golden baselines for the `train_worker.py` refactor

These pin the *current* behavior of `workers/train_worker.py` so the staged split into
`workers/krea2/` can be proven not to change it. They were captured **before** any code
moved, so they encode the existing contract rather than the refactor's assumptions.

Nothing here runs in CI: `pyproject.toml` scopes pytest to `tests/` but the worker needs
torch, which lives in `training_runtime/venv`, not the hub's environment. These are
run by hand at every stage boundary.

## Running

```bash
training_runtime/venv/bin/python tests/workers/golden/capture_config.py --check
```

```bash
training_runtime/venv/bin/python tests/workers/golden/capture_behavior.py --check
```

Both exit non-zero and print per-key diffs on any mismatch. Drop `--check` to
*re-record* the goldens — only do that when a behavior change is intended and
understood, never to make a failing check go away.

`make_fixtures.py` regenerates the input fixtures; the recorded outputs in `expected/`
are what matters.

## What each file covers

| Script | Pins |
|---|---|
| `capture_config.py` | The ~83 module-level globals, resolved for 16 settings fixtures. One fresh interpreter per fixture, since resolution happens at import time. Targets the 21 globals that get reassigned after first definition — precedence layering, warmup unit conversion and clamping, `compact_text` force-off at batch > 1, `project_name` directory derivation, and every `_validate_choice` fallback. Also records `_CFG_SOURCE`, so a value that ends up right *for the wrong reason* still fails. |
| `capture_behavior.py` | Numeric and sequencing behavior: latent pack/unpack (including exact round-trip), `calculate_shift`, `prepare_position_ids`, `timestep_weight` across all three modes, `sample_sigma` across all six of its config globals, both samplers (sequence, epoch count, and mid-stream save/restore fidelity), EMA shadow evolution, curation weighting, and checkpoint rotation. |

## Sensitivity

Tensors are compared by sha256 over their float64 bytes. That is lossless for the
float32 tensors these helpers produce, and it is what actually gates a run — the
rounded `sum`/`min`/`max`/`head` fields exist to make a failure readable, not to
detect one.

Verified by negative test: perturbing `-2.0` to `-2.001` in `timestep_weight`'s bell
term and `u ** 3` to `u ** 3.0001` in `sample_sigma`'s content skew fails exactly the
two affected goldens and leaves the other eight passing. A 1-ULP change to a float64
constant such as `_BELL_MEAN` does **not** register, because it cannot survive the
float32 arithmetic downstream — that is the intended limit, not a gap.

## What these do not cover

The training loop itself, checkpoint save/restore round-trips, NF4 quantization, and
LoRA export all need a model and a populated cache. Those are covered by the Stage 5
smoke run: ~20 steps at a fixed seed, comparing `train_log.csv` before and after.
