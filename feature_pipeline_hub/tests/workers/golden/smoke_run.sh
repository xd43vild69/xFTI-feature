#!/usr/bin/env bash
# End-to-end smoke run: a few real training steps against the real model, recorded so a
# refactor of the training loop can be proven not to change what it computes.
#
# The goldens next door cover the pieces. This covers the loop -- the one part where a
# mistake produces a slightly worse LoRA hours later instead of an error.
#
#   ./smoke_run.sh record     # write the baseline (run before touching the loop)
#   ./smoke_run.sh check      # re-run and diff the losses against it
#
# Needs a GPU, the NF4 model in training_runtime/model, and a populated cache. Takes a
# couple of minutes, most of it loading the 12B transformer.
set -euo pipefail

MODE="${1:-check}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUB="$(cd "$HERE/../../.." && pwd)"
RUNTIME="$HUB/training_runtime"
CACHE_NAME="${SMOKE_CACHE:-mg-bd-v1}"

BASELINE="$HERE/expected/smoke_train_log.csv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [[ ! -x "$RUNTIME/venv/bin/python" ]]; then
  echo "no training runtime at $RUNTIME/venv -- see scripts/setup_training_runtime.sh" >&2
  exit 1
fi
if [[ ! -d "$RUNTIME/cache/$CACHE_NAME" ]]; then
  echo "no cache at $RUNTIME/cache/$CACHE_NAME (set SMOKE_CACHE to another one)" >&2
  exit 1
fi

# Deliberate settings: 12 steps is three optimizer updates at grad_accum 4, enough for
# warmup, an accumulation boundary and a save to all happen. A fixed seed and the epoch
# sampler make the batch sequence reproducible; previews and validation stay off so the
# run stays short and the log holds only training rows.
cat > "$WORK/settings.json" <<JSON
{
  "model_id": "$RUNTIME/model",
  "cache_dir": "$RUNTIME/cache/$CACHE_NAME",
  "output_dir": "$WORK/out",
  "dataset_path": "$RUNTIME/datasets/$CACHE_NAME",
  "total_steps": 12,
  "grad_accum_steps": 4,
  "warmup_steps": 1,
  "save_every": 8,
  "seed": 1234,
  "sampler": "epoch",
  "lora_rank": 8,
  "lora_alpha": 16,
  "csv_log": true,
  "preview_every": 0,
  "validate_every": 0,
  "max_checkpoints_to_keep": 0
}
JSON

echo "running 12 steps against $CACHE_NAME ..."
TRAIN_SETTINGS_PATH="$WORK/settings.json" \
TRAIN_ADVANCED_PATH="$WORK/absent.json" \
  "$RUNTIME/venv/bin/python" -u "$HUB/workers/train_worker.py" > "$WORK/stdout.log" 2>&1 || {
    echo "run failed:" >&2; tail -30 "$WORK/stdout.log" >&2; exit 1; }

LOG="$WORK/out/train_log.csv"
[[ -f "$LOG" ]] || { echo "no train_log.csv produced" >&2; tail -30 "$WORK/stdout.log" >&2; exit 1; }

# Drop secs and vram_peak_gb: wall-clock and allocator state, which differ between
# identical runs and say nothing about what was computed.
cut -d, -f1-10 "$LOG" > "$WORK/trimmed.csv"

if [[ "$MODE" == "record" ]]; then
  mkdir -p "$HERE/expected"
  cp "$WORK/trimmed.csv" "$BASELINE"
  echo "baseline recorded at $BASELINE"
  cat "$BASELINE"
  exit 0
fi

[[ -f "$BASELINE" ]] || { echo "no baseline -- run: $0 record" >&2; exit 1; }

# Compared with a tolerance, not by diff: this loop is not bit-reproducible even against
# itself. See compare_smoke.py.
exec python3 "$HERE/compare_smoke.py" "$BASELINE" "$WORK/trimmed.csv"
