"""Compare a smoke-run train_log.csv against the recorded baseline.

Exact diff does not work here: the training loop is not bit-reproducible on the same
GPU with the same seed. `cudnn.benchmark = True` picks kernels by timing, TF32 is on for
matmuls, and reductions in attention and backward have no fixed order — so two identical
runs drift by ~1e-6 on loss. That drift is real and worth knowing about, but it is not
what this check is looking for.

So: columns that must be identical are compared exactly, and computed quantities are
compared with a tolerance well below what any wiring mistake would produce. For scale, a
swapped sigma_min/sigma_max during the Stage 3 refactor moved values by 0.5 — five
hundred times the tolerance used here.
"""
from __future__ import annotations

import csv
import sys

# Driven by the RNG and the config, not by float arithmetic: any difference is a real
# change in which batch was drawn or how the schedule was computed.
EXACT_COLUMNS = ("step", "update", "epoch", "lr", "sigma", "bucket_h", "bucket_w")

# Per-column relative tolerance, measured from repeated runs of unchanged code.
#
# loss is the sensitive detector — it drifts about 1e-4 between identical runs, so a
# tight bound still leaves two orders of magnitude of headroom before a real change
# would hide. grad_norm is a global norm over every LoRA gradient, accumulating the
# backward pass's non-deterministic reduction order, and the CSV rounds it to four
# decimals; it moves ~0.7% run to run, so it is checked only for gross change.
RTOL = {"loss": 5e-3, "loss_avg": 5e-3, "grad_norm": 1e-1}
DEFAULT_RTOL = 5e-3

# Absolute floor, so values near zero do not fail on noise alone.
ATOL = 1e-4


def read(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close_enough(column: str, expected: float, actual: float) -> bool:
    rtol = RTOL.get(column, DEFAULT_RTOL)
    return abs(expected - actual) <= max(ATOL, rtol * abs(expected))


def main() -> int:
    if len(sys.argv) != 3:
        return int(bool(sys.stderr.write("usage: compare_smoke.py BASELINE ACTUAL\n")))

    baseline, actual = read(sys.argv[1]), read(sys.argv[2])
    problems: list[str] = []

    if len(baseline) != len(actual):
        print(f"FAIL — {len(baseline)} logged updates in the baseline, "
              f"{len(actual)} now", file=sys.stderr)
        return 1

    worst: dict[str, float] = {}
    for i, (want, got) in enumerate(zip(baseline, actual)):
        for column in want:
            if column not in got:
                problems.append(f"  row {i}: column {column} missing")
                continue
            if column in EXACT_COLUMNS:
                if want[column] != got[column]:
                    problems.append(
                        f"  row {i} {column}: {want[column]} -> {got[column]}  (must be exact)")
                continue
            try:
                a, b = float(want[column]), float(got[column])
            except ValueError:
                continue
            drift = abs(a - b) / max(abs(a), 1e-12)
            worst[column] = max(worst.get(column, 0.0), drift)
            if not close_enough(column, a, b):
                problems.append(f"  row {i} {column}: {a} -> {b}  ({drift:.2%} drift, "
                                f"tolerance {RTOL.get(column, DEFAULT_RTOL):.1%})")

    if problems:
        print("FAIL — the training loop no longer computes the same thing:",
              file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1

    summary = ", ".join(f"{c} {d:.2e}" for c, d in sorted(worst.items()))
    print(f"PASS — {len(baseline)} updates match. Largest drift per column: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
