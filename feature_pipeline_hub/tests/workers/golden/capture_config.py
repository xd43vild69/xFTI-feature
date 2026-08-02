"""Capture train_worker's resolved config for every fixture, as the refactor's contract.

Config resolution happens at *module import time* and lands in ~93 module-level
globals, so each fixture needs its own fresh interpreter. This script runs one
subprocess per fixture, imports the worker with TRAIN_SETTINGS_PATH pointed at that
fixture, and dumps every UPPER_CASE global to golden/expected/<name>.json.

Run it with the training runtime's interpreter, which is the one that has torch:

    training_runtime/venv/bin/python tests/workers/golden/capture_config.py

Re-running it after the refactor must produce byte-identical output.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
FIXTURES = HERE / "fixtures"
EXPECTED = HERE / "expected"
WORKERS = HERE.parents[2] / "workers"

# Values that are noise rather than resolved config: absolute paths that move with
# the checkout or the temp dir, and the static tables the fixtures never change.
SKIP = {"PROJECT_ROOT", "CACHE_ROOT", "OUTPUT_ROOT", "CONFIG_PATH", "ADVANCED_PATH",
        "DEFAULTS", "PRESETS", "SKIP_QUANT", "RESUME_DIR", "OPT_FILE", "STEP_FILE",
        "RUN_ID_FILE", "_BELL_MEAN", "_HALF_BELL_MEAN"}

# Runs inside the child interpreter, after train_worker has been imported.
CHILD = r"""
import json, os, sys
sys.path.insert(0, os.environ["WORKERS_DIR"])
import train_worker as tw

SKIP = set(json.loads(os.environ["SKIP_KEYS"]))
root = os.environ["FAKE_ROOT"]

def norm(v):
    # Paths are captured relative to the sandbox root so the golden is portable.
    if isinstance(v, str) and root in v:
        return v.replace(root, "<ROOT>")
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return f"<{type(v).__name__}:{v}>"

out = {k: norm(v) for k, v in sorted(vars(tw).items())
       if k.isupper() and k not in SKIP and not callable(v)}
out["_CFG_SOURCE"] = dict(sorted(tw._CFG_SOURCE.items()))
with open(os.environ["DUMP_PATH"], "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
"""


def capture(fixture: pathlib.Path, sandbox: pathlib.Path) -> dict:
    """Import train_worker against one fixture in a fresh interpreter; return its globals."""
    dump = sandbox / f"{fixture.stem}.dump.json"
    advanced = fixture.with_suffix(".advanced.json")

    env = {
        **os.environ,
        "TRAIN_SETTINGS_PATH": str(fixture),
        # Point the advanced sidecar at a real file only when the fixture has one,
        # otherwise at a path that does not exist so the worker skips it.
        "TRAIN_ADVANCED_PATH": str(advanced if advanced.is_file() else sandbox / "none.json"),
        "CACHE_DIR": str(sandbox / "cache"),
        "OUTPUT_DIR": str(sandbox / "out"),
        "WORKERS_DIR": str(WORKERS),
        "DUMP_PATH": str(dump),
        "FAKE_ROOT": str(sandbox),
        "SKIP_KEYS": json.dumps(sorted(SKIP)),
        "CUDA_VISIBLE_DEVICES": "",
    }
    result = subprocess.run([sys.executable, "-c", CHILD], env=env,
                            capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"{fixture.name} failed:\n{result.stderr[-3000:]}")
    return json.loads(dump.read_text())


def diff_keys(expected: dict, actual: dict) -> list[str]:
    """Human-readable per-key differences between a golden and a fresh capture."""
    problems = []
    for key in sorted(set(expected) | set(actual)):
        if key not in actual:
            problems.append(f"      {key}: MISSING (golden had {expected[key]!r})")
        elif key not in expected:
            problems.append(f"      {key}: UNEXPECTED (now {actual[key]!r})")
        elif expected[key] != actual[key]:
            problems.append(f"      {key}: {expected[key]!r} -> {actual[key]!r}")
    return problems


def main():
    check = "--check" in sys.argv
    fixtures = sorted(f for f in FIXTURES.glob("*.json")
                      if not f.name.endswith(".advanced.json"))
    if not fixtures:
        sys.exit("no fixtures — run make_fixtures.py first")

    EXPECTED.mkdir(parents=True, exist_ok=True)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = pathlib.Path(tmp)
        for fixture in fixtures:
            resolved = capture(fixture, sandbox)
            target = EXPECTED / f"{fixture.stem}.json"

            if not check:
                target.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
                print(f"  {fixture.stem:<24} {len(resolved)} keys")
                continue

            if not target.is_file():
                print(f"  MISSING GOLDEN  {fixture.stem}")
                failures += 1
                continue
            problems = diff_keys(json.loads(target.read_text()), resolved)
            if problems:
                failures += 1
                print(f"  FAIL  {fixture.stem}")
                print("\n".join(problems))
            else:
                print(f"  ok    {fixture.stem}")

    if not check:
        print(f"\ncaptured {len(fixtures)} fixtures into {EXPECTED}")
        return
    print(f"\n{len(fixtures) - failures}/{len(fixtures)} fixtures match")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
