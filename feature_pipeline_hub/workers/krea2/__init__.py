"""Krea 2 LoRA trainer, split out of the single-file workers/train_worker.py.

`train_worker.py` remains the entrypoint — training_service.py launches it by path and
training_runner.py runs it as a plain script, so the file must stay where it is. What
moved here is everything that file used to hold as module-level globals and top-level
functions.

Two layers, split by whether they import torch:

- **torch-free** — `config`, `sampling`, `curation`, `checkpoints`. These are covered
  by mypy and pytest in the hub environment, which is why the split runs along this
  line rather than a purely thematic one.
- **torch-dependent** — everything else, importable only under the training runtime's
  virtualenv (`training_runtime/venv`), where torch and diffusers live.

Nothing is re-exported here on purpose: importing this package must not drag torch in.
Import the submodule you need.
"""
