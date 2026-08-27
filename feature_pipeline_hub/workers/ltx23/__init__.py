"""LTX 2.3 LoRA trainer, split out of the single-file 2_train_lora_LTX23.py.

`train_ltx23_worker.py` remains the entrypoint — training_service launches it by path and
training_runner runs it as a plain script. What moved here is everything that file used
to hold as module-level globals and top-level functions.

Two layers, split by whether they import torch:

- **torch-free** — `config`, `cache_index`, `metrics`, `checkpoints`. These are covered
  by mypy and pytest in the hub environment.
- **torch-dependent** — `math_ops`, `quantization`, `lora_io`, `dataset`, `preview`, `state`,
  importable only under the training runtime's virtualenv (`training_runtime/venv`).

Nothing is re-exported here on purpose: importing this package must not drag torch in.
Import the submodule you need.
"""
