"""Checkpoint management, training state restoration, and signal handling for LTX 2.3.

Imports torch: runs in the training_runtime environment.
"""
from __future__ import annotations

import json
import os
import signal
import sys
from typing import Any, Callable

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file

from ltx23.checkpoints import atomic_write, rotate
from ltx23.config import LTX23TrainConfig
from ltx23.lora_io import save_lora
from ltx23.metrics import CheckpointLog

Logger = Callable[[str], None]


class CheckpointManager:
    """Manages atomic checkpoint saving, rotation, and resumption for LTX 2.3."""

    def __init__(self, cfg: LTX23TrainConfig, log: Logger = print) -> None:
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.resume_dir = os.path.join(self.output_dir, "resume_checkpoint")
        self.adapter_path = os.path.join(self.resume_dir, "adapter_model.safetensors")
        self.opt_file = os.path.join(self.output_dir, "optimizer.pt")
        self.step_file = os.path.join(self.output_dir, "current_step.txt")
        self.run_id_file = os.path.join(self.output_dir, "run_id.txt")
        self.checkpoint_log = CheckpointLog(self.output_dir)
        self.log = log

    def has_checkpoint(self) -> bool:
        """Return True if restorable checkpoint files exist."""
        return os.path.exists(self.adapter_path) and os.path.exists(self.step_file)

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> int:
        """Restore adapter weights and optimizer state. Return the start step."""
        if not self.has_checkpoint():
            return 0

        # Check rank compatibility
        cfg_json_path = os.path.join(self.resume_dir, "adapter_config.json")
        if os.path.exists(cfg_json_path):
            try:
                with open(cfg_json_path, "r", encoding="utf-8") as f:
                    acfg = json.load(f)
                saved_r = int(acfg.get("r", -1))
                if saved_r != self.cfg.lora_rank:
                    self.log(f"[!] Checkpoint incompatible: saved rank={saved_r}, config rank={self.cfg.lora_rank}. Starting fresh.")
                    return 0
            except Exception as e:
                self.log(f"[!] Warning reading adapter config: {e}")

        try:
            with open(self.step_file, "r", encoding="utf-8") as f:
                start_step = int(f.read().strip())

            state = load_file(self.adapter_path, device="cpu")
            set_peft_model_state_dict(model, state)

            if optimizer is not None and os.path.exists(self.opt_file):
                try:
                    optimizer.load_state_dict(torch.load(self.opt_file, map_location="cpu", weights_only=False))
                    self.log("✓ Optimizer state restored.")
                except Exception as e:
                    self.log(f"[!] Could not restore optimizer: {e}")

            self.log(f"✓ Resuming LTX 2.3 training from step {start_step}")
            return start_step
        except Exception as e:
            self.log(f"[!] Error restoring checkpoint: {e}; starting fresh.")
            return 0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        reason: str = "periodic",
        is_final: bool = False,
    ) -> None:
        """Atomically persist adapter state, optimizer moments, step marker, and exported LoRA."""
        if step <= 0:
            return

        self.log(f"\nSaving checkpoint at step {step} ({reason})...")
        os.makedirs(self.resume_dir, exist_ok=True)

        try:
            model.save_pretrained(self.resume_dir)
        except Exception as e:
            self.log(f"[!] Warning saving PEFT pretrained: {e}")

        if optimizer is not None:
            try:
                atomic_write(self.opt_file, lambda tmp: torch.save(optimizer.state_dict(), tmp))
            except Exception as e:
                self.log(f"[!] Warning saving optimizer state: {e}")

        try:
            atomic_write(self.step_file, lambda tmp: open(tmp, "w", encoding="utf-8").write(str(step)))
        except Exception as e:
            self.log(f"[!] Warning writing current step: {e}")

        # Export standalone LoRA .safetensors
        prefix = f"LTX23_{self.cfg.project_name}_" if self.cfg.project_name else "LTX23_LoRA_"
        tag = "FINAL" if is_final else f"step_{step}"
        out_name = f"{prefix}{tag}.safetensors"
        out_path = os.path.join(self.output_dir, out_name)

        try:
            save_lora(model, out_path, self.cfg, step=step)
            self.log(f"✓ LoRA exported: {out_path}")
        except Exception as e:
            self.log(f"[!] Warning exporting LoRA: {e}")

        self.checkpoint_log.record(step=step, reason=reason)

    def close(self) -> None:
        self.checkpoint_log.close()


def register_signal_handlers(on_signal: Callable[[int], None]) -> None:
    """Register graceful termination signal handlers."""
    def _handler(sig: int, frame: Any) -> None:
        on_signal(sig)
        sys.exit(0)

    for sig_name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, sig_name):
            try:
                signal.signal(getattr(signal, sig_name), _handler)
            except Exception:
                pass
