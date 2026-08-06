"""Saving and restoring a run: adapter weights, optimizer, RNG, sampler position.

A training run is far more often killed than finished — SIGTERM from the UI, an OOM, a
closed laptop — so the checkpoint is the primary artifact, not a safety net. Two rules
shape everything here:

**Write order is a commit protocol.** Weights, then optimizer and RNG state, then
`current_step.txt` last. That file's existence is the assertion that everything it
refers to is complete, so a crash mid-save leaves the *previous* checkpoint intact
rather than a half-written one that loads and trains to garbage.

**Restoring must be all-or-nothing.** A checkpoint that only partly applies is worse
than none: the run continues, reports normal-looking losses, and produces a worse
adapter. Failures abort by default rather than silently restarting from step 0.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import signal
import sys
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable

import torch

from krea2 import checkpoints, lora_io
from krea2.config import TrainConfig

Logger = Callable[[str], None]

# Bumped when the saved dict's shape changes. Version 1 was the bare optimizer
# state_dict with the step living only in current_step.txt.
FORMAT_VERSION = 2

ADAPTER_FILENAME = "adapter_model.safetensors"

# Written by the hub to ask for a weights-only start, never by the trainer. See
# _restore_warm_start for why its absence has to keep meaning "no checkpoint".
WARM_START_FILENAME = "warm_start.json"
WARM_START_FORMAT_VERSION = 1


@dataclass
class RestoreResult:
    """Outcome of looking for a checkpoint to resume from."""

    start_step: int = 0
    # Sampler/EMA/RNG state, applied once those objects exist. None when there was
    # nothing to restore or the checkpoint predates format version 2.
    pending: dict[str, Any] | None = None
    already_complete: bool = False


class CheckpointManager:
    """Owns everything that reads or writes the run's on-disk state."""

    def __init__(self, cfg: TrainConfig, model: torch.nn.Module, optimizer: Any,
                 log: Logger = print, ckpt_log: Any = None) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.fingerprint = lora_io.fingerprint(cfg)
        self.log = log
        self._busy = False
        self.saved_on_exit = False
        # metrics.CheckpointLog, or None outside the hub. Optional so the standalone
        # workflow keeps working unchanged.
        self._ckpt_log = ckpt_log

    # ── paths ───────────────────────────────────────────────────────────────
    @property
    def adapter_path(self) -> str:
        return os.path.join(self.cfg.resume_dir, ADAPTER_FILENAME)

    @property
    def _checkpoint_prefix(self) -> str:
        # Falls back to the historical name for the standalone workflow (outside
        # the hub, cfg.checkpoint_prefix stays "" unless someone sets it by hand).
        return self.cfg.checkpoint_prefix or "Krea2"

    def has_checkpoint(self) -> bool:
        return (os.path.exists(self.cfg.step_file)
                and os.path.exists(self.cfg.optimizer_state_file)
                and os.path.exists(self.adapter_path))

    # ── restore ─────────────────────────────────────────────────────────────
    def restore(self, ema: Any = None) -> RestoreResult:
        """Resume from this run's checkpoint, if there is a usable one.

        Exits with code 2 on a corrupt checkpoint unless `resume_on_corrupt` says to
        restart: silently starting over would retrain a whole run over already-trained
        weights with the LR reset to warmup, and nothing would look wrong.
        """
        if not self.has_checkpoint():
            warm = self._restore_warm_start()
            if warm.start_step <= 0:
                return warm
            return self._flag_if_complete(warm)

        if not checkpoints.belongs_to_run(self.cfg.run_id_file, self.cfg.run_id):
            self.log("=" * 65)
            self.log("[i] Checkpoint from a previous pipeline run; discarding it / "
                     "Checkpoint de una corrida anterior del pipeline; se descarta.")
            self.log(f"    {self.cfg.resume_dir}")
            self.log("=" * 65)
            return RestoreResult()

        self.log("=" * 65)
        self.log("¡Checkpoint detected! Restoring state... / "
                 "¡Checkpoint detectado! Restaurando estado...")
        try:
            result = self._load_state(ema)
            with open(self.adapter_path, "rb") as handle:
                lora_io.load_lora_weights(self.model, handle.read(),
                                          self.adapter_path, log=self.log)
            self.log(f"Resuming training from step / Reanudando entrenamiento desde el "
                     f"paso {result.start_step}...")
        except Exception as exc:
            self.log(f"[!] ERROR: checkpoint exists but could not be restored / "
                     f"existe pero no se pudo restaurar: {exc}")
            if self.cfg.resume_on_corrupt != "restart":
                self.log("[!] Refusing to silently restart from step 0. Delete the "
                         "checkpoint or set resume_on_corrupt='restart' / Me niego a "
                         "reiniciar en silencio desde 0.")
                sys.exit(2)
            self.log("[!] resume_on_corrupt='restart': starting from step 0 / "
                     "empezando desde 0.")
            result = RestoreResult()
        self.log("=" * 65)

        return self._flag_if_complete(result)

    def _flag_if_complete(self, result: RestoreResult) -> RestoreResult:
        """Mark a restore that has nothing left to run, rather than looping zero times."""
        if result.start_step >= self.cfg.total_steps:
            self.log(f"[!] Checkpoint step {result.start_step} >= total_steps "
                     f"{self.cfg.total_steps}; nothing to do / nada que hacer. "
                     f"Increase total_steps to continue.")
            result.already_complete = True
        return result

    def _restore_warm_start(self) -> RestoreResult:
        """Start from exported weights alone, when the hub explicitly asked for it.

        `resume_checkpoint/` and `optimizer.pt` are rewritten in place on every save, so
        a finished run keeps exactly one resumable state — its last. The per-step
        `{prefix}_step_N.safetensors` exports are the only surviving record of the steps
        in between, and they carry weights and nothing else: no optimizer moments, no
        RNG streams, no sampler position, no EMA shadow. Starting from one is therefore
        a warm start, not a resume, and it is the caller's job to know the difference —
        branches forked this way are comparable to each other (they are all equally
        cold) but not to the parent's own curve past that step.

        Gated on a marker file so that a missing optimizer.pt keeps meaning "no
        checkpoint" everywhere else: without the gate, a run interrupted between the
        adapter write and the optimizer write would come back as a silent cold restart
        at the old step, which is the exact failure the commit protocol exists to
        prevent. The marker stays behind afterwards but goes inert — once this branch
        saves once, has_checkpoint() is true and this path is never consulted again.
        """
        marker = os.path.join(self.cfg.output_dir, WARM_START_FILENAME)
        if not os.path.exists(marker):
            return RestoreResult()

        self.log("=" * 65)
        self.log("Warm start requested / Arranque en caliente solicitado.")
        try:
            with open(marker, "r", encoding="utf-8") as handle:
                spec = json.load(handle)
            version = int(spec.get("format_version", 0))
            if version != WARM_START_FORMAT_VERSION:
                raise ValueError(f"unsupported warm_start.json version {version}")
            start_step = int(spec["step"])
            if start_step <= 0:
                raise ValueError(f"warm start step must be positive, got {start_step}")
            # sys.exit(1) from load_lora_weights on a rank/target mismatch is a
            # SystemExit, so it passes through this handler untouched — a branch whose
            # adapter does not fit must abort, not fall back to step 0.
            with open(self.adapter_path, "rb") as handle:
                lora_io.load_lora_weights(self.model, handle.read(),
                                          self.adapter_path, log=self.log)
        except Exception as exc:
            self.log(f"[!] ERROR: warm start requested but could not be applied / "
                     f"se pidió pero no se pudo aplicar: {exc}")
            if self.cfg.resume_on_corrupt != "restart":
                self.log("[!] Refusing to silently train from step 0 a branch that asked "
                         "to start elsewhere / Me niego a entrenar desde 0 en silencio.")
                sys.exit(2)
            self.log("[!] resume_on_corrupt='restart': starting from step 0 / "
                     "empezando desde 0.")
            self.log("=" * 65)
            return RestoreResult()

        self.log(f"    source / origen: {spec.get('source') or '?'}")
        self.log("[i] Weights only — optimizer moments, RNG streams, sampler position "
                 "and EMA all start cold / sólo pesos: optimizador, RNG, posición del "
                 "sampler y EMA arrancan en frío.")
        self.log(f"Starting from step / Empezando desde el paso {start_step}...")
        self.log("=" * 65)
        return RestoreResult(start_step=start_step)

    def _load_state(self, ema: Any) -> RestoreResult:
        """Read optimizer.pt, tolerating the version-1 format still in flight."""
        state = torch.load(self.cfg.optimizer_state_file, weights_only=True)

        if not isinstance(state, dict) or "format_version" not in state:
            self.log("[i] Legacy checkpoint format / formato antiguo: "
                     "no RNG or sampler state.")
            with open(self.cfg.step_file, "r", encoding="utf-8") as handle:
                start_step = int(handle.read().strip())
            self.optimizer.load_state_dict(state)
            return RestoreResult(start_step=start_step)

        if state.get("fingerprint") != self.fingerprint:
            # Not fatal: rank/target changes make the weights not fit, which
            # load_lora_weights catches, but a changed batch size or cache is merely
            # suspect. Say so and let the operator judge.
            self.log("[!] WARNING: checkpoint fingerprint differs from current config.")
            self.log(f"    ckpt = {state.get('fingerprint')}")
            self.log(f"    now  = {self.fingerprint}")

        self.optimizer.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema"):
            ema.load_state_dict(state["ema"], log=self.log)
        return RestoreResult(start_step=int(state["step"]), pending=state)

    def apply_pending(self, pending: dict[str, Any] | None, sampler: Any) -> None:
        """Restore RNG streams and the sampler's position, once both objects exist.

        Best-effort: a mismatch here costs reproducibility, not correctness, so it warns
        and continues rather than aborting a run that is otherwise fine.
        """
        if pending is None:
            return
        try:
            if pending.get("sampler"):
                sampler.load_state_dict(pending["sampler"])
            raw = json.loads(pending["rng_python"])
            random.setstate((raw[0], tuple(raw[1]), raw[2]))
            torch.set_rng_state(pending["rng_torch"].to(torch.uint8))
            if torch.cuda.is_available() and pending.get("rng_cuda"):
                torch.cuda.set_rng_state_all(
                    [s.to(torch.uint8) for s in pending["rng_cuda"]])
            self.log(f"[OK] RNG and sampler state restored (epoch {sampler.epoch}) / "
                     f"estado RNG y del sampler restaurado.")
        except Exception as exc:
            self.log(f"[!] Could not restore RNG/sampler state: {exc} — continuing with "
                     f"a fresh stream / continuando con un flujo nuevo.")

    def load_previous_phase(self) -> None:
        """Seed the adapter from the previous progressive phase's weights.

        Only the adapter carries over. The optimizer starts fresh and the step counter
        at 0, so each phase re-warms against its own gradient scale instead of inheriting
        momentum tuned for a different resolution.
        """
        if not self.cfg.init_lora_from:
            return
        adapter = os.path.join(self.cfg.init_lora_from, ADAPTER_FILENAME)
        if not os.path.exists(adapter):
            self.log(f"[!] init_lora_from set but adapter not found / "
                     f"adapter no encontrado: {adapter}")
            sys.exit(1)

        self.log("=" * 65)
        self.log(f"Phase hand-off: loading LoRA weights from previous phase / "
                 f"Cargando LoRA de la fase previa:\n  {adapter}")
        with open(adapter, "rb") as handle:
            lora_io.load_lora_weights(self.model, handle.read(), adapter, log=self.log)
        self.log("[OK] LoRA initialized from previous phase; optimizer starts fresh / "
                 "LoRA inicializada; optimizador desde cero.")
        self.log("=" * 65)

    # ── save ────────────────────────────────────────────────────────────────
    def save(self, step: int, sampler: Any = None, ema: Any = None,
             num_images: int | None = None, reason: str = "periodic") -> None:
        """Write the full resumable state atomically, and non-reentrantly.

        Re-entrancy matters: a signal arriving mid-save would otherwise start a second
        save over the first one's staging directory.

        `reason` only reaches the checkpoint log — it says whether this was a full
        save_every span or a stretch cut short. The timing is recorded here rather
        than at the call site precisely because of the two guards below: a save that
        returned early wrote nothing, and timing it would invent an interval.
        """
        if step <= 0 or self._busy:
            return
        self._busy = True
        try:
            self.log(f"\nSaving checkpoint state at step / "
                     f"Guardando estado en paso {step}...")
            self._save_adapter()
            self._save_run_state(step, sampler, ema)

            path = os.path.join(self.cfg.output_dir,
                                f"{self._checkpoint_prefix}_step_{step}.safetensors")
            self._export(path, step, sampler, ema, num_images)

            # The commit: written last, so its presence implies everything above.
            checkpoints.atomic_write(
                self.cfg.step_file,
                lambda p: open(p, "w", encoding="utf-8").write(str(step)))
            checkpoints.rotate(self.cfg.output_dir, self.cfg.max_checkpoints_to_keep,
                               self._checkpoint_prefix, log=self.log)
            self.log(f"✓ Checkpoint saved successfully at step / "
                     f"Checkpoint guardado en paso {step}: {path}")
            if self._ckpt_log is not None:
                self._ckpt_log.record(
                    step=step,
                    epoch=sampler.epoch if sampler is not None else 0,
                    reason=reason,
                    num_images=num_images,
                )
        finally:
            self._busy = False

    def _save_adapter(self) -> None:
        """Publish the adapter through a staging directory.

        `save_pretrained` writes in place, so a process dying halfway leaves a truncated
        adapter_model.safetensors — which still loads. Staging plus per-file os.replace
        makes the swap atomic.
        """
        stage = self.cfg.resume_dir + ".stage"
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage, exist_ok=True)
        self.model.save_pretrained(stage)
        os.makedirs(self.cfg.resume_dir, exist_ok=True)
        for name in os.listdir(stage):
            os.replace(os.path.join(stage, name), os.path.join(self.cfg.resume_dir, name))
        shutil.rmtree(stage, ignore_errors=True)

    def _save_run_state(self, step: int, sampler: Any, ema: Any) -> None:
        """Write optimizer.pt: optimizer, sampler position, EMA shadow and RNG streams."""
        py_rng = random.getstate()
        state = {
            "format_version": FORMAT_VERSION,
            "step": int(step),
            "optimizer": self.optimizer.state_dict(),
            "sampler": sampler.state_dict() if sampler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
            # Python's RNG state goes as JSON and torch's as a ByteTensor, so that
            # torch.load(weights_only=True) still accepts the file.
            "rng_python": json.dumps([py_rng[0], list(py_rng[1]), py_rng[2]]),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "fingerprint": self.fingerprint,
        }
        checkpoints.atomic_write(self.cfg.optimizer_state_file,
                                 lambda p: torch.save(state, p))
        if self.cfg.run_id:
            checkpoints.atomic_write(
                self.cfg.run_id_file,
                lambda p: open(p, "w", encoding="utf-8").write(self.cfg.run_id))

    def _export(self, path: str, step: int, sampler: Any, ema: Any,
                num_images: int | None) -> None:
        """Export the per-step LoRA, smoothed when EMA is on.

        The shipped adapter is the EMA copy while `resume_checkpoint` keeps the raw
        weights — otherwise every restart would smooth an already-smoothed set.
        """
        if ema is not None:
            ema.apply()
        try:
            checkpoints.atomic_write(path, lambda p: lora_io.export_lora(
                self.model, p, self.cfg, step=step,
                epoch=sampler.epoch if sampler is not None else 0,
                num_images=num_images))
        finally:
            if ema is not None:
                ema.restore()

    def export_final(self, step: int, sampler: Any = None, ema: Any = None,
                     num_images: int | None = None) -> str:
        """Write {prefix}_FINAL.safetensors — the run's deliverable."""
        path = os.path.join(self.cfg.output_dir, f"{self._checkpoint_prefix}_FINAL.safetensors")
        self._export(path, step, sampler, ema, num_images)
        return path

    # ── signals ─────────────────────────────────────────────────────────────
    def install_signal_handlers(self, current_step: Callable[[], int],
                                on_save: Callable[[int], None]) -> None:
        """Checkpoint and exit cleanly on SIGTERM/SIGINT/SIGHUP.

        `current_step` is a callable rather than a value because the step advances after
        the handler is installed. SIGHUP matters for a run started over SSH without
        tmux — closing the terminal should save, not kill. Launched through the UI it
        never arrives, since the process gets its own session.
        """
        def handle(sig: int, frame: FrameType | None) -> None:
            self.log(f"\n[!] Signal received / Señal de detención recibida ({sig}).")
            on_save(current_step())
            # The outer SystemExit handler would otherwise save a second time.
            self.saved_on_exit = True
            sys.exit(0)

        try:
            signal.signal(signal.SIGTERM, handle)
            signal.signal(signal.SIGINT, handle)
            for name in ("SIGBREAK", "SIGHUP"):
                if hasattr(signal, name):
                    signal.signal(getattr(signal, name), handle)
        except Exception:
            # Signal registration fails off the main thread; not worth aborting a run.
            pass
