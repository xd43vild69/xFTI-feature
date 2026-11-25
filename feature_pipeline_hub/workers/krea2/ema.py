"""Exponential moving average of the trainable LoRA weights.

Averaging late-training weights smooths out the per-step noise that a small batch and a
low LR leave behind, so the shipped adapter is steadier than any single step. The shadow
is kept in fp32 regardless of the adapter's own dtype: accumulating an average in bf16
would lose exactly the small updates the average exists to capture.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch

Logger = Callable[[str], None]


class EMA:
    """Shadow copy of the trainable weights, updated as a decaying average.

    Wrap an export in `apply()` / `restore()` — the shipped LoRA is the smoothed copy
    while the resume checkpoint keeps the raw weights, so restarts never compound the
    smoothing.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], decay: float = 0.99,
                 device: str | torch.device = "cpu", start_update: int = 0) -> None:
        self.decay = float(decay)
        self.params = list(params)
        self.device = torch.device(device)
        self.start_update = max(0, int(start_update))
        self.shadow = [p.detach().to(self.device, torch.float32).clone()
                       for p in self.params]
        self.backup: list[torch.Tensor] | None = None
        self.updates = 0
        self._seen_updates = 0

    @torch.no_grad()
    def update(self) -> None:
        """Fold the current weights into the shadow. Call once per *optimizer* update.

        Calling it per micro-batch instead would make the effective decay
        `decay ** grad_accum_steps` — a silent, and badly wrong, horizon.

        When `start_update > 0` (e.g. set to `warmup_updates`), `shadow` mirrors
        live weights until `start_update` is reached so step-0 zeros (`lora_B = 0`)
        never pollute the moving average. When shadow and live parameters share the
        same device and float32 dtype, uses fused `_foreach` kernels.
        """
        self._seen_updates += 1
        if self._seen_updates <= self.start_update:
            for shadow, param in zip(self.shadow, self.params):
                shadow.copy_(param.detach().to(self.device, torch.float32))
            return

        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        if (
            self.shadow
            and self.params[0].device == self.device
            and self.params[0].dtype == torch.float32
        ):
            live = [p.detach() for p in self.params]
            torch._foreach_mul_(self.shadow, decay)
            torch._foreach_add_(self.shadow, live, alpha=1.0 - decay)
            return
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(decay).add_(param.detach().to(self.device, torch.float32),
                                    alpha=1.0 - decay)

    @torch.no_grad()
    def apply(self) -> None:
        """Install the EMA weights into the model, stashing the live ones for `restore`."""
        self.backup = [p.detach().clone() for p in self.params]
        for shadow, param in zip(self.shadow, self.params):
            param.copy_(shadow.to(param.device, param.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        """Put the live weights back. Safe to call when `apply` never ran."""
        if self.backup is None:
            return
        for backup, param in zip(self.backup, self.params):
            param.copy_(backup)
        self.backup = None

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "updates": self.updates,
                "seen_updates": self._seen_updates,
                "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, state: Mapping[str, Any], log: Logger = print) -> None:
        """Restore the shadow, keeping the current weights if the saved shape disagrees.

        A size mismatch means the adapter's shape changed between runs, so the saved
        average describes different parameters and must be dropped rather than
        partially applied.
        """
        self.updates = int(state.get("updates", 0))
        self._seen_updates = int(state.get("seen_updates", self.updates + self.start_update))
        loaded = state.get("shadow") or []
        if len(loaded) != len(self.shadow):
            log("[!] EMA state size mismatch; reinitializing from current weights / "
                "tamaño de estado EMA distinto; reinicializando.")
            return
        for dst, src in zip(self.shadow, loaded):
            dst.copy_(src.to(dst.device, dst.dtype))


def horizon_warning(decay: float, total_updates: float) -> str | None:
    """Warn when the decay implies an averaging window longer than the run itself.

    At decay d the average spans roughly 1/(1-d) updates. If that exceeds a third of the
    run, the EMA never leaves its initialization and the exported LoRA is closer to the
    starting weights than to what was trained.
    """
    span = 1.0 / max(1e-9, 1.0 - decay)
    if span <= total_updates / 3.0:
        return None
    return (f"[!] ema_decay {decay} implies a ~{span:.0f}-update horizon but this run is "
            f"only {total_updates:.0f} updates: the EMA will barely leave its "
            f"initialization / el EMA apenas saldrá de su inicialización.")
