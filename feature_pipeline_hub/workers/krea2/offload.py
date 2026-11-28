"""Gradient checkpointing whose saved block inputs live in pinned host RAM, not VRAM.

Why this exists rather than `save_on_cpu`: with checkpointing on, almost nothing the
backward needs goes through saved-tensor hooks. Non-reentrant `torch.utils.checkpoint`
installs its own hooks inside each block, which shadow any outer `save_on_cpu`, and it
keeps the block's *inputs* alive by plain reference so it can recompute — those never
pass through a hook at all. Those inputs are the residual stream entering each of the
28 blocks: at 1280² (~6528 tokens × 6144 wide, bf16) that is ~80 MB per block, ~2.2 GB
in all, and it stayed on the GPU with `activation_offload` on. Measured: turning the
flag on moved the OOM point by ~0.2 GB.

`offloaded_checkpoint` keeps the same contract as `torch.utils.checkpoint.checkpoint`
for the call diffusers makes (`func(block, hidden_states, *args)`): run the block
without building a graph, then recompute it with grad in the backward. The one
difference is that `hidden_states` is copied to pinned host memory on the way in and
copied back for the recompute. Copies are exact, so gradients are bit-identical to
standard checkpointing — `tests/workers/golden/verify_offload_checkpoint.py` checks it.

A second saving falls out of running the forward under `no_grad`: autocast only caches
its bf16 casts of weights that require grad while grad is enabled, and standard
checkpointing runs its forward with grad on. Here those are the fp32 LoRA weights —
161M parameters at rank 48 attn+ff, so ~0.3 GB of cached casts that are never made.
The recompute in the backward casts again, as standard checkpointing's does.

Stream ordering is what makes the non-blocking copies safe: both run on the current
stream, so the device→host copy is ordered before any kernel that could reuse the
freed VRAM, and the host→device copy before the recompute that reads it.

Not covered, deliberately: RNG state is not stashed and replayed. Krea 2's blocks have
no dropout, so the recompute is deterministic without it; a model that did would need
the RNG handling `torch.utils.checkpoint` does.

Imports torch, so this module only loads under the training runtime.
"""
from __future__ import annotations

from typing import Any, Callable

import torch


class _OffloadedCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, run_function: Callable[..., Any],  # type: ignore[override]
                hidden_states: torch.Tensor, *args: Any) -> Any:
        ctx.run_function = run_function
        ctx.device = hidden_states.device
        ctx.hidden_requires_grad = hidden_states.requires_grad
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))

        host = torch.empty(hidden_states.shape, dtype=hidden_states.dtype,
                           device="cpu", pin_memory=True)
        host.copy_(hidden_states, non_blocking=True)
        ctx.hidden_host = host

        # The other inputs (temb, rotary tables, mask) are shared by every block and
        # already alive for the whole forward, so holding a reference costs no memory.
        # Held as attributes rather than via save_for_backward so that an enclosing
        # `save_on_cpu` does not copy the same shared tensors to host once per block.
        ctx.args = args

        with torch.no_grad():
            return run_function(hidden_states, *args)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        hidden = ctx.hidden_host.to(ctx.device, non_blocking=True)
        hidden = hidden.detach().requires_grad_(ctx.hidden_requires_grad)
        ctx.hidden_host = None

        args = []
        for a in ctx.args:
            if isinstance(a, torch.Tensor):
                d = a.detach()
                d.requires_grad_(a.requires_grad)
                args.append(d)
            else:
                args.append(a)

        autocast_on, autocast_dtype = ctx.autocast
        with torch.enable_grad(), torch.amp.autocast("cuda", dtype=autocast_dtype,
                                                     enabled=autocast_on):
            outputs = ctx.run_function(hidden, *args)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        pairs = [(o, g) for o, g in zip(outputs, grad_outputs)
                 if isinstance(o, torch.Tensor) and o.requires_grad and g is not None]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])

        arg_grads = [a.grad if isinstance(a, torch.Tensor) and a.requires_grad else None
                     for a in args]
        return (None, hidden.grad if ctx.hidden_requires_grad else None, *arg_grads)


def offloaded_checkpoint(module: Callable[..., Any], hidden_states: torch.Tensor,
                         *args: Any) -> Any:
    """Drop-in for diffusers' `_gradient_checkpointing_func(block, hidden_states, ...)`."""
    return _OffloadedCheckpoint.apply(module.__call__, hidden_states, *args)
