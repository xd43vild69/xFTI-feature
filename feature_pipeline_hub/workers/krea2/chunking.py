"""Splitting each block's SwiGLU feed-forward along the sequence to cap its backward.

Measured at 1280² (6486 tokens), rank 48 attn+ff: with checkpointing, VRAM peaks at
the *end* of the backward — all fp32 LoRA grads exist by then (~0.7 GB) and the last
block's recompute+backward adds ~3.2 GB on top. A large share of that transient is the
FFN: every intermediate is tokens × 16384 bf16, ~212 MB at this size (gate, up, the
SiLU, their product, and the gradients of each). Holding fewer tokens at once is the
lever.

Splitting alone does nothing for that — autograd would still keep every chunk's
intermediates until the backward reaches them. Each chunk therefore runs under its own
(nested) checkpoint: only the chunk's input is kept, and its intermediates exist for
one chunk at a time, during that chunk's backward. The FFN is position-wise, so the
result is the same function; it is not bit-identical, since GEMMs over fewer rows can
pick other kernels and the weight gradients are summed over chunks in a different
order. The difference is ~8e-3 relative on the weight grads, against ~5e-3 between two
identical non-deterministic runs and ~8e-3 between bf16 and fp32 —
`tests/workers/golden/verify_ffn_chunking.py` has the measurements and the bound.

Costs: the FFN forward runs once more per step (it is recomputed inside the outer
block recompute), and bitsandbytes dequantizes each NF4 weight once per chunk.

Only active while grad is enabled. Under `krea2.offload`'s no-grad first pass, and at
inference (previews, validation), the FFN runs whole: nothing is saved there, so
chunking would only cost time.

Imports torch, so this module only loads under the training runtime.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.utils.checkpoint


@torch._dynamo.disable
def _run_chunked(whole: Any, hidden_states: torch.Tensor, chunks: int) -> torch.Tensor:
    # Kept out of torch.compile: dynamo (2.14) asserts on a nested checkpoint whose
    # function closes over PEFT's LoRA layers ("lift_tracked_freevar_to_input should not
    # be called on root SubgraphTracer"). The graph break costs the fusion of the two
    # cheap elementwise ops around the FFN; its GEMMs are bitsandbytes NF4 kernels,
    # which inductor would not generate anyway.
    return torch.cat(
        [torch.utils.checkpoint.checkpoint(whole, part, use_reentrant=False)
         for part in hidden_states.chunk(chunks, dim=1)],
        dim=1)


def chunk_feed_forward(blocks: Iterable[Any], chunks: int) -> int:
    """Patch each block's `ff.forward` to run in `chunks` sequence slices. Returns blocks patched."""
    if chunks <= 1:
        return 0
    patched = 0
    for block in blocks:
        ff = getattr(block, "ff", None)
        if ff is None:
            continue
        whole = ff.forward

        def forward(hidden_states: torch.Tensor, _whole: Any = whole) -> torch.Tensor:
            if not torch.is_grad_enabled() or hidden_states.shape[1] < chunks:
                return _whole(hidden_states)
            return _run_chunked(_whole, hidden_states, chunks)

        ff.forward = forward
        patched += 1
    return patched
