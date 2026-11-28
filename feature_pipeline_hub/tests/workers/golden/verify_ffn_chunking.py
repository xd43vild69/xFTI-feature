"""Checks krea2.chunking's sequence-split FFN against the whole FFN on real Krea 2 blocks.

`ffn_chunks` defaults to 1, so the smoke run never reaches this path:

    training_runtime/venv/bin/python tests/workers/golden/verify_ffn_chunking.py

Needs a GPU. Builds a short stack of diffusers' `Krea2TransformerBlock` at reduced
width, runs forward+backward under bf16 autocast with and without chunking — through
both standard and offloaded checkpointing, since the trainer can combine either with
it — and compares loss and gradients.

Chunking is the same function but not the same arithmetic (GEMMs over fewer rows,
weight grads summed over chunks), so the bound is a tolerance, not equality. Measured
on this setup, as worst per-tensor max|a-b| / max|a| over the parameter grads:

    standard checkpointing vs itself, non-deterministic kernels   5.3e-3
    bf16 whole FFN vs an fp32 reference                           7.9e-3
    bf16 chunked FFN vs the same fp32 reference                   8.8e-3
    chunked vs whole (this script)                                ~8e-3

So chunking moves the grads by about what bf16 and the trainer's own non-determinism
already do between two identical runs. REL_TOL leaves room above that, not below.

Under offloaded checkpointing the first pass runs the FFN whole (no grad) and only the
recompute is chunked, so the loss matches exactly while the grads carry the same small
difference: the backward is that of a function equal to the forward up to bf16 rounding.

It also checks the reason it exists: the peak VRAM of the backward must drop.

Exits non-zero on any failure.
"""
import os
import pathlib
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # feature_pipeline_hub/
sys.path.insert(0, str(ROOT / "workers"))

import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402
from diffusers.models.transformers.transformer_krea2 import Krea2TransformerBlock  # noqa: E402
from krea2 import chunking, offload  # noqa: E402

FAILURES = []

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 512, 4096, 4, 2, 128
N_BLOCKS, SEQ, CHUNKS = 4, 4096, 4
REL_TOL = 2e-2  # max |a-b| relative to max |a|, per tensor


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def standard(module, *args):
    return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=False)


def build_blocks(chunks):
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        Krea2TransformerBlock(HIDDEN, INTER, HEADS, KV_HEADS, norm_eps=1e-5)
        for _ in range(N_BLOCKS))
    for b in blocks:
        torch.nn.init.normal_(b.scale_shift_table, std=0.1)
    chunking.chunk_feed_forward(blocks, chunks)
    return blocks.to("cuda", torch.float32)


def run(blocks, ckpt):
    for p in blocks.parameters():
        p.grad = None
    g = torch.Generator(device="cuda").manual_seed(1)
    hidden = torch.randn(1, SEQ, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    hidden.requires_grad_(True)
    temb = torch.randn(1, 1, 6 * HIDDEN, device="cuda", generator=g)
    angles = torch.randn(SEQ, HEAD_DIM // 2, device="cuda", generator=g)
    rope = (torch.cos(angles).repeat_interleave(2, -1), torch.sin(angles).repeat_interleave(2, -1))
    torch.cuda.synchronize()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = hidden
        for b in blocks:
            h = ckpt(b, h, temb, rope, None)
        loss = h.float().pow(2).mean()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    loss.backward()
    torch.cuda.synchronize()
    return {"loss": loss.detach(), "hidden": hidden.grad.clone(),
            "params": {n: p.grad.clone() for n, p in blocks.named_parameters()
                       if p.grad is not None},
            "bwd_peak": torch.cuda.max_memory_allocated() - start}


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / a.float().abs().max().clamp_min(1e-30)).item()


def compare(label, ref, got):
    check(f"{label}: loss", rel(ref["loss"], got["loss"]) <= 1e-5,
          f"{ref['loss'].item():.8f} vs {got['loss'].item():.8f}")
    r = rel(ref["hidden"], got["hidden"])
    check(f"{label}: input grad", r <= REL_TOL, f"rel {r:.2e}")
    check(f"{label}: same params got grads", ref["params"].keys() == got["params"].keys())
    worst = max((rel(ref["params"][n], got["params"][n]), n) for n in ref["params"])
    check(f"{label}: param grads", worst[0] <= REL_TOL, f"worst rel {worst[0]:.2e} ({worst[1]})")
    saved = ref["bwd_peak"] - got["bwd_peak"]
    check(f"{label}: backward peak drops", saved > 0,
          f"{ref['bwd_peak'] / 2**20:.1f} -> {got['bwd_peak'] / 2**20:.1f} MiB")


if not torch.cuda.is_available():
    print("needs a GPU")
    sys.exit(1)

torch.use_deterministic_algorithms(True)
whole, split = build_blocks(1), build_blocks(CHUNKS)
check("identical starting weights",
      all(torch.equal(a, b) for a, b in zip(whole.parameters(), split.parameters())))
for label, ckpt in (("standard", standard), ("offloaded", offload.offloaded_checkpoint)):
    compare(label, run(whole, ckpt), run(split, ckpt))

# Inference must take the whole-FFN path: nothing to save, so chunking would only cost.
calls = []
_orig = torch.utils.checkpoint.checkpoint
torch.utils.checkpoint.checkpoint = lambda *a, **k: calls.append(1) or _orig(*a, **k)
with torch.no_grad():
    split[0].ff(torch.randn(1, SEQ, HIDDEN, device="cuda"))
torch.utils.checkpoint.checkpoint = _orig
check("no chunking under no_grad", not calls, f"{len(calls)} chunk checkpoints")

print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall checks passed")
sys.exit(1 if FAILURES else 0)
