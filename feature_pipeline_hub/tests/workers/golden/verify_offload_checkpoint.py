"""Checks krea2.offload's checkpointing against torch.utils.checkpoint on real Krea 2 blocks.

`offloaded_checkpoint` replaces the checkpoint function only when `activation_offload`
is on, so the smoke run (which leaves it off) never reaches it. This covers that path:

    training_runtime/venv/bin/python tests/workers/golden/verify_offload_checkpoint.py

Needs a GPU (pinned memory and the CUDA autocast state are the point). Builds a short
stack of diffusers' own `Krea2TransformerBlock` at reduced width, runs forward+backward
through both checkpoint functions from identical weights and inputs, under the bf16
autocast the trainer uses, and requires the loss, the input gradient, the temb gradient
and every parameter gradient to be **bit-identical** in eager mode. The copies to host
and back are exact, so anything short of identical is a bug, not noise.

Bit-identity needs deterministic kernels, which this script turns on: without them the
attention backward's atomics make standard checkpointing differ from *itself* by ~5e-7
between two runs, and that noise would hide a real divergence.

It also checks the reason the module exists: the VRAM still held when the forward ends
(what the backward will need) must drop by at least most of the block inputs.

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
from krea2 import offload  # noqa: E402

FAILURES = []

HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM = 512, 1536, 4, 2, 128
N_BLOCKS, SEQ = 6, 2048


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def standard(module, *args):
    return torch.utils.checkpoint.checkpoint(module.__call__, *args, use_reentrant=False)


def build_blocks():
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        Krea2TransformerBlock(HIDDEN, INTER, HEADS, KV_HEADS, norm_eps=1e-5)
        for _ in range(N_BLOCKS))
    # scale_shift_table starts at zero; randomize so modulation is exercised.
    for b in blocks:
        torch.nn.init.normal_(b.scale_shift_table, std=0.1)
    return blocks.to("cuda", torch.float32)


def make_inputs(temb_grad):
    g = torch.Generator(device="cuda").manual_seed(1)
    hidden = torch.randn(1, SEQ, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    hidden.requires_grad_(True)
    temb = torch.randn(1, 1, 6 * HIDDEN, device="cuda", generator=g)
    temb.requires_grad_(temb_grad)
    half = HEAD_DIM // 2
    angles = torch.randn(SEQ, half, device="cuda", generator=g)
    rope = (torch.cos(angles).repeat_interleave(2, -1), torch.sin(angles).repeat_interleave(2, -1))
    return hidden, temb, rope


def run(blocks, ckpt, temb_grad):
    for p in blocks.parameters():
        p.grad = None
    hidden, temb, rope = make_inputs(temb_grad)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = hidden
        for b in blocks:
            h = ckpt(b, h, temb, rope, None)
        loss = h.float().pow(2).mean()
    torch.cuda.synchronize()
    # Measured before backward: the peak itself lands late in the backward, once the
    # fp32 parameter grads exist and most block inputs are already released, so it
    # says little about what checkpointing retained.
    retained = torch.cuda.memory_allocated() - base
    loss.backward()
    torch.cuda.synchronize()
    grads = {n: p.grad.clone() for n, p in blocks.named_parameters() if p.grad is not None}
    return {"loss": loss.detach(), "hidden_grad": hidden.grad.clone(),
            "temb_grad": None if temb.grad is None else temb.grad.clone(),
            "params": grads, "retained": retained}


def identical(a, b):
    return a is None and b is None or (
        a is not None and b is not None and a.dtype == b.dtype and torch.equal(a, b))


def compare(label, ref, got):
    check(f"{label}: loss", identical(ref["loss"], got["loss"]),
          f"{ref['loss'].item():.8f} vs {got['loss'].item():.8f}")
    check(f"{label}: input grad", identical(ref["hidden_grad"], got["hidden_grad"]))
    check(f"{label}: temb grad", identical(ref["temb_grad"], got["temb_grad"]))
    check(f"{label}: same params got grads", ref["params"].keys() == got["params"].keys(),
          f"{len(got['params'])} params")
    bad = [n for n in ref["params"] if not identical(ref["params"][n], got["params"].get(n))]
    check(f"{label}: param grads bit-identical", not bad, ", ".join(bad[:3]))


if not torch.cuda.is_available():
    print("needs a GPU")
    sys.exit(1)

torch.use_deterministic_algorithms(True)
blocks = build_blocks()
for temb_grad in (False, True):
    label = f"temb.requires_grad={temb_grad}"
    ref = run(blocks, standard, temb_grad)
    got = run(blocks, offload.offloaded_checkpoint, temb_grad)
    compare(label, ref, got)
    if not temb_grad:
        # Each block input is SEQ x HIDDEN bf16. The first is `hidden` itself, which the
        # caller holds either way; offload should move the other N-1 off the device.
        moved = (N_BLOCKS - 1) * SEQ * HIDDEN * 2
        saved = ref["retained"] - got["retained"]
        check("VRAM retained after forward drops by the block inputs", saved >= 0.9 * moved,
              f"standard {ref['retained'] / 2**20:.1f} MiB, offload "
              f"{got['retained'] / 2**20:.1f} MiB, saved {saved / 2**20:.1f} of "
              f"{moved / 2**20:.1f} MiB")

# Selective checkpointing mixes plain and offloaded blocks in one graph.
def mixed(module, *args):
    idx = next(i for i, b in enumerate(blocks) if b is module)
    return module(*args) if idx % 2 else offload.offloaded_checkpoint(module, *args)


def mixed_ref(module, *args):
    idx = next(i for i, b in enumerate(blocks) if b is module)
    return module(*args) if idx % 2 else standard(module, *args)


compare("interval=2", run(blocks, mixed_ref, False), run(blocks, mixed, False))

print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall checks passed")
sys.exit(1 if FAILURES else 0)
