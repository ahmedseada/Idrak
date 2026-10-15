"""Writes tests/Idrak.Tests/data/pytorch-checkpoints: small PyTorch checkpoints (torch.save's zip format) and the same
tensors as safetensors, so Idrak's PyTorchCheckpointReader (src/Idrak/Models/PyTorchCheckpoint.cs) can be checked
against PyTorch's own files without running any pickle.

Run: python tools/pytorch/checkpoint_reference.py   (torch 2.14.1 CPU, safetensors 0.8). Reruns write the same tensors.
Checked by `IDRAK_FILTER="pytorch checkpoint"`.

  state.pth        an OrderedDict state dict: float32, float16, bfloat16, float64, int64, int8, bool tensors, a 0-d
                   scalar, an nn.Parameter, a transposed (strided) view, two tensors sharing one storage at an offset
  state-p4.pth     the same, saved with pickle protocol 4 (FRAME, MEMOIZE, SHORT_BINUNICODE, STACK_GLOBAL)
  wrapped.pth      {"model": the state dict, "optimizer": {...}, "iteration": 5, "lr": 0.1} as a training script saves it
  nested.pth       {"encoder": {"w": ...}, "layers": [t0, t1]}: names joined by '.'
  legacy.pth       the pre-1.6 (non-zip) format, which is refused
  evil.pth         a zip whose data.pkl would call os.system: refused, never run
  reference.safetensors   every tensor of state.pth as contiguous float32 (bool and ints as their values)
"""
import io
import os
import pickle
import zipfile
from collections import OrderedDict

import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "Idrak.Tests", "data", "pytorch-checkpoints")


def state():
    g = torch.Generator().manual_seed(2032)
    base = torch.randn(6, 5, generator=g)
    shared = torch.randn(20, generator=g)
    s = OrderedDict()
    s["dense.weight"] = torch.randn(4, 3, generator=g)
    s["dense.bias"] = torch.randn(4, generator=g)
    s["half"] = torch.randn(3, 2, generator=g).half()
    s["bf16"] = torch.randn(2, 5, generator=g).bfloat16()
    s["double"] = torch.randn(5, generator=g).double()
    s["steps"] = torch.arange(-3, 4, dtype=torch.int64)
    s["small"] = torch.tensor([-128, -1, 0, 7, 127], dtype=torch.int8)
    s["mask"] = torch.tensor([True, False, True])
    s["scalar"] = torch.tensor(2.5)
    s["param"] = torch.nn.Parameter(torch.randn(2, 2, generator=g))
    s["transposed"] = base.t()                                                   # strides (1, 5): not contiguous
    s["shared.first"] = shared[2:8].view(2, 3)                                   # one storage, offset 2
    s["shared.second"] = shared[10:14]                                           # the same storage, offset 10
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    s = state()
    torch.save(s, os.path.join(OUT, "state.pth"))
    torch.save(s, os.path.join(OUT, "state-p4.pth"), pickle_protocol=4)
    torch.save({"model": s, "optimizer": {"state": {}, "param_groups": [{"lr": 0.1, "betas": (0.9, 0.99)}]}, "iteration": 5, "lr": 0.1},
               os.path.join(OUT, "wrapped.pth"))
    torch.save({"encoder": {"w": s["dense.weight"]}, "layers": [s["dense.bias"], s["double"]]}, os.path.join(OUT, "nested.pth"))
    torch.save({"w": s["dense.weight"]}, os.path.join(OUT, "legacy.pth"), _use_new_zipfile_serialization=False)

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo this must never run",))

    with zipfile.ZipFile(os.path.join(OUT, "evil.pth"), "w", zipfile.ZIP_STORED) as z:
        z.writestr("evil/data.pkl", pickle.dumps({"w": Evil()}, protocol=2))
        z.writestr("evil/version", "3\n")
        z.writestr("evil/byteorder", "little")

    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in s.items()}, os.path.join(OUT, "reference.safetensors"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
