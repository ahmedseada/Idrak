"""Trains XOR in PyTorch (on the GPU when PyTorch sees one), exports it with PyTorch's ONNX exporter and saves
PyTorch's own outputs next to the file, so Idrak can check its import against them:

    pip install torch onnx            (add onnxscript for --dynamo)
    python tools/pytorch/xor_to_onnx.py --out xor.onnx
    dotnet run -c Release --project samples/Idrak.Samples.OnnxImport -- --onnx xor.onnx --cuda

--dynamo uses PyTorch's newer exporter (torch.export) instead of the classic TorchScript one.
"""
import argparse
import json

import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="xor.onnx")
parser.add_argument("--dynamo", action="store_true", help="use the torch.export-based exporter")
parser.add_argument("--steps", type=int, default=2000)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(args.steps):
    loss = nn.functional.binary_cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(x).cpu()
print(f"PyTorch {torch.__version__} on {device}: loss {loss.item():.6f}")
for inp, out in zip(x.cpu().tolist(), outputs.tolist()):
    print(f"  {inp} -> {out[0]:.6f}")

model = model.cpu()
if args.dynamo:
    # The torch.export-based exporter (PyTorch's default since 2.9) targets opset 18 and takes dynamic_shapes.
    batch = torch.export.Dim("batch")
    torch.onnx.export(model, (x.cpu(),), args.out, dynamo=True, input_names=["input"], output_names=["output"],
                      opset_version=18, dynamic_shapes=({0: batch},))
else:
    torch.onnx.export(model, (x.cpu(),), args.out, dynamo=False, input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
with open(args.out + ".expected.json", "w") as f:
    json.dump({"inputs": x.cpu().tolist(), "outputs": outputs.tolist(), "torch": torch.__version__,
               "trainedOn": device, "exporter": "dynamo" if args.dynamo else "torchscript"}, f, indent=2)
print(f"Wrote {args.out} and {args.out}.expected.json")
