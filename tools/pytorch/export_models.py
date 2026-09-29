"""Builds a small CNN or ResNet in PyTorch, exports it to ONNX with PyTorch's exporter and saves PyTorch's own outputs
for a few random images next to the file, so Idrak can check its import against them (on its CPU or CUDA
backend):

    pip install torch onnx onnxscript
    python tools/pytorch/export_models.py --model resnet --out resnet.onnx            (add --dynamo for torch.export)
    dotnet run -c Release --project samples/Idrak.Samples.OnnxImport -- --onnx resnet.onnx --cuda

The weights are random (no training is needed to check that two implementations compute the same function), and the
BatchNorm statistics are randomized so normalization is really exercised.
"""
import argparse
import json

import torch
from torch import nn


class BasicBlock(nn.Module):
    """The ResNet-18/34 block: two 3x3 convolutions with BatchNorm, and a skip connection (1x1 conv when the shape changes)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.shortcut(x))


def build(name):
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
    if name == "resnet":
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            BasicBlock(16, 16), BasicBlock(16, 32, stride=2), BasicBlock(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    raise ValueError(name)


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["cnn", "resnet"], default="resnet")
parser.add_argument("--out", default=None)
parser.add_argument("--dynamo", action="store_true", help="use the torch.export-based exporter")
args = parser.parse_args()
out = args.out or f"{args.model}.onnx"

torch.manual_seed(0)
model = build(args.model)
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.uniform_(0.5, 1.5)
            m.bias.uniform_(-0.2, 0.2)
            m.running_mean.uniform_(-0.3, 0.3)
            m.running_var.uniform_(0.5, 2.0)
model.eval()
x = torch.randn(4, 3, 32, 32)
with torch.no_grad():
    outputs = model(x)
print(f"PyTorch {torch.__version__}: {args.model}, {sum(p.numel() for p in model.parameters()):,} parameters, output {tuple(outputs.shape)}")

if args.dynamo:
    batch = torch.export.Dim("batch")
    torch.onnx.export(model, (x,), out, dynamo=True, input_names=["input"], output_names=["output"],
                      opset_version=18, dynamic_shapes=({0: batch},))
else:
    torch.onnx.export(model, (x,), out, dynamo=False, input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
with open(out + ".expected.json", "w") as f:
    json.dump({"inputs": x.reshape(4, -1).tolist(), "outputs": outputs.reshape(4, -1).tolist(), "inputShape": list(x.shape[1:]),
               "tolerance": 1e-4, "torch": torch.__version__, "trainedOn": "random weights",
               "exporter": "dynamo" if args.dynamo else "torchscript"}, f)
print(f"Wrote {out} and {out}.expected.json")
