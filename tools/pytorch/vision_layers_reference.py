"""Writes tests/Idrak.Tests/data/vision-layers: PyTorch's results for the layers modern image backbones and decoders use
(plan 13, step 2: transposed convolution, upsampling, adaptive pooling, group norm, ceil-mode pooling), so Idrak can check
its own against them.

Run: python tools/pytorch/vision_layers_reference.py   (torch 2.14.1 CPU; onnx 1.23.2 for the .onnx files).
Reruns write the same bytes. Checked by `IDRAK_FILTER="vision layers"`.

reference.json holds, for each case, its inputs and weights (drawn from torch.manual_seed(0) and written out, so the
test does not reproduce PyTorch's random numbers), PyTorch's outputs and, for "upstream" weights w, the gradients of
sum(output · w) with respect to the input and the parameters:

  convt_*        nn.ConvTranspose2d: rectangular, strided, padded, with output padding, dilation and groups; a square
                 decoder step (kernel 4, stride 2, padding 1)
  interp_*       F.interpolate: nearest and bilinear (corners aligned or not), up and down, by scale factors (2, 1.5,
                 (1.7, 2.3)) and to sizes
  adaptive_*     nn.AdaptiveAvgPool2d / nn.AdaptiveMaxPool2d with uneven, overlapping windows
  groupnorm*     nn.GroupNorm with random affine parameters over [N, C, H, W], and without affine over [N, C, L]
  maxpool_ceil, avgpool_ceil*   ceil_mode pooling (padding counted and not)

decoder.onnx is PyTorch's export (torch.onnx.export, dynamo=False, opset 17) of a small decoder: Conv, GroupNorm, ReLU,
ConvTranspose, nearest upsampling by 2, bilinear upsampling to a size (corners aligned), ceil-mode MaxPool and AveragePool,
AdaptiveAvgPool2d(1); its output for the stored input is "onnx_decoder". windows.onnx is written with onnx.helper (opset
21): MaxPool and AveragePool with more padding below and right than above and left, GroupNormalization with per-channel
scale and bias, and a Resize to constant sizes; its output, from onnx.reference.ReferenceEvaluator, is "onnx_windows".
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "Idrak.Tests", "data", "vision-layers")


def values(t):
    # The shortest text that reads back as the same float32, so the file stays small and exact.
    return [float(str(np.float32(v))) for v in t.detach().reshape(-1).tolist()]


def tensor(t):
    return {"shape": list(t.shape), "values": values(t)}


def with_gradients(run, inp, parameters=None):
    """Runs `run` on a copy of `inp` that needs gradients; returns the output, the upstream weights and the gradients
    (of the input and of each named parameter)."""
    parameters = parameters or {}
    x = inp.clone().requires_grad_(True)
    out = run(x)
    w = torch.randn(out.shape)
    for p in parameters.values():
        p.grad = None
    (out * w).sum().backward()
    case = {"input": tensor(inp), "output": tensor(out), "upstream": tensor(w), "grad_input": tensor(x.grad)}
    for name, p in parameters.items():
        case["grad_" + name] = tensor(p.grad)
    return case


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    reference = {"torch": torch.__version__}

    # ---- transposed convolutions
    def convt_case(cin, cout, kernel, stride, padding, output_padding, dilation, groups, shape):
        conv = nn.ConvTranspose2d(cin, cout, kernel, stride, padding, output_padding, groups, True, dilation)
        with torch.no_grad():
            conv.bias.uniform_(-0.5, 0.5)
        case = {"in": cin, "out": cout, "kernel": list(conv.kernel_size), "stride": list(conv.stride), "padding": list(conv.padding),
                "output_padding": list(conv.output_padding), "dilation": list(conv.dilation), "groups": groups,
                "weight": tensor(conv.weight), "bias": tensor(conv.bias)}
        case.update(with_gradients(conv, torch.randn(*shape), {"weight": conv.weight, "bias": conv.bias}))
        return case

    reference["convt_rect"] = convt_case(4, 6, (3, 2), (2, 1), (1, 0), (1, 0), (1, 2), 2, (1, 4, 4, 5))
    reference["convt_square"] = convt_case(3, 2, 4, 2, 1, 0, 1, 1, (1, 3, 4, 5))

    # ---- interpolation
    def interp_case(shape, **kwargs):
        case = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()}
        case.update(with_gradients(lambda x: F.interpolate(x, **kwargs), torch.randn(*shape)))
        return case

    reference["interp_nearest_2"] = interp_case((1, 2, 4, 5), scale_factor=2.0, mode="nearest")
    reference["interp_nearest_1_5"] = interp_case((1, 2, 5, 6), scale_factor=1.5, mode="nearest")
    reference["interp_nearest_size"] = interp_case((1, 2, 5, 6), size=(7, 9), mode="nearest")
    reference["interp_nearest_down"] = interp_case((1, 2, 7, 9), size=(3, 4), mode="nearest")
    reference["interp_bilinear_2"] = interp_case((1, 2, 4, 5), scale_factor=2.0, mode="bilinear", align_corners=False)
    reference["interp_bilinear_pair"] = interp_case((1, 2, 5, 6), scale_factor=(1.7, 2.3), mode="bilinear", align_corners=False)
    reference["interp_bilinear_aligned"] = interp_case((1, 2, 5, 6), size=(7, 9), mode="bilinear", align_corners=True)
    reference["interp_bilinear_down"] = interp_case((1, 2, 7, 9), size=(3, 4), mode="bilinear", align_corners=False)

    # ---- adaptive pooling (uneven, overlapping windows)
    for name, module, shape in [("adaptive_avg", nn.AdaptiveAvgPool2d((3, 4)), (1, 2, 7, 9)), ("adaptive_avg_small", nn.AdaptiveAvgPool2d((2, 3)), (1, 2, 5, 5)),
                                ("adaptive_max", nn.AdaptiveMaxPool2d((3, 4)), (1, 2, 7, 9)), ("adaptive_max_small", nn.AdaptiveMaxPool2d((2, 3)), (1, 2, 5, 5))]:
        case = {"size": list(module.output_size)}
        case.update(with_gradients(module, torch.randn(*shape)))
        reference[name] = case

    # ---- group norm
    norm = nn.GroupNorm(3, 6)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.5, 0.5)
    case = {"groups": 3, "channels": 6, "eps": norm.eps, "weight": tensor(norm.weight), "bias": tensor(norm.bias)}
    case.update(with_gradients(norm, torch.randn(2, 6, 4, 5) * 2 + 0.5, {"weight": norm.weight, "bias": norm.bias}))
    reference["groupnorm"] = case
    plain = nn.GroupNorm(2, 4, eps=1e-3, affine=False)
    case = {"groups": 2, "channels": 4, "eps": plain.eps}
    case.update(with_gradients(plain, torch.randn(3, 4, 7)))
    reference["groupnorm_plain"] = case

    # ---- ceil-mode pooling
    case = {"kernel": [3, 3], "stride": [2, 2], "padding": [1, 1]}
    case.update(with_gradients(nn.MaxPool2d(3, 2, 1, ceil_mode=True), torch.randn(1, 2, 8, 9)))
    reference["maxpool_ceil"] = case
    for name, include in [("avgpool_ceil", True), ("avgpool_ceil_nopad", False)]:
        case = {"kernel": [3, 2], "stride": [2, 2], "padding": [1, 1], "count_include_pad": include}
        case.update(with_gradients(nn.AvgPool2d((3, 2), (2, 2), (1, 1), ceil_mode=True, count_include_pad=include), torch.randn(1, 2, 8, 7)))
        reference[name] = case

    # ---- ONNX: PyTorch's export of a small decoder
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 4, 3, padding=1)
            self.norm = nn.GroupNorm(2, 4)
            self.up = nn.ConvTranspose2d(4, 3, (3, 2), (2, 2), (1, 0), (1, 1))
            self.pool = nn.MaxPool2d(3, 2, 1, ceil_mode=True)
            self.average = nn.AvgPool2d(2, 2, 1, ceil_mode=True)
            self.head = nn.AdaptiveAvgPool2d(1)
            with torch.no_grad():
                self.norm.weight.uniform_(0.5, 1.5)
                self.norm.bias.uniform_(-0.5, 0.5)

        def forward(self, x):
            x = self.up(torch.relu(self.norm(self.conv(x))))
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = F.interpolate(x, size=(17, 15), mode="bilinear", align_corners=True)
            return self.head(self.average(self.pool(x)))

    decoder = Decoder().eval()
    decoder_in = torch.randn(2, 2, 5, 6)
    torch.onnx.export(decoder, (decoder_in,), os.path.join(OUT, "decoder.onnx"), dynamo=False, opset_version=17, input_names=["input"], output_names=["output"])
    reference["onnx_decoder"] = {"input": tensor(decoder_in), "output": tensor(decoder(decoder_in))}

    # ---- ONNX: windows PyTorch does not export (padding below and right, GroupNormalization), run by onnx's reference
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    scale = torch.rand(4) + 0.5
    bias = torch.rand(4) - 0.5
    nodes = [
        helper.make_node("MaxPool", ["input"], ["pooled"], kernel_shape=[3, 2], strides=[2, 2], pads=[0, 1, 1, 1]),
        helper.make_node("AveragePool", ["pooled"], ["averaged"], kernel_shape=[2, 2], strides=[1, 2], pads=[1, 0, 0, 1], count_include_pad=0),
        helper.make_node("GroupNormalization", ["averaged", "scale", "bias"], ["normalized"], num_groups=2, epsilon=1e-5),
        helper.make_node("Resize", ["normalized", "", "", "sizes"], ["output"], mode="linear", coordinate_transformation_mode="half_pixel"),
    ]
    graph = helper.make_graph(nodes, "windows", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4, 9, 8])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 4, 7, 6])],
                              [numpy_helper.from_array(scale.numpy(), "scale"), numpy_helper.from_array(bias.numpy(), "bias"),
                               numpy_helper.from_array(np.array([2, 4, 7, 6], dtype=np.int64), "sizes")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], producer_name="vision_layers_reference")
    model.ir_version = 10
    onnx.checker.check_model(model)
    windows_in = torch.randn(2, 4, 9, 8)
    with open(os.path.join(OUT, "windows.onnx"), "wb") as f:
        f.write(model.SerializeToString())
    windows_out = ReferenceEvaluator(model).run(None, {"input": windows_in.numpy()})[0]
    reference["onnx_windows"] = {"input": tensor(windows_in), "output": tensor(torch.from_numpy(windows_out))}

    # One case a line.
    with open(os.path.join(OUT, "reference.json"), "w", newline="\n") as f:
        f.write("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(reference[k], sort_keys=True, separators=(',', ':'))}" for k in sorted(reference)) + "\n}\n")
    print(f"PyTorch {torch.__version__}: wrote {OUT}")


if __name__ == "__main__":
    main()
