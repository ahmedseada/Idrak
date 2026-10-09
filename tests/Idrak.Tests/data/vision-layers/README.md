# Image layer fixtures

Written by `tools/pytorch/vision_layers_reference.py` (torch 2.14.1 on the CPU, onnx 1.23.2 for the ONNX files); reruns
write the same bytes. Checked by `IDRAK_FILTER="vision layers"`.

| File | What |
|---|---|
| `reference.json` | one case a line, inputs and weights included, with PyTorch's outputs and, for random upstream weights w, the gradients of sum(output · w) on the input and the parameters: `ConvTranspose2d` (rectangular, strided, padded, output padding, dilation, groups; a square kernel-4 stride-2 decoder step); `F.interpolate` nearest and bilinear (corners aligned or not; scale factors 2, 1.5 and (1.7, 2.3); sizes up and down); `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d` with uneven windows; `GroupNorm` with random affine parameters, and without affine over [N, C, L]; ceil-mode `MaxPool2d` and `AvgPool2d` (padding counted and not); the outputs of the two ONNX files |
| `decoder.onnx` | PyTorch's export (opset 17) of Conv, GroupNorm (as Reshape, InstanceNormalization, Reshape, Mul, Add), ReLU, ConvTranspose with output padding, nearest Resize by 2, bilinear Resize to a size computed from the input's shape (corners aligned), ceil-mode MaxPool and AveragePool, GlobalAveragePool |
| `windows.onnx` | written with `onnx.helper` (opset 21), run by `onnx.reference`: MaxPool and AveragePool with more padding below and right than above and left, GroupNormalization with per-channel scale and bias, a linear Resize to constant sizes |
