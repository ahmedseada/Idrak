# Vision and sequence layer fixtures

Written by `tools/pytorch/vision_sequence_reference.py` (torch 2.14.1 on the CPU, onnx 1.23.2 for the exports); reruns
write the same bytes. Checked by `IDRAK_FILTER="vision sequence"`.

| File | What |
|---|---|
| `reference.json` | one case a line, inputs and weights included: CTC losses (none, sum, mean; with and without `zero_infinity`; an impossible sequence; time-major and batch-first with blank 4) and the gradient of the mean loss on the logits; `Conv2d` with rectangular kernels, strides and padding, with dilation and groups, and depthwise; `MaxPool2d` and `AvgPool2d` (padding counted and not) with rectangular windows; a bidirectional two-layer `LSTM` and `GRU` (batch first, random biases) with every parameter by PyTorch's name, the output and the last states; the outputs of the two ONNX files |
| `conv.onnx` | PyTorch's export (opset 17) of Conv 3x5 stride 1x2, ReLU, MaxPool 2x1, a grouped dilated Conv 3x1, ReLU, AveragePool 2x3 |
| `recurrent.onnx` | PyTorch's export (opset 17) of a bidirectional two-layer LSTM, batch first: its zero initial states are expanded to a shape computed from the input |
