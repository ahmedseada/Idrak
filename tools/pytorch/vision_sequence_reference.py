"""Writes tests/Idrak.Tests/data/vision-sequence: PyTorch's results for the layers and losses a convolutional and
recurrent sequence reader (a text-line recognizer) is built from, so Idrak can check its own against them.

Run: python tools/pytorch/vision_sequence_reference.py   (torch 2.14.1 CPU; onnx 1.23.2 for the two .onnx files).
Reruns write the same bytes. Checked by `IDRAK_FILTER="vision sequence"`.

reference.json holds, for each case, its inputs and weights (drawn from torch.manual_seed(0) and written out, so the
test does not reproduce PyTorch's random numbers) and PyTorch's outputs:

  ctc            F.ctc_loss on log_softmax(logits) [T, N, C] with padded targets: the losses with reduction none, sum
                 and mean, with and without zero_infinity (one sequence is impossible), and d(mean loss)/d(logits)
  ctc_batch_first the same losses for [N, T, C] logits transposed to [T, N, C] (PyTorch takes time-major only)
  conv_*         nn.Conv2d with rectangular kernels, strides, padding, dilation and groups (depthwise among them)
  maxpool, avgpool, avgpool_nopad   rectangular windows and strides; AvgPool2d with count_include_pad on and off
  lstm, gru      nn.LSTM / nn.GRU (batch_first, bidirectional, stacked) with random biases (the GRU's b_hn too):
                 every parameter by PyTorch's name, the output and the last hidden states

conv.onnx and recurrent.onnx are the PyTorch exports (torch.onnx.export, dynamo=False, opset 17) of a small convolutional
stack and of a bidirectional two-layer LSTM, whose outputs for the stored inputs are in reference.json ("onnx_conv",
"onnx_recurrent").
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "Idrak.Tests", "data", "vision-sequence")


def values(t):
    # The shortest text that reads back as the same float32, so the file stays small and exact; an infinite loss as the
    # string "Infinity" (JSON has no infinity).
    return [float(str(np.float32(v))) if np.isfinite(v) else ("Infinity" if v > 0 else "-Infinity") for v in t.detach().reshape(-1).tolist()]


def tensor(t):
    return {"shape": list(t.shape), "values": values(t)}


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    reference = {"torch": torch.__version__}

    # ---- CTC: three sequences, one of them impossible (5 labels with a repeat need 6 steps; it has 5).
    T, N, C = 12, 4, 6
    logits = torch.randn(T, N, C, dtype=torch.float32) * 2
    targets = torch.tensor([[1, 2, 2, 3, 1], [4, 5, 1, 0, 0], [0, 0, 0, 0, 0], [3, 3, 2, 3, 3]], dtype=torch.long)
    input_lengths = torch.tensor([12, 9, 4, 5])
    target_lengths = torch.tensor([5, 3, 0, 5])
    log_probs = logits.log_softmax(2)
    case = {"logits": tensor(logits), "targets": tensor(targets.float()), "input_lengths": input_lengths.tolist(),
            "target_lengths": target_lengths.tolist(), "blank": 0}
    for zero in (False, True):
        key = "zero_infinity" if zero else "plain"
        case[key] = {r: values(F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction=r, zero_infinity=zero).reshape(-1))
                     for r in ("none", "sum", "mean")}
    x = logits.clone().requires_grad_(True)
    F.ctc_loss(x.log_softmax(2), targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=True).backward()
    case["zero_infinity"]["grad_logits_mean"] = values(x.grad)
    reference["ctc"] = case

    # Batch-first logits and a blank other than 0.
    logits_bf = torch.randn(3, 7, 5) * 1.5
    targets_bf = torch.tensor([[0, 1, 0], [3, 3, 0], [2, 0, 0]])                      # blank 4
    lengths_bf, tlengths_bf = torch.tensor([7, 6, 3]), torch.tensor([3, 2, 1])
    lp_bf = logits_bf.log_softmax(2).transpose(0, 1)
    reference["ctc_batch_first"] = {
        "logits": tensor(logits_bf), "targets": tensor(targets_bf.float()), "input_lengths": lengths_bf.tolist(), "target_lengths": tlengths_bf.tolist(),
        "blank": 4, "none": values(F.ctc_loss(lp_bf, targets_bf, lengths_bf, tlengths_bf, blank=4, reduction="none")),
    }

    # ---- convolutions
    def conv_case(cin, cout, kernel, stride, padding, dilation, groups, shape):
        conv = nn.Conv2d(cin, cout, kernel, stride, padding, dilation, groups, bias=True)
        with torch.no_grad():
            conv.bias.uniform_(-0.5, 0.5)
        inp = torch.randn(*shape)
        return {"in": cin, "out": cout, "kernel": list(conv.kernel_size), "stride": list(conv.stride), "padding": list(conv.padding),
                "dilation": list(conv.dilation), "groups": groups, "weight": tensor(conv.weight), "bias": tensor(conv.bias),
                "input": tensor(inp), "output": tensor(conv(inp))}

    reference["conv_rect"] = conv_case(3, 4, (3, 5), (2, 1), (1, 2), (1, 1), 1, (2, 3, 9, 11))
    reference["conv_dilated_grouped"] = conv_case(4, 6, (3, 2), (1, 2), (2, 1), (2, 3), 2, (2, 4, 10, 13))
    reference["conv_depthwise"] = conv_case(3, 3, (3, 3), (1, 1), (1, 1), (1, 1), 3, (1, 3, 6, 7))

    # ---- pooling
    pool_in = torch.randn(2, 3, 9, 10)
    reference["maxpool"] = {"kernel": [3, 2], "stride": [2, 1], "padding": [1, 0], "input": tensor(pool_in),
                            "output": tensor(nn.MaxPool2d((3, 2), (2, 1), (1, 0))(pool_in))}
    reference["avgpool"] = {"kernel": [2, 3], "stride": [2, 1], "padding": [1, 1], "count_include_pad": True, "input": tensor(pool_in),
                            "output": tensor(nn.AvgPool2d((2, 3), (2, 1), (1, 1), count_include_pad=True)(pool_in))}
    reference["avgpool_nopad"] = {"kernel": [2, 3], "stride": [2, 1], "padding": [1, 1], "count_include_pad": False, "input": tensor(pool_in),
                                  "output": tensor(nn.AvgPool2d((2, 3), (2, 1), (1, 1), count_include_pad=False)(pool_in))}

    # ---- recurrent layers (random biases, so b_ih and b_hh both count)
    def rnn_case(module, inp):
        with torch.no_grad():
            for name, p in module.named_parameters():
                if name.startswith("bias"):
                    p.uniform_(-0.5, 0.5)
        out, state = module(inp)
        h_n = state[0] if isinstance(state, tuple) else state
        return {"input_size": module.input_size, "hidden_size": module.hidden_size, "layers": module.num_layers, "bidirectional": module.bidirectional,
                "parameters": {name: tensor(p) for name, p in module.named_parameters()}, "input": tensor(inp), "output": tensor(out), "h_n": tensor(h_n)}

    reference["lstm"] = rnn_case(nn.LSTM(3, 4, num_layers=2, batch_first=True, bidirectional=True), torch.randn(2, 5, 3))
    reference["gru"] = rnn_case(nn.GRU(3, 4, num_layers=2, batch_first=True, bidirectional=True), torch.randn(2, 5, 3))

    # ---- ONNX exports
    convs = nn.Sequential(
        nn.Conv2d(1, 4, (3, 5), (1, 2), (1, 2)), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
        nn.Conv2d(4, 4, (3, 1), padding=(2, 0), dilation=(2, 1), groups=2), nn.ReLU(), nn.AvgPool2d((2, 3), (2, 1), (1, 1)))
    conv_in = torch.randn(2, 1, 8, 12)
    torch.onnx.export(convs, (conv_in,), os.path.join(OUT, "conv.onnx"), dynamo=False, opset_version=17, input_names=["input"], output_names=["output"])
    reference["onnx_conv"] = {"input": tensor(conv_in), "output": tensor(convs(conv_in))}

    class Recurrent(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(3, 4, num_layers=2, batch_first=True, bidirectional=True)

        def forward(self, x):
            return self.rnn(x)[0]

    recurrent = Recurrent()
    rec_in = torch.randn(1, 6, 3)
    torch.onnx.export(recurrent, (rec_in,), os.path.join(OUT, "recurrent.onnx"), dynamo=False, opset_version=17, input_names=["input"], output_names=["output"])
    reference["onnx_recurrent"] = {"input": tensor(rec_in), "output": tensor(recurrent(rec_in))}

    # One case a line.
    with open(os.path.join(OUT, "reference.json"), "w", newline="\n") as f:
        f.write("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(reference[k], sort_keys=True, separators=(',', ':'))}" for k in sorted(reference)) + "\n}\n")
    print(f"PyTorch {torch.__version__}: wrote {OUT}")


if __name__ == "__main__":
    main()
