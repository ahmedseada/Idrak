// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Long short-term memory over [batch, time, features]. Gates: input, forget, cell, output (PyTorch's order). The
/// forget-gate bias starts at 1 so the layer remembers by default, which helps long sequences train. Optionally
/// bidirectional and stacked (<see cref="RecurrentModule"/>).
/// </summary>
public sealed class LSTM : RecurrentModule
{
    /// <summary>Creates a one-direction, one-layer LSTM.</summary>
    /// <param name="inputSize">Features per step.</param>
    /// <param name="hiddenSize">Hidden/cell state size.</param>
    /// <param name="returnSequences">Output every step ([N, T, H]) or just the last ([N, H]).</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public LSTM(int inputSize, int hiddenSize, bool returnSequences = false, Device? device = null, Random? random = null)
        : base(inputSize, hiddenSize, 4, returnSequences, device, random)
    {
    }

    /// <summary>Creates an LSTM, optionally bidirectional and stacked.</summary>
    /// <param name="inputSize">Features per step.</param>
    /// <param name="hiddenSize">Hidden/cell state size per direction.</param>
    /// <param name="returnSequences">Output every step ([N, T, H · directions]) or the last state of each direction ([N, H · directions]).</param>
    /// <param name="bidirectional">Also read each sequence backwards; the two hidden states are concatenated, forward first.</param>
    /// <param name="layers">Stacked layers, each reading the previous one's every step.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public LSTM(int inputSize, int hiddenSize, bool returnSequences, bool bidirectional, int layers = 1, Device? device = null, Random? random = null)
        : base(inputSize, hiddenSize, 4, returnSequences, bidirectional, layers, false, device, random)
    {
    }

    /// <inheritdoc />
    protected override float[] InitialBias(int gates, int hiddenSize)
    {
        var bias = new float[gates * hiddenSize];
        Array.Fill(bias, 1f, hiddenSize, hiddenSize);
        return bias;
    }

    /// <inheritdoc />
    protected override int States => 2;

    /// <inheritdoc />
    protected override Tensor[] Cell(Tensor projected, IReadOnlyList<Tensor> state, RecurrentWeights weights)
    {
        int h = HiddenSize;
        var z = projected + state[0].MatMul(weights.HiddenWeight);
        var inputGate = z.Narrow(1, 0, h).Sigmoid();
        var forgetGate = z.Narrow(1, h, h).Sigmoid();
        var candidate = z.Narrow(1, 2 * h, h).Tanh();
        var outputGate = z.Narrow(1, 3 * h, h).Sigmoid();
        var cell = forgetGate * state[1] + inputGate * candidate;
        return [outputGate * cell.Tanh(), cell];
    }

    /// <inheritdoc />
    public override string ToString() => $"LSTM({InputSize} -> {HiddenSize}{Options})";
}

/// <summary>
/// Gated recurrent unit over [batch, time, features]: like an LSTM with fewer gates (reset, update, candidate, PyTorch's
/// order) and no separate cell state; often trains as well with fewer parameters. The reset gate scales the candidate's
/// recurrent term after its product (PyTorch's GRU, ONNX's <c>linear_before_reset</c>). Optionally bidirectional and
/// stacked (<see cref="RecurrentModule"/>), and with the candidate's recurrent bias kept apart (PyTorch's <c>b_hn</c>), so
/// PyTorch's weights load exactly.
/// </summary>
public sealed class GRU : RecurrentModule
{
    /// <summary>Creates a one-direction, one-layer GRU without a separate candidate bias.</summary>
    /// <param name="inputSize">Features per step.</param>
    /// <param name="hiddenSize">Hidden state size.</param>
    /// <param name="returnSequences">Output every step ([N, T, H]) or just the last ([N, H]).</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public GRU(int inputSize, int hiddenSize, bool returnSequences = false, Device? device = null, Random? random = null)
        : base(inputSize, hiddenSize, 3, returnSequences, device, random)
    {
    }

    /// <summary>Creates a GRU, optionally bidirectional and stacked.</summary>
    /// <param name="inputSize">Features per step.</param>
    /// <param name="hiddenSize">Hidden state size per direction.</param>
    /// <param name="returnSequences">Output every step ([N, T, H · directions]) or the last state of each direction ([N, H · directions]).</param>
    /// <param name="bidirectional">Also read each sequence backwards; the two hidden states are concatenated, forward first.</param>
    /// <param name="layers">Stacked layers, each reading the previous one's every step.</param>
    /// <param name="candidateBias">Keep the candidate gate's recurrent bias apart, inside the reset gate's product (PyTorch's
    /// <c>b_hn</c>; one more [hidden] parameter per layer and direction).</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public GRU(int inputSize, int hiddenSize, bool returnSequences, bool bidirectional, int layers = 1, bool candidateBias = false, Device? device = null, Random? random = null)
        : base(inputSize, hiddenSize, 3, returnSequences, bidirectional, layers, candidateBias, device, random)
    {
    }

    /// <summary>Whether the candidate gate's recurrent bias is kept apart (<see cref="RecurrentWeights.HiddenBias"/>).</summary>
    public bool CandidateBias => Weights[0].HiddenBias is not null;

    /// <inheritdoc />
    protected override int States => 1;

    /// <inheritdoc />
    protected override bool ScalesLastHiddenBias => true;

    /// <inheritdoc />
    protected override Tensor[] Cell(Tensor projected, IReadOnlyList<Tensor> state, RecurrentWeights weights)
    {
        int h = HiddenSize;
        var hidden = state[0];
        var r = hidden.MatMul(weights.HiddenWeight);
        var reset = (projected.Narrow(1, 0, h) + r.Narrow(1, 0, h)).Sigmoid();
        var update = (projected.Narrow(1, h, h) + r.Narrow(1, h, h)).Sigmoid();
        var recurrent = r.Narrow(1, 2 * h, h);
        if (weights.HiddenBias is { } bias)
        {
            recurrent += bias;
        }

        var candidate = (projected.Narrow(1, 2 * h, h) + reset * recurrent).Tanh();
        return [(1f - update) * candidate + update * hidden];
    }

    /// <inheritdoc />
    public override string ToString() => $"GRU({InputSize} -> {HiddenSize}{Options}{(CandidateBias ? ", candidate bias" : "")})";
}
