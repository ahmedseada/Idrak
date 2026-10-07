// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// Long short-term memory over [batch, time, features]. Gates: input, forget, cell, output. The forget-gate
/// bias starts at 1 so the layer remembers by default, which helps long sequences train.
/// </summary>
/// <param name="inputSize">Features per step.</param>
/// <param name="hiddenSize">Hidden/cell state size.</param>
/// <param name="returnSequences">Output every step ([N, T, H]) or just the last ([N, H]).</param>
/// <param name="device">Where the parameters live.</param>
/// <param name="random">Seed source for the initial weights.</param>
public sealed class LSTM(int inputSize, int hiddenSize, bool returnSequences = false, Device? device = null, Random? random = null)
    : RecurrentModule(inputSize, hiddenSize, 4, returnSequences, device, random)
{
    /// <inheritdoc />
    protected override float[] InitialBias(int gates, int hiddenSize)
    {
        var bias = new float[gates * hiddenSize];
        Array.Fill(bias, 1f, hiddenSize, hiddenSize);
        return bias;
    }

    /// <inheritdoc />
    protected override IEnumerable<Tensor> Run(Tensor projected, int batch, int steps)
    {
        int h = HiddenSize;
        var hidden = Tensor.Zeros([batch, h], projected.Device);
        var cell = Tensor.Zeros([batch, h], projected.Device);
        for (int t = 0; t < steps; t++)
        {
            var z = Step(projected, t, batch) + hidden.MatMul(HiddenWeight);
            var inputGate = z.Narrow(1, 0, h).Sigmoid();
            var forgetGate = z.Narrow(1, h, h).Sigmoid();
            var candidate = z.Narrow(1, 2 * h, h).Tanh();
            var outputGate = z.Narrow(1, 3 * h, h).Sigmoid();
            cell = forgetGate * cell + inputGate * candidate;
            hidden = outputGate * cell.Tanh();
            yield return hidden;
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"LSTM({InputSize} -> {HiddenSize}{(ReturnSequences ? ", sequences" : "")})";
}

/// <summary>
/// Gated recurrent unit over [batch, time, features]: like an LSTM with fewer gates (reset, update) and no
/// separate cell state; often trains as well with fewer parameters.
/// </summary>
/// <param name="inputSize">Features per step.</param>
/// <param name="hiddenSize">Hidden state size.</param>
/// <param name="returnSequences">Output every step ([N, T, H]) or just the last ([N, H]).</param>
/// <param name="device">Where the parameters live.</param>
/// <param name="random">Seed source for the initial weights.</param>
public sealed class GRU(int inputSize, int hiddenSize, bool returnSequences = false, Device? device = null, Random? random = null)
    : RecurrentModule(inputSize, hiddenSize, 3, returnSequences, device, random)
{
    /// <inheritdoc />
    protected override IEnumerable<Tensor> Run(Tensor projected, int batch, int steps)
    {
        int h = HiddenSize;
        var hidden = Tensor.Zeros([batch, h], projected.Device);
        for (int t = 0; t < steps; t++)
        {
            var x = Step(projected, t, batch);
            var r = hidden.MatMul(HiddenWeight);
            var reset = (x.Narrow(1, 0, h) + r.Narrow(1, 0, h)).Sigmoid();
            var update = (x.Narrow(1, h, h) + r.Narrow(1, h, h)).Sigmoid();
            var candidate = (x.Narrow(1, 2 * h, h) + reset * r.Narrow(1, 2 * h, h)).Tanh();
            hidden = (1f - update) * candidate + update * hidden;
            yield return hidden;
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"GRU({InputSize} -> {HiddenSize}{(ReturnSequences ? ", sequences" : "")})";
}
