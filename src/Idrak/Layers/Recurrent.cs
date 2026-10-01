// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>Shared plumbing for recurrent layers over [batch, time, features] input.</summary>
public abstract class RecurrentModule : Module
{
    /// <summary>Creates the layer.</summary>
    protected RecurrentModule(int inputSize, int hiddenSize, int gates, bool returnSequences, Device? device, Random? random)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        ReturnSequences = returnSequences;
        device ??= Device.Default;
        random ??= Random.Shared;
        float bound = 1f / MathF.Sqrt(hiddenSize);
        InputWeight = CreateParameter(UniformValues(inputSize * gates * hiddenSize, bound, random), [inputSize, gates * hiddenSize], device);
        HiddenWeight = CreateParameter(UniformValues(hiddenSize * gates * hiddenSize, bound, random), [hiddenSize, gates * hiddenSize], device);
        Bias = CreateParameter(InitialBias(gates, hiddenSize), [gates * hiddenSize], device);
    }

    /// <summary>Features per time step.</summary>
    public int InputSize { get; }

    /// <summary>Size of the hidden state.</summary>
    public int HiddenSize { get; }

    /// <summary>True: output every step, [N, T, H]. False: only the last hidden state, [N, H].</summary>
    public bool ReturnSequences { get; }

    /// <summary>Input-to-gates weights, [input, gates · hidden].</summary>
    public Tensor InputWeight { get; private set; }

    /// <summary>Hidden-to-gates weights, [hidden, gates · hidden].</summary>
    public Tensor HiddenWeight { get; private set; }

    /// <summary>Gate biases, [gates · hidden].</summary>
    public Tensor Bias { get; private set; }

    /// <summary>The initial bias vector.</summary>
    protected virtual float[] InitialBias(int gates, int hiddenSize) => new float[gates * hiddenSize];

    /// <inheritdoc />
    protected sealed override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 3 || input.Shape[2] != InputSize)
        {
            throw new ArgumentException($"{GetType().Name} expects [batch, time, {InputSize}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int batch = input.Shape[0], steps = input.Shape[1];

        // Project every time step's input at once (one large matrix product) instead of once per step.
        var projected = input.MatMul(InputWeight) + Bias;
        var outputs = ReturnSequences ? new List<Tensor>(steps) : null;
        Tensor? last = null;
        foreach (var h in Run(projected, batch, steps))
        {
            outputs?.Add(h);
            last = h;
        }

        return ReturnSequences ? Tensor.Stack(outputs!, 1) : last!;
    }

    /// <summary>Yields the hidden state after each step, given the [N, T, gates·H] input projections.</summary>
    private protected abstract IEnumerable<Tensor> Run(Tensor projected, int batch, int steps);

    /// <summary>The projection of time step t: [N, gates·H].</summary>
    private protected static Tensor Step(Tensor projected, int t, int batch) =>
        projected.Narrow(1, t, 1).Reshape(batch, projected.Shape[2]);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [InputWeight, HiddenWeight, Bias];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        InputWeight = MoveTensor(InputWeight, device);
        HiddenWeight = MoveTensor(HiddenWeight, device);
        Bias = MoveTensor(Bias, device);
    }
}

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

    private protected override IEnumerable<Tensor> Run(Tensor projected, int batch, int steps)
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
    private protected override IEnumerable<Tensor> Run(Tensor projected, int batch, int steps)
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
