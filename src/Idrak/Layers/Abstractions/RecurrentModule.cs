// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

/// <summary>The weights of one layer and direction of a recurrent layer.</summary>
public sealed class RecurrentWeights
{
    internal RecurrentWeights(Tensor inputWeight, Tensor hiddenWeight, Tensor bias, Tensor? hiddenBias)
    {
        InputWeight = inputWeight;
        HiddenWeight = hiddenWeight;
        Bias = bias;
        HiddenBias = hiddenBias;
    }

    /// <summary>Input-to-gates weights, [input, gates · hidden].</summary>
    public Tensor InputWeight { get; private set; }

    /// <summary>Hidden-to-gates weights, [hidden, gates · hidden].</summary>
    public Tensor HiddenWeight { get; private set; }

    /// <summary>Gate biases, [gates · hidden] (the input and recurrent biases added, except a <see cref="HiddenBias"/> kept apart).</summary>
    public Tensor Bias { get; private set; }

    /// <summary>The recurrent bias of the last gate kept apart, [hidden], where the cell scales it (a GRU's candidate gate); null when the layer has none.</summary>
    public Tensor? HiddenBias { get; private set; }

    internal IEnumerable<Tensor> Parameters() => HiddenBias is null ? [InputWeight, HiddenWeight, Bias] : [InputWeight, HiddenWeight, Bias, HiddenBias];

    internal void MoveTo(Func<Tensor, Tensor> move)
    {
        InputWeight = move(InputWeight);
        HiddenWeight = move(HiddenWeight);
        Bias = move(Bias);
        HiddenBias = HiddenBias is null ? null : move(HiddenBias);
    }
}

/// <summary>
/// Shared plumbing for recurrent layers over [batch, time, features] input: the time loop, a backward direction
/// (bidirectional: the sequence read forward and backward, the two hidden states concatenated, as PyTorch's
/// <c>bidirectional=True</c>), and stacked layers (each reading the previous one's every step). A layer type supplies one
/// step of its cell (<see cref="Cell"/>).
/// </summary>
public abstract class RecurrentModule : Module
{
    private readonly RecurrentWeights[] _weights;

    /// <summary>Creates the layer: one direction, one layer.</summary>
    protected RecurrentModule(int inputSize, int hiddenSize, int gates, bool returnSequences, Device? device, Random? random)
        : this(inputSize, hiddenSize, gates, returnSequences, false, 1, false, device, random)
    {
    }

    /// <summary>Creates the layer.</summary>
    /// <param name="inputSize">Features per step.</param>
    /// <param name="hiddenSize">Hidden state size (per direction).</param>
    /// <param name="gates">Gates of the cell (4 for an LSTM, 3 for a GRU).</param>
    /// <param name="returnSequences">Output every step or only the last state.</param>
    /// <param name="bidirectional">Also read each sequence backwards.</param>
    /// <param name="layers">Stacked layers.</param>
    /// <param name="hiddenBias">Keep the last gate's recurrent bias apart (<see cref="RecurrentWeights.HiddenBias"/>).</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    protected RecurrentModule(int inputSize, int hiddenSize, int gates, bool returnSequences, bool bidirectional, int layers, bool hiddenBias, Device? device, Random? random)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(gates);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        Gates = gates;
        ReturnSequences = returnSequences;
        Bidirectional = bidirectional;
        Layers = layers;
        device ??= Device.Default;
        random ??= Random.Shared;
        float bound = 1f / MathF.Sqrt(hiddenSize);
        _weights = new RecurrentWeights[layers * Directions];
        for (int i = 0; i < _weights.Length; i++)
        {
            int inputs = i < Directions ? inputSize : Directions * hiddenSize;
            _weights[i] = new RecurrentWeights(
                CreateParameter(UniformValues(inputs * gates * hiddenSize, bound, random), [inputs, gates * hiddenSize], device),
                CreateParameter(UniformValues(hiddenSize * gates * hiddenSize, bound, random), [hiddenSize, gates * hiddenSize], device),
                CreateParameter(InitialBias(gates, hiddenSize), [gates * hiddenSize], device),
                hiddenBias ? CreateParameter(new float[hiddenSize], [hiddenSize], device) : null);
        }
    }

    /// <summary>Features per time step.</summary>
    public int InputSize { get; }

    /// <summary>Size of the hidden state of one direction.</summary>
    public int HiddenSize { get; }

    /// <summary>Gates of the cell.</summary>
    public int Gates { get; }

    /// <summary>True: output every step, [N, T, <see cref="OutputSize"/>]. False: only the last hidden state, [N, <see cref="OutputSize"/>].</summary>
    public bool ReturnSequences { get; }

    /// <summary>Whether each sequence is also read backwards (the forward and backward hidden states concatenated, forward first).</summary>
    public bool Bidirectional { get; }

    /// <summary>Stacked layers.</summary>
    public int Layers { get; }

    /// <summary>Directions: 2 when <see cref="Bidirectional"/>, else 1.</summary>
    public int Directions => Bidirectional ? 2 : 1;

    /// <summary>Features of each output: <see cref="HiddenSize"/> times <see cref="Directions"/>.</summary>
    public int OutputSize => HiddenSize * Directions;

    /// <summary>
    /// The weights of each layer and direction: layer 0 forward, layer 0 backward (when bidirectional), layer 1 forward, ...
    /// (PyTorch's order: <c>weight_ih_l0</c>, <c>weight_ih_l0_reverse</c>, <c>weight_ih_l1</c>, ...).
    /// </summary>
    public IReadOnlyList<RecurrentWeights> Weights => _weights;

    /// <summary>Input-to-gates weights of the first layer's forward direction, [input, gates · hidden].</summary>
    public Tensor InputWeight => _weights[0].InputWeight;

    /// <summary>Hidden-to-gates weights of the first layer's forward direction, [hidden, gates · hidden].</summary>
    public Tensor HiddenWeight => _weights[0].HiddenWeight;

    /// <summary>Gate biases of the first layer's forward direction, [gates · hidden].</summary>
    public Tensor Bias => _weights[0].Bias;

    /// <summary>The initial bias vector.</summary>
    protected virtual float[] InitialBias(int gates, int hiddenSize) => new float[gates * hiddenSize];

    /// <summary>The states a step carries (the hidden state first), each [N, hidden] and zero at the start: 1 for most cells, 2 for an LSTM (hidden and cell).</summary>
    protected abstract int States { get; }

    /// <summary>
    /// One step of the cell: <paramref name="projected"/> is the step's input already through <see cref="RecurrentWeights.InputWeight"/>
    /// plus <see cref="RecurrentWeights.Bias"/>, [N, gates · hidden]; <paramref name="state"/> the states before the step. Returns
    /// the states after it, the new hidden state first.
    /// </summary>
    protected abstract Tensor[] Cell(Tensor projected, IReadOnlyList<Tensor> state, RecurrentWeights weights);

    /// <summary>
    /// The whole time loop of one layer and direction as one operation, from the projected input [N, T, gates · hidden]
    /// (every step's input through <see cref="RecurrentWeights.InputWeight"/> plus <see cref="RecurrentWeights.Bias"/>),
    /// with zero initial states: every step's hidden state, [N, T, hidden] (<paramref name="reverse"/>: the steps are taken
    /// from the last to the first, each state still at its own step). Null (the default) when the layer type or the device
    /// has no such operation: the steps are then composed from <see cref="Cell"/>. An override returns what the composed
    /// steps return, to float rounding, and their gradients.
    /// </summary>
    protected virtual Tensor? Sequence(Tensor projected, RecurrentWeights weights, bool reverse) => null;

    // Tests: compose the steps from Cell even where Sequence has a fused loop (the reference the fused loop is checked against).
    internal static bool ComposedOnly { get; set; }

    /// <inheritdoc />
    protected sealed override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 3 || input.Shape[2] != InputSize)
        {
            throw new ArgumentException($"{GetType().Name} expects [batch, time, {InputSize}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int batch = input.Shape[0], steps = input.Shape[1];
        var x = input;
        var last = new Tensor[Directions];
        for (int layer = 0; layer < Layers; layer++)
        {
            bool sequences = ReturnSequences || layer < Layers - 1;
            var outputs = new Tensor[Directions];
            for (int direction = 0; direction < Directions; direction++)
            {
                var weights = _weights[layer * Directions + direction];

                // Project every time step's input at once (one large matrix product) instead of once per step.
                var projected = x.MatMul(weights.InputWeight) + weights.Bias;

                // The whole time loop as one operation where the layer type and the device have one; else step by step.
                if (!ComposedOnly && Sequence(projected, weights, direction == 1) is { } fused)
                {
                    if (sequences)
                    {
                        outputs[direction] = fused;
                    }
                    else
                    {
                        // The last state: the forward direction's after the last step, the backward one's after the first.
                        last[direction] = fused.Narrow(1, direction == 0 ? steps - 1 : 0, 1).Reshape(batch, HiddenSize);
                    }

                    continue;
                }

                Tensor[] state = [.. Enumerable.Range(0, States).Select(_ => Tensor.Zeros([batch, HiddenSize], input.Device))];
                var hidden = sequences ? new Tensor[steps] : null;
                for (int i = 0; i < steps; i++)
                {
                    int t = direction == 0 ? i : steps - 1 - i;
                    state = Cell(projected.Narrow(1, t, 1).Reshape(batch, projected.Shape[2]), state, weights);
                    if (hidden is not null)
                    {
                        hidden[t] = state[0];
                    }
                }

                last[direction] = state[0];
                outputs[direction] = hidden is null ? null! : Tensor.Stack(hidden, 1);
            }

            if (sequences)
            {
                x = Directions == 1 ? outputs[0] : Tensor.Concat(outputs, 2);
            }
        }

        return ReturnSequences ? x : Directions == 1 ? last[0] : Tensor.Concat(last, 1);
    }

    /// <summary>
    /// Loads one layer and direction from weights in PyTorch's layout (as <c>torch.nn.LSTM</c> and <c>torch.nn.GRU</c> store
    /// them): <paramref name="inputWeights"/> [gates · hidden, inputs] (<c>weight_ih_l{layer}</c>, with <c>_reverse</c> for
    /// the backward direction), <paramref name="hiddenWeights"/> [gates · hidden, hidden] (<c>weight_hh</c>), and the biases
    /// <c>bias_ih</c> and <c>bias_hh</c>, [gates · hidden] (empty: zeros). The gates are in the cell's order (LSTM: input,
    /// forget, cell, output; GRU: reset, update, candidate), which is PyTorch's. The two biases add up, except the last
    /// gate's recurrent bias in a layer that keeps it apart (<see cref="RecurrentWeights.HiddenBias"/>); a layer without one
    /// takes only a zero recurrent bias there when the cell scales it (<see cref="ScalesLastHiddenBias"/>).
    /// </summary>
    public void LoadGateWeights(int layer, bool reverse, ReadOnlySpan<float> inputWeights, ReadOnlySpan<float> hiddenWeights, ReadOnlySpan<float> inputBias,
        ReadOnlySpan<float> hiddenBias)
    {
        if ((uint)layer >= (uint)Layers || reverse && !Bidirectional)
        {
            throw new ArgumentOutOfRangeException(nameof(layer), $"The layer has {Layers} layer(s){(Bidirectional ? " in two directions" : " in one direction")}; asked for layer {layer}{(reverse ? " reversed" : "")}.");
        }

        var weights = _weights[layer * Directions + (reverse ? 1 : 0)];
        int h = HiddenSize, size = Gates * h, inputs = weights.InputWeight.Shape[0];
        if (inputWeights.Length != size * inputs || hiddenWeights.Length != size * h || inputBias.Length != 0 && inputBias.Length != size
            || hiddenBias.Length != 0 && hiddenBias.Length != size)
        {
            throw new ArgumentException($"Layer {layer} takes weights [{size}, {inputs}] and [{size}, {h}] and biases [{size}]; got {inputWeights.Length}, {hiddenWeights.Length}, "
                + $"{inputBias.Length} and {hiddenBias.Length} values.");
        }

        var bias = new float[size];
        var apart = new float[h];
        for (int i = 0; i < size; i++)
        {
            float recurrent = hiddenBias.Length > 0 ? hiddenBias[i] : 0f;
            bias[i] = inputBias.Length > 0 ? inputBias[i] : 0f;
            if (i >= size - h && (weights.HiddenBias is not null || ScalesLastHiddenBias))
            {
                apart[i - (size - h)] = recurrent;
            }
            else
            {
                bias[i] += recurrent;
            }
        }

        if (weights.HiddenBias is null && apart.Any(v => v != 0f))
        {
            throw new NotSupportedException($"{GetType().Name} scales the last gate's recurrent bias, which this layer does not keep apart: create it with that bias "
                + "(GRU: candidateBias: true) to load these weights.");
        }

        weights.InputWeight.Load(Transposed(inputWeights, size, inputs));
        weights.HiddenWeight.Load(Transposed(hiddenWeights, size, h));
        weights.Bias.Load(bias);
        weights.HiddenBias?.Load(apart);
    }

    /// <summary>Whether the cell scales the last gate's recurrent bias (a GRU's reset gate multiplies it), so it cannot be added to the input bias.</summary>
    protected virtual bool ScalesLastHiddenBias => false;

    private static float[] Transposed(ReadOnlySpan<float> values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => _weights.SelectMany(w => w.Parameters());

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        foreach (var weights in _weights)
        {
            weights.MoveTo(t => MoveTensor(t, device));
        }
    }

    /// <summary>The layer's options for <see cref="object.ToString"/>: ", sequences", ", bidirectional", ", 2 layers".</summary>
    protected string Options => (ReturnSequences ? ", sequences" : "") + (Bidirectional ? ", bidirectional" : "") + (Layers > 1 ? $", {Layers} layers" : "");
}
