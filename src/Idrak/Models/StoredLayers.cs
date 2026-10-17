// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Models;

/// <summary>A tensor a model trains, under the name and in the layout its checkpoint stores it (a linear weight transposed back to [out, in] on saving).</summary>
/// <param name="Name">The stored name.</param>
/// <param name="Tensor">The parameter.</param>
/// <param name="Stored">The stored shape.</param>
/// <param name="Transposed">Whether the parameter is the stored [out, in] matrix transposed.</param>
public sealed record StoredParameter(string Name, Tensor Tensor, int[] Stored, bool Transposed)
{
    /// <summary>The values in the stored layout (of <paramref name="of"/> when given, a tensor of the parameter's shape).</summary>
    public float[] Values(Tensor? of = null)
    {
        var values = (of ?? Tensor).ToArray();
        return Transposed ? HostParallel.Transpose(values, Tensor.Shape[0], Tensor.Shape[1]) : values;
    }
}

/// <summary>
/// A model's trained tensors as a <see cref="Module"/> (what the trainers take): its layers compute, this lists; the
/// layers own the tensors. Parts (the first segment of the stored names: "text_encoder", "flow", "decoder", …) can be
/// frozen by name.
/// </summary>
public sealed class ParameterSet(string name, IReadOnlyList<StoredParameter> parameters) : Module
{
    private readonly HashSet<string> _frozen = new(StringComparer.Ordinal);

    /// <summary>Every tensor, frozen or not (what a checkpoint saves).</summary>
    public IReadOnlyList<StoredParameter> Named => parameters;

    /// <summary>The parts: the first segment of the tensors' names, in reading order.</summary>
    public IReadOnlyList<string> Parts => [.. parameters.Select(p => Part(p.Name)).Distinct(StringComparer.Ordinal)];

    /// <summary>Keeps these parts' tensors as they are: no gradient, and out of <see cref="Parameters"/>.</summary>
    /// <exception cref="ArgumentException">A part is not one of <see cref="Parts"/>.</exception>
    public void Freeze(IReadOnlyCollection<string> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        var known = Parts;
        foreach (string part in parts)
        {
            if (!known.Contains(part, StringComparer.Ordinal))
            {
                throw new ArgumentException($"{name} has no part '{part}' to freeze; its parts are {string.Join(", ", known)}.", nameof(parts));
            }
        }

        foreach (var p in parameters.Where(p => parts.Contains(Part(p.Name))))
        {
            p.Tensor.RequiresGrad = false;
            _frozen.Add(p.Name);
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => throw new NotSupportedException($"{name} lists the trained tensors; the model's layers run them.");

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => parameters.Where(p => !_frozen.Contains(p.Name)).Select(p => p.Tensor);

    private static string Part(string tensor) => tensor.IndexOf('.') is int dot and > 0 ? tensor[..dot] : tensor;

    /// <inheritdoc />
    public override string ToString() => name;

    /// <inheritdoc />
    public override void Dispose() => GC.SuppressFinalize(this);
}

/// <summary>
/// Reads a checkpoint's tensors into the library's layers on one device, by their stored names: linear layers (the
/// [out, in] weight read transposed to the library's [in, out]), layer norms, convolutions over time and 2-D ones (a
/// weight-normed one folded at load, g · v / ‖v‖, once), transposed convolutions, 1 × 1 convolutions as linear layers.
/// Loaded to be trained (<see cref="Trainable"/>), every tensor read becomes a parameter that takes gradients, a weight
/// norm stays its gains and directions (<see cref="WeightNorm"/>: made per pass, or bound once a step), and each is listed
/// in <see cref="Trained"/> under its stored name, so the model saves back to the checkpoint's own layout.
/// </summary>
public sealed class StoredLayers(ITensorStore store, Device device, bool trainable = false)
{
    private readonly List<StoredParameter> _trained = [];
    private readonly List<WeightNorm> _normed = [];

    /// <summary>The device the layers live on.</summary>
    public Device Device { get; } = device;

    /// <summary>Whether the tensors are read as trained parameters.</summary>
    public bool Trainable { get; } = trainable;

    /// <summary>Every tensor read as a parameter (when <see cref="Trainable"/>), in reading order.</summary>
    public IReadOnlyList<StoredParameter> Trained => _trained;

    /// <summary>The weight-normed layers read (training), to bind for a step (<see cref="WeightNorm"/>).</summary>
    public IReadOnlyList<WeightNorm> Normed => _normed;

    /// <summary>Whether the store holds <paramref name="name"/>.</summary>
    public bool Contains(string name) => store.Contains(name);

    /// <summary>The values of <paramref name="name"/>.</summary>
    /// <exception cref="InvalidDataException">The store has no such tensor.</exception>
    public float[] Read(string name) =>
        store.Contains(name) ? store.Read(name) : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");

    /// <summary>The shape of <paramref name="name"/>.</summary>
    /// <exception cref="InvalidDataException">The store has no such tensor.</exception>
    public int[] Shape(string name) =>
        store.Contains(name) ? store.ShapeOf(name) : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");

    /// <summary>The names of <paramref name="prefix"/>'s weight-norm gains and directions when the store holds them (<see cref="StoredWeights.WeightNormNames"/>), else null.</summary>
    public (string Gains, string Directions)? WeightNormNames(string prefix) => StoredWeights.WeightNormNames(store, prefix);

    /// <summary>The tensor <paramref name="name"/> on the device (a parameter when trainable), checked against <paramref name="expected"/> when given.</summary>
    /// <exception cref="InvalidDataException">The store has no such tensor, or its shape is not the expected one.</exception>
    public Tensor Tensor(string name, params int[] expected)
    {
        var shape = Shape(name);
        if (expected.Length > 0 && !shape.AsSpan().SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{name}' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says {Abstraction.Tensor.FormatShape(expected)}.");
        }

        return Parameter(name, Read(name), shape, shape);
    }

    /// <summary>A tensor of <paramref name="shape"/> holding the values stored as <paramref name="name"/> (of <paramref name="stored"/> shape), a parameter when trainable.</summary>
    public Tensor Parameter(string name, float[] values, int[] shape, int[] stored, bool transposed = false)
    {
        var tensor = Abstraction.Tensor.Persistent(values, shape, Device, requiresGrad: Trainable);
        if (Trainable)
        {
            _trained.Add(new StoredParameter(name, tensor, stored, transposed));
        }

        return tensor;
    }

    /// <summary>A linear layer from <c>{prefix}.weight</c> [out, in] and <c>{prefix}.bias</c> when present.</summary>
    public Linear Linear(string prefix, int inFeatures, int outFeatures)
    {
        var shape = Shape(prefix + ".weight");
        if (shape.Length != 2 || shape[0] != outFeatures || shape[1] != inFeatures)
        {
            throw new InvalidDataException($"'{prefix}.weight' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says [{outFeatures}, {inFeatures}].");
        }

        var weight = Parameter(prefix + ".weight", store.ReadTransposed(prefix + ".weight"), [inFeatures, outFeatures], shape, transposed: true);
        return Layers.Linear.FromWeights(weight, store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outFeatures) : null);
    }

    /// <summary>A 1 × 1 convolution <c>{prefix}.weight</c> [out, in, 1] read as a linear layer [in, out], with <c>{prefix}.bias</c> when present.</summary>
    public Linear ProjectionLinear(string prefix, int inFeatures, int outFeatures)
    {
        var shape = Shape(prefix + ".weight");
        if (shape.Length != 3 || shape[0] != outFeatures || shape[1] != inFeatures || shape[2] != 1)
        {
            throw new InvalidDataException($"'{prefix}.weight' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says [{outFeatures}, {inFeatures}, 1].");
        }

        var weight = Parameter(prefix + ".weight", HostParallel.Transpose(Read(prefix + ".weight"), outFeatures, inFeatures), [inFeatures, outFeatures], shape, transposed: true);
        return Layers.Linear.FromWeights(weight, Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outFeatures) : null);
    }

    /// <summary>A layer norm from <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public LayerNorm Norm(string prefix, int features, float epsilon) =>
        LayerNorm.FromWeights(Tensor(prefix + ".weight", features), Tensor(prefix + ".bias", features), epsilon);

    /// <summary>
    /// A convolution over time from <c>{prefix}.weight</c> [out, in / groups, k], or a weight norm's gains and directions
    /// when the store holds those (<c>parametrizations.weight.original0</c> and <c>original1</c>, or <c>weight_g</c> and
    /// <c>weight_v</c>), and <c>{prefix}.bias</c> when present.
    /// </summary>
    public Conv1d Conv(string prefix, int inChannels, int outChannels, int kernel, int padding, int dilation = 1, int groups = 1) =>
        Conv2(prefix, inChannels, outChannels, (1, kernel), (1, 1), (0, padding), (1, dilation), groups);

    /// <summary>A 2-D convolution (a discriminator's (k, 1) filters over a period's rows) from <c>{prefix}</c>'s filters [out, in / groups, kh, kw] (or [out, in / groups, kw] when kh is 1).</summary>
    public Conv1d Conv2(string prefix, int inChannels, int outChannels, (int Height, int Width) kernel, (int Height, int Width) stride, (int Height, int Width) padding,
        (int Height, int Width) dilation, int groups = 1)
    {
        int columns = inChannels / groups * kernel.Height * kernel.Width;
        int[] expected = kernel.Height == 1 ? [outChannels, inChannels / groups, kernel.Width] : [outChannels, inChannels / groups, kernel.Height, kernel.Width];
        var (weight, gain) = Filters(prefix, expected, outChannels, columns);
        var bias = store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outChannels) : null;
        var conv = Conv2d.FromWeights(weight, bias, kernel, stride, padding, dilation, groups);
        return new Conv1d(conv, Register(gain, conv.Weight));
    }

    /// <summary>A transposed convolution over time from <c>{prefix}.weight</c> [in, out, k] (or a weight norm's gains and directions) and <c>{prefix}.bias</c>.</summary>
    public ConvTranspose1d ConvTranspose(string prefix, int inChannels, int outChannels, int kernel, int stride, int padding)
    {
        var (weight, gain) = Filters(prefix, [inChannels, outChannels, kernel], inChannels, outChannels * kernel);
        var bias = store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outChannels) : null;
        var conv = ConvTranspose2d.FromWeights(weight, bias, (1, kernel), (1, stride), (0, padding));
        return new ConvTranspose1d(conv, Register(gain, conv.Weight));
    }

    private WeightNorm? Register(Tensor? gain, Tensor direction)
    {
        if (gain is null)
        {
            return null;
        }

        var normed = new WeightNorm(gain, direction);
        _normed.Add(normed);
        return normed;
    }

    // A convolution's filters as [rows, columns] (rows the first stored dimension): {prefix}.weight, else a weight norm's
    // gains and directions, folded once (inference) or kept as the directions and the gains [rows, 1] (training).
    private (Tensor Weight, Tensor? Gain) Filters(string prefix, int[] expected, int rows, int columns)
    {
        var names = StoredWeights.WeightNormNames(store, prefix);
        string name = names?.Directions ?? prefix + ".weight";
        var shape = Shape(name);
        if (!shape.AsSpan().SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{name}' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says {Abstraction.Tensor.FormatShape(expected)}.");
        }

        if (names is not { } norm)
        {
            return (Parameter(name, Read(name), [rows, columns], shape), null);
        }

        if (!Trainable)
        {
            return (Abstraction.Tensor.Persistent(StoredWeights.FoldWeightNorm(Read(norm.Gains), Read(norm.Directions), shape), [rows, columns], Device), null);
        }

        var gains = Read(norm.Gains);
        if (gains.Length != rows)
        {
            throw new InvalidDataException($"'{norm.Gains}' holds {gains.Length} gains for {rows} filters.");
        }

        var direction = Parameter(norm.Directions, Read(norm.Directions), [rows, columns], shape);
        return (direction, Parameter(norm.Gains, gains, [rows, 1], Shape(norm.Gains)));
    }
}
