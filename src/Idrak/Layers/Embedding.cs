// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Maps integer token ids to learned vectors: [..., ] ids → [..., dim]. Ids are stored as floats
/// (exact up to 16,777,216) and must lie in [0, vocabulary). On the CPU an invalid id throws; on the GPU
/// it is clamped to the valid range, so validate ids when loading data.
/// </summary>
public sealed class Embedding : Module, INamedWeights, IOffloadableLayer
{
    /// <summary>Creates a table of <paramref name="vocabulary"/> vectors of size <paramref name="dim"/>, initialized N(0, 1).</summary>
    public Embedding(int vocabulary, int dim, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabulary);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        Vocabulary = vocabulary;
        Dim = dim;
        random ??= Random.Shared;
        var values = new float[vocabulary * dim];
        for (int i = 0; i < values.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2 * PortableMath.Log(u1)) * PortableMath.Cos(2 * Math.PI * u2));
        }

        Weight = CreateParameter(values, [vocabulary, dim], device ?? Device.Default);
    }

    private Embedding(Tensor weight)
    {
        Vocabulary = weight.Shape[0];
        Dim = weight.Shape[1];
        Weight = weight;
    }

    /// <summary>A lookup table around existing weights [vocabulary, dim]; the layer takes ownership.</summary>
    public static Embedding FromWeights(Tensor weight) =>
        weight.Rank == 2 ? new Embedding(weight) : throw new ArgumentException($"Embedding weights must be [vocabulary, dim], got {Tensor.FormatShape(weight.Shape)}.");

    /// <summary>Number of distinct ids.</summary>
    public int Vocabulary { get; }

    /// <summary>Vector size.</summary>
    public int Dim { get; }

    /// <summary>The [vocabulary, dim] table.</summary>
    public Tensor Weight
    {
        get => _weight!;
        private set => _weight = value;
    }

    private Tensor? _weight;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Weight.EmbeddingLookup(input);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => _weight is null ? [] : [_weight];

    internal Device Device => Weight.Device;

    // The table's values [vocabulary, dim] on the host.
    internal float[] WeightValues() => Weight.ToArray();

    /// <inheritdoc />
    protected override void MoveTo(Device device) => _weight = _weight is null ? null : MoveTensor(_weight, device);

    /// <inheritdoc />
    /// <remarks>"weight", the [vocabulary, dim] table.</remarks>
    IEnumerable<(string Name, Tensor Tensor)> INamedWeights.NamedWeights() => [("weight", Weight)];

    /// <inheritdoc />
    /// <remarks>None: a lookup reads only the rows it needs, which system memory serves without copying the whole table.</remarks>
    IEnumerable<Tensor> IOffloadableLayer.StagedParameters() => [];

    /// <inheritdoc />
    IEnumerable<Tensor> IOffloadableLayer.StagedBuffers() => [];

    /// <inheritdoc />
    public override string ToString() => $"Embedding({Vocabulary} -> {Dim})";
}
