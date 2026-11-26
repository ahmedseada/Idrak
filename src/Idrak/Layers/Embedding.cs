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

    /// <summary>
    /// A lookup table around existing weights [vocabulary, dim]: float32, or a table of another type as its checkpoint
    /// stores it (<see cref="Tensor.Stored(IElementType, ReadOnlySpan{byte}, ReadOnlySpan{int}, Device?)"/>, never widened; each
    /// row looked up read through its type, for inference). The layer takes ownership.
    /// </summary>
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
    protected override Tensor ForwardCore(Tensor input) => Stored ? StoredLookup(input) : Weight.EmbeddingLookup(input);

    // Whether the table is stored in a type of its own.
    private bool Stored => _weight is { } w && w.Storage.Typed && w.Type.Native != NativeKind.Float32;

    // The rows of a table stored in its own type: each id's row read from the table's words and through its type (each
    // row whole words of whole groups of the type's blocks).
    private Tensor StoredLookup(Tensor input)
    {
        var table = Weight;
        var type = table.Type;
        long rowBytes = type.ByteCount(Dim);
        if (rowBytes % sizeof(float) != 0 || type.ByteCount(2L * Dim) != 2 * rowBytes)
        {
            throw new NotSupportedException($"{this}: a row of {Dim} values of {type.Name} is not whole words of whole groups; its rows are not read one by one.");
        }

        int words = (int)(rowBytes / sizeof(float));
        var ids = input.ToArray();
        var values = new float[ids.Length * Dim];
        var row = new float[words];
        for (int i = 0; i < ids.Length; i++)
        {
            int id = (int)ids[i];
            if (id < 0 || id >= Vocabulary || id != ids[i])
            {
                throw new ArgumentOutOfRangeException(nameof(input), $"Id {ids[i]} (element {i}) is not one of the {Vocabulary} rows.");
            }

            table.Backend.DownloadRange(table.Storage, id * words, row);
            type.Read(System.Runtime.InteropServices.MemoryMarshal.AsBytes(row.AsSpan()), values.AsSpan(i * Dim, Dim));
        }

        return Tensor.From(values, [.. input.Shape, Dim], table.Device);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => _weight is null || Stored ? [] : [_weight];

    /// <inheritdoc />
    /// <remarks>A table stored in its own type, which takes no gradient.</remarks>
    public override IEnumerable<Tensor> Buffers() => Stored ? [_weight!] : [];

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
