// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// Maps integer token ids to learned vectors: [..., ] ids → [..., dim]. Ids are stored as floats
/// (exact up to 16,777,216) and must lie in [0, vocabulary). On the CPU an invalid id throws; on the GPU
/// it is clamped to the valid range, so validate ids when loading data.
/// </summary>
public sealed class Embedding : Module
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
            values[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }

        Weight = CreateParameter(values, [vocabulary, dim], device ?? Device.Default);
    }

    private Embedding(Tensor weight)
    {
        Vocabulary = weight.Shape[0];
        Dim = weight.Shape[1];
        Weight = weight;
    }

    private Embedding(BFloat16Weight packed)
    {
        Vocabulary = packed.Rows;
        Dim = packed.Columns;
        BFloat16 = packed;
    }

    /// <summary>
    /// A lookup table around existing bfloat16 weights [vocabulary, dim] (half the memory of float32; lossless for tables
    /// stored in bfloat16, as most checkpoints are); the layer takes ownership. The table is fixed (not trained).
    /// </summary>
    public static Embedding FromBFloat16(BFloat16Weight weight) => new(weight);

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
        get => _weight ?? throw new InvalidOperationException($"{this} holds a bfloat16 table (see BFloat16); call ToFloat32() on the model to get float weights back.");
        private set => _weight = value;
    }

    private Tensor? _weight;

    /// <summary>The bfloat16 table when the layer holds one (see <see cref="FromBFloat16"/>), else null.</summary>
    public BFloat16Weight? BFloat16 { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => BFloat16 is { } h ? Tensor.EmbeddingLookup(h, input) : Weight.EmbeddingLookup(input);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => _weight is null ? [] : [_weight];

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => BFloat16 is { } h ? [h.Packed] : [];

    internal Device Device => (_weight ?? BFloat16!.Packed).Device;

    // The table's values [vocabulary, dim] on the host.
    internal float[] WeightValues()
    {
        if (_weight is not null)
        {
            return _weight.ToArray();
        }

        using var w = BFloat16!.Dequantize();
        return w.ToArray();
    }

    internal void ToBFloat16()
    {
        if (_weight is null)
        {
            return;
        }

        BFloat16 = BFloat16Weight.Convert(_weight);
        _weight.Dispose();
        _weight = null;
    }

    // Loading a bfloat16 table (Module.Load): room for it, without rounding the current values the file replaces.
    internal void LoadAsBFloat16()
    {
        if (_weight is null)
        {
            return;
        }

        BFloat16 = BFloat16Weight.Empty(Vocabulary, Dim, _weight.Device);
        _weight.Dispose();
        _weight = null;
    }

    internal void ToFloat32(bool trainable)
    {
        if (BFloat16 is not { } h)
        {
            return;
        }

        using (var w = h.Dequantize())
        {
            _weight = Tensor.Persistent(w.ToArray(), [Vocabulary, Dim], w.Device, trainable);
        }

        h.Dispose();
        BFloat16 = null;
    }

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        _weight = _weight is null ? null : MoveTensor(_weight, device);
        BFloat16?.MoveWeights(device, MoveTensor);
    }

    /// <inheritdoc />
    public override string ToString() => $"Embedding({Vocabulary} -> {Dim}{(BFloat16 is null ? "" : ", bf16")})";
}
