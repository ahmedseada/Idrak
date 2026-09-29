namespace Idrak.Layers;

/// <summary>
/// An FP8 (e4m3) copy of a frozen layer's weight [rows = in, columns = out] for fast forward products on tensor cores
/// (compute capability 8.9 and newer): quantized once, per output column, k-major. The layer keeps its own weights for
/// everything else (the backward pass, merging, saving); see <see cref="Linear.AttachFloat8"/>.
/// </summary>
internal sealed class Float8Weight : IDisposable
{
    private Float8Weight(Tensor values, Tensor scales, int rows, int columns)
    {
        Values = values;
        Scales = scales;
        Rows = rows;
        Columns = columns;
    }

    /// <summary>The quantized values: [columns, padded rows] bytes.</summary>
    public Tensor Values { get; }

    /// <summary>One scale per column.</summary>
    public Tensor Scales { get; }

    public int Rows { get; }

    public int Columns { get; }

    /// <summary>The quantized copy of <paramref name="weight"/> [rows, columns] (float32), or null when its device has no FP8 products.</summary>
    public static Float8Weight? Create(Tensor weight)
    {
        int rows = weight.Shape[0], columns = weight.Shape[1];
        var backend = weight.Device.Backend;
        int padded = backend.Float8PaddedK(rows);
        if (padded == 0)
        {
            return null;
        }

        var values = Tensor.Persistent(new float[Math.Max(1, columns * padded / 4)], [Math.Max(1, columns * padded / 4)], weight.Device, requiresGrad: false);
        var scales = Tensor.Persistent(new float[columns], [columns], weight.Device, requiresGrad: false);
        if (!backend.Float8QuantizeWeight(weight.Storage, rows, columns, values.Storage, scales.Storage))
        {
            values.Dispose();
            scales.Dispose();
            return null;
        }

        return new Float8Weight(values, scales, rows, columns);
    }

    public void Dispose()
    {
        Values.Dispose();
        Scales.Dispose();
    }
}
