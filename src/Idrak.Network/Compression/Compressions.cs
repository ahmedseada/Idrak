// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Training;

namespace Idrak.Networking.Compression;

/// <summary>"none" in <see cref="GradientCompressions"/>: every value, as float32.</summary>
internal sealed class PlainCompression : IGradientCompression
{
    public string Name => "none";

    public string Summary => "every gradient value travels as float32";

    public IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options) =>
        new DenseCompressor(Name, collectives);
}

/// <summary>A bucket weighted, then all-reduced: the plain compression.</summary>
internal sealed class DenseCompressor(string name, ICollectives collectives) : IGradientCompressor
{
    private readonly HostValues _values = new(collectives);

    public string Name => name;

    public void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        Scale(values, weight);
        _values.AllReduce(values);
    }

    public void Dispose() => _values.Dispose();

    // values *= factor: a vector loop and a scalar tail.
    internal static void Scale(Span<float> values, float factor)
    {
        if (factor == 1f)
        {
            return;
        }

        if (factor == 0f)
        {
            values.Clear();
            return;
        }

        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var f = new Vector<float>(factor);
            for (; i <= values.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(values[i..]) * f).CopyTo(values[i..]);
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] *= factor;
        }
    }
}
