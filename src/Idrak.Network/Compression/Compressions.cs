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
        new DenseCompressor(Name, collectives, WireType.Float32);
}

/// <summary>"bf16" in <see cref="GradientCompressions"/>: every value, as bfloat16 (half the bytes); the sums are made in float32.</summary>
internal sealed class BFloat16Compression : IGradientCompression
{
    public string Name => "bf16";

    public string Summary => "every gradient value travels as bfloat16, half the bytes; the sums are made in float32";

    public IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options) =>
        new DenseCompressor(Name, collectives, WireType.BFloat16);
}

/// <summary>A bucket weighted, then all-reduced on a wire type: the plain and bfloat16 compressions.</summary>
internal sealed class DenseCompressor(string name, ICollectives collectives, WireType wire) : IGradientCompressor
{
    private readonly HostValues _values = new(collectives);

    public string Name => name;

    public void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        Scale(values, weight);
        _values.AllReduce(values, wire);
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
