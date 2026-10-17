// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.VitsSpeech;

/// <summary>
/// Pieces the plug-in's layers share, written from public tensor operations so they run on every device. The layers
/// themselves are the library's: <c>StoredLayers</c> reads the checkpoint, <c>WaveNet</c>, <c>ResidualCouplingBlock</c>,
/// <c>PosteriorEncoder</c>, <c>HifiGanGenerator</c> and <c>HifiGanDiscriminators</c> (Idrak.Audio),
/// <c>DilatedDepthSeparableConv</c>, <c>ConvolutionalSplineFlow</c> and <c>RelativePositionAttention</c> (Idrak).
/// </summary>
internal static class VitsOps
{
    /// <summary>The feed-forward activation a config.json names.</summary>
    public static Tensor Activation(string name, Tensor x) => name switch
    {
        "relu" => x.Relu(),
        "gelu" => ExactGELU.Apply(x),
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => x.Gelu(),
        "silu" or "swish" => x.Silu(),
        _ => throw new NotSupportedException($"hidden_act '{name}' is not supported (relu, gelu, gelu_new, silu)."),
    };

    /// <summary>A column [C, 1] repeated over <paramref name="columns"/> columns: [C, columns].</summary>
    public static Tensor ExpandColumns(Tensor column, int columns) => column.MatMul(Tensor.Ones([1, columns], column.Device));

    /// <summary>A float tensor of integer indices [count] on <paramref name="device"/> (what the library's gathers take).</summary>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        var values = new float[indices.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = indices[i];
        }

        return Tensor.From(values, [values.Length], device);
    }
}
