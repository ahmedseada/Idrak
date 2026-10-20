// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Audio.Plugins.Vits;

/// <summary>
/// Pieces the plug-in's layers share, written from public tensor operations so they run on every device. The layers
/// themselves are the library's: <c>StoredLayers</c> reads the checkpoint, <c>WaveNet</c>, <c>ResidualCouplingBlock</c>,
/// <c>PosteriorEncoder</c>, <c>HifiGanGenerator</c> and <c>HifiGanDiscriminators</c> (Idrak.Audio),
/// <c>DilatedDepthSeparableConv</c>, <c>ConvolutionalSplineFlow</c>, <c>ConvolutionalFeedForward</c>, <c>ElementwiseAffineFlow</c>
/// and <c>RelativePositionAttention</c> (Idrak).
/// </summary>
internal static class VitsOps
{
    /// <summary>A column [C, 1] repeated over <paramref name="columns"/> columns: [C, columns].</summary>
    public static Tensor ExpandColumns(Tensor column, int columns) => column.MatMul(Tensor.Ones([1, columns], column.Device));
}
