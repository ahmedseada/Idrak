// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What a device's kernels can do, read by layers and tensors to choose a path instead of checking which device they run
/// on or reading one backend's kernel constants. Each backend fills it in once; a new backend (AMD, Intel, Apple) states
/// its own limits here and the layers follow without changes. What a device has beyond its limits (matrix units, fused
/// packed products, kernel timing) is a capability interface it implements (<see cref="IMatrixUnitBackend"/>,
/// <see cref="IFusedPackedKernels"/>, <see cref="IKernelProfiler"/>), read through <see cref="Backend.MatrixUnits"/>,
/// <see cref="Backend.FusedKernels"/> and the interface itself.
/// </summary>
public sealed record BackendCapabilities
{
    /// <summary>
    /// Rows the few-row (decoding) products take through their row kernels; more rows go through the prompt-sized products.
    /// </summary>
    public required int FewRows { get; init; }

    /// <summary>Largest head size of the one-pass decoding attention (a cached step's query rows over the cache).</summary>
    public required int DecodeAttentionHeadDim { get; init; }

    /// <summary>Largest head size of the tiled (flash) attention over many query rows.</summary>
    public required int TiledAttentionHeadDim { get; init; }

    /// <summary>
    /// Largest head size the device's own attention over key ranges takes (<c>Backend.AttentionSpansKernel</c> and its
    /// gradient); a larger head runs the base <c>Backend</c>'s fallback, on the host. Above it,
    /// <c>Tensor.AttentionFastest</c> composes the attention on the device (the full scores, a softmax with the ranges as
    /// a mask, the values) whenever the device reports the memory for the scores, and layers that know their ranges pass
    /// that mask. Default: every head size (a device that does not say keeps its own kernel, or its fallback, for all).
    /// </summary>
    public int SpanAttentionHeadDim { get; init; } = int.MaxValue;
}
