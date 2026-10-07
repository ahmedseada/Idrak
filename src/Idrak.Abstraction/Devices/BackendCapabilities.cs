// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What a device's kernels can do, read by layers and tensors to choose a path instead of checking which device they run
/// on or reading one backend's kernel constants. Each backend fills it in once; a new backend (AMD, Intel, Apple) states
/// its own limits here and the layers follow without changes.
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
    /// Matrix units the <c>MixedPrecision</c> modes run on (tensor cores): the fused training products, the LoRA
    /// products and the bfloat16 copies of tied heads are worth it only where these exist.
    /// </summary>
    public required bool MatrixUnits { get; init; }

    /// <summary>Head sizes with matrix-unit flash attention for training (none when the device has none).</summary>
    public required Func<int, bool> MatrixUnitAttentionHeadDim { get; init; }

    /// <summary>
    /// Fused decoding kernels: several few-row products in one pass, gate/up with the activation, a projection with its
    /// residual and normalization, norms and rotary positions written into the cache, and recorded packed training products.
    /// Without them every product runs on its own (same results, more passes).
    /// </summary>
    public required bool FusedKernels { get; init; }

    /// <summary>Kernel timing (<c>Idrak.Diagnostics.GpuProfiler</c>).</summary>
    public required bool Profiling { get; init; }
}
