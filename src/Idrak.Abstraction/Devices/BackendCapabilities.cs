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

    /// <summary>
    /// The operations whose kernels on this device read strided input (plan 37, V0): a view's storage through its layout
    /// rather than a contiguous copy. A device that lists <see cref="Operations.Ops.StridedCopy"/> keeps views
    /// (<see cref="TensorViews"/>): permutes, narrows, selects and expands share their input's storage, and an operation
    /// not listed here reads a view through one contiguous copy made by that device's own strided copy (counted:
    /// <see cref="TensorViews.Scratch"/>, <see cref="TensorViews.Materialized"/>). <see cref="Operations.Ops.BatchedMatMul"/>
    /// listed: a product reads a transposed view in place (its transpose flags). <see cref="Operations.Ops.BatchedMatMulStrided"/>,
    /// <see cref="Operations.Ops.AttentionTiledStrided"/> and <see cref="Operations.Ops.AttentionTiledStridedBackward"/>
    /// listed (plan 37, V1b): products, forward and backward, read both operands through their layouts, and the tiled
    /// attention reads q, keys and values (heads split and permuted) where they are, forward and backward. Default: none
    /// (every tensor a copy, as before views).
    /// </summary>
    public IReadOnlySet<Operations.Operation> StridedInputs { get; init; } = new HashSet<Operations.Operation>();

    /// <summary>
    /// The element type this device's kernels read and write indices, token ids and offsets in (plan 37, T1): the storages an
    /// operation lists in <see cref="Operations.Operation.IndexParameters"/>. A storage of another type is converted for the
    /// call by the dispatcher, once, and counted (<see cref="ElementTypes.Conversions"/>); index tensors the library makes
    /// (<see cref="Tensor.Indices(ReadOnlySpan{int}, Device)"/>, a top-k's indices, a sampler's ids, the routing of a mixture
    /// of experts) are made in it, so they need none. Default: float32 (indices exact up to 2²⁴, as before int32 kernels).
    /// </summary>
    public ElementType IndexType { get; init; } = ElementType.Float32;

    /// <summary>
    /// Operations whose kernels on this device take another index type than <see cref="IndexType"/> (a device whose integer
    /// kernels cover some operations so far). Default: none.
    /// </summary>
    public IReadOnlyDictionary<Operations.Operation, ElementType> IndexTypes { get; init; } = new Dictionary<Operations.Operation, ElementType>();

    /// <summary>
    /// Whether this device multiplies weights kept in their file's format (plan 37, Q0: <c>StoredWeight</c>) of every
    /// registered element type that widens to float32, with kernels of its own (<c>Backend.StoredMatMul</c>,
    /// <c>Backend.ExpandStored</c>). Default: false.
    /// </summary>
    public bool StoresAnyFormat { get; init; }

    /// <summary>
    /// The element types (by name) whose stored weights this device multiplies with kernels of its own when it does not all
    /// (<see cref="StoresAnyFormat"/>); a load expands weights of any other type to float32 for it. Default: none.
    /// </summary>
    public IReadOnlySet<string> StoredFormats { get; init; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// The element types this device computes natively (plan 37, T2), read by each backend at run time from what the device
    /// and its driver report (a compute capability, cooperative matrix types), never from a list of cards, and only where
    /// the backend has kernels that compute in them: <see cref="ElementTypes.Choose"/>'s auto picks bfloat16 where it is
    /// listed, float16 where only it is, float32 otherwise. float32 is always computed. Default: float32 alone.
    /// </summary>
    public IReadOnlyList<ElementType> NativeTypes { get; init; } = [ElementType.Float32];

    /// <summary>
    /// What the device reported that <see cref="NativeTypes"/> follows from, for people (the device's own report of the
    /// 16-bit and 8-bit arithmetic it runs): the telemetry says it with auto's pick.
    /// </summary>
    public string NativeReport { get; init; } = "the device reports no 16-bit arithmetic the backend computes in";

    /// <summary>
    /// Per operation, the value types its kernel on this device reads and writes without a conversion (plan 37, decision 8:
    /// activations in the run's type), each backend's own report from its kernels and what the device reports at run time;
    /// an operation not listed takes <see cref="DefaultValueTypes"/>. The dispatcher converts a storage of another type for
    /// the call once, counts it and says so in the telemetry (<see cref="ElementTypes.Conversions"/>); an operation with no
    /// types reported here or in <see cref="DefaultValueTypes"/> takes its storages as given. Default: none listed.
    /// </summary>
    public IReadOnlyDictionary<Operations.Operation, IReadOnlyList<ElementType>> ValueTypes { get; init; } = new Dictionary<Operations.Operation, IReadOnlyList<ElementType>>();

    /// <summary>The value types an operation not listed in <see cref="ValueTypes"/> takes on this device, as its backend reports them. Default: none reported.</summary>
    public IReadOnlyList<ElementType> DefaultValueTypes { get; init; } = [];

    /// <summary>
    /// Per operation, the types its kernel on this device can sum in (plan 37, decision 7: no fixed sum type), the
    /// device's preferred first, each backend's own report from what the device reports at run time; an operation not
    /// listed takes <see cref="DefaultSumTypes"/>. <see cref="Abstraction.SumTypes"/>' choice picks among them. Default: none listed.
    /// </summary>
    public IReadOnlyDictionary<Operations.Operation, IReadOnlyList<ElementType>> SumTypes { get; init; } = new Dictionary<Operations.Operation, IReadOnlyList<ElementType>>();

    /// <summary>The sum types an operation not listed in <see cref="SumTypes"/> takes on this device, as its backend reports them. Default: none reported.</summary>
    public IReadOnlyList<ElementType> DefaultSumTypes { get; init; } = [];

    /// <summary>The value types <paramref name="operation"/>'s kernel reads and writes on this device (<see cref="ValueTypes"/>, else <see cref="DefaultValueTypes"/>).</summary>
    public IReadOnlyList<ElementType> ValueTypesOf(Operations.Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return ValueTypes.TryGetValue(operation, out var types) ? types : DefaultValueTypes;
    }

    /// <summary>The types <paramref name="operation"/>'s kernel can sum in on this device (<see cref="SumTypes"/>, else <see cref="DefaultSumTypes"/>).</summary>
    public IReadOnlyList<ElementType> SumTypesOf(Operations.Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return SumTypes.TryGetValue(operation, out var types) ? types : DefaultSumTypes;
    }
}
