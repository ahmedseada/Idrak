// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// One step of a <c>GraphModule</c>: an operation on named values that produces a named value.
/// </summary>
/// <param name="Op">"layer" (runs <paramref name="Layer"/> on the first input) or one of the structural operations of <c>GraphModule</c>.</param>
/// <param name="Inputs">Names of the input values: the graph input, earlier outputs or constants.</param>
/// <param name="Output">Name of the value produced.</param>
/// <param name="Attributes">Settings of the operation (for example "axis" or "perm"), or null.</param>
/// <param name="Layer">The layer run by a "layer" node.</param>
public sealed record GraphNode(string Op, IReadOnlyList<string> Inputs, string Output, JsonObject? Attributes = null, Module? Layer = null);

/// <summary>
/// Values flowing through a graph that are computed on the host rather than as tensors (shapes, axes, indices): what an
/// operator given a shape or indices (a reshape's target, a gather's axis) may receive instead of a tensor.
/// </summary>
public static class GraphValues
{
    /// <summary>An integer value computed on the host (a shape, axes, indices).</summary>
    /// <param name="Values">The integers, row-major.</param>
    /// <param name="Dims">Their shape.</param>
    public sealed record HostValue(long[] Values, int[] Dims);

    /// <summary>The integers a value holds: a host value's, or a tensor's elements.</summary>
    /// <exception cref="InvalidOperationException">The value is neither.</exception>
    public static long[] AsIntegers(object value) => value switch
    {
        HostValue h => h.Values,
        Tensor t => t.ToArray().Select(v => (long)v).ToArray(),
        _ => throw new InvalidOperationException("expected integers"),
    };
}
