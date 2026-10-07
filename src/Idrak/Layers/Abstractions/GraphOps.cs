// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers.Abstractions;

/// <summary>
/// Computes one node of a <c>GraphModule</c> from its inputs and attributes (<see cref="GraphOpContext"/>) and
/// returns the output tensor. Register it with <see cref="GraphOps.Register"/>; build it from public tensor operations
/// so it runs on every device and gradients flow through it.
/// </summary>
public delegate Tensor GraphOp(GraphOpContext context);

/// <summary>The node being run and its input values, for a <see cref="GraphOp"/>.</summary>
public sealed class GraphOpContext
{
    private readonly Func<int, object> _arg;

    /// <summary>The context of <paramref name="node"/>, whose input i is <c>arg(i)</c> (a <see cref="Tensor"/> or a <see cref="GraphValues.HostValue"/>).</summary>
    public GraphOpContext(GraphNode node, Func<int, object> arg)
    {
        Node = node;
        _arg = arg;
    }

    /// <summary>The node: its operation, input and output names and attributes.</summary>
    public GraphNode Node { get; }

    /// <summary>The node's operation name.</summary>
    public string Op => Node.Op;

    /// <summary>The number of inputs the node lists (an omitted optional input counts, as "").</summary>
    public int Count => Node.Inputs.Count;

    /// <summary>Whether input <paramref name="index"/> is given (listed and not "").</summary>
    public bool Has(int index) => index < Node.Inputs.Count && Node.Inputs[index].Length > 0;

    /// <summary>Input <paramref name="index"/> as a tensor (a computed value or a float constant).</summary>
    public Tensor Input(int index) => _arg(index) as Tensor ?? throw new InvalidOperationException($"input {index} must be a tensor");

    /// <summary>Input <paramref name="index"/> as integers (an integer constant, a shape computed on the host, or a tensor's values).</summary>
    public long[] Integers(int index) => GraphValues.AsIntegers(_arg(index));

    /// <summary>The integer attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public long Int(string name, long fallback) => Node.Attributes?[name] is JsonValue v ? (long)v : fallback;

    /// <summary>The number attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public float Float(string name, float fallback) => Node.Attributes?[name] is JsonValue v ? (float)v : fallback;

    /// <summary>The integer list attribute <paramref name="name"/>, or null.</summary>
    public long[]? Ints(string name) => Node.Attributes?[name] is JsonArray a ? [.. a.Select(v => (long)v!)] : null;

    /// <summary>The number list attribute <paramref name="name"/>, or null.</summary>
    public float[]? Floats(string name) => Node.Attributes?[name] is JsonArray a ? [.. a.Select(v => (float)v!)] : null;

    /// <summary>The string attribute <paramref name="name"/>, or null.</summary>
    public string? String(string name) => Node.Attributes?[name] is JsonValue v && v.TryGetValue(out string? s) ? s : null;
}

/// <summary>
/// The operations a <c>GraphModule</c> node can run, by name. The element-wise and tensor operations are
/// registered here (relu, exp, softmax, transpose, clip, pow, ...); add your own with <see cref="Register"/>, so a graph
/// using them runs, is written to JSON and is read back. The structural operations that also work on integer values
/// computed on the host (shapes, axes, indices) are part of <c>GraphModule</c> and cannot be replaced: layer,
/// identity, cast, add, sub, mul, div, concat, shape, gather, slice, unsqueeze, squeeze and reshape.
/// </summary>
public static class GraphOps
{
    internal static readonly HashSet<string> Core = new(StringComparer.Ordinal)
    {
        "layer", "identity", "cast", "add", "sub", "mul", "div", "concat", "shape", "gather", "slice", "unsqueeze", "squeeze", "reshape",
    };

    private static readonly Dictionary<string, GraphOp> Registry = new(StringComparer.Ordinal);

    // Idrak's operations (the element-wise and tensor operations ONNX names) are registered before the first use.
    static GraphOps() => LibraryGraphOps.RegisterAll();   // the built-in operations, on first use

    /// <summary>Registers (or replaces) the graph operation <paramref name="name"/>.</summary>
    public static void Register(string name, GraphOp op)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(op);
        if (Core.Contains(name))
        {
            throw new ArgumentException($"'{name}' is a structural operation of GraphModule and cannot be replaced.", nameof(name));
        }

        lock (Registry)
        {
            Registry[name] = op;
        }
    }

    /// <summary>Removes the graph operation <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The operation names a graph node can use: the structural ones and the registered ones.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Core, .. Registry.Keys];
            }
        }
    }

    /// <summary>The operation registered as <paramref name="name"/>.</summary>
    public static GraphOp Get(string name) =>
        TryGet(name) ?? throw new NotSupportedException($"No graph operation '{name}' is registered ({string.Join(", ", Names)}); add it with GraphOps.Register.");

    /// <summary>The operation registered as <paramref name="name"/>, or null.</summary>
    public static GraphOp? TryGet(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var op) ? op : null;
        }
    }

    /// <summary>Whether <paramref name="name"/> is an operation graphs can use: a structural one of GraphModule or a registered one.</summary>
    public static bool Contains(string name) => Core.Contains(name) || TryGet(name) is not null;
}

/// <summary>
/// One step of a <c>GraphModule</c>: an operation on named values that produces a named value.
/// </summary>
/// <param name="Op">"layer" (runs <paramref name="Layer"/> on the first input) or an operation (see <c>GraphModule</c> and <see cref="GraphOps"/>).</param>
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
