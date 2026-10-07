// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// Translates one module into ONNX nodes: <paramref name="input"/> is the module's input value, <paramref name="outputShape"/>
/// the shape it produces (measured by running the module, -1 for the batch dimension). Returns the output value.
/// Register it for every export with <see cref="OnnxExportOps.Register{T}"/>, or for one export (<c>OnnxExporter.Module</c> in Idrak.Onnx).
/// </summary>
public delegate OnnxValue OnnxTranslator<in T>(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) where T : Module;

/// <summary>
/// Translates one operation node of a graph module (<see cref="GraphNode"/>, <see cref="GraphOps"/>) into ONNX nodes and
/// returns the output value. Register it with <see cref="OnnxExportOps.RegisterGraphOp"/> (or for one export,
/// <c>OnnxExporter.GraphOp</c> in Idrak.Onnx) under the operation's name.
/// </summary>
public delegate OnnxValue OnnxGraphOpTranslator(OnnxGraphOpContext context);

/// <summary>The graph node being exported, its input values and its measured output shape, for an <see cref="OnnxGraphOpTranslator"/>.</summary>
public sealed class OnnxGraphOpContext
{
    private readonly Func<int, long[]?> _integers;

    /// <summary>
    /// The context of exporting <paramref name="node"/> into <paramref name="graph"/>: its inputs as ONNX values, its
    /// measured output shape, and <paramref name="integers"/>, which gives the integer values of an input by its index
    /// (null for a tensor). Exporters create it; translators read it.
    /// </summary>
    public OnnxGraphOpContext(OnnxGraph graph, GraphNode node, IReadOnlyList<OnnxValue?> inputs, IReadOnlyList<int>? outputShape, Func<int, long[]?> integers)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(integers);
        Graph = graph;
        Node = node;
        Inputs = inputs;
        OutputShape = outputShape;
        _integers = integers;
    }

    /// <summary>The ONNX graph to add nodes to.</summary>
    public OnnxGraph Graph { get; }

    /// <summary>The graph node: its operation, inputs, output and attributes.</summary>
    public GraphNode Node { get; }

    /// <summary>The node's inputs as ONNX values (null for an omitted optional input).</summary>
    public IReadOnlyList<OnnxValue?> Inputs { get; }

    /// <summary>The shape of the node's output (-1 for the batch dimension), or null when it is an integer value (a shape, axes or indices).</summary>
    public IReadOnlyList<int>? OutputShape { get; }

    /// <summary>The integer values of input <paramref name="index"/> (a constant, or a shape computed for the export's sample), or null for a tensor.</summary>
    public long[]? Integers(int index) => _integers(index);

    /// <summary>The integer attribute <paramref name="name"/>, or null when the node does not set it.</summary>
    public long? Int(string name) => Node.Attributes?[name] is JsonValue v ? (long)v : null;

    /// <summary>The number attribute <paramref name="name"/>, or null when the node does not set it.</summary>
    public float? Float(string name) => Node.Attributes?[name] is JsonValue v ? (float)v : null;

    /// <summary>The integer list attribute <paramref name="name"/>, or null.</summary>
    public long[]? Ints(string name) => Node.Attributes?[name] is JsonArray a ? [.. a.Select(v => (long)v!)] : null;

    /// <summary>Adds the ONNX node <paramref name="op"/> on <see cref="Inputs"/> with <see cref="OutputShape"/>.</summary>
    public OnnxValue Operator(string op, params OnnxAttribute[] attributes) => Graph.Node(op, Inputs, OutputShape, attributes);
}

/// <summary>
/// How modules, lambdas and graph operations are written to ONNX, for every export. Idrak.Onnx registers the built-in
/// layers (Linear, Conv2d, BatchNorm, LSTM, Sequential, GraphModule, ...), the builder's lambdas (MeanOverTime,
/// FirstStep, LastStep, Reshape) and every built-in graph operation (<see cref="GraphOps"/>); add or replace one with
/// <see cref="Register{T}"/>, <see cref="RegisterLambda"/> or <see cref="RegisterGraphOp"/>. A module uses the translator
/// registered for its own type or its nearest registered base type (<see cref="Find"/>). Translators given to one
/// exporter (<c>OnnxExporter.Module</c>, <c>Lambda</c> and <c>GraphOp</c> in Idrak.Onnx) take precedence over these.
/// </summary>
public static class OnnxExportOps
{
    private static readonly Dictionary<Type, (Delegate Original, OnnxTranslator<Module> Translate)> Modules = [];
    private static readonly Dictionary<string, OnnxTranslator<Module>> Lambdas = new(StringComparer.Ordinal);
    private static readonly Dictionary<string, OnnxGraphOpTranslator> GraphOpTranslators = new(StringComparer.Ordinal);

    // The built-ins of the first-party assemblies (Idrak.Onnx) are registered before the first use.
    static OnnxExportOps() => LibraryDefaults.Ensure();

    /// <summary>Registers (or replaces) how modules of type <typeparamref name="T"/> (and types derived from it without their own translator) are exported.</summary>
    public static void Register<T>(OnnxTranslator<T> translate) where T : Module
    {
        ArgumentNullException.ThrowIfNull(translate);
        lock (Modules)
        {
            Modules[typeof(T)] = (translate, (g, m, x, s) => translate(g, (T)m, x, s));
        }
    }

    /// <summary>Removes the translator of modules of type <typeparamref name="T"/>; returns whether one was registered.</summary>
    public static bool Unregister<T>() where T : Module
    {
        lock (Modules)
        {
            return Modules.Remove(typeof(T));
        }
    }

    /// <summary>The module types with a registered translator.</summary>
    public static IReadOnlyCollection<Type> Types
    {
        get
        {
            lock (Modules)
            {
                return [.. Modules.Keys];
            }
        }
    }

    /// <summary>The translator registered for modules of exactly the type <typeparamref name="T"/>.</summary>
    public static OnnxTranslator<T> Get<T>() where T : Module
    {
        lock (Modules)
        {
            return Modules.TryGetValue(typeof(T), out var entry) ? (OnnxTranslator<T>)entry.Original
                : throw new NotSupportedException($"No ONNX export translator for {typeof(T).Name} is registered ({string.Join(", ", Types.Select(t => t.Name).Order())}); add it with OnnxExportOps.Register<{typeof(T).Name}>.");
        }
    }

    /// <summary>Registers (or replaces) how the lambdas named <paramref name="name"/> (their <c>ToString()</c>) are exported.</summary>
    public static void RegisterLambda(string name, OnnxTranslator<Module> translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(translate);
        lock (Lambdas)
        {
            Lambdas[name] = translate;
        }
    }

    /// <summary>Removes the lambda translator <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool UnregisterLambda(string name)
    {
        lock (Lambdas)
        {
            return Lambdas.Remove(name);
        }
    }

    /// <summary>The lambda names with a registered translator.</summary>
    public static IReadOnlyCollection<string> LambdaNames
    {
        get
        {
            lock (Lambdas)
            {
                return [.. Lambdas.Keys];
            }
        }
    }

    /// <summary>Registers (or replaces) how graph module nodes (<see cref="GraphNode"/>) running the operation <paramref name="op"/> are exported.</summary>
    public static void RegisterGraphOp(string op, OnnxGraphOpTranslator translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(op);
        ArgumentNullException.ThrowIfNull(translate);
        lock (GraphOpTranslators)
        {
            GraphOpTranslators[op] = translate;
        }
    }

    /// <summary>Removes the graph operation translator <paramref name="op"/>; returns whether it was registered.</summary>
    public static bool UnregisterGraphOp(string op)
    {
        lock (GraphOpTranslators)
        {
            return GraphOpTranslators.Remove(op);
        }
    }

    /// <summary>The graph operations with a registered translator.</summary>
    public static IReadOnlyCollection<string> GraphOpNames
    {
        get
        {
            lock (GraphOpTranslators)
            {
                return [.. GraphOpTranslators.Keys];
            }
        }
    }

    /// <summary>The translator registered for the graph operation <paramref name="op"/>.</summary>
    public static OnnxGraphOpTranslator GetGraphOp(string op) => FindGraphOp(op)
        ?? throw new NotSupportedException($"No ONNX export translator for the graph operation '{op}' is registered ({string.Join(", ", GraphOpNames)}); add it with OnnxExportOps.RegisterGraphOp.");

    /// <summary>The translator of modules of type <paramref name="type"/>: the one registered for it or for its nearest registered base type, or null.</summary>
    public static OnnxTranslator<Module>? Find(Type type)
    {
        lock (Modules)
        {
            for (Type? t = type; t is not null && t != typeof(object); t = t.BaseType)
            {
                if (Modules.TryGetValue(t, out var entry))
                {
                    return entry.Translate;
                }
            }

            return null;
        }
    }

    /// <summary>The translator registered for the lambdas named <paramref name="name"/>, or null.</summary>
    public static OnnxTranslator<Module>? FindLambda(string name)
    {
        lock (Lambdas)
        {
            return Lambdas.TryGetValue(name, out var translate) ? translate : null;
        }
    }

    /// <summary>The translator registered for the graph operation <paramref name="op"/>, or null.</summary>
    public static OnnxGraphOpTranslator? FindGraphOp(string op)
    {
        lock (GraphOpTranslators)
        {
            return GraphOpTranslators.TryGetValue(op, out var translate) ? translate : null;
        }
    }
}
