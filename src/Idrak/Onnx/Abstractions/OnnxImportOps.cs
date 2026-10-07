// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Onnx.Abstractions;

/// <summary>
/// Translates one ONNX node of a chain into network steps (<see cref="OnnxImportContext.Add"/>) and returns the value
/// the chain continues from, usually <see cref="OnnxImportContext.Add"/>'s result (or <see cref="OnnxImportContext.Output"/>
/// for a node that is skipped). The import counterpart of <see cref="OnnxTranslator{T}"/>; register it with
/// <see cref="OnnxImportOps.Register"/>.
/// </summary>
public delegate string OnnxImportTranslator(OnnxImportContext context);

/// <summary>
/// The node being imported and what a translator may do with it: read its attributes and constant inputs, add network
/// steps (with the weights to load into the layers they create), add graph operations, add notes, and report what
/// cannot be imported. The same translator serves a chain of layers and a graph (<see cref="IsGraph"/>): in a graph,
/// the steps it adds become a layer node and <see cref="AddGraphOp"/> adds an operation node. The importer
/// (<c>OnnxImport</c> in Idrak) implements it.
/// </summary>
public abstract class OnnxImportContext
{
    /// <summary>The node's operator type (for example "Relu").</summary>
    public abstract string Op { get; }

    /// <summary>The node's operator domain ("" for the standard operators).</summary>
    public abstract string Domain { get; }

    /// <summary>The value the chain reached: the node's input that comes from the previous layer (in a graph, its first computed input).</summary>
    public abstract string Input { get; }

    /// <summary>Whether the node is imported into a graph (a graph module of <see cref="GraphNode"/>s) rather than a chain of layers.</summary>
    public abstract bool IsGraph { get; }

    /// <summary>The node's inputs (value names; "" for an omitted optional input).</summary>
    public abstract IReadOnlyList<string> Inputs { get; }

    /// <summary>The node's first output, where the chain usually continues.</summary>
    public abstract string Output { get; }

    /// <summary>The model's ONNX opset (attribute defaults change with it, for example Softmax's axis before opset 13).</summary>
    public abstract long Opset { get; }

    /// <summary>
    /// The shape of one sample where the next step continues (at first <see cref="Input"/>), without the batch dimension.
    /// In a graph it is measured by running the graph so far, which needs the model's sample shape.
    /// </summary>
    public abstract IReadOnlyList<int> CurrentShape { get; }

    /// <summary>The integer attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public abstract long Int(string name, long fallback);

    /// <summary>The float attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public abstract float Float(string name, float fallback);

    /// <summary>The integer list attribute <paramref name="name"/>, or null.</summary>
    public abstract long[]? Ints(string name);

    /// <summary>The string attribute <paramref name="name"/>, or null.</summary>
    public abstract string? String(string name);

    /// <summary>The values of the constant (initializer) <paramref name="value"/> as floats, or null when it is computed.</summary>
    public abstract float[]? Constant(string value);

    /// <summary>The dimensions of the constant <paramref name="value"/>, or null when it is computed.</summary>
    public abstract int[]? ConstantShape(string value);

    /// <summary>
    /// Adds the network step registered as <paramref name="step"/> (a built-in one such as "linear", "relu" or "reshape",
    /// or one of your own, registered in <c>NetworkOps</c>) with <paramref name="arguments"/> (the keys of its JSON besides
    /// "op", for example <c>{"out": 10, "bias": true}</c>), and returns <paramref name="output"/> (the node's
    /// <see cref="Output"/> when null). <paramref name="load"/>, when given, fills the first layer the step creates with
    /// the file's weights. Since every step is registered by name, the imported network can be written to JSON and
    /// saved in a model package.
    /// </summary>
    public abstract string Add(string step, JsonObject? arguments = null, Action<Module>? load = null, string? output = null);

    /// <summary>
    /// Adds a graph node running the operation <paramref name="op"/> (registered in <see cref="GraphOps"/>) on
    /// <paramref name="inputs"/> (the node's <see cref="Inputs"/> when null; constants among them become the graph's
    /// constants) with <paramref name="attributes"/> (the node's own when null), and returns <paramref name="output"/>
    /// (the node's <see cref="Output"/> when null). A chain of layers cannot hold it, so the model is then imported as
    /// a graph.
    /// </summary>
    public abstract string AddGraphOp(string op, JsonObject? attributes = null, IReadOnlyList<string>? inputs = null, string? output = null);

    /// <summary>Records an approximation made during import (it ends up in the imported network's notes).</summary>
    public abstract void Note(string note);

    /// <summary>The exception for a node this translator cannot import; <paramref name="what"/> says what is missing.</summary>
    public abstract NotSupportedException Unsupported(string what);
}

/// <summary>
/// The ONNX operators the importer (<c>OnnxImport</c> in Idrak) turns into Idrak layers one node at a time, by
/// operator type. Idrak registers the built-in ones (Relu, Conv, BatchNormalization, Reshape, ...); add or replace
/// one with <see cref="Register"/>. Patterns of several nodes (Linear, GELU, attention, transformer layers, LSTM and
/// GRU) are matched before a node is looked up. A model that is not a chain of layers is imported as a graph: there the
/// built-in operators become graph operations (<see cref="GraphOps"/>), and a translator you registered (a new operator
/// or one replacing a built-in) runs as well, its network steps becoming layer nodes and its
/// <see cref="OnnxImportContext.AddGraphOp"/> calls operation nodes.
/// </summary>
public static class OnnxImportOps
{
    private static readonly Dictionary<string, OnnxImportTranslator> Registry = new(StringComparer.Ordinal);

    static OnnxImportOps() => OnnxBuiltIns.RegisterImports();   // the built-in translators, on first use

    /// <summary>Registers (or replaces) how to import nodes of the operator type <paramref name="opType"/>.</summary>
    public static void Register(string opType, OnnxImportTranslator translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(opType);
        ArgumentNullException.ThrowIfNull(translate);
        lock (Registry)
        {
            Registry[opType] = translate;
        }
    }

    /// <summary>Removes the operator type <paramref name="opType"/>; returns whether it was registered.</summary>
    public static bool Unregister(string opType)
    {
        lock (Registry)
        {
            return Registry.Remove(opType);
        }
    }

    /// <summary>The registered operator types.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The translator registered for <paramref name="opType"/>.</summary>
    public static OnnxImportTranslator Get(string opType) =>
        Find(opType) ?? throw new NotSupportedException($"No ONNX import op '{opType}' is registered ({string.Join(", ", Names)}); add it with OnnxImportOps.Register.");

    /// <summary>The translator registered for <paramref name="opType"/>, or null.</summary>
    public static OnnxImportTranslator? Find(string opType)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(opType, out var translate) ? translate : null;
        }
    }
}
