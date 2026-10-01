// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Onnx;

/// <summary>
/// Translates one ONNX node of a chain into builder steps (<see cref="OnnxImportContext.Add"/>) and returns the value
/// the chain continues from, usually <see cref="OnnxImportContext.Add"/>'s result (or <see cref="OnnxImportContext.Output"/>
/// for a node that is skipped). The import counterpart of <see cref="OnnxTranslator{T}"/>; register it with
/// <see cref="OnnxImportOps.Register"/>.
/// </summary>
public delegate string OnnxImportTranslator(OnnxImportContext context);

/// <summary>
/// The node being imported and what a translator may do with it: read its attributes and constant inputs, add builder
/// steps (with the weights to load into the layers they create), add notes, and report what cannot be imported.
/// </summary>
public sealed class OnnxImportContext
{
    internal OnnxImportContext(Importer importer, OnnxNode node, string input)
    {
        Importer = importer;
        Node = node;
        Input = input;
    }

    internal Importer Importer { get; }

    internal OnnxNode Node { get; }

    /// <summary>The node's operator type (for example "Relu").</summary>
    public string Op => Node.Op;

    /// <summary>The node's operator domain ("" for the standard operators).</summary>
    public string Domain => Node.Domain;

    /// <summary>The value the chain reached: the node's input that comes from the previous layer.</summary>
    public string Input { get; }

    /// <summary>The node's inputs (value names; "" for an omitted optional input).</summary>
    public IReadOnlyList<string> Inputs => Node.Inputs;

    /// <summary>The node's first output, where the chain usually continues.</summary>
    public string Output => Node.Outputs[0];

    /// <summary>The model's ONNX opset (attribute defaults change with it, for example Softmax's axis before opset 13).</summary>
    public long Opset => Importer.Opset;

    /// <summary>The shape of one sample at <see cref="Input"/> (without the batch dimension).</summary>
    public IReadOnlyList<int> CurrentShape => Importer.CurrentShape;

    /// <summary>The integer attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public long Int(string name, long fallback) => Node.Int(name, fallback);

    /// <summary>The float attribute <paramref name="name"/>, or <paramref name="fallback"/> when the node does not set it.</summary>
    public float Float(string name, float fallback) => Node.Float(name, fallback);

    /// <summary>The integer list attribute <paramref name="name"/>, or null.</summary>
    public long[]? Ints(string name) => Node.Ints(name);

    /// <summary>The string attribute <paramref name="name"/>, or null.</summary>
    public string? String(string name) => Node.String(name);

    /// <summary>The values of the constant (initializer) <paramref name="value"/> as floats, or null when it is computed.</summary>
    public float[]? Constant(string value) => Importer.Constant(value)?.AsFloats();

    /// <summary>The dimensions of the constant <paramref name="value"/>, or null when it is computed.</summary>
    public int[]? ConstantShape(string value) => Importer.Constant(value)?.Dims;

    /// <summary>
    /// Adds builder steps (for example <c>b =&gt; b.Linear(10)</c> or a registered <c>b =&gt; b.Op("scale", ...)</c>) and
    /// returns <paramref name="output"/> (the node's <see cref="Output"/> when null). <paramref name="load"/>, when given,
    /// fills the first layer they create with the file's weights.
    /// </summary>
    public string Add(Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load = null, string? output = null) =>
        Importer.Push(step, load, output ?? Output);

    /// <summary>Records an approximation made during import (it ends up in <see cref="ImportedNetwork.Notes"/>).</summary>
    public void Note(string note) => Importer.Note(note);

    /// <summary>The exception for a node this translator cannot import; <paramref name="what"/> says what is missing.</summary>
    public NotSupportedException Unsupported(string what) => Importer.Unsupported(Node, what);
}

/// <summary>
/// The ONNX operators <see cref="OnnxImport"/> turns into Idrak layers one node at a time, by operator type. The
/// built-in ones are registered here (Relu, Conv, BatchNormalization, Reshape, ...); add or replace one with
/// <see cref="Register"/>. Patterns of several nodes (Linear, GELU, attention, transformer layers, LSTM and GRU) are
/// matched before a node is looked up, and a model that is not a chain of layers is imported as a graph of fixed
/// primitives, where translators do not apply.
/// </summary>
public static class OnnxImportOps
{
    private static readonly Dictionary<string, OnnxImportTranslator> Registry = new(Importer.BuiltInOps(), StringComparer.Ordinal);

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
        TryGet(opType) ?? throw new NotSupportedException($"No ONNX import op '{opType}' is registered ({string.Join(", ", Names)}); add it with OnnxImportOps.Register.");

    internal static OnnxImportTranslator? TryGet(string opType)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(opType, out var translate) ? translate : null;
        }
    }
}
