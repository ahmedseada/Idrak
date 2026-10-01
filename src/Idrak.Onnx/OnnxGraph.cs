// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Onnx;

/// <summary>A value in an ONNX graph: its name and, when known, its shape (-1 for the batch dimension).</summary>
/// <param name="Name">The value's name in the graph.</param>
/// <param name="Shape">Its shape, with -1 for the dynamic batch dimension; null when not tracked.</param>
public sealed record OnnxValue(string Name, IReadOnlyList<int>? Shape);

/// <summary>An attribute of an ONNX node; create with the <c>Of</c> methods.</summary>
public sealed class OnnxAttribute
{
    private OnnxAttribute(string name, object value)
    {
        Name = name;
        Value = value;
    }

    /// <summary>The attribute's name.</summary>
    public string Name { get; }

    internal object Value { get; }

    /// <summary>An integer attribute.</summary>
    public static OnnxAttribute Of(string name, long value) => new(name, value);

    /// <summary>A float attribute.</summary>
    public static OnnxAttribute Of(string name, float value) => new(name, value);

    /// <summary>A string attribute.</summary>
    public static OnnxAttribute Of(string name, string value) => new(name, value);

    /// <summary>An integer-list attribute.</summary>
    public static OnnxAttribute Of(string name, long[] values) => new(name, values);

    /// <summary>A float-list attribute.</summary>
    public static OnnxAttribute Of(string name, float[] values) => new(name, values);

    internal ProtoWriter Write()
    {
        var a = new ProtoWriter().String(1, Name);
        switch (Value)
        {
            case long i:
                a.Int(3, i).Int(20, 2);
                break;
            case float f:
                a.Float(2, f).Int(20, 1);
                break;
            case string s:
                a.String(4, s).Int(20, 3);
                break;
            case long[] ints:
                a.PackedInts(8, ints).Int(20, 7);
                break;
            case float[] floats:
                a.PackedFloats(7, floats).Int(20, 6);
                break;
        }

        return a;
    }
}

/// <summary>
/// Builds an ONNX graph: constants (initializers) and operator nodes. The exporter uses it for the built-in layers;
/// custom translators (<see cref="OnnxExporter.Lambda"/>, <see cref="OnnxExporter.Module{T}"/>) use it for theirs.
/// Operators follow ONNX opset <see cref="Opset"/>.
/// </summary>
public sealed class OnnxGraph
{
    /// <summary>The opset the exporter targets.</summary>
    public const int Opset = 17;

    private readonly List<ProtoWriter> _nodes = [];
    private readonly List<ProtoWriter> _initializers = [];
    private readonly HashSet<string> _names = [];
    private int _counter;

    internal OnnxGraph()
    {
    }

    /// <summary>A float constant with the given dimensions (an empty <paramref name="dims"/> for a scalar).</summary>
    public OnnxValue Constant(string hint, ReadOnlySpan<float> values, params int[] dims)
    {
        CheckSize(values.Length, dims);
        string name = Unique(hint);
        var raw = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), values[i]);
        }

        _initializers.Add(new ProtoWriter().PackedInts(1, dims.Select(d => (long)d)).Int(2, 1).String(8, name).Bytes(9, raw));
        return new OnnxValue(name, dims);
    }

    /// <summary>An int64 constant (shapes, axes, indices) with the given dimensions (an empty <paramref name="dims"/> for a scalar).</summary>
    public OnnxValue Constant(string hint, ReadOnlySpan<long> values, params int[] dims)
    {
        CheckSize(values.Length, dims);
        string name = Unique(hint);
        var raw = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8), values[i]);
        }

        _initializers.Add(new ProtoWriter().PackedInts(1, dims.Select(d => (long)d)).Int(2, 7).String(8, name).Bytes(9, raw));
        return new OnnxValue(name, dims);
    }

    /// <summary>A 1-D int64 constant.</summary>
    public OnnxValue Ints(string hint, params long[] values) => Constant(hint, values, values.Length);

    /// <summary>A scalar float constant.</summary>
    public OnnxValue Scalar(string hint, float value) => Constant(hint, [value]);

    /// <summary>
    /// Adds a node with one output and returns it. <paramref name="inputs"/> may contain null for an omitted optional input;
    /// <paramref name="shape"/> is the output's shape when known (it is only recorded, not checked).
    /// </summary>
    public OnnxValue Node(string op, IReadOnlyList<OnnxValue?> inputs, IReadOnlyList<int>? shape = null, params OnnxAttribute[] attributes) =>
        Nodes(op, inputs, 1, attributes)[0] with { Shape = shape };

    /// <summary>Adds a node with <paramref name="outputs"/> outputs (for example LSTM's Y and Y_h).</summary>
    public IReadOnlyList<OnnxValue> Nodes(string op, IReadOnlyList<OnnxValue?> inputs, int outputs, params OnnxAttribute[] attributes)
    {
        var results = Enumerable.Range(0, outputs).Select(_ => new OnnxValue(Unique(op.ToLowerInvariant()), null)).ToList();
        var node = new ProtoWriter();
        foreach (var input in inputs)
        {
            node.String(1, input?.Name ?? "");
        }

        foreach (var output in results)
        {
            node.String(2, output.Name);
        }

        node.String(3, results[0].Name + "_node").String(4, op);
        foreach (var attribute in attributes)
        {
            node.Message(5, attribute.Write());
        }

        _nodes.Add(node);
        return results;
    }

    // The graph output: an Identity node whose output has the requested name.
    internal OnnxValue Output(OnnxValue value, string name, IReadOnlyList<int> shape)
    {
        var output = new OnnxValue(Unique(name), shape);
        _nodes.Add(new ProtoWriter().String(1, value.Name).String(2, output.Name).String(3, output.Name + "_node").String(4, "Identity"));
        return output;
    }

    internal string Unique(string hint)
    {
        string name = hint;
        while (!_names.Add(name))
        {
            name = $"{hint}_{++_counter}";
        }

        return name;
    }

    internal byte[] ToModel(OnnxValue input, IReadOnlyList<int> inputShape, OnnxValue output, IReadOnlyList<int> outputShape, string graphName,
        string producer, string producerVersion, IReadOnlyDictionary<string, string> metadata)
    {
        var graph = new ProtoWriter();
        foreach (var node in _nodes)
        {
            graph.Message(1, node);
        }

        graph.String(2, graphName);
        foreach (var initializer in _initializers)
        {
            graph.Message(5, initializer);
        }

        graph.Message(11, ValueInfo(input.Name, inputShape));
        graph.Message(12, ValueInfo(output.Name, outputShape));

        var model = new ProtoWriter().Int(1, 8).String(2, producer).String(3, producerVersion).Message(7, graph)
            .Message(8, new ProtoWriter().String(1, "").Int(2, Opset));
        foreach (var (key, value) in metadata)
        {
            model.Message(14, new ProtoWriter().String(1, key).String(2, value));
        }

        return model.ToArray();
    }

    // A float tensor value; -1 dimensions become the symbolic dimension "batch".
    private static ProtoWriter ValueInfo(string name, IReadOnlyList<int> shape)
    {
        var dims = new ProtoWriter();
        foreach (int d in shape)
        {
            dims.Message(1, d < 0 ? new ProtoWriter().String(2, "batch") : new ProtoWriter().Int(1, d));
        }

        var tensor = new ProtoWriter().Int(1, 1).Message(2, dims);
        return new ProtoWriter().String(1, name).Message(2, new ProtoWriter().Message(1, tensor));
    }

    private static void CheckSize(int count, int[] dims)
    {
        if (dims.Aggregate(1, (a, b) => a * b) != count)
        {
            throw new ArgumentException($"{count} values do not fill a tensor of shape [{string.Join(", ", dims)}].");
        }
    }
}
