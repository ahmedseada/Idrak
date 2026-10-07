// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using Idrak.Onnx.Abstractions;

namespace Idrak.Onnx;

/// <summary>
/// The <see cref="OnnxGraph"/> an <see cref="OnnxExporter"/> writes: nodes and initializers as Protocol Buffers messages,
/// then the whole model (<see cref="ToModel"/>).
/// </summary>
internal sealed class OnnxGraphWriter(OnnxExporter exporter) : OnnxGraph
{
    private readonly List<ProtoWriter> _nodes = [];
    private readonly List<ProtoWriter> _initializers = [];
    private readonly HashSet<string> _names = [];
    private int _counter;

    // The export this graph belongs to (for Module and the graph operations' translators).
    public OnnxExporter Exporter => exporter;

    public override OnnxValue Module(Module module, OnnxValue input)
    {
        ArgumentNullException.ThrowIfNull(module);
        ArgumentNullException.ThrowIfNull(input);
        return exporter.Emit(this, module, input);
    }

    public override OnnxValue Constant(string hint, ReadOnlySpan<float> values, params int[] dims)
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

    public override OnnxValue Constant(string hint, ReadOnlySpan<long> values, params int[] dims)
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

    public override IReadOnlyList<OnnxValue> Nodes(string op, IReadOnlyList<OnnxValue?> inputs, int outputs, params OnnxAttribute[] attributes)
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
            node.Message(5, Write(attribute));
        }

        _nodes.Add(node);
        return results;
    }

    // The graph output: an Identity node whose output has the requested name.
    public OnnxValue Output(OnnxValue value, string name, IReadOnlyList<int> shape)
    {
        var output = new OnnxValue(Unique(name), shape);
        _nodes.Add(new ProtoWriter().String(1, value.Name).String(2, output.Name).String(3, output.Name + "_node").String(4, "Identity"));
        return output;
    }

    public string Unique(string hint)
    {
        string name = hint;
        while (!_names.Add(name))
        {
            name = $"{hint}_{++_counter}";
        }

        return name;
    }

    public byte[] ToModel(OnnxValue input, IReadOnlyList<int> inputShape, OnnxValue output, IReadOnlyList<int> outputShape, string graphName,
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

    // An AttributeProto: the name, the value in the field of its type, and the type.
    private static ProtoWriter Write(OnnxAttribute attribute)
    {
        var a = new ProtoWriter().String(1, attribute.Name);
        switch (attribute.Value)
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
