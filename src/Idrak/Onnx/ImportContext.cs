// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Onnx.Abstractions;

namespace Idrak.Onnx;

/// <summary>The <see cref="OnnxImportContext"/> the importer gives translators: one node, in a chain or in a graph.</summary>
internal sealed class ImportContext : OnnxImportContext
{
    public ImportContext(Importer importer, OnnxNode node, string input, bool graph = false)
    {
        Importer = importer;
        Node = node;
        Input = input;
        IsGraph = graph;
        Current = input;
    }

    public Importer Importer { get; }

    public OnnxNode Node { get; }

    public override string Op => Node.Op;

    public override string Domain => Node.Domain;

    public override string Input { get; }

    public override bool IsGraph { get; }

    // The value the next step continues from: the input, then the output of each step added.
    public string Current { get; private set; }

    public override IReadOnlyList<string> Inputs => Node.Inputs;

    public override string Output => Node.Outputs[0];

    public override long Opset => Importer.Opset;

    public override IReadOnlyList<int> CurrentShape => IsGraph ? Importer.SampleShape(Node, Current) : Importer.CurrentShape;

    public override long Int(string name, long fallback) => Node.Int(name, fallback);

    public override float Float(string name, float fallback) => Node.Float(name, fallback);

    public override long[]? Ints(string name) => Node.Ints(name);

    public override string? String(string name) => Node.String(name);

    public override float[]? Constant(string value) => Importer.Constant(value)?.AsFloats();

    public override int[]? ConstantShape(string value) => Importer.Constant(value)?.Dims;

    public override string Add(string step, JsonObject? arguments = null, Action<Module>? load = null, string? output = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(step);
        return Add(b => b.Op(step, arguments), load, output);
    }

    // Builder steps of the built-in translators (for example b => b.ReLU()).
    public string Add(Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load = null, string? output = null)
    {
        ArgumentNullException.ThrowIfNull(step);
        Current = IsGraph ? Importer.PushGraph(Node, Current, step, load, output ?? Output) : Importer.Push(step, load, output ?? Output);
        return Current;
    }

    public override string AddGraphOp(string op, JsonObject? attributes = null, IReadOnlyList<string>? inputs = null, string? output = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(op);
        if (!IsGraph)
        {
            throw Unsupported($"the graph operation '{op}' in a chain of layers");
        }

        Current = Importer.AddGraphOp(Node, op, inputs ?? Node.Inputs, attributes ?? Importer.Attributes(Node), output ?? Output);
        return Current;
    }

    public override void Note(string note) => Importer.Note(note);

    public override NotSupportedException Unsupported(string what) => Importer.Unsupported(Node, what);
}
