// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Onnx.Abstractions;

namespace Idrak.Onnx;

/// <summary>The <see cref="OnnxImportContext"/> the importer gives its translators: one node of a chain.</summary>
internal sealed class ImportContext : OnnxImportContext
{
    public ImportContext(Importer importer, OnnxNode node, string input)
    {
        Importer = importer;
        Node = node;
        Input = input;
        Current = input;
    }

    public Importer Importer { get; }

    public OnnxNode Node { get; }

    public override string Op => Node.Op;

    public override string Domain => Node.Domain;

    public override string Input { get; }

    // The value the next step continues from: the input, then the output of each step added.
    public string Current { get; private set; }

    public override IReadOnlyList<string> Inputs => Node.Inputs;

    public override string Output => Node.Outputs[0];

    public override long Opset => Importer.Opset;

    public override IReadOnlyList<int> CurrentShape => Importer.CurrentShape;

    public override long Int(string name, long fallback) => Node.Int(name, fallback);

    public override float Float(string name, float fallback) => Node.Float(name, fallback);

    public override long[]? Ints(string name) => Node.Ints(name);

    public override string? String(string name) => Node.String(name);

    public override float[]? Constant(string value) => Importer.Constant(value)?.AsFloats();

    public override int[]? ConstantShape(string value) => Importer.Constant(value)?.Dims;

    // Builder steps of the built-in translators (for example b => b.ReLU()).
    public string Add(Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load = null, string? output = null)
    {
        ArgumentNullException.ThrowIfNull(step);
        Current = Importer.Push(step, load, output ?? Output);
        return Current;
    }

    public override void Note(string note) => Importer.Note(note);

    public override NotSupportedException Unsupported(string what) => Importer.Unsupported(Node, what);
}
