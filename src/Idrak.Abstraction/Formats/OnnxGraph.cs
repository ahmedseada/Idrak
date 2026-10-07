// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

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

    /// <summary>The attribute's value: a <see cref="long"/>, <see cref="float"/>, <see cref="string"/>, <c>long[]</c> or <c>float[]</c>.</summary>
    public object Value { get; }

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
}

/// <summary>
/// An ONNX graph being written: constants (initializers) and operator nodes. Export translators
/// (<see cref="OnnxExportOps"/>, or one exporter's own) add their nodes to it; the exporter (<c>OnnxExporter</c> in
/// Idrak.Onnx) implements it and writes the result as an .onnx file. Operators follow ONNX opset <see cref="Opset"/>.
/// </summary>
public abstract class OnnxGraph
{
    /// <summary>The opset exports target: the operators and attributes translators write follow it.</summary>
    public const int Opset = 17;

    /// <summary>
    /// Adds the nodes of <paramref name="module"/> applied to <paramref name="input"/> (whose shape must be known) and
    /// returns its output, translated as the export translates every module (its own translators first, then
    /// <see cref="OnnxExportOps"/>). For translators of modules made of other modules (a custom residual block).
    /// </summary>
    public abstract OnnxValue Module(Module module, OnnxValue input);

    /// <summary>A float constant with the given dimensions (an empty <paramref name="dims"/> for a scalar).</summary>
    public abstract OnnxValue Constant(string hint, ReadOnlySpan<float> values, params int[] dims);

    /// <summary>An int64 constant (shapes, axes, indices) with the given dimensions (an empty <paramref name="dims"/> for a scalar).</summary>
    public abstract OnnxValue Constant(string hint, ReadOnlySpan<long> values, params int[] dims);

    /// <summary>Adds a node with <paramref name="outputs"/> outputs (for example LSTM's Y and Y_h) and returns them.</summary>
    public abstract IReadOnlyList<OnnxValue> Nodes(string op, IReadOnlyList<OnnxValue?> inputs, int outputs, params OnnxAttribute[] attributes);

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
}
