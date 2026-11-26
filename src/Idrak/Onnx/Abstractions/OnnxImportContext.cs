// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Onnx.Abstractions;

/// <summary>
/// Translates one ONNX node of a chain into network steps and returns the value the chain continues from: one of the
/// importer's built-in translators (<c>OnnxImport</c> in Idrak).
/// </summary>
public delegate string OnnxImportTranslator(OnnxImportContext context);

/// <summary>
/// The node being imported and what a translator may do with it: read its attributes and constant inputs, add notes,
/// and report what cannot be imported. The importer (<c>OnnxImport</c> in Idrak) implements it.
/// </summary>
public abstract class OnnxImportContext
{
    /// <summary>The node's operator type (for example "Relu").</summary>
    public abstract string Op { get; }

    /// <summary>The node's operator domain ("" for the standard operators).</summary>
    public abstract string Domain { get; }

    /// <summary>The value the chain reached: the node's input that comes from the previous layer.</summary>
    public abstract string Input { get; }

    /// <summary>The node's inputs (value names; "" for an omitted optional input).</summary>
    public abstract IReadOnlyList<string> Inputs { get; }

    /// <summary>The node's first output, where the chain usually continues.</summary>
    public abstract string Output { get; }

    /// <summary>The model's ONNX opset (attribute defaults change with it, for example Softmax's axis before opset 13).</summary>
    public abstract long Opset { get; }

    /// <summary>The shape of one sample where the next step continues (at first <see cref="Input"/>), without the batch dimension.</summary>
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

    /// <summary>Records an approximation made during import (it ends up in the imported network's notes).</summary>
    public abstract void Note(string note);

    /// <summary>The exception for a node this translator cannot import; <paramref name="what"/> says what is missing.</summary>
    public abstract NotSupportedException Unsupported(string what);
}
