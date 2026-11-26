// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Onnx.Abstractions;

/// <summary>
/// Translates one module into ONNX nodes: <paramref name="input"/> is the module's input value, <paramref name="outputShape"/>
/// the shape it produces (measured by running the module, -1 for the batch dimension). Returns the output value. A
/// translator given to one export (<c>OnnxExporter.Module</c> and <c>OnnxExporter.Lambda</c> in Idrak).
/// </summary>
public delegate OnnxValue OnnxTranslator<in T>(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) where T : Module;
