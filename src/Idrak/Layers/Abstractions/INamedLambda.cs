// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers.Abstractions;

/// <summary>
/// A layer that runs a function it is told by name (Idrak's <c>Lambda</c>): what is written for it in other formats (an
/// ONNX export's translator, <c>OnnxExportOps.RegisterLambda</c>) is found by that name, never by the layer's text.
/// </summary>
public interface INamedLambda
{
    /// <summary>The function's name ("MeanOverTime", "Reshape", one of one's own), as its translators are registered by.</summary>
    string Name { get; }
}
