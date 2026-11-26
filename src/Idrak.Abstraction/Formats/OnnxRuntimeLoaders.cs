// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// A runtime that runs an ONNX model as a <see cref="Module"/>, outside Idrak's own import of it (Idrak.Onnx.Runtime
/// registers "onnxruntime": ONNX Runtime).
/// </summary>
public interface IOnnxRuntimeLoader
{
    /// <summary>The model whose .onnx file's bytes are <paramref name="model"/>, run by this runtime (inference only).</summary>
    /// <exception cref="NotSupportedException">The model has inputs or outputs the runtime does not take.</exception>
    Module Load(byte[] model);
}

