// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>
/// Tensors by name (Hugging Face names), read as float32: a safetensors checkpoint, a GGUF file, or whatever an
/// <see cref="ICheckpointFormat"/> opens.
/// </summary>
public interface ITensorStore : IDisposable
{
    /// <summary>The names of the stored tensors.</summary>
    IEnumerable<string> Names { get; }

    /// <summary>Whether a tensor named <paramref name="name"/> exists.</summary>
    bool Contains(string name);

    /// <summary>The shape of the tensor <paramref name="name"/>, outermost first ([rows, columns]).</summary>
    int[] ShapeOf(string name);

    /// <summary>The tensor <paramref name="name"/> as float32 values (row-major, as stored).</summary>
    float[] Read(string name);

    /// <summary>
    /// The 2-D tensor <paramref name="name"/> [rows, columns] transposed to [columns, rows]. Stores override this to
    /// transpose while reading, so the stored-order values never exist as a second full array.
    /// </summary>
    float[] ReadTransposed(string name)
    {
        var shape = ShapeOf(name);
        return Devices.HostParallel.Transpose(Read(name), shape[0], shape[1]);
    }
}
