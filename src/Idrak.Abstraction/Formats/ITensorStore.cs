// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

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
    /// Whether the tensors <paramref name="first"/> and <paramref name="second"/> have the same shape and the same float32
    /// values as <see cref="Read"/> gives them (each pair compared as <c>float.Equals</c> does: a NaN equals a NaN, 0 equals
    /// -0), for example a tied head checked against the embedding a checkpoint may repeat it as. The default reads both
    /// whole; stores override it to compare piece by piece, so neither tensor is held whole on the host and the reading
    /// stops at the first difference.
    /// </summary>
    bool SameValues(string first, string second) =>
        ShapeOf(first).AsSpan().SequenceEqual(ShapeOf(second)) && Read(first).AsSpan().SequenceEqual(Read(second));

    /// <summary>
    /// The 2-D tensor <paramref name="name"/> [rows, columns] transposed to [columns, rows]. Stores override this to
    /// transpose while reading, so the stored-order values never exist as a second full array.
    /// </summary>
    float[] ReadTransposed(string name)
    {
        var shape = ShapeOf(name);
        return HostParallel.Transpose(Read(name), shape[0], shape[1]);
    }

    /// <summary>
    /// Reads rows [<paramref name="first"/>, <paramref name="first"/> + <paramref name="count"/>) of the tensor
    /// <paramref name="name"/> (its outermost dimension) as float32 values into <paramref name="destination"/>, row-major, as
    /// <see cref="Read"/> gives them: so a large tensor (an embedding table) is read a block of rows at a time and never held
    /// whole. Returns false, having written nothing, when the store cannot read part of a tensor (the default): the caller
    /// reads it whole.
    /// </summary>
    /// <param name="name">The tensor.</param>
    /// <param name="first">The first row.</param>
    /// <param name="count">The rows.</param>
    /// <param name="destination">At least <paramref name="count"/> rows of values.</param>
    bool ReadRows(string name, int first, int count, Span<float> destination) => false;

    /// <summary>
    /// A text that changes whenever the stored values of the tensor <paramref name="name"/> may change (the file holding it,
    /// its size and time of change, the tensor's place and type in it), for caches of what is made from it (packed weights);
    /// null when the store cannot say (the default), and nothing made from the tensor is cached.
    /// </summary>
    string? IdentityOf(string name) => null;

    /// <summary>
    /// The time spent reading tensor bytes from storage so far, apart from converting them, or null when the store does
    /// not measure it (the default): <c>PretrainedModel.LoadTimes</c> then counts reading and converting as one.
    /// </summary>
    TimeSpan? ReadTime => null;

    /// <summary>
    /// The bytes of tensor values read from storage so far, or null when the store does not count them (the default): a
    /// rank that loads only its own tensors (plan 23, C8b) shows what it read against the whole checkpoint.
    /// </summary>
    long? BytesRead => null;
}
