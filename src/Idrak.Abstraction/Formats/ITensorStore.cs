// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Formats;

/// <summary>
/// Tensors by name (Hugging Face names), read as float32 (and bfloat16 ones also as their stored bits, where the store
/// copies them: <see cref="IBFloat16TensorStore"/>): a safetensors checkpoint, a GGUF file, or whatever an
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
    /// The format the tensor <paramref name="name"/> is stored in (float32, float16, bfloat16), or null when the store
    /// does not say (quantized GGUF types, a store of one's own). <see cref="Read"/> gives float32 values whatever it is;
    /// this tells a reader whether a narrower copy would hold the same values (a bfloat16 tensor kept as bfloat16).
    /// </summary>
    WeightFormat? FormatOf(string name) => null;

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

/// <summary>
/// A tensor store that copies bfloat16 tensors as their stored 16-bit values, so a bfloat16 weight is made without a
/// float32 copy of the tensor (a safetensors checkpoint does). A store that is not one is read as float32 values.
/// </summary>
public interface IBFloat16TensorStore : ITensorStore
{
    /// <summary>
    /// Copies the 2-D tensor <paramref name="name"/> [rows, columns], stored as bfloat16, as its 16-bit values (transposed
    /// to [columns, rows] when <paramref name="transposed"/>) into <paramref name="destination"/>, row r of the result at
    /// [r · <paramref name="stride"/>], the values after each row's end left as they are; each value as
    /// <c>BFloat16Weight.FromValues</c> rounds its float32 copy (<c>BFloat16Weight.CopyStoredBits</c>). So a bfloat16
    /// weight is made without a float32 copy of the tensor. Returns false, having written nothing, when the store does not
    /// hold the tensor as bfloat16 or cannot copy its values: the caller reads float32 values then.
    /// </summary>
    /// <param name="name">The tensor.</param>
    /// <param name="transposed">Whether the result is the tensor's transpose.</param>
    /// <param name="destination">At least (result rows - 1) · stride + result columns values.</param>
    /// <param name="stride">The distance between the starts of the result's rows, at least its columns.</param>
    bool ReadBFloat16(string name, bool transposed, Span<ushort> destination, int stride);
}
