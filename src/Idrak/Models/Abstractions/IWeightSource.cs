// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// Provides weights by name for <c>DecoderSpec.Build</c>, in Idrak's layout (Linear weights [in, out]). The
/// names are listed in <c>DecoderSpec</c>. Implementations read files (for example a pretrained-model package
/// translating another framework's names and layouts) or anything else.
/// </summary>
public interface IWeightSource
{
    /// <summary>The values of the tensor <paramref name="name"/> with <paramref name="shape"/>, or null when the source does not have it.</summary>
    float[]? Read(string name, IReadOnlyList<int> shape);

    /// <summary>
    /// Reads rows [<paramref name="first"/>, <paramref name="first"/> + <paramref name="count"/>) of the matrix
    /// <paramref name="name"/> [rows, columns] (<paramref name="shape"/>, as <see cref="Read"/> takes it) into
    /// <paramref name="destination"/>: the values <see cref="Read"/> gives for those rows. So a large matrix (an embedding
    /// table a tied head is packed from) is read a block at a time. False, having written nothing, when the source cannot
    /// (the default, or a source that has not the matrix): the caller reads it whole.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [rows, columns].</param>
    /// <param name="first">The first row.</param>
    /// <param name="count">The rows.</param>
    /// <param name="destination">At least <paramref name="count"/> rows of values.</param>
    bool ReadRows(string name, IReadOnlyList<int> shape, int first, int count, Span<float> destination) => false;

    /// <summary>
    /// The matrix <paramref name="name"/> [rows, columns] (<paramref name="shape"/>, as <see cref="Read"/> takes it), or its
    /// transpose [columns, rows] when <paramref name="transposed"/>, as bfloat16 weights on <paramref name="device"/>; null
    /// when the source does not have it. Always the weights <see cref="BFloat16Weight.FromValues(ReadOnlySpan{float}, int, int, Device)"/>
    /// makes from <see cref="Read"/>'s values, bit for bit; the default makes exactly those. A source that holds the values
    /// as bfloat16 (a checkpoint's stored bits) copies them instead (<see cref="BFloat16Weight.FromBits"/>), so the build
    /// never holds a float32 copy of the matrix: <c>DecoderSpec.Build</c> reads its bfloat16 projections, head and
    /// embedding table through this.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [rows, columns].</param>
    /// <param name="device">Where the weights are created.</param>
    /// <param name="transposed">Make the weights of the transpose, [columns, rows] (a tied head from the embedding table).</param>
    BFloat16Weight? ReadBFloat16(string name, IReadOnlyList<int> shape, Device device, bool transposed = false)
    {
        if (Read(name, shape) is not { } values)
        {
            return null;
        }

        return transposed
            ? BFloat16Weight.FromValues(HostParallel.Transpose(values, shape[0], shape[1]), shape[1], shape[0], device)
            : BFloat16Weight.FromValues(values, shape[0], shape[1], device);
    }
}
