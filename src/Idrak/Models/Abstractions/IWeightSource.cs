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
    /// Reads output rows [<paramref name="first"/>, <paramref name="first"/> + <paramref name="count"/>) of the weight
    /// matrix <paramref name="name"/> [inputs, outputs] (<paramref name="shape"/>, as <see cref="Read"/> takes it) into
    /// <paramref name="destination"/>, each of inputs values: the columns of what <see cref="Read"/> gives, read in the order
    /// a file storing the matrix [outputs, inputs] holds them. So a format keeping a row an output (a stored format, the
    /// product routes') packs a block of rows at a time (<c>PackedWeightFormats.PackRows</c>), with no float32 copy of the
    /// matrix and no transposition. False, having written nothing, when the source cannot (the default): the caller reads
    /// the matrix whole.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [inputs, outputs].</param>
    /// <param name="first">The first output row.</param>
    /// <param name="count">The rows.</param>
    /// <param name="destination">At least <paramref name="count"/> rows of inputs values.</param>
    bool ReadOutputRows(string name, IReadOnlyList<int> shape, int first, int count, Span<float> destination) => false;

    /// <summary>
    /// The bfloat16 bits of the weight matrix <paramref name="name"/> [inputs, outputs] as its output rows (outputs rows of
    /// inputs values) into <paramref name="destination"/>, where the source holds the matrix as bfloat16 (its bits copied:
    /// no float32 and no transposition); with an empty <paramref name="destination"/>, only whether it can. False, having
    /// written nothing, otherwise (the default).
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [inputs, outputs].</param>
    /// <param name="destination">Inputs × outputs bits, or empty to ask.</param>
    bool ReadOutputBFloat16(string name, IReadOnlyList<int> shape, Span<ushort> destination) => false;

    /// <summary>
    /// A text that changes whenever the values <see cref="Read"/> gives for <paramref name="name"/> [<paramref name="shape"/>]
    /// may change (where the stored tensor is, its file's size and time of change), for a cache of what is made from them
    /// (packed weights): the tensor counts as used, as a read would make it. Null when the source cannot say (the default,
    /// or a source that has not the tensor), and nothing made from it is cached.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape.</param>
    string? IdentityOf(string name, IReadOnlyList<int> shape) => null;

    /// <summary>
    /// The matrix <paramref name="name"/> [rows, columns] (<paramref name="shape"/>) kept in its file's format on
    /// <paramref name="device"/> (plan 37, Q0: a <see cref="Abstraction.Generation.StoredWeight"/>, whose products equal the
    /// float32 weights' bit for bit), when the source stores it so, <paramref name="keep"/> allows it and the device
    /// multiplies that format; null otherwise (the default), and the caller reads float32 values. A source that tallies
    /// what it kept and expanded counts the weight either way.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [rows, columns] (inputs, outputs).</param>
    /// <param name="device">Where the weights are created.</param>
    /// <param name="keep">Whether weights may be kept as stored (<c>DecoderBuildOptions.KeepPacked</c>).</param>
    Abstraction.Generation.IPackedWeight? ReadStored(string name, IReadOnlyList<int> shape, Device device, bool keep) => null;

    /// <summary>
    /// The matrix <paramref name="name"/> [rows, columns] (<paramref name="shape"/>) read in place from the checkpoint's file
    /// mapping on <paramref name="device"/> (plan 37, W3: a <see cref="Abstraction.Generation.StoredWeight"/> over the
    /// mapped bytes, whose products equal the float32 weights' bit for bit, with no second copy in memory), when the file
    /// holds it as the product reads it (float32, or a format of its own <paramref name="keep"/> allows and the device
    /// multiplies, as <see cref="ReadStored"/>) and the device reads a mapping in place
    /// (<see cref="IMappedStorageBackend"/>); null otherwise (the default), and the caller reads it as before. A source that
    /// counts copies counts the weight's then (<see cref="MappedWeights"/>), with why.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape, [rows, columns] (inputs, outputs).</param>
    /// <param name="device">Where the weights are created.</param>
    /// <param name="keep">Whether weights stored in a format of their own may be kept in it (<c>DecoderBuildOptions.KeepPacked</c>).</param>
    Abstraction.Generation.IPackedWeight? ReadMapped(string name, IReadOnlyList<int> shape, Device device, bool keep) => null;

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
