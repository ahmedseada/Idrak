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
    /// The tensor <paramref name="name"/> as its source stores it, a tensor of its stored type on <paramref name="device"/>
    /// (its bytes as they lie, never widened), where its stored layout is <paramref name="stored"/> (a projection's
    /// [out, in] rows, as a product of stored weights reads them; a table's [rows, columns]) and its type is not the
    /// device's own float32; null otherwise (the default: the caller reads its float32 values with <see cref="Read"/>).
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="stored">Its shape in the stored layout.</param>
    /// <param name="device">Where the tensor goes.</param>
    Tensor? ReadStored(string name, IReadOnlyList<int> stored, Device device) => null;

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
    /// A text that changes whenever the values <see cref="Read"/> gives for <paramref name="name"/> [<paramref name="shape"/>]
    /// may change (where the stored tensor is, its file's size and time of change), for a cache of what is made from them:
    /// the tensor counts as used, as a read would make it. Null when the source cannot say (the default,
    /// or a source that has not the tensor), and nothing made from it is cached.
    /// </summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">Its shape.</param>
    string? IdentityOf(string name, IReadOnlyList<int> shape) => null;
}
