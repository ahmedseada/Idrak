// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// A tensor store that hands out a weight's stored bytes (plan 37, Q0), so the weight is kept in its file's format in
/// memory (<see cref="Abstraction.Generation.StoredWeight"/>) instead of expanded to float32 at load: a GGUF file's block
/// formats (q4_k, iq2_xxs, mxfp4, ...), a safetensors checkpoint's F8 and I8 weights with their scales. A store that is not
/// one is read as float32 values.
/// </summary>
public interface IStoredWeightStore : Abstraction.Formats.ITensorStore
{
    /// <summary>
    /// The registered element type (<see cref="ElementTypes"/>) the 2-D tensor <paramref name="name"/>'s rows are stored in,
    /// whole rows of it one after another (a block format's blocks along the last dimension), when its stored bytes, with
    /// <see cref="ScalingOf"/>, give exactly the values <see cref="Abstraction.Formats.ITensorStore.Read"/> gives; null
    /// otherwise (a float tensor, a tensor whose values the store changes on reading in another way).
    /// </summary>
    ElementType? StoredTypeOf(string name);

    /// <summary>
    /// Copies the stored bytes of the 2-D tensor <paramref name="name"/> [rows, columns] into <paramref name="destination"/>
    /// (<see cref="ElementType.ByteCount"/> of its values, rows in the order <see cref="Abstraction.Formats.ITensorStore.Read"/>
    /// gives them); false, having written nothing, when it cannot.
    /// </summary>
    bool ReadStored(string name, Span<byte> destination);

    /// <summary>The scales the tensor <paramref name="name"/> is stored with, or null when it has none (the default).</summary>
    Abstraction.Formats.StoredScaling? ScalingOf(string name) => null;
}
