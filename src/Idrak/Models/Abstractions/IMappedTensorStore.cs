// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// Where a tensor's stored bytes lie in a checkpoint's file (plan 37, W3): <paramref name="Length"/> bytes from
/// <paramref name="Offset"/> in <paramref name="File"/>, its rows one after another in the order the store's reads give
/// them, of <paramref name="Type"/>, whose widening to float32 (with <see cref="IStoredWeightStore.ScalingOf"/> when the
/// store has scales for it) gives exactly the values the store's reads give.
/// </summary>
/// <param name="File">The file's path.</param>
/// <param name="Offset">Where the bytes start in it.</param>
/// <param name="Length">Their count.</param>
/// <param name="Type">The registered element type they are stored in (float32, or a format of their own: q4_k, float8_e4m3, ...).</param>
public sealed record StoredLocation(string File, long Offset, long Length, ElementType Type);

/// <summary>
/// A tensor store whose weights can be read in place from its file (plan 37, W3): it says where a matrix's stored bytes
/// lie (<see cref="Locate"/>), so a device that reads a file's mapping in place (<see cref="IMappedStorageBackend"/>: the
/// CPU) multiplies them there, with no second copy in memory (<see cref="Abstraction.Generation.StoredWeight.FromMapped"/>).
/// A safetensors checkpoint and a GGUF file are such stores.
/// </summary>
public interface IMappedTensorStore : Abstraction.Formats.ITensorStore
{
    /// <summary>
    /// Where the 2-D tensor <paramref name="name"/>'s stored bytes lie, as its reads give it; null, with
    /// <paramref name="why"/> for people, when the file does not hold it so (its rows put back in another order on reading,
    /// a value added on reading, a type with no widening, float32 bytes on a machine of the other byte order).
    /// </summary>
    StoredLocation? Locate(string name, out string? why);
}
