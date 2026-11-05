// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Host-value collectives that also exchange a block of values with every rank (an all-to-all, plan 37 N0b): each rank
/// sends a different block to each other rank and receives one from each. A compression that reduces its own encoding
/// sends each owner its piece this way (each owner sums its piece, then the sums are all-gathered), so every rank sends
/// about 2·(W−1)/W of its encoded bucket, as a ring all-reduce does, instead of all of it to every rank. Collectives
/// without it are asked <c>is IAllToAllHostValueCollectives</c> once, and the caller takes another way.
/// </summary>
public interface IAllToAllHostValueCollectives : IHostValueCollectives
{
    /// <summary>Whether <see cref="TryAllToAll"/> works here (it returns false otherwise); the same for the whole run.</summary>
    bool SupportsAllToAll { get; }

    /// <summary>
    /// <paramref name="send"/> holds world · <paramref name="count"/> values, block r (values [r · count, r · count + count))
    /// for rank r; afterwards block r of <paramref name="receive"/> holds the block rank r sent this rank (block Rank this
    /// rank's own), every value's bits unchanged (a value travels as its float32 bits, so any bytes may ride in them). The
    /// two spans do not overlap. Returns false, having moved nothing, when it cannot this time.
    /// </summary>
    bool TryAllToAll(ReadOnlySpan<float> send, Span<float> receive, int count);
}
