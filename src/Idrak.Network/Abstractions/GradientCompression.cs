// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Networking.Abstractions;

/// <summary>
/// One piece of a gradient bucket: values [<paramref name="Offset"/>, <paramref name="Offset"/> + <paramref name="Count"/>)
/// of parameter <paramref name="Parameter"/> (its index in the optimizer's list), at <paramref name="At"/> in the bucket.
/// </summary>
public readonly record struct GradientPiece(int Parameter, int Offset, int Count, int At);

/// <summary>
/// One rank's compression of gradient buckets on their way across the ranks (plan 23, phase C4): it sums a bucket of this
/// rank's gradient values over every rank, sending less than the values where it can. Made once per run by its
/// <see cref="IGradientCompression"/>; any state it keeps (the error-feedback residual of top-k) is allocated
/// once and lives with it. <see cref="Reduce"/> is called one bucket at a time, in bucket order, the same buckets on every
/// rank, from the thread that reduces the step's buckets (not the training thread while the backward pass overlaps): it
/// works on the host values it is given and on the collectives' host values (<see cref="IHostValueCollectives"/>).
/// </summary>
public interface IGradientCompressor : IDisposable
{
    /// <summary>The compression's name.</summary>
    string Name { get; }

    /// <summary>
    /// Replaces <paramref name="values"/> (this rank's gradient values of a bucket, as <paramref name="pieces"/> lay them out)
    /// with the sum over every rank of each rank's values times its <paramref name="weight"/> (its rows in the step, or 1),
    /// as compressed: the same values on every rank afterwards.
    /// </summary>
    void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces);

    /// <summary>
    /// The pieces were summed again in full, uncompressed (their gradient changed after they were sent): whatever the
    /// compression kept of them this step is forgotten (top-k's residual).
    /// </summary>
    void Forget(ReadOnlySpan<GradientPiece> pieces)
    {
    }
}

/// <summary>One kind of gradient compression ("none", "top-k").</summary>
public interface IGradientCompression
{
    /// <summary>The name ("none", "top-k").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>A rank's compressor of the gradients of <paramref name="parameters"/> over <paramref name="collectives"/>.</summary>
    /// <exception cref="ArgumentException">A setting it needs is missing (top-k's share).</exception>
    IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options);
}

