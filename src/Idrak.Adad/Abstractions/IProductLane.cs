// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Adad.Abstractions;

/// <summary>
/// The operands of y = x · Wᵀ on a device: x (m × k float32), W as stored (n rows of k values in its type), y (m × n
/// float32), as device addresses.
/// </summary>
public readonly record struct ProductOperands(ulong X, ulong W, ulong Y, int M, int N, int K);

/// <summary>
/// A lane of an operation for weights of a description (decision 7, kept minimal for N0): one default per operation and
/// overrides matched by the description's fields, never by a type's name. A lane offers variants (its own shapes and
/// splits), measured per shape; the operation measures its lanes' choices against the default.
/// </summary>
/// <typeparam name="TDescription">The description the lane reads its weights through.</typeparam>
public interface IProductLane<in TDescription>
{
    /// <summary>The lane's name (reports only; never a key).</summary>
    string Name { get; }

    /// <summary>Whether the lane reads weights of <paramref name="description"/> (its fields alone).</summary>
    bool Fits(TDescription description);

    /// <summary>
    /// Readies the lane's kernels for <paramref name="description"/> (once a description); false, with the reason, when the
    /// device refuses every one of them (the lane is then no candidate).
    /// </summary>
    bool Prepare(TDescription description, out string reason);

    /// <summary>The lane's variants for a shape (made when it is measured, never on a call) and the one it runs unmeasured.</summary>
    int[] Variants(int m, int n, int k, out int fallback);

    /// <summary>Enqueues y = x · Wᵀ by variant <paramref name="variant"/>.</summary>
    void Run(int variant, in ProductOperands operands);
}
