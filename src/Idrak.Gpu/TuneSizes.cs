// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu;

/// <summary>
/// The size classes of measured choices. A size that follows the data (a batch, rows of a product, a sequence's length, an
/// image's height or width) is keyed by its class, not its value: work whose sizes change from step to step (lines of
/// text of every width, prompts of every length) would otherwise measure its candidates again at nearly every step. Up to
/// <see cref="Exact"/> a size is its own class (decoding and small batches choose differently at each size); above it the
/// classes are four a doubling (1, 1.25, 1.5 and 1.75 times a power of two), a size rounded up to its class, so the
/// sizes of a class are within a quarter of each other and their fastest kernel is the same as a rule. The choice is
/// still measured on the device in use, on the first shape of the class met; a later shape of the class uses it when it
/// is one of that shape's own candidates (else the formula's choice). Sizes a model fixes (channels, features, a
/// window) stay exact in the keys.
/// </summary>
internal static class TuneSizes
{
    /// <summary>Sizes up to this are their own class.</summary>
    public const int Exact = 32;

    /// <summary>The class of <paramref name="size"/>: the size itself up to <see cref="Exact"/>, else the next of 1, 1.25, 1.5 or 1.75 times a power of two.</summary>
    public static int Class(int size)
    {
        if (size <= Exact)
        {
            return size;
        }

        long power = 1L << (63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)size)); // the largest power of two ≤ size
        long quarter = power / 4;
        long rounded = (size + quarter - 1) / quarter * quarter;                               // up to the next quarter of the octave
        return (int)Math.Min(rounded, int.MaxValue);
    }
}
