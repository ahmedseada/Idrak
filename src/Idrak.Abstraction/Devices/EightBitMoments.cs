// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// The layout of 8-bit optimizer moments the devices update (<c>Idrak.Optimizers.AdamW8Bit</c>): a code per element
/// into a sorted 256-entry map, scaled per block of <see cref="BlockSize"/> elements.
/// </summary>
internal static class EightBitMoments
{
    /// <summary>Elements sharing one scale.</summary>
    public const int BlockSize = 256;

    /// <summary>The index of the code in the sorted <paramref name="map"/> nearest to <paramref name="x"/> (as the GPU kernel finds it).</summary>
    public static byte Nearest(ReadOnlySpan<float> map, float x)
    {
        int lo = 0;
        for (int step = 128; step > 0; step >>= 1)
        {
            if (map[lo + step] <= x)
            {
                lo += step;
            }
        }

        if (lo < 255 && map[lo + 1] - x < x - map[lo])
        {
            lo++;
        }

        return (byte)lo;
    }
}
