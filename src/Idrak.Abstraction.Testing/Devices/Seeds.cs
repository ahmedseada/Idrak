// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Testing;

// Seeds derived from a run's seed, the same in every process (HashCode.Combine is not), so a reported seed reproduces.
internal static class Seeds
{
    // A non-negative seed from the values (a MurmurHash3 finalizer over each in turn).
    public static int Mix(params ReadOnlySpan<int> values)
    {
        uint h = 0x9E3779B9u;
        foreach (int value in values)
        {
            h ^= (uint)value;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
        }

        return (int)(h & int.MaxValue);
    }
}
