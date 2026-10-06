// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>Rounding to bfloat16, shared by the devices and the packed bfloat16 weights.</summary>
internal static class BFloat16Bits
{
    /// <summary>A value's bfloat16 bits (round to nearest, ties to even; NaN stays NaN).</summary>
    public static ushort Round(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
        {
            return (ushort)((bits >> 16) | 0x40);
        }

        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }
}
