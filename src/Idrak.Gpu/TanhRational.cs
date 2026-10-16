// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu;

// tanh as the CPU computes it (CpuMath.Tanh, whose coefficients these repeat): an odd rational of the input clamped to
// ±Clamp, x·a(x²) (degree 13) over b(x²) (degree 6), a few ulp from the true tanh at every size, ±1 past |x| = Saturated,
// NaN kept. It keeps a small tanh's relative precision (1 - 2 / (e^(2x) + 1) left an absolute error near 1e-7 whatever x)
// and needs no exponential. A[i] multiplies x^(2i + 1), B[i] x^(2i).
internal static class TanhRational
{
    public const float Clamp = 7.90531110763549805f, Saturated = 9f;

    public static readonly float[] A =
        [4.89352455891786e-03f, 6.37261928875436e-04f, 1.48572235717979e-05f, 5.12229709037114e-08f, -8.60467152213735e-11f, 2.00018790482477e-13f, -2.76076847742355e-16f];

    public static readonly float[] B = [4.89352518554385e-03f, 2.26843463243900e-03f, 1.18534705686654e-04f, 1.19825839466702e-06f];
}
