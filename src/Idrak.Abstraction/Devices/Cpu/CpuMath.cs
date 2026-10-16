// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// Vectorized row operations shared by the CPU kernels: the widest vectors the machine runs well (512-bit where
// accelerated, else Vector<float>), with scalar tails.
internal static class CpuMath
{
    // tanh as an odd rational of the input clamped to ±7.905: x·a(x²) (degree 13) over b(x²) (degree 6), within a few ulp
    // of the true tanh at every size; ±1 past |x| = 9 (where the float tanh is 1); NaN kept. 1 - 2 / (e^(2x) + 1) left an
    // absolute error near 1e-7 whatever x, so a small tanh lost its digits; this needs no exponential either. The lanes and
    // the scalar form round alike (the same products, FMAs and division), so an element's bits do not depend on whether it
    // falls in a vector or the tail. The devices compute the same rational (Idrak.Gpu's TanhRational holds its coefficients).
    private const float TanhClamp = 7.90531110763549805f, TanhSaturated = 9f;
    private const float TanhA1 = 4.89352455891786e-03f, TanhA3 = 6.37261928875436e-04f, TanhA5 = 1.48572235717979e-05f, TanhA7 = 5.12229709037114e-08f,
        TanhA9 = -8.60467152213735e-11f, TanhA11 = 2.00018790482477e-13f, TanhA13 = -2.76076847742355e-16f;
    private const float TanhB0 = 4.89352518554385e-03f, TanhB2 = 2.26843463243900e-03f, TanhB4 = 1.18534705686654e-04f, TanhB6 = 1.19825839466702e-06f;

    /// <summary>tanh of each lane (see the rational above).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Tanh(Vector<float> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<float>(-TanhClamp)), new Vector<float>(TanhClamp));
        var x2 = c * c;
        var p = Vector.FusedMultiplyAdd(x2, new Vector<float>(TanhA13), new Vector<float>(TanhA11));
        p = Vector.FusedMultiplyAdd(x2, p, new Vector<float>(TanhA9));
        p = Vector.FusedMultiplyAdd(x2, p, new Vector<float>(TanhA7));
        p = Vector.FusedMultiplyAdd(x2, p, new Vector<float>(TanhA5));
        p = Vector.FusedMultiplyAdd(x2, p, new Vector<float>(TanhA3));
        p = Vector.FusedMultiplyAdd(x2, p, new Vector<float>(TanhA1));
        p *= c;
        var q = Vector.FusedMultiplyAdd(x2, new Vector<float>(TanhB6), new Vector<float>(TanhB4));
        q = Vector.FusedMultiplyAdd(x2, q, new Vector<float>(TanhB2));
        q = Vector.FusedMultiplyAdd(x2, q, new Vector<float>(TanhB0));
        var r = p / q;
        r = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<float>(TanhSaturated)), Vector<float>.One, r);
        r = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<float>(-TanhSaturated)), -Vector<float>.One, r);
        return Vector.ConditionalSelect(Vector.Equals(x, x), r, x);
    }

    /// <summary>tanh of one value, rounded as a lane of <see cref="Tanh(Vector{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Tanh(float x)
    {
        if (float.IsNaN(x))
        {
            return x;
        }

        if (x > TanhSaturated || x < -TanhSaturated)
        {
            return x > 0f ? 1f : -1f;
        }

        float c = MathF.Min(MathF.Max(x, -TanhClamp), TanhClamp);
        float x2 = c * c;
        float p = MathF.FusedMultiplyAdd(x2, TanhA13, TanhA11);
        p = MathF.FusedMultiplyAdd(x2, p, TanhA9);
        p = MathF.FusedMultiplyAdd(x2, p, TanhA7);
        p = MathF.FusedMultiplyAdd(x2, p, TanhA5);
        p = MathF.FusedMultiplyAdd(x2, p, TanhA3);
        p = MathF.FusedMultiplyAdd(x2, p, TanhA1);
        p *= c;
        float q = MathF.FusedMultiplyAdd(x2, TanhB6, TanhB4);
        q = MathF.FusedMultiplyAdd(x2, q, TanhB2);
        q = MathF.FusedMultiplyAdd(x2, q, TanhB0);
        return p / q;
    }
    /// <summary>The largest value (negative infinity when empty).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Max(ReadOnlySpan<float> x)
    {
        int i = 0;
        float max = float.NegativeInfinity;
        ref float r = ref MemoryMarshal.GetReference(x);
        if (Vector512.IsHardwareAccelerated && x.Length >= 16)
        {
            var m = Vector512.Create(float.NegativeInfinity);
            for (; i <= x.Length - 16; i += 16)
            {
                m = Vector512.Max(m, Vector512.LoadUnsafe(ref r, (nuint)i));
            }

            var h = Vector256.Max(m.GetLower(), m.GetUpper());
            max = MaxLanes(Vector128.Max(h.GetLower(), h.GetUpper()));
        }
        else if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var m = new Vector<float>(float.NegativeInfinity);
            for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                m = Vector.Max(m, Vector.LoadUnsafe(ref r, (nuint)i));
            }

            for (int j = 0; j < Vector<float>.Count; j++)
            {
                max = MathF.Max(max, m[j]);
            }
        }

        for (; i < x.Length; i++)
        {
            max = MathF.Max(max, x[i]);
        }

        return max;
    }

    private static float MaxLanes(Vector128<float> v) => MathF.Max(MathF.Max(v[0], v[1]), MathF.Max(v[2], v[3]));

    /// <summary>x[i] = exp(x[i] - shift) in place; returns the sum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float ExpShifted(Span<float> x, float shift)
    {
        int i = 0;
        float sum = 0f;
        ref float r = ref MemoryMarshal.GetReference(x);
        if (Vector512.IsHardwareAccelerated && x.Length >= 16)
        {
            var s = Vector512.Create(shift);
            var total = Vector512<float>.Zero;
            for (; i <= x.Length - 16; i += 16)
            {
                var e = Vector512.Exp(Vector512.LoadUnsafe(ref r, (nuint)i) - s);
                e.StoreUnsafe(ref r, (nuint)i);
                total += e;
            }

            sum = Vector512.Sum(total);
        }
        else if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var s = new Vector<float>(shift);
            var total = Vector<float>.Zero;
            for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var e = Vector.Exp(Vector.LoadUnsafe(ref r, (nuint)i) - s);
                e.StoreUnsafe(ref r, (nuint)i);
                total += e;
            }

            sum = Vector.Sum(total);
        }

        for (; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - shift);
            sum += x[i];
        }

        return sum;
    }

    /// <summary>x[i] *= factor in place.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scale(Span<float> x, float factor)
    {
        int i = 0;
        ref float r = ref MemoryMarshal.GetReference(x);
        if (Vector.IsHardwareAccelerated)
        {
            var f = new Vector<float>(factor);
            for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (Vector.LoadUnsafe(ref r, (nuint)i) * f).StoreUnsafe(ref r, (nuint)i);
            }
        }

        for (; i < x.Length; i++)
        {
            x[i] *= factor;
        }
    }

    /// <summary>y = act(gate) · up for the gated activations (0: SiLU, 1: tanh GELU, else ReLU).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Gated(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> y, int kind)
    {
        int i = 0;
        ref float g = ref MemoryMarshal.GetReference(gate);
        ref float u = ref MemoryMarshal.GetReference(up);
        ref float o = ref MemoryMarshal.GetReference(y);
        if (Vector.IsHardwareAccelerated)
        {
            var one = Vector<float>.One;
            var half = new Vector<float>(0.5f);
            var k = new Vector<float>(0.7978845608f);
            var c = new Vector<float>(0.044715f);
            for (; i <= y.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var x = Vector.LoadUnsafe(ref g, (nuint)i);
                Vector<float> a;
                if (kind == 0)
                {
                    a = x / (one + Vector.Exp(-x));
                }
                else if (kind == 1)
                {
                    var t = Tanh(k * Vector.FusedMultiplyAdd(c * x * x, x, x));
                    a = half * x * (one + t);
                }
                else
                {
                    a = Vector.Max(x, Vector<float>.Zero);
                }

                (a * Vector.LoadUnsafe(ref u, (nuint)i)).StoreUnsafe(ref o, (nuint)i);
            }
        }

        for (; i < y.Length; i++)
        {
            float x = gate[i];
            float a = kind switch
            {
                0 => x / (1f + MathF.Exp(-x)),
                1 => 0.5f * x * (1f + Tanh(0.7978845608f * (x + 0.044715f * x * x * x))),
                _ => x > 0f ? x : 0f,
            };
            y[i] = a * up[i];
        }
    }
}
