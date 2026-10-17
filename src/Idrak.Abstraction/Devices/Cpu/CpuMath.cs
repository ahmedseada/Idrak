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
    // e^x as 2^k · e^f with k = round(x · log2 e) and f = x − k · ln 2 (ln 2 in two parts) in [−ln 2 / 2, ln 2 / 2], e^f a
    // degree-6 polynomial (Cephes' expf coefficients, within 8e-8 of the true value): the lanes' own code, inlined into the
    // optimized kernels, so e^x is fast from the first call. The runtime's Vector.Exp is a method of its own that the JIT
    // compiles unoptimized at first and does not inline; a kernel calling it ran 4 to 7 times slower for its first hundreds
    // of calls (exp over 262k elements: 1,300 µs against 180 µs settled). Below ExpUnderflow the result is 0, above
    // ExpOverflow +∞, NaN stays NaN; k is held within [−126, 127], so 2^k is a normal float.
    private const float ExpUnderflow = -87.3365f, ExpOverflow = 88.7228f, ExpLog2E = 1.44269504f, ExpLn2High = 0.693359375f, ExpLn2Low = -2.12194440e-4f;
    private const float ExpRoundMagic = 12582912f;                               // 1.5 · 2^23: adding it rounds to the nearest integer in the low mantissa bits
    private const float ExpP0 = 1.9875691500E-4f, ExpP1 = 1.3981999507E-3f, ExpP2 = 8.3334519073E-3f, ExpP3 = 4.1665795894E-2f, ExpP4 = 1.6666665459E-1f, ExpP5 = 5.0000001201E-1f;

    /// <summary>e^x of each lane (see the notes above): 0 below −87.34, +∞ above 88.72, NaN kept.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Exp(Vector<float> x)
    {
        var c = Vector.Min(Vector.Max(x, new Vector<float>(ExpUnderflow)), new Vector<float>(ExpOverflow));
        var magic = new Vector<float>(ExpRoundMagic);
        var kf = Vector.FusedMultiplyAdd(c, new Vector<float>(ExpLog2E), magic);
        var k = Vector.Min(Vector.Max(Vector.AsVectorInt32(kf) - Vector.AsVectorInt32(magic), new Vector<int>(-126)), new Vector<int>(127));
        var kFloat = Vector.ConvertToSingle(k);
        var f = Vector.FusedMultiplyAdd(kFloat, new Vector<float>(-ExpLn2High), c);
        f = Vector.FusedMultiplyAdd(kFloat, new Vector<float>(-ExpLn2Low), f);
        var p = Vector.FusedMultiplyAdd(new Vector<float>(ExpP0), f, new Vector<float>(ExpP1));
        p = Vector.FusedMultiplyAdd(p, f, new Vector<float>(ExpP2));
        p = Vector.FusedMultiplyAdd(p, f, new Vector<float>(ExpP3));
        p = Vector.FusedMultiplyAdd(p, f, new Vector<float>(ExpP4));
        p = Vector.FusedMultiplyAdd(p, f, new Vector<float>(ExpP5));
        p = Vector.FusedMultiplyAdd(p, f * f, f + Vector<float>.One);
        var result = p * Vector.AsVectorSingle((k + new Vector<int>(127)) << 23);
        result = Vector.ConditionalSelect(Vector.LessThan(x, new Vector<float>(ExpUnderflow)), Vector<float>.Zero, result);
        result = Vector.ConditionalSelect(Vector.GreaterThan(x, new Vector<float>(ExpOverflow)), new Vector<float>(float.PositiveInfinity), result);
        return Vector.ConditionalSelect(Vector.Equals(x, x), result, x);
    }

    /// <summary>e^x of each lane of a 512-bit vector, as <see cref="Exp(Vector{float})"/> computes it.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        var c = Vector512.Min(Vector512.Max(x, Vector512.Create(ExpUnderflow)), Vector512.Create(ExpOverflow));
        var magic = Vector512.Create(ExpRoundMagic);
        var kf = Vector512.FusedMultiplyAdd(c, Vector512.Create(ExpLog2E), magic);
        var k = Vector512.Min(Vector512.Max(kf.AsInt32() - magic.AsInt32(), Vector512.Create(-126)), Vector512.Create(127));
        var kFloat = Vector512.ConvertToSingle(k);
        var f = Vector512.FusedMultiplyAdd(kFloat, Vector512.Create(-ExpLn2High), c);
        f = Vector512.FusedMultiplyAdd(kFloat, Vector512.Create(-ExpLn2Low), f);
        var p = Vector512.FusedMultiplyAdd(Vector512.Create(ExpP0), f, Vector512.Create(ExpP1));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(ExpP2));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(ExpP3));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(ExpP4));
        p = Vector512.FusedMultiplyAdd(p, f, Vector512.Create(ExpP5));
        p = Vector512.FusedMultiplyAdd(p, f * f, f + Vector512<float>.One);
        var result = p * ((k + Vector512.Create(127)) << 23).AsSingle();
        result = Vector512.ConditionalSelect(Vector512.LessThan(x, Vector512.Create(ExpUnderflow)), Vector512<float>.Zero, result);
        result = Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512.Create(ExpOverflow)), Vector512.Create(float.PositiveInfinity), result);
        return Vector512.ConditionalSelect(Vector512.Equals(x, x), result, x);
    }

    /// <summary>erf of each lane within 1.5e-7 (Abramowitz and Stegun 7.1.26): sign(z) · (1 − t · poly(t) · e^(−z²)), t = 1 / (1 + p |z|).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Erf(Vector<float> z)
    {
        var a = Vector.Abs(z);
        var t = Vector<float>.One / Vector.FusedMultiplyAdd(new Vector<float>(0.3275911f), a, Vector<float>.One);
        var poly = Vector.FusedMultiplyAdd(new Vector<float>(1.061405429f), t, new Vector<float>(-1.453152027f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(1.421413741f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(-0.284496736f));
        poly = Vector.FusedMultiplyAdd(poly, t, new Vector<float>(0.254829592f));
        poly *= t;
        var e = Vector<float>.One - poly * Exp(-a * a);
        return Vector.ConditionalSelect(Vector.LessThan(z, Vector<float>.Zero), -e, e);
    }

    /// <summary>erf of one value, the same approximation as the lanes'.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Erf(float z)
    {
        float a = MathF.Abs(z), t = 1f / MathF.FusedMultiplyAdd(0.3275911f, a, 1f);
        float poly = MathF.FusedMultiplyAdd(1.061405429f, t, -1.453152027f);
        poly = MathF.FusedMultiplyAdd(poly, t, 1.421413741f);
        poly = MathF.FusedMultiplyAdd(poly, t, -0.284496736f);
        poly = MathF.FusedMultiplyAdd(poly, t, 0.254829592f);
        poly *= t;
        float e = 1f - poly * MathF.Exp(-a * a);
        return z < 0 ? -e : e;
    }

    /// <summary>1 / (1 + e^(−x)) of each lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Sigmoid(Vector<float> x) => Vector<float>.One / (Vector<float>.One + Exp(-x));

    /// <summary>y = x · scale + mask (mask empty: none), and the largest y (the softmax's shift), in one pass.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float ScaleAddMax(ReadOnlySpan<float> x, ReadOnlySpan<float> mask, Span<float> y, float scale)
    {
        int i = 0;
        float max = float.NegativeInfinity;
        ref float rx = ref MemoryMarshal.GetReference(x);
        ref float ry = ref MemoryMarshal.GetReference(y);
        if (Vector.IsHardwareAccelerated && x.Length >= Vector<float>.Count)
        {
            var s = new Vector<float>(scale);
            var m = new Vector<float>(float.NegativeInfinity);
            if (mask.IsEmpty)
            {
                for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    var v = Vector.LoadUnsafe(ref rx, (nuint)i) * s;
                    v.StoreUnsafe(ref ry, (nuint)i);
                    m = Vector.Max(m, v);
                }
            }
            else
            {
                ref float rm = ref MemoryMarshal.GetReference(mask);
                for (; i <= x.Length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    var v = Vector.LoadUnsafe(ref rx, (nuint)i) * s + Vector.LoadUnsafe(ref rm, (nuint)i);
                    v.StoreUnsafe(ref ry, (nuint)i);
                    m = Vector.Max(m, v);
                }
            }

            for (int l = 0; l < Vector<float>.Count; l++)
            {
                max = MathF.Max(max, m[l]);
            }
        }

        for (; i < x.Length; i++)
        {
            y[i] = x[i] * scale + (mask.IsEmpty ? 0f : mask[i]);
            max = MathF.Max(max, y[i]);
        }

        return max;
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
                var e = Exp(Vector512.LoadUnsafe(ref r, (nuint)i) - s);
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
                var e = Exp(Vector.LoadUnsafe(ref r, (nuint)i) - s);
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

    /// <summary>y = act(gate) · up for the gated activations (0: SiLU, 1: tanh GELU, 2: ReLU, 3: tanh(gate) · sigmoid(up)).</summary>
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
                var v = Vector.LoadUnsafe(ref u, (nuint)i);
                Vector<float> a;
                if (kind == 0)
                {
                    a = x / (one + Exp(-x));
                }
                else if (kind == 1)
                {
                    var t = Tanh(k * Vector.FusedMultiplyAdd(c * x * x, x, x));
                    a = half * x * (one + t);
                }
                else if (kind == 3)
                {
                    a = Tanh(x);
                    v = one / (one + Exp(-v));
                }
                else
                {
                    a = Vector.Max(x, Vector<float>.Zero);
                }

                (a * v).StoreUnsafe(ref o, (nuint)i);
            }
        }

        for (; i < y.Length; i++)
        {
            float x = gate[i], v = up[i];
            float a = kind switch
            {
                0 => x / (1f + MathF.Exp(-x)),
                1 => 0.5f * x * (1f + Tanh(0.7978845608f * (x + 0.044715f * x * x * x))),
                3 => Tanh(x),
                _ => x > 0f ? x : 0f,
            };
            y[i] = a * (kind == 3 ? 1f / (1f + MathF.Exp(-v)) : v);
        }
    }
}
