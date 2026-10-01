// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Backends.Cpu;

// Vectorized row operations shared by the CPU kernels: the widest vectors the machine runs well (512-bit where
// accelerated, else Vector<float>), with scalar tails.
internal static class CpuMath
{
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
            var two = new Vector<float>(2f);
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
                    var t = one - two / (Vector.Exp(two * k * Vector.FusedMultiplyAdd(c * x * x, x, x)) + one);
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
                1 => 0.5f * x * (1f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x))),
                _ => x > 0f ? x : 0f,
            };
            y[i] = a * up[i];
        }
    }
}
