// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Tests.Gcn;

/// <summary>
/// The float arithmetic of gfx9's vector ALU as the interpreter models it. Adds, multiplies, fused multiply-adds,
/// conversions and square roots round as IEEE 754 does (round to nearest even), as the hardware does. The
/// transcendentals the hardware computes to about 1 ULP are computed in double precision and rounded once:
/// v_exp_f32 is 2^x, v_log_f32 log2(x), v_sin_f32 and v_cos_f32 take revolutions (sin(2πx)), v_rcp_f32 1/x,
/// v_rsq_f32 1/√x; a kernel's result may so differ from the device's by about an ULP per such step, which the
/// device-against-CPU tolerance covers. v_mad_f32 and v_mac_f32 round the product before the add and flush
/// denormals, as gfx9 does.
/// </summary>
internal static class GcnFloat
{
    internal static float F(ulong bits) => BitConverter.UInt32BitsToSingle((uint)bits);

    internal static ulong B(float value) => BitConverter.SingleToUInt32Bits(value);

    internal static double D(ulong bits) => BitConverter.UInt64BitsToDouble(bits);

    internal static ulong B(double value) => BitConverter.DoubleToUInt64Bits(value);

    internal static Half H(ulong bits) => BitConverter.UInt16BitsToHalf((ushort)bits);

    internal static ulong B(Half value) => BitConverter.HalfToUInt16Bits(value);

    /// <summary>A denormal flushed to a zero of its sign.</summary>
    internal static float Flush(float x) => float.IsSubnormal(x) ? (float.IsNegative(x) ? -0f : 0f) : x;

    internal static float Quiet(float x) => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(x) | 0x00400000);

    /// <summary>IEEE minNum (IEEE mode): a NaN operand yields the other; -0 is less than +0.</summary>
    internal static float Min(float a, float b)
    {
        if (float.IsNaN(a))
        {
            return float.IsNaN(b) ? Quiet(a) : b;
        }

        if (float.IsNaN(b))
        {
            return a;
        }

        if (a == b)
        {
            return float.IsNegative(a) ? a : b;
        }

        return a < b ? a : b;
    }

    internal static float Max(float a, float b)
    {
        if (float.IsNaN(a))
        {
            return float.IsNaN(b) ? Quiet(a) : b;
        }

        if (float.IsNaN(b))
        {
            return a;
        }

        if (a == b)
        {
            return float.IsNegative(a) ? b : a;
        }

        return a > b ? a : b;
    }

    internal static double Min(double a, double b) => double.IsNaN(a) ? b : double.IsNaN(b) ? a : a == b ? (double.IsNegative(a) ? a : b) : Math.Min(a, b);

    internal static double Max(double a, double b) => double.IsNaN(a) ? b : double.IsNaN(b) ? a : a == b ? (double.IsNegative(a) ? b : a) : Math.Max(a, b);

    /// <summary>v_med3_f32: the middle value; with a NaN, the least of the three (as min3).</summary>
    internal static float Med3(float a, float b, float c)
    {
        if (float.IsNaN(a) || float.IsNaN(b) || float.IsNaN(c))
        {
            return Min(Min(a, b), c);
        }

        float max = Max(Max(a, b), c);
        return max == a ? Max(b, c) : max == b ? Max(a, c) : Max(a, b);
    }

    internal static float Mad(float a, float b, float c) => Flush(Flush(Flush(a) * Flush(b)) + Flush(c));

    internal static float MulLegacy(float a, float b) => a == 0 || b == 0 ? 0f : a * b;

    internal static float Exp(float x) => (float)Math.Pow(2, x);

    internal static float Log(float x) => x == 0 ? float.NegativeInfinity : x < 0 ? float.NaN : (float)Math.Log2(x);

    internal static float Sin(float x) => float.IsFinite(x) ? (float)Math.Sin(2 * Math.PI * (x - Math.Round((double)x))) : float.NaN;

    internal static float Cos(float x) => float.IsFinite(x) ? (float)Math.Cos(2 * Math.PI * (x - Math.Round((double)x))) : float.NaN;

    internal static float Rcp(float x) => (float)(1.0 / x);

    internal static float Rsq(float x) => (float)(1.0 / Math.Sqrt(x));

    internal static float Fract(float x)
    {
        if (float.IsNaN(x))
        {
            return x;
        }

        if (float.IsInfinity(x))
        {
            return float.NaN;
        }

        return Math.Min(x - MathF.Floor(x), BitConverter.UInt32BitsToSingle(0x3F7FFFFF));
    }

    internal static double Fract(double x) => double.IsNaN(x) ? x : double.IsInfinity(x) ? double.NaN : Math.Min(x - Math.Floor(x), BitConverter.UInt64BitsToDouble(0x3FEFFFFFFFFFFFFF));

    internal static float Ldexp(float x, int e) => MathF.ScaleB(x, Math.Clamp(e, -1000, 1000));

    internal static int Exponent(float x) => (int)(BitConverter.SingleToUInt32Bits(x) >> 23 & 0xFF);

    internal static int FrexpExp(float x) => float.IsFinite(x) && x != 0 ? MathF.ILogB(x) + 1 : 0;

    internal static float FrexpMant(float x) => float.IsFinite(x) && x != 0 ? MathF.ScaleB(x, -FrexpExp(x)) : x;

    internal static int FrexpExp(double x) => double.IsFinite(x) && x != 0 ? Math.ILogB(x) + 1 : 0;

    internal static double FrexpMant(double x) => double.IsFinite(x) && x != 0 ? Math.ScaleB(x, -FrexpExp(x)) : x;

    /// <summary>A float32 to int32: truncated toward zero, saturated, NaN to 0.</summary>
    internal static int ToInt(double x) => double.IsNaN(x) ? 0 : x >= int.MaxValue ? int.MaxValue : x <= int.MinValue ? int.MinValue : (int)Math.Truncate(x);

    internal static uint ToUInt(double x) => double.IsNaN(x) || x <= 0 ? 0 : x >= uint.MaxValue ? uint.MaxValue : (uint)Math.Truncate(x);

    internal static ushort ToUShort(double x) => double.IsNaN(x) || x <= 0 ? (ushort)0 : x >= ushort.MaxValue ? ushort.MaxValue : (ushort)Math.Truncate(x);

    internal static short ToShort(double x) => double.IsNaN(x) ? (short)0 : x >= short.MaxValue ? short.MaxValue : x <= short.MinValue ? short.MinValue : (short)Math.Truncate(x);

    /// <summary>A float32 to float16 rounded toward zero (v_cvt_pkrtz_f16_f32).</summary>
    internal static Half ToHalfTowardZero(float x)
    {
        if (float.IsNaN(x) || float.IsInfinity(x))
        {
            return (Half)x;
        }

        var h = (Half)x;
        if (Half.IsInfinity(h) || Math.Abs((float)h) > Math.Abs(x))
        {
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            bits = Half.IsInfinity(h) ? (ushort)((bits & 0x8000) | 0x7BFF) : (ushort)(bits - 1);
            return BitConverter.UInt16BitsToHalf(bits);
        }

        return h;
    }

    /// <summary>
    /// v_div_scale_f32 (ISA pseudocode): scales the numerator or the denominator (S0 is one of them; S1 the denominator,
    /// S2 the numerator) by 2^±64 so the division in between stays in range; the flag (VCC's lane) asks v_div_fmas_f32 to
    /// undo a scale of the quotient.
    /// </summary>
    internal static float DivScale(float s0, float s1, float s2, out bool flag)
    {
        flag = false;
        if (s2 == 0 || s1 == 0)
        {
            return float.NaN;
        }

        if (Exponent(s2) - Exponent(s1) >= 96)
        {
            flag = true;
            return s0 == s1 ? MathF.ScaleB(s0, 64) : s0;
        }

        if (float.IsSubnormal(s1))
        {
            return MathF.ScaleB(s0, 64);
        }

        if (float.IsSubnormal((float)(1.0 / s1)) && float.IsSubnormal(s2 / s1))
        {
            // The reference's pseudocode reads ldexp(S0, 64) here, which overflows every such denominator (1/S1 is a
            // denormal only past 2^126) and would make LLVM's 1 / 2.7e38 infinite; its comment ("only scale the
            // denominator") and v_div_fmas's 2^-64 for a small quotient mean the denominator scaled down.
            flag = true;
            return s0 == s1 ? MathF.ScaleB(s0, -64) : s0;
        }

        if (float.IsSubnormal((float)(1.0 / s1)))
        {
            return MathF.ScaleB(s0, -64);
        }

        if (float.IsSubnormal(s2 / s1))
        {
            flag = true;
            return s0 == s2 ? MathF.ScaleB(s0, 64) : s0;
        }

        if (Exponent(s2) <= 23)
        {
            return MathF.ScaleB(s0, 64);
        }

        return s0;
    }

    /// <summary>v_div_fmas_f32: fma(S0, S1, S2), and with the flag the quotient's scale undone (2^64 when S2 is large, else 2^-64).</summary>
    internal static float DivFmas(float s0, float s1, float s2, bool flag)
    {
        float fma = MathF.FusedMultiplyAdd(s0, s1, s2);
        return flag ? MathF.ScaleB(fma, Exponent(s2) >= 127 ? 64 : -64) : fma;
    }

    /// <summary>v_div_fixup_f32 (ISA pseudocode): the special cases of S2 / S1 (NaN, zero, infinity), else S0 with the quotient's sign.</summary>
    internal static float DivFixup(float s0, float s1, float s2)
    {
        bool negative = float.IsNegative(s1) ^ float.IsNegative(s2);
        if (float.IsNaN(s2))
        {
            return Quiet(s2);
        }

        if (float.IsNaN(s1))
        {
            return Quiet(s1);
        }

        if (s1 == 0 && s2 == 0 || float.IsInfinity(s1) && float.IsInfinity(s2))
        {
            return float.NaN;
        }

        if (s1 == 0 || float.IsInfinity(s2))
        {
            return negative ? float.NegativeInfinity : float.PositiveInfinity;
        }

        if (float.IsInfinity(s1) || s2 == 0)
        {
            return negative ? -0f : 0f;
        }

        float magnitude = float.IsNaN(s0) ? float.PositiveInfinity : Math.Abs(s0);
        return negative ? -magnitude : magnitude;
    }

    internal static double DivScale(double s0, double s1, double s2, out bool flag)
    {
        flag = false;
        if (s2 == 0 || s1 == 0)
        {
            return double.NaN;
        }

        int e1 = (int)(BitConverter.DoubleToUInt64Bits(s1) >> 52 & 0x7FF), e2 = (int)(BitConverter.DoubleToUInt64Bits(s2) >> 52 & 0x7FF);
        if (e2 - e1 >= 768)
        {
            flag = true;
            return s0 == s1 ? Math.ScaleB(s0, 128) : s0;
        }

        if (double.IsSubnormal(s1))
        {
            return Math.ScaleB(s0, 128);
        }

        if (double.IsSubnormal(1.0 / s1) && double.IsSubnormal(s2 / s1))
        {
            flag = true;
            return s0 == s1 ? Math.ScaleB(s0, 128) : s0;
        }

        if (double.IsSubnormal(1.0 / s1))
        {
            return Math.ScaleB(s0, -128);
        }

        if (double.IsSubnormal(s2 / s1))
        {
            flag = true;
            return s0 == s2 ? Math.ScaleB(s0, 128) : s0;
        }

        return e2 <= 53 ? Math.ScaleB(s0, 128) : s0;
    }

    internal static double DivFmas(double s0, double s1, double s2, bool flag)
    {
        double fma = Math.FusedMultiplyAdd(s0, s1, s2);
        int e = (int)(BitConverter.DoubleToUInt64Bits(s2) >> 52 & 0x7FF);
        return flag ? Math.ScaleB(fma, e >= 1023 ? 128 : -128) : fma;
    }

    internal static double DivFixup(double s0, double s1, double s2)
    {
        bool negative = double.IsNegative(s1) ^ double.IsNegative(s2);
        if (double.IsNaN(s2) || double.IsNaN(s1))
        {
            return double.IsNaN(s2) ? s2 : s1;
        }

        if (s1 == 0 && s2 == 0 || double.IsInfinity(s1) && double.IsInfinity(s2))
        {
            return double.NaN;
        }

        if (s1 == 0 || double.IsInfinity(s2))
        {
            return negative ? double.NegativeInfinity : double.PositiveInfinity;
        }

        if (double.IsInfinity(s1) || s2 == 0)
        {
            return negative ? -0.0 : 0.0;
        }

        double magnitude = double.IsNaN(s0) ? double.PositiveInfinity : Math.Abs(s0);
        return negative ? -magnitude : magnitude;
    }

    /// <summary>v_cmp_class: which of the ten classes (signaling NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf) the value is in.</summary>
    internal static int Class(double x, bool signaling)
    {
        if (double.IsNaN(x))
        {
            return signaling ? 0 : 1;
        }

        bool negative = double.IsNegative(x);
        if (double.IsInfinity(x))
        {
            return negative ? 2 : 9;
        }

        if (x == 0)
        {
            return negative ? 5 : 6;
        }

        return negative ? 3 : 8;
    }
}
