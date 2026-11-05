// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// Vectorized row operations shared by the CPU kernels: the widest vectors the machine runs well (512-bit where
// accelerated, else Vector<float>), with scalar tails. One result on every CPU (rule 70): e^x and ln x are Idrak's own
// (no C runtime's, whose last bits differ between platforms), a scalar tail rounds as a lane does, and every sum keeps
// the same fixed lanes whatever the vector width (CpuMath.Lanes.cs).
internal static partial class CpuMath
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

    /// <summary>
    /// e^x of one value, rounded as a lane of <see cref="Exp(Vector{float})"/> (the same clamp, FMAs and scaling), so an
    /// element's bits do not depend on whether it falls in a vector or a tail, nor on the platform's C runtime.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Exp(float x)
    {
        if (float.IsNaN(x))
        {
            return x;
        }

        if (x < ExpUnderflow)
        {
            return 0f;
        }

        if (x > ExpOverflow)
        {
            return float.PositiveInfinity;
        }

        float kf = MathF.FusedMultiplyAdd(x, ExpLog2E, ExpRoundMagic);
        int k = Math.Clamp(BitConverter.SingleToInt32Bits(kf) - BitConverter.SingleToInt32Bits(ExpRoundMagic), -126, 127);
        float kFloat = k;
        float f = MathF.FusedMultiplyAdd(kFloat, -ExpLn2High, x);
        f = MathF.FusedMultiplyAdd(kFloat, -ExpLn2Low, f);
        float p = MathF.FusedMultiplyAdd(ExpP0, f, ExpP1);
        p = MathF.FusedMultiplyAdd(p, f, ExpP2);
        p = MathF.FusedMultiplyAdd(p, f, ExpP3);
        p = MathF.FusedMultiplyAdd(p, f, ExpP4);
        p = MathF.FusedMultiplyAdd(p, f, ExpP5);
        p = MathF.FusedMultiplyAdd(p, f * f, f + 1f);
        return p * BitConverter.Int32BitsToSingle((k + 127) << 23);
    }

    // ln x as e · ln 2 + ln m with x = m · 2^e, m in [√½, √2) (Cephes' logf: m − 1 into a degree-8 polynomial, ln 2 in two
    // parts; within 1e-7 relative of the true value): the lanes' own code, inlined, the scalar form the same operations.
    // A subnormal x is scaled by 2^25 first; x < 0 and NaN give NaN, ±0 gives −∞, +∞ stays +∞.
    private const float LogSqrtHalf = 0.707106781186547524f, LogLn2High = 0.693359375f, LogLn2Low = -2.12194440e-4f, LogSubnormalScale = 33554432f, LogMinNormal = 1.17549435E-38f;
    private const float LogP0 = 7.0376836292E-2f, LogP1 = -1.1514610310E-1f, LogP2 = 1.1676998740E-1f, LogP3 = -1.2420140846E-1f, LogP4 = 1.4249322787E-1f,
        LogP5 = -1.6668057665E-1f, LogP6 = 2.0000714765E-1f, LogP7 = -2.4999993993E-1f, LogP8 = 3.3333331174E-1f;

    /// <summary>ln x of each lane (see the notes above): NaN below 0 and for NaN, −∞ at ±0, +∞ kept.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Log(Vector<float> x)
    {
        var subnormal = Vector.LessThan(x, new Vector<float>(LogMinNormal));
        var scaled = Vector.ConditionalSelect(subnormal, x * new Vector<float>(LogSubnormalScale), x);
        var bits = Vector.AsVectorInt32(scaled);
        var e = (Vector.ShiftRightLogical(bits, 23) & new Vector<int>(0xff)) - Vector.ConditionalSelect(subnormal, new Vector<int>(126 + 25), new Vector<int>(126));
        var m = Vector.AsVectorSingle((bits & new Vector<int>(0x007fffff)) | new Vector<int>(0x3f000000));       // [½, 1)
        var low = Vector.LessThan(m, new Vector<float>(LogSqrtHalf));
        e += Vector.ConditionalSelect(low, new Vector<int>(-1), Vector<int>.Zero);
        m = Vector.ConditionalSelect(low, m + m - Vector<float>.One, m - Vector<float>.One);
        var z = m * m;
        var y = Vector.FusedMultiplyAdd(new Vector<float>(LogP0), m, new Vector<float>(LogP1));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP2));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP3));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP4));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP5));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP6));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP7));
        y = Vector.FusedMultiplyAdd(y, m, new Vector<float>(LogP8));
        y = y * m * z;
        var ef = Vector.ConvertToSingle(e);
        y = Vector.FusedMultiplyAdd(ef, new Vector<float>(LogLn2Low), y);
        y = Vector.FusedMultiplyAdd(z, new Vector<float>(-0.5f), y);
        var result = Vector.FusedMultiplyAdd(ef, new Vector<float>(LogLn2High), m + y);
        result = Vector.ConditionalSelect(Vector.Equals(x, new Vector<float>(float.PositiveInfinity)), x, result);
        result = Vector.ConditionalSelect(Vector.Equals(x, Vector<float>.Zero), new Vector<float>(float.NegativeInfinity), result);
        return Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, Vector<float>.Zero), result, new Vector<float>(float.NaN));
    }

    /// <summary>ln x of one value, rounded as a lane of <see cref="Log(Vector{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Log(float x)
    {
        if (!(x >= 0f))
        {
            return float.NaN;                                                // below 0, or NaN
        }

        if (x == 0f)
        {
            return float.NegativeInfinity;
        }

        if (x == float.PositiveInfinity)
        {
            return x;
        }

        bool subnormal = x < LogMinNormal;
        int bits = BitConverter.SingleToInt32Bits(subnormal ? x * LogSubnormalScale : x);
        int e = (int)((uint)bits >> 23 & 0xff) - (subnormal ? 126 + 25 : 126);
        float m = BitConverter.Int32BitsToSingle(bits & 0x007fffff | 0x3f000000);
        if (m < LogSqrtHalf)
        {
            e -= 1;
            m = m + m - 1f;
        }
        else
        {
            m -= 1f;
        }

        float z = m * m;
        float y = MathF.FusedMultiplyAdd(LogP0, m, LogP1);
        y = MathF.FusedMultiplyAdd(y, m, LogP2);
        y = MathF.FusedMultiplyAdd(y, m, LogP3);
        y = MathF.FusedMultiplyAdd(y, m, LogP4);
        y = MathF.FusedMultiplyAdd(y, m, LogP5);
        y = MathF.FusedMultiplyAdd(y, m, LogP6);
        y = MathF.FusedMultiplyAdd(y, m, LogP7);
        y = MathF.FusedMultiplyAdd(y, m, LogP8);
        y = y * m * z;
        float ef = e;
        y = MathF.FusedMultiplyAdd(ef, LogLn2Low, y);
        y = MathF.FusedMultiplyAdd(z, -0.5f, y);
        return MathF.FusedMultiplyAdd(ef, LogLn2High, m + y);
    }

    // e^x and ln x in double precision, for the sums the kernels keep in double (a softmax's normalizer, CTC's log-space
    // recursions, a cross-entropy's log-sum-exp): fdlibm's __ieee754_exp and __ieee754_log in plain IEEE double
    // arithmetic (within 1 ulp), so their bits are the same on every machine where the C runtime's Math.Exp and Math.Log
    // are not.
    private const double DExpLn2High = 6.93147180369123816490e-01, DExpLn2Low = 1.90821492927058770002e-10, DExpInvLn2 = 1.44269504088896338700e+00;
    private const double DExpP1 = 1.66666666666666019037e-01, DExpP2 = -2.77777777770155933842e-03, DExpP3 = 6.61375632143793436117e-05,
        DExpP4 = -1.65339022054652515390e-06, DExpP5 = 4.13813679705723846039e-08;
    private const double DExpOverflow = 7.09782712893383973096e+02, DExpUnderflow = -7.45133219101941108420e+02;
    private const double DLogLg1 = 6.666666666666735130e-01, DLogLg2 = 3.999999999940941908e-01, DLogLg3 = 2.857142874366239149e-01,
        DLogLg4 = 2.222219843214978396e-01, DLogLg5 = 1.818357216161805012e-01, DLogLg6 = 1.531383769920937332e-01, DLogLg7 = 1.479819860511658591e-01;

    /// <summary>e^x in double precision (fdlibm's exp, within 1 ulp): 0 below −745.13, +∞ above 709.78, NaN kept.</summary>
    public static double ExpDouble(double x)
    {
        if (double.IsNaN(x))
        {
            return x;
        }

        if (x > DExpOverflow)
        {
            return double.PositiveInfinity;
        }

        if (x < DExpUnderflow)
        {
            return 0;
        }

        double ax = Math.Abs(x);
        if (ax < 3.725290298461914e-09)                                     // |x| < 2^-28: 1 + x is e^x rounded
        {
            return 1 + x;
        }

        int k = 0;
        double hi = 0, lo = 0;
        if (ax > 0.5 * 0.6931471805599453)
        {
            k = (int)(DExpInvLn2 * x + (x < 0 ? -0.5 : 0.5));               // round half away from zero
            hi = x - k * DExpLn2High;                                       // exact: k · ln2_hi has trailing zero bits
            lo = k * DExpLn2Low;
            x = hi - lo;
        }

        double t = x * x;
        double c = x - t * (DExpP1 + t * (DExpP2 + t * (DExpP3 + t * (DExpP4 + t * DExpP5))));
        if (k == 0)
        {
            return 1 - (x * c / (c - 2.0) - x);
        }

        double y = 1 - (lo - x * c / (2.0 - c) - hi);
        return ScaleByPowerOfTwo(y, k);
    }

    // y · 2^k by the exponent's bits (k from −1075 to 1024 here), in at most two exact steps but for a subnormal result.
    private static double ScaleByPowerOfTwo(double y, int k)
    {
        if (k > 1023)
        {
            return y * BitConverter.Int64BitsToDouble((long)(1023 + 1023) << 52) * BitConverter.Int64BitsToDouble((long)(k - 1023 + 1023) << 52);
        }

        if (k < -1021)
        {
            return y * BitConverter.Int64BitsToDouble((long)(k + 1000 + 1023) << 52) * BitConverter.Int64BitsToDouble((long)(-1000 + 1023) << 52);
        }

        return y * BitConverter.Int64BitsToDouble((long)(k + 1023) << 52);
    }

    /// <summary>ln x in double precision (fdlibm's log, within 1 ulp): NaN below 0 and for NaN, −∞ at ±0, +∞ kept.</summary>
    public static double LogDouble(double x)
    {
        if (!(x >= 0))
        {
            return double.NaN;
        }

        if (x == 0)
        {
            return double.NegativeInfinity;
        }

        if (double.IsPositiveInfinity(x))
        {
            return x;
        }

        int k = 0;
        if (x < 2.2250738585072014e-308)                                    // subnormal: scaled by 2^54
        {
            k -= 54;
            x *= 1.80143985094819840000e+16;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        int hx = (int)(bits >> 32);
        k += (hx >> 20) - 1023;
        hx &= 0x000fffff;
        int i = (hx + 0x95f64) & 0x100000;
        x = BitConverter.Int64BitsToDouble((bits & 0xffffffffL) | (long)(hx | (i ^ 0x3ff00000)) << 32);   // x or x / 2, in [√½, √2)
        k += i >> 20;
        double f = x - 1.0, dk = k;
        if ((0x000fffff & (2 + hx)) < 3)                                    // |f| < 2^-20
        {
            if (f == 0)
            {
                return k == 0 ? 0 : dk * DExpLn2High + dk * DExpLn2Low;
            }

            double r0 = f * f * (0.5 - 0.33333333333333333 * f);
            return k == 0 ? f - r0 : dk * DExpLn2High - (r0 - dk * DExpLn2Low - f);
        }

        double s = f / (2.0 + f), z = s * s, w = z * z;
        int i0 = hx - 0x6147a, j0 = 0x6b851 - hx;
        double t1 = w * (DLogLg2 + w * (DLogLg4 + w * DLogLg6));
        double t2 = z * (DLogLg1 + w * (DLogLg3 + w * (DLogLg5 + w * DLogLg7)));
        double r = t2 + t1;
        if ((i0 | j0) > 0)
        {
            double hfsq = 0.5 * f * f;
            return k == 0 ? f - (hfsq - s * (hfsq + r)) : dk * DExpLn2High - (hfsq - (s * (hfsq + r) + dk * DExpLn2Low) - f);
        }

        return k == 0 ? f - s * (f - r) : dk * DExpLn2High - (s * (f - r) - dk * DExpLn2Low - f);
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
        float e = 1f - poly * Exp(-a * a);
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
            y[i] = mask.IsEmpty ? x[i] * scale : x[i] * scale + mask[i];   // as the lanes: no + 0 without a mask (−0 stays −0)
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

    /// <summary>
    /// x[i] = exp(x[i] - shift) in place; returns the sum, in the fixed lanes (<see cref="LaneCount"/>): one 512-bit
    /// accumulator where those are accelerated, else <see cref="Fold{T}"/>'s, the same bits either way.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float ExpShifted(Span<float> x, float shift)
    {
        ref float r = ref MemoryMarshal.GetReference(x);
        if (!Vector512.IsHardwareAccelerated || x.Length < LaneCount)
        {
            return Fold(new ExpShiftedTerms(ref r, shift), x.Length);
        }

        FloatLanes lanes = default;
        ref float s = ref lanes[0];
        var shifts = Vector512.Create(shift);
        var total = Vector512<float>.Zero;
        int i = 0;
        for (; i <= x.Length - LaneCount; i += LaneCount)
        {
            var e = Exp(Vector512.LoadUnsafe(ref r, (nuint)i) - shifts);
            e.StoreUnsafe(ref r, (nuint)i);
            total += e;
        }

        total.StoreUnsafe(ref s);
        var terms = new ExpShiftedTerms(ref r, shift);
        for (; i < x.Length; i++)
        {
            ref float lane = ref Unsafe.Add(ref s, i & (LaneCount - 1));
            lane = terms.Add(lane, i);
        }

        return Reduce(ref s);
    }

    // e^(x − shift) written back over x and added to the sum.
    private readonly ref struct ExpShiftedTerms : ILaneSum
    {
        private readonly ref float _x;
        private readonly float _shift;

        public ExpShiftedTerms(ref float x, float shift)
        {
            _x = ref x;
            _shift = shift;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i)
        {
            var e = Exp(Vector.LoadUnsafe(ref _x, (nuint)i) - new Vector<float>(_shift));
            e.StoreUnsafe(ref _x, (nuint)i);
            return sum + e;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i)
        {
            ref float x = ref Unsafe.Add(ref _x, i);
            x = Exp(x - _shift);
            return sum + x;
        }
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

        for (; i < y.Length; i++)                                           // the lanes' operations, in their order
        {
            float x = gate[i], v = up[i];
            float a = kind switch
            {
                0 => x / (1f + Exp(-x)),
                1 => 0.5f * x * (1f + Tanh(0.7978845608f * MathF.FusedMultiplyAdd(0.044715f * x * x, x, x))),
                3 => Tanh(x),
                _ => MathF.Max(x, 0f),
            };
            y[i] = a * (kind == 3 ? 1f / (1f + Exp(-v)) : v);
        }
    }
}
