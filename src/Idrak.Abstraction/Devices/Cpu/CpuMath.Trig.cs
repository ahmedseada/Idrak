// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// sin, cos, tan, atan, atan2, tanh, pow, log2 and log10 in plain IEEE arithmetic (rule 70: one result on every CPU). The
// C runtime's (Math, MathF) and the runtime's vector ones (Vector.Sin, Cos) differ in their last bits between platforms and
// between a vector and its scalar tail; these are the same operations everywhere. The double forms are fdlibm's (within an
// ulp or two); the float sin and cos reduce and evaluate in double and round once, lanes and scalar alike.
internal static partial class CpuMath
{
    // π/2 in three parts of 33 bits (fdlibm's pio2_1..3 and their tails) and 2/π.
    private const double Pio2High = 1.57079632673412561417e+00, Pio2HighTail = 6.07710050650619224932e-11;
    private const double Pio2Mid = 6.07710050630396597660e-11, Pio2MidTail = 2.02226624879595063154e-21;
    private const double Pio2Low = 2.02226624871116645580e-21, Pio2LowTail = 8.47842766036889956997e-32;
    private const double InvPio2 = 6.36619772367581382433e-01, RoundMagic = 6755399441055744.0;   // 1.5 · 2^52

    // fdlibm's __kernel_sin and __kernel_cos coefficients, on [−π/4, π/4].
    private const double KS1 = -1.66666666666666324348e-01, KS2 = 8.33333333332248946124e-03, KS3 = -1.98412698298579493134e-04,
        KS4 = 2.75573137070700676789e-06, KS5 = -2.50507602534068634195e-08, KS6 = 1.58969099521155010221e-10;
    private const double KC1 = 4.16666666666666019037e-02, KC2 = -1.38888888888741095749e-03, KC3 = 2.48015872894767294178e-05,
        KC4 = -2.75573143513906633035e-07, KC5 = 2.08757232129817482790e-09, KC6 = -1.13596475577881948265e-11;

    // ------------------------------------------------------------ float sin and cos (lanes and scalar alike)

    // x as n · π/2 + r: n = round(x · 2/π) by the 1.5 · 2^52 trick (its low bits are n mod 4), r = x − n · π/2 with π/2
    // in two parts (n · Pio2High exact while |n| < 2^20, |x| below about 1.6e6; beyond, the same bits, less accurate).
    // sin r and cos r in double (fdlibm's kernels without a tail), the quadrant's one chosen, rounded to float once.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> SinCosLanes(Vector<double> x, bool cosine)
    {
        var shifted = x * new Vector<double>(InvPio2) + new Vector<double>(RoundMagic);
        var n = shifted - new Vector<double>(RoundMagic);
        var quadrant = Vector.AsVectorInt64(shifted) + (cosine ? Vector<long>.One : Vector<long>.Zero);
        var r = x - n * new Vector<double>(Pio2High) - n * new Vector<double>(Pio2HighTail);
        var z = r * r;
        var s = new Vector<double>(KS5) + z * new Vector<double>(KS6);
        s = new Vector<double>(KS4) + z * s;
        s = new Vector<double>(KS3) + z * s;
        s = new Vector<double>(KS2) + z * s;
        var sine = r + z * r * (new Vector<double>(KS1) + z * s);
        var c = new Vector<double>(KC5) + z * new Vector<double>(KC6);
        c = new Vector<double>(KC4) + z * c;
        c = new Vector<double>(KC3) + z * c;
        c = new Vector<double>(KC2) + z * c;
        c = z * (new Vector<double>(KC1) + z * c);
        var hz = new Vector<double>(0.5) * z;
        var w = Vector<double>.One - hz;
        var cos = w + (Vector<double>.One - w - hz + z * c);
        var odd = Vector.Equals(quadrant & Vector<long>.One, Vector<long>.One);
        var negative = Vector.Equals(quadrant & new Vector<long>(2), new Vector<long>(2));
        var value = Vector.ConditionalSelect(odd, cos, sine);
        return Vector.ConditionalSelect(negative, -value, value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double SinCosOne(double x, bool cosine)
    {
        double shifted = x * InvPio2 + RoundMagic;
        double n = shifted - RoundMagic;
        long quadrant = BitConverter.DoubleToInt64Bits(shifted) + (cosine ? 1 : 0);
        double r = x - n * Pio2High - n * Pio2HighTail;
        double z = r * r;
        double s = KS5 + z * KS6;
        s = KS4 + z * s;
        s = KS3 + z * s;
        s = KS2 + z * s;
        double sine = r + z * r * (KS1 + z * s);
        double c = KC5 + z * KC6;
        c = KC4 + z * c;
        c = KC3 + z * c;
        c = KC2 + z * c;
        c = z * (KC1 + z * c);
        double hz = 0.5 * z, w = 1.0 - hz;
        double cos = w + (1.0 - w - hz + z * c);
        double value = (quadrant & 1) != 0 ? cos : sine;
        return (quadrant & 2) != 0 ? -value : value;
    }

    /// <summary>sin x of each lane: reduced and evaluated in double, rounded once (the scalar form's bits).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Sin(Vector<float> x)
    {
        Vector.Widen(x, out var low, out var high);
        return Vector.Narrow(SinCosLanes(low, false), SinCosLanes(high, false));
    }

    /// <summary>cos x of each lane, as <see cref="Sin(Vector{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Cos(Vector<float> x)
    {
        Vector.Widen(x, out var low, out var high);
        return Vector.Narrow(SinCosLanes(low, true), SinCosLanes(high, true));
    }

    /// <summary>sin x of one value, rounded as a lane of <see cref="Sin(Vector{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sin(float x) => (float)SinCosOne(x, false);

    /// <summary>cos x of one value, rounded as a lane of <see cref="Cos(Vector{float})"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Cos(float x) => (float)SinCosOne(x, true);

    /// <summary>x^y in float: the double power (<see cref="PowDouble"/>) rounded once, IEEE's special cases.</summary>
    public static float Pow(float x, float y) => (float)PowDouble(x, y);

    // ------------------------------------------------------------ double sin, cos and tan (fdlibm)

    // x = n · π/2 + (y0 + y1), fdlibm's __ieee754_rem_pio2 medium case (three refinements of π/2 as cancellation needs);
    // past |x| ≈ 1.6e6, where fdlibm turns to Payne-Hanek, the same steps (the same bits everywhere, less accurate).
    private static long ReducePio2(double x, out double y0, out double y1)
    {
        double t = Math.Abs(x);
        double shifted = t * InvPio2 + RoundMagic;
        double fn = shifted - RoundMagic;
        long n = (long)fn;
        double r = t - fn * Pio2High, w = fn * Pio2HighTail;
        y0 = r - w;
        int j = (int)(BitConverter.DoubleToInt64Bits(t) >> 52) & 0x7ff;
        if (j - ((int)(BitConverter.DoubleToInt64Bits(y0) >> 52) & 0x7ff) > 16)
        {
            double t2 = r;
            w = fn * Pio2Mid;
            r = t2 - w;
            w = fn * Pio2MidTail - (t2 - r - w);
            y0 = r - w;
            if (j - ((int)(BitConverter.DoubleToInt64Bits(y0) >> 52) & 0x7ff) > 49)
            {
                t2 = r;
                w = fn * Pio2Low;
                r = t2 - w;
                w = fn * Pio2LowTail - (t2 - r - w);
                y0 = r - w;
            }
        }

        y1 = r - y0 - w;
        if (x < 0)
        {
            y0 = -y0;
            y1 = -y1;
            return -n;
        }

        return n;
    }

    // fdlibm's __kernel_sin(x, y, 1) and __kernel_cos(x, y) on [−π/4, π/4], y the tail of x.
    private static double KernelSin(double x, double y)
    {
        double z = x * x, v = z * x;
        double r = KS2 + z * (KS3 + z * (KS4 + z * (KS5 + z * KS6)));
        return x - (z * (0.5 * y - v * r) - y - v * KS1);
    }

    private static double KernelCos(double x, double y)
    {
        double z = x * x;
        double r = z * (KC1 + z * (KC2 + z * (KC3 + z * (KC4 + z * (KC5 + z * KC6)))));
        double hz = 0.5 * z, w = 1.0 - hz;
        return w + (1.0 - w - hz + (z * r - x * y));
    }

    /// <summary>sin x in double (fdlibm's, within an ulp below |x| ≈ 1.6e6): NaN for NaN and ±∞.</summary>
    public static double SinDouble(double x)
    {
        if (!double.IsFinite(x))
        {
            return double.NaN;
        }

        if (Math.Abs(x) <= 0.7853981633974483)
        {
            return KernelSin(x, 0);
        }

        long n = ReducePio2(x, out double y0, out double y1);
        return (n & 3) switch
        {
            0 => KernelSin(y0, y1),
            1 => KernelCos(y0, y1),
            2 => -KernelSin(y0, y1),
            _ => -KernelCos(y0, y1),
        };
    }

    /// <summary>cos x in double (fdlibm's, within an ulp below |x| ≈ 1.6e6): NaN for NaN and ±∞.</summary>
    public static double CosDouble(double x)
    {
        if (!double.IsFinite(x))
        {
            return double.NaN;
        }

        if (Math.Abs(x) <= 0.7853981633974483)
        {
            return KernelCos(x, 0);
        }

        long n = ReducePio2(x, out double y0, out double y1);
        return (n & 3) switch
        {
            0 => KernelCos(y0, y1),
            1 => -KernelSin(y0, y1),
            2 => -KernelCos(y0, y1),
            _ => KernelSin(y0, y1),
        };
    }

    /// <summary>tan x in double: sin x / cos x of the one reduction (within a few ulp).</summary>
    public static double TanDouble(double x)
    {
        if (!double.IsFinite(x))
        {
            return double.NaN;
        }

        if (Math.Abs(x) <= 0.7853981633974483)
        {
            return KernelSin(x, 0) / KernelCos(x, 0);
        }

        long n = ReducePio2(x, out double y0, out double y1);
        double s = KernelSin(y0, y1), c = KernelCos(y0, y1);
        return (n & 1) == 0 ? s / c : -c / s;
    }

    // ------------------------------------------------------------ double atan, atan2 and tanh

    private static readonly double[] AtanHigh = [4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00];
    private static readonly double[] AtanLow = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17];
    private const double AT0 = 3.33333333333329318027e-01, AT1 = -1.99999999998764832476e-01, AT2 = 1.42857142725034663711e-01,
        AT3 = -1.11111104054623557880e-01, AT4 = 9.09088713343650656196e-02, AT5 = -7.69187620504482999495e-02, AT6 = 6.66107313738753120669e-02,
        AT7 = -5.83357013379057348645e-02, AT8 = 4.97687799461593236017e-02, AT9 = -3.65315727442169155270e-02, AT10 = 1.62858201153657823623e-02;

    /// <summary>atan x in double (fdlibm's, within an ulp): NaN kept, ±π/2 at ±∞.</summary>
    public static double AtanDouble(double x)
    {
        if (double.IsNaN(x))
        {
            return x;
        }

        double ax = Math.Abs(x);
        bool negative = x < 0;
        if (ax >= 7.378697629483821e19)                                     // |x| ≥ 2^66: ±π/2
        {
            return x > 0 ? AtanHigh[3] + AtanLow[3] : -AtanHigh[3] - AtanLow[3];
        }

        int id;
        if (ax < 0.4375)
        {
            if (ax < 1.862645149230957e-09)                                  // 2^-29: atan x is x
            {
                return x;
            }

            id = -1;
        }
        else if (ax < 1.1875)
        {
            (id, x) = ax < 0.6875 ? (0, (2.0 * ax - 1.0) / (2.0 + ax)) : (1, (ax - 1.0) / (ax + 1.0));
        }
        else
        {
            (id, x) = ax < 2.4375 ? (2, (ax - 1.5) / (1.0 + 1.5 * ax)) : (3, -1.0 / ax);
        }

        double z = x * x, w = z * z;
        double s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))));
        double s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))));
        if (id < 0)
        {
            return x - x * (s1 + s2);
        }

        double result = AtanHigh[id] - (x * (s1 + s2) - AtanLow[id] - x);
        return negative ? -result : result;                                 // the reduced forms work on |x|
    }

    /// <summary>atan(y / x) in double over the four quadrants (IEEE's special cases), from <see cref="AtanDouble"/>.</summary>
    public static double Atan2Double(double y, double x)
    {
        const double Pi = 3.1415926535897931160E+00, PiLow = 1.2246467991473531772E-16, HalfPi = 1.5707963267948965580E+00;
        if (double.IsNaN(x) || double.IsNaN(y))
        {
            return double.NaN;
        }

        bool yNegative = double.IsNegative(y);
        if (y == 0)
        {
            return double.IsNegative(x) ? (yNegative ? -Pi : Pi) : y;      // ±0 against x ≥ +0; ±π against x ≤ −0
        }

        if (x == 0)
        {
            return yNegative ? -HalfPi : HalfPi;
        }

        if (double.IsInfinity(x))
        {
            if (double.IsInfinity(y))
            {
                double quarter = x > 0 ? Pi / 4 : 3 * Pi / 4;
                return yNegative ? -quarter : quarter;
            }

            return x > 0 ? (yNegative ? -0.0 : 0.0) : yNegative ? -Pi : Pi;
        }

        if (double.IsInfinity(y))
        {
            return yNegative ? -HalfPi : HalfPi;
        }

        double z = AtanDouble(Math.Abs(y / x));
        if (x > 0)
        {
            return yNegative ? -z : z;
        }

        double away = Pi - (z - PiLow);
        return yNegative ? -away : away;
    }

    /// <summary>tanh x in double (within about 5e-15 relative): its series to x¹³ below 0.1, 1 − 2 / (e^2|x| + 1) above.</summary>
    public static double TanhDouble(double x)
    {
        if (double.IsNaN(x))
        {
            return x;
        }

        double ax = Math.Abs(x), result;
        if (ax > 22)
        {
            result = 1;
        }
        else if (ax < 0.1)
        {
            double x2 = ax * ax;
            result = ax * (1 + x2 * (-1.0 / 3 + x2 * (2.0 / 15 + x2 * (-17.0 / 315 + x2 * (62.0 / 2835 + x2 * (-1382.0 / 155925 + x2 * (21844.0 / 6081075)))))));
        }
        else
        {
            result = 1 - 2 / (ExpDouble(2 * ax) + 1);
        }

        return double.IsNegative(x) ? -result : result;
    }

    // ------------------------------------------------------------ double pow, log2 and log10

    /// <summary>
    /// x^y in double with IEEE's special cases: integer y up to 2^30 by repeated squaring (exact where the result is,
    /// 1024³ and 2⁻³ included), any other y as e^(y · ln |x|) (<see cref="ExpDouble"/>, <see cref="LogDouble"/>).
    /// </summary>
    public static double PowDouble(double x, double y)
    {
        if (y == 0 || x == 1)
        {
            return 1;
        }

        if (double.IsNaN(x) || double.IsNaN(y))
        {
            return double.NaN;
        }

        bool integer = y == Math.Floor(y), odd = integer && Math.Abs(y) < 9007199254740992.0 && Math.Abs(y) % 2 == 1;
        if (double.IsInfinity(y))
        {
            double ax = Math.Abs(x);
            return ax == 1 ? 1 : (ax > 1) == (y > 0) ? double.PositiveInfinity : 0;
        }

        if (x == 0)
        {
            return y < 0 ? (odd ? 1 / x : double.PositiveInfinity) : odd ? x : 0;
        }

        if (double.IsInfinity(x))
        {
            if (x > 0)
            {
                return y > 0 ? double.PositiveInfinity : 0;
            }

            return y > 0 ? (odd ? double.NegativeInfinity : double.PositiveInfinity) : odd ? -0.0 : 0;
        }

        if (x < 0 && !integer)
        {
            return double.NaN;
        }

        double magnitude = Math.Abs(x), value;
        if (integer && Math.Abs(y) <= 1 << 30)
        {
            value = PowInteger(magnitude, (long)y);
            if (double.IsInfinity(value) && y < 0 || value == 0 && y > 0)
            {
                value = ExpDouble(y * LogDouble(magnitude));                // an intermediate overflowed: the logarithm's way
            }
        }
        else
        {
            value = ExpDouble(y * LogDouble(magnitude));
        }

        return x < 0 && odd ? -value : value;
    }

    // b^n by repeated squaring (n ≥ 0 directly; n < 0 as 1 / b^|n|).
    private static double PowInteger(double b, long n)
    {
        ulong e = (ulong)Math.Abs(n);
        double result = 1;
        while (e > 0)
        {
            if ((e & 1) != 0)
            {
                result *= b;
            }

            e >>= 1;
            if (e > 0)
            {
                b *= b;
            }
        }

        return n < 0 ? 1 / result : result;
    }

    /// <summary>log₂ x in double: the exponent plus ln m · log₂ e for the mantissa m in [√½, √2), so a power of two's is exact.</summary>
    public static double Log2Double(double x)
    {
        if (!(x > 0) || double.IsPositiveInfinity(x))
        {
            return LogDouble(x);                                            // NaN, −∞ at 0, +∞
        }

        int e = 0;
        if (x < 2.2250738585072014e-308)
        {
            x *= 1.80143985094819840000e+16;                                // subnormal: scaled by 2^54
            e = -54;
        }

        long bits = BitConverter.DoubleToInt64Bits(x);
        e += (int)(bits >> 52) - 1023;
        double m = BitConverter.Int64BitsToDouble(bits & 0x000fffffffffffffL | 0x3ff0000000000000L);   // [1, 2)
        if (m > 1.4142135623730951)
        {
            m *= 0.5;
            e++;
        }

        return e + LogDouble(m) * 1.4426950408889634;
    }

    /// <summary>log₁₀ x in double: ln x · log₁₀ e, a power of ten (10^k exactly representable) giving k exactly.</summary>
    public static double Log10Double(double x)
    {
        double value = LogDouble(x) * 0.43429448190325182;
        double k = Math.Round(value);
        return Math.Abs(value - k) < 1e-9 && Math.Abs(k) <= 22 && PowInteger(10, (long)k) == x ? k : value;
    }
}
