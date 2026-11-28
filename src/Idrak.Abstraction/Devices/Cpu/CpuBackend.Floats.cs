// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

// The float32 lanes every CPU operation is written over once: IFloats, the operations of one vector width (512, 256 and
// 128 bits, and one float), each the same IEEE operation lane by lane (fused multiply-adds rounded once, comparisons and
// selects, no instruction whose result differs between sets), so an operation gives the same bits by every width and a
// shape's measured choice among them changes its speed alone. Reductions add in sixteen canonical lanes (Reduce16):
// element i into lane i mod 16, each lane in order, the lanes added in a fixed tree; every width keeps those lanes (one
// 512-bit vector, two 256-bit, four 128-bit or sixteen floats), so sums have the same bits by every width too.
internal sealed partial class CpuBackend
{
    /// <summary>The float lanes of one vector width: the operations the CPU's kernels are written over (see above).</summary>
    internal unsafe interface IFloats<TV> where TV : unmanaged
    {
        /// <summary>Floats a vector holds.</summary>
        static abstract int Count { get; }

        static abstract TV Load(float* p);

        static abstract void Store(float* p, TV v);

        static abstract TV All(float v);

        static abstract TV Add(TV a, TV b);

        static abstract TV Sub(TV a, TV b);

        static abstract TV Mul(TV a, TV b);

        static abstract TV Div(TV a, TV b);

        /// <summary>a · b + c rounded once.</summary>
        static abstract TV Fma(TV a, TV b, TV c);

        static abstract TV Sqrt(TV a);

        /// <summary>All bits set where a &lt; b (IEEE: false with a NaN), else 0.</summary>
        static abstract TV Less(TV a, TV b);

        static abstract TV LessOrEqual(TV a, TV b);

        static abstract TV Equal(TV a, TV b);

        /// <summary>mask's bits of a, the others of b.</summary>
        static abstract TV Select(TV mask, TV a, TV b);

        static abstract TV And(TV a, TV b);

        static abstract TV Or(TV a, TV b);

        static abstract TV Xor(TV a, TV b);

        /// <summary>The lanes' bits as 32-bit integers, added.</summary>
        static abstract TV IntAdd(TV a, TV b);

        static abstract TV IntSub(TV a, TV b);

        static abstract TV IntMin(TV a, TV b);

        static abstract TV IntMax(TV a, TV b);

        /// <summary>The integer each lane's bits hold, as a float.</summary>
        static abstract TV IntToFloat(TV a);

        static abstract TV ShiftLeft(TV a, int bits);

        static abstract TV ShiftRightLogical(TV a, int bits);
    }

    // The widths the measured choices pick among (a choice's value), the widest first.
    internal enum Lanes : byte
    {
        Wide512 = 0,
        Wide256 = 1,
        Wide128 = 2,
        One = 3,
    }

    // A width's name in the recorded choices.
    internal static string LanesName(Lanes lanes) => lanes switch
    {
        Lanes.Wide512 => "512-bit lanes",
        Lanes.Wide256 => "256-bit lanes",
        Lanes.Wide128 => "128-bit lanes",
        _ => "one float at a time",
    };

    // Whether a width runs here: its vectors accelerated (512 bits with the FMA instructions), and the vector paths setting
    // (IDRAK_CPU_VECTORS) allowing vectors at all (scalar: one float at a time alone).
    internal static bool LanesRun(Lanes lanes) => lanes switch
    {
        Lanes.Wide512 => VectorsAllowed && Vector512.IsHardwareAccelerated && Fma.IsSupported,
        Lanes.Wide256 => VectorsAllowed && Vector256.IsHardwareAccelerated,
        Lanes.Wide128 => VectorsAllowed && Vector128.IsHardwareAccelerated,
        _ => true,
    };

    private static bool VectorsAllowed => ComputeResources.CpuVectors != CpuVectorPaths.Scalar;

    // The widest width that runs here: the choice without measuring (CpuTuning.Autotune off).
    internal static Lanes WidestLanes => LanesRun(Lanes.Wide512) ? Lanes.Wide512 : LanesRun(Lanes.Wide256) ? Lanes.Wide256 : LanesRun(Lanes.Wide128) ? Lanes.Wide128 : Lanes.One;

    // A 32-bit integer's bits as a float lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Bits(int value) => BitConverter.Int32BitsToSingle(value);

    internal unsafe struct F512 : IFloats<Vector512<float>>
    {
        public static int Count => 16;

        public static Vector512<float> Load(float* p) => Vector512.Load(p);

        public static void Store(float* p, Vector512<float> v) => v.Store(p);

        public static Vector512<float> All(float v) => Vector512.Create(v);

        public static Vector512<float> Add(Vector512<float> a, Vector512<float> b) => a + b;

        public static Vector512<float> Sub(Vector512<float> a, Vector512<float> b) => a - b;

        public static Vector512<float> Mul(Vector512<float> a, Vector512<float> b) => a * b;

        public static Vector512<float> Div(Vector512<float> a, Vector512<float> b) => a / b;

        public static Vector512<float> Fma(Vector512<float> a, Vector512<float> b, Vector512<float> c) => Vector512.FusedMultiplyAdd(a, b, c);

        public static Vector512<float> Sqrt(Vector512<float> a) => Vector512.Sqrt(a);

        public static Vector512<float> Less(Vector512<float> a, Vector512<float> b) => Vector512.LessThan(a, b);

        public static Vector512<float> LessOrEqual(Vector512<float> a, Vector512<float> b) => Vector512.LessThanOrEqual(a, b);

        public static Vector512<float> Equal(Vector512<float> a, Vector512<float> b) => Vector512.Equals(a, b);

        public static Vector512<float> Select(Vector512<float> mask, Vector512<float> a, Vector512<float> b) => Vector512.ConditionalSelect(mask, a, b);

        public static Vector512<float> And(Vector512<float> a, Vector512<float> b) => a & b;

        public static Vector512<float> Or(Vector512<float> a, Vector512<float> b) => a | b;

        public static Vector512<float> Xor(Vector512<float> a, Vector512<float> b) => a ^ b;

        public static Vector512<float> IntAdd(Vector512<float> a, Vector512<float> b) => (a.AsInt32() + b.AsInt32()).AsSingle();

        public static Vector512<float> IntSub(Vector512<float> a, Vector512<float> b) => (a.AsInt32() - b.AsInt32()).AsSingle();

        public static Vector512<float> IntMin(Vector512<float> a, Vector512<float> b) => Vector512.Min(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector512<float> IntMax(Vector512<float> a, Vector512<float> b) => Vector512.Max(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector512<float> IntToFloat(Vector512<float> a) => Vector512.ConvertToSingle(a.AsInt32());

        public static Vector512<float> ShiftLeft(Vector512<float> a, int bits) => Vector512.ShiftLeft(a.AsInt32(), bits).AsSingle();

        public static Vector512<float> ShiftRightLogical(Vector512<float> a, int bits) => Vector512.ShiftRightLogical(a.AsInt32(), bits).AsSingle();
    }

    internal unsafe struct F256 : IFloats<Vector256<float>>
    {
        public static int Count => 8;

        public static Vector256<float> Load(float* p) => Vector256.Load(p);

        public static void Store(float* p, Vector256<float> v) => v.Store(p);

        public static Vector256<float> All(float v) => Vector256.Create(v);

        public static Vector256<float> Add(Vector256<float> a, Vector256<float> b) => a + b;

        public static Vector256<float> Sub(Vector256<float> a, Vector256<float> b) => a - b;

        public static Vector256<float> Mul(Vector256<float> a, Vector256<float> b) => a * b;

        public static Vector256<float> Div(Vector256<float> a, Vector256<float> b) => a / b;

        // The CPU's fused multiply-add where it has one, else the exact emulation (FusedInHardware: the same bits).
        public static Vector256<float> Fma(Vector256<float> a, Vector256<float> b, Vector256<float> c) =>
            FusedInHardware.Single ? Vector256.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        public static Vector256<float> Sqrt(Vector256<float> a) => Vector256.Sqrt(a);

        public static Vector256<float> Less(Vector256<float> a, Vector256<float> b) => Vector256.LessThan(a, b);

        public static Vector256<float> LessOrEqual(Vector256<float> a, Vector256<float> b) => Vector256.LessThanOrEqual(a, b);

        public static Vector256<float> Equal(Vector256<float> a, Vector256<float> b) => Vector256.Equals(a, b);

        public static Vector256<float> Select(Vector256<float> mask, Vector256<float> a, Vector256<float> b) => Vector256.ConditionalSelect(mask, a, b);

        public static Vector256<float> And(Vector256<float> a, Vector256<float> b) => a & b;

        public static Vector256<float> Or(Vector256<float> a, Vector256<float> b) => a | b;

        public static Vector256<float> Xor(Vector256<float> a, Vector256<float> b) => a ^ b;

        public static Vector256<float> IntAdd(Vector256<float> a, Vector256<float> b) => (a.AsInt32() + b.AsInt32()).AsSingle();

        public static Vector256<float> IntSub(Vector256<float> a, Vector256<float> b) => (a.AsInt32() - b.AsInt32()).AsSingle();

        public static Vector256<float> IntMin(Vector256<float> a, Vector256<float> b) => Vector256.Min(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector256<float> IntMax(Vector256<float> a, Vector256<float> b) => Vector256.Max(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector256<float> IntToFloat(Vector256<float> a) => Vector256.ConvertToSingle(a.AsInt32());

        public static Vector256<float> ShiftLeft(Vector256<float> a, int bits) => Vector256.ShiftLeft(a.AsInt32(), bits).AsSingle();

        public static Vector256<float> ShiftRightLogical(Vector256<float> a, int bits) => Vector256.ShiftRightLogical(a.AsInt32(), bits).AsSingle();
    }

    internal unsafe struct F128 : IFloats<Vector128<float>>
    {
        public static int Count => 4;

        public static Vector128<float> Load(float* p) => Vector128.Load(p);

        public static void Store(float* p, Vector128<float> v) => v.Store(p);

        public static Vector128<float> All(float v) => Vector128.Create(v);

        public static Vector128<float> Add(Vector128<float> a, Vector128<float> b) => a + b;

        public static Vector128<float> Sub(Vector128<float> a, Vector128<float> b) => a - b;

        public static Vector128<float> Mul(Vector128<float> a, Vector128<float> b) => a * b;

        public static Vector128<float> Div(Vector128<float> a, Vector128<float> b) => a / b;

        public static Vector128<float> Fma(Vector128<float> a, Vector128<float> b, Vector128<float> c) =>
            FusedInHardware.Single ? Vector128.FusedMultiplyAdd(a, b, c) : ExactFused(a, b, c);

        public static Vector128<float> Sqrt(Vector128<float> a) => Vector128.Sqrt(a);

        public static Vector128<float> Less(Vector128<float> a, Vector128<float> b) => Vector128.LessThan(a, b);

        public static Vector128<float> LessOrEqual(Vector128<float> a, Vector128<float> b) => Vector128.LessThanOrEqual(a, b);

        public static Vector128<float> Equal(Vector128<float> a, Vector128<float> b) => Vector128.Equals(a, b);

        public static Vector128<float> Select(Vector128<float> mask, Vector128<float> a, Vector128<float> b) => Vector128.ConditionalSelect(mask, a, b);

        public static Vector128<float> And(Vector128<float> a, Vector128<float> b) => a & b;

        public static Vector128<float> Or(Vector128<float> a, Vector128<float> b) => a | b;

        public static Vector128<float> Xor(Vector128<float> a, Vector128<float> b) => a ^ b;

        public static Vector128<float> IntAdd(Vector128<float> a, Vector128<float> b) => (a.AsInt32() + b.AsInt32()).AsSingle();

        public static Vector128<float> IntSub(Vector128<float> a, Vector128<float> b) => (a.AsInt32() - b.AsInt32()).AsSingle();

        public static Vector128<float> IntMin(Vector128<float> a, Vector128<float> b) => Vector128.Min(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector128<float> IntMax(Vector128<float> a, Vector128<float> b) => Vector128.Max(a.AsInt32(), b.AsInt32()).AsSingle();

        public static Vector128<float> IntToFloat(Vector128<float> a) => Vector128.ConvertToSingle(a.AsInt32());

        public static Vector128<float> ShiftLeft(Vector128<float> a, int bits) => Vector128.ShiftLeft(a.AsInt32(), bits).AsSingle();

        public static Vector128<float> ShiftRightLogical(Vector128<float> a, int bits) => Vector128.ShiftRightLogical(a.AsInt32(), bits).AsSingle();
    }

    internal unsafe struct F1 : IFloats<float>
    {
        public static int Count => 1;

        public static float Load(float* p) => *p;

        public static void Store(float* p, float v) => *p = v;

        public static float All(float v) => v;

        public static float Add(float a, float b) => a + b;

        public static float Sub(float a, float b) => a - b;

        public static float Mul(float a, float b) => a * b;

        public static float Div(float a, float b) => a / b;

        public static float Fma(float a, float b, float c) => MathF.FusedMultiplyAdd(a, b, c);

        public static float Sqrt(float a) => MathF.Sqrt(a);

        public static float Less(float a, float b) => a < b ? Bits(-1) : 0f;

        public static float LessOrEqual(float a, float b) => a <= b ? Bits(-1) : 0f;

        public static float Equal(float a, float b) => a == b ? Bits(-1) : 0f;

        public static float Select(float mask, float a, float b)
        {
            int m = BitConverter.SingleToInt32Bits(mask);
            return Bits((m & BitConverter.SingleToInt32Bits(a)) | (~m & BitConverter.SingleToInt32Bits(b)));
        }

        public static float And(float a, float b) => Bits(BitConverter.SingleToInt32Bits(a) & BitConverter.SingleToInt32Bits(b));

        public static float Or(float a, float b) => Bits(BitConverter.SingleToInt32Bits(a) | BitConverter.SingleToInt32Bits(b));

        public static float Xor(float a, float b) => Bits(BitConverter.SingleToInt32Bits(a) ^ BitConverter.SingleToInt32Bits(b));

        public static float IntAdd(float a, float b) => Bits(BitConverter.SingleToInt32Bits(a) + BitConverter.SingleToInt32Bits(b));

        public static float IntSub(float a, float b) => Bits(BitConverter.SingleToInt32Bits(a) - BitConverter.SingleToInt32Bits(b));

        public static float IntMin(float a, float b) => Bits(Math.Min(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b)));

        public static float IntMax(float a, float b) => Bits(Math.Max(BitConverter.SingleToInt32Bits(a), BitConverter.SingleToInt32Bits(b)));

        public static float IntToFloat(float a) => BitConverter.SingleToInt32Bits(a);

        public static float ShiftLeft(float a, int bits) => Bits(BitConverter.SingleToInt32Bits(a) << bits);

        public static float ShiftRightLogical(float a, int bits) => Bits((int)((uint)BitConverter.SingleToInt32Bits(a) >> bits));
    }

    // ------------------------------------------------------------------ the functions every width computes alike

    /// <summary>The CPU's float functions written once over a width's lanes (the same operations in the same order by every width).</summary>
    internal static class Fn<TV, TF> where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public static TV Zero => TF.All(0f);

        public static TV One => TF.All(1f);

        // a where a > b, else b (b also where either is NaN): a comparison and a select, alike in every width.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Larger(TV a, TV b) => TF.Select(TF.Less(b, a), a, b);

        // a where a < b, else b.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Smaller(TV a, TV b) => TF.Select(TF.Less(a, b), a, b);

        // max(a, b) as MathF.Max: NaN where either is, else the larger, +0 over -0 (equal values' bits ANDed: a zero's sign
        // kept only where both have it).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Max(TV a, TV b) =>
            TF.Select(TF.Equal(a, a), TF.Select(TF.Equal(b, b), TF.Select(TF.Less(b, a), a, TF.Select(TF.Less(a, b), b, TF.And(a, b))), b), a);

        // min(a, b) as MathF.Min: NaN where either is, else the smaller, -0 under +0 (equal values' bits ORed).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Min(TV a, TV b) =>
            TF.Select(TF.Equal(a, a), TF.Select(TF.Equal(b, b), TF.Select(TF.Less(a, b), a, TF.Select(TF.Less(b, a), b, TF.Or(a, b))), b), a);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Abs(TV a) => TF.And(a, TF.All(Bits(int.MaxValue)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Neg(TV a) => TF.Xor(a, TF.All(-0f));

        // Where a is not NaN.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV IsNumber(TV a) => TF.Equal(a, a);

        // e^x: 2^k · e^f with k = round(x · log2 e) and f = x − k · ln 2 (ln 2 in two parts), e^f a degree-6 polynomial
        // (Cephes' expf), k held within [−126, 127]; 0 below −87.3365, +∞ above 88.7228, NaN kept (CpuMath.Exp's constants).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Exp(TV x)
        {
            const float Under = -87.3365f, Over = 88.7228f;
            var c = Smaller(Larger(x, TF.All(Under)), TF.All(Over));
            var magic = TF.All(12582912f);
            var kf = TF.Fma(c, TF.All(1.44269504f), magic);
            var k = TF.IntMin(TF.IntMax(TF.IntSub(kf, magic), TF.All(Bits(-126))), TF.All(Bits(127)));
            var kFloat = TF.IntToFloat(k);
            var f = TF.Fma(kFloat, TF.All(-0.693359375f), c);
            f = TF.Fma(kFloat, TF.All(2.12194440e-4f), f);
            var p = TF.Fma(TF.All(1.9875691500E-4f), f, TF.All(1.3981999507E-3f));
            p = TF.Fma(p, f, TF.All(8.3334519073E-3f));
            p = TF.Fma(p, f, TF.All(4.1665795894E-2f));
            p = TF.Fma(p, f, TF.All(1.6666665459E-1f));
            p = TF.Fma(p, f, TF.All(5.0000001201E-1f));
            p = TF.Fma(p, TF.Mul(f, f), TF.Add(f, One));
            var result = TF.Mul(p, TF.ShiftLeft(TF.IntAdd(k, TF.All(Bits(127))), 23));
            result = TF.Select(TF.Less(x, TF.All(Under)), Zero, result);
            result = TF.Select(TF.Less(TF.All(Over), x), TF.All(float.PositiveInfinity), result);
            return TF.Select(IsNumber(x), result, x);
        }

        // ln x: e · ln 2 + ln m with x = m · 2^e, m in [√½, √2) (Cephes' logf, CpuMath.Log's constants); a subnormal x
        // scaled by 2^25 first; NaN below 0 and for NaN, −∞ at ±0, +∞ kept.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Log(TV x)
        {
            var subnormal = TF.Less(x, TF.All(1.17549435E-38f));
            var scaled = TF.Select(subnormal, TF.Mul(x, TF.All(33554432f)), x);
            var e = TF.IntSub(TF.And(TF.ShiftRightLogical(scaled, 23), TF.All(Bits(0xff))), TF.Select(subnormal, TF.All(Bits(126 + 25)), TF.All(Bits(126))));
            var m = TF.Or(TF.And(scaled, TF.All(Bits(0x007fffff))), TF.All(Bits(0x3f000000)));
            var low = TF.Less(m, TF.All(0.707106781186547524f));
            e = TF.IntAdd(e, TF.And(low, TF.All(Bits(-1))));
            m = TF.Select(low, TF.Sub(TF.Add(m, m), One), TF.Sub(m, One));
            var z = TF.Mul(m, m);
            var y = TF.Fma(TF.All(7.0376836292E-2f), m, TF.All(-1.1514610310E-1f));
            y = TF.Fma(y, m, TF.All(1.1676998740E-1f));
            y = TF.Fma(y, m, TF.All(-1.2420140846E-1f));
            y = TF.Fma(y, m, TF.All(1.4249322787E-1f));
            y = TF.Fma(y, m, TF.All(-1.6668057665E-1f));
            y = TF.Fma(y, m, TF.All(2.0000714765E-1f));
            y = TF.Fma(y, m, TF.All(-2.4999993993E-1f));
            y = TF.Fma(y, m, TF.All(3.3333331174E-1f));
            y = TF.Mul(TF.Mul(y, m), z);
            var ef = TF.IntToFloat(e);
            y = TF.Fma(ef, TF.All(-2.12194440e-4f), y);
            y = TF.Fma(z, TF.All(-0.5f), y);
            var result = TF.Fma(ef, TF.All(0.693359375f), TF.Add(m, y));
            result = TF.Select(TF.Equal(x, TF.All(float.PositiveInfinity)), x, result);
            result = TF.Select(TF.Equal(x, Zero), TF.All(float.NegativeInfinity), result);
            return TF.Select(TF.LessOrEqual(Zero, x), result, TF.All(float.NaN));
        }

        // tanh: an odd rational of x clamped to ±7.905 (CpuMath.Tanh's coefficients), ±1 past |x| = 9, NaN kept.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Tanh(TV x)
        {
            var c = Smaller(Larger(x, TF.All(-7.90531110763549805f)), TF.All(7.90531110763549805f));
            var x2 = TF.Mul(c, c);
            var p = TF.Fma(x2, TF.All(-2.76076847742355e-16f), TF.All(2.00018790482477e-13f));
            p = TF.Fma(x2, p, TF.All(-8.60467152213735e-11f));
            p = TF.Fma(x2, p, TF.All(5.12229709037114e-08f));
            p = TF.Fma(x2, p, TF.All(1.48572235717979e-05f));
            p = TF.Fma(x2, p, TF.All(6.37261928875436e-04f));
            p = TF.Fma(x2, p, TF.All(4.89352455891786e-03f));
            p = TF.Mul(p, c);
            var q = TF.Fma(x2, TF.All(1.19825839466702e-06f), TF.All(1.18534705686654e-04f));
            q = TF.Fma(x2, q, TF.All(2.26843463243900e-03f));
            q = TF.Fma(x2, q, TF.All(4.89352518554385e-03f));
            var r = TF.Div(p, q);
            r = TF.Select(TF.Less(TF.All(9f), x), One, r);
            r = TF.Select(TF.Less(x, TF.All(-9f)), TF.All(-1f), r);
            return TF.Select(IsNumber(x), r, x);
        }

        // 1 / (1 + e^-x).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Sigmoid(TV x) => TF.Div(One, TF.Add(One, Exp(Neg(x))));

        // x / (1 + e^-x).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Silu(TV x) => TF.Div(x, TF.Add(One, Exp(Neg(x))));

        // The tanh GELU: 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³))).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Gelu(TV x)
        {
            var inner = TF.Mul(TF.All(0.7978845608028654f), TF.Fma(TF.Mul(TF.All(0.044715f), TF.Mul(x, x)), x, x));
            return TF.Mul(TF.Mul(TF.All(0.5f), x), TF.Add(One, Tanh(inner)));
        }

        // erf within 1.5e-7 (Abramowitz and Stegun 7.1.26): sign(z) · (1 − t · poly(t) · e^(−z²)), t = 1 / (1 + p |z|).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Erf(TV z)
        {
            var a = Abs(z);
            var t = TF.Div(One, TF.Fma(TF.All(0.3275911f), a, One));
            var poly = TF.Fma(TF.All(1.061405429f), t, TF.All(-1.453152027f));
            poly = TF.Fma(poly, t, TF.All(1.421413741f));
            poly = TF.Fma(poly, t, TF.All(-0.284496736f));
            poly = TF.Fma(poly, t, TF.All(0.254829592f));
            poly = TF.Mul(poly, t);
            var e = TF.Sub(One, TF.Mul(poly, Exp(Neg(TF.Mul(a, a)))));
            return TF.Select(TF.Less(z, Zero), Neg(e), e);
        }

        // The exact GELU: 0.5 · x · (1 + erf(x / √2)).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV GeluErf(TV x) => TF.Mul(TF.Mul(TF.All(0.5f), x), TF.Add(One, Erf(TF.Mul(x, TF.All(0.70710678118654752f)))));

        // -1, 0 or 1 (NaN kept, a zero's sign kept).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TV Sign(TV x)
        {
            var s = TF.Select(TF.Less(Zero, x), One, TF.Select(TF.Less(x, Zero), TF.All(-1f), x));
            return s;
        }
    }

    // ------------------------------------------------------------------ sums in sixteen canonical lanes

    /// <summary>A term of a sum over [0, n): each lane's accumulator taking element i's term (an addition or a fused multiply-add).</summary>
    internal interface ITerms
    {
        TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV>;
    }

    // The sum of `terms` over [0, n): element i into lane i mod 16, each lane in order (the full blocks of 16 by the width,
    // the tail a float at a time into its lanes), then the lanes added in a fixed tree. The same bits by every width.
    // (Its lanes a local of the caller's frame, not a stackalloc: a method with a stackalloc is not inlined into a loop, so
    // a row kernel's sums in a loop were calls, compiled unoptimized for their first calls and tiered up while the rows ran.)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe float Reduce16<TV, TF, TT>(ref TT terms, nint n) where TV : unmanaged where TF : struct, IFloats<TV> where TT : struct, ITerms
    {
        Unsafe.SkipInit(out SixteenLanes lanes);
        float* at = (float*)&lanes;
        Lanes16<TV, TF, TT>(ref terms, n, at);
        return Tree16(at);
    }

    // Reduce16's sixteen lanes on the stack.
    [InlineArray(16)]
    internal struct SixteenLanes
    {
        private float _lane;
    }

    // The sixteen lanes of Reduce16 before they are added.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void Lanes16<TV, TF, TT>(ref TT terms, nint n, float* lanes) where TV : unmanaged where TF : struct, IFloats<TV> where TT : struct, ITerms
    {
        nint blocks = n & ~(nint)15, i = 0;
        if (TF.Count == 16)
        {
            var a = TF.All(0f);
            for (; i < blocks; i += 16)
            {
                a = terms.Take<TV, TF>(a, i);
            }

            TF.Store(lanes, a);
        }
        else if (TF.Count == 8)
        {
            TV a0 = TF.All(0f), a1 = a0;
            for (; i < blocks; i += 16)
            {
                a0 = terms.Take<TV, TF>(a0, i);
                a1 = terms.Take<TV, TF>(a1, i + 8);
            }

            TF.Store(lanes, a0);
            TF.Store(lanes + 8, a1);
        }
        else if (TF.Count == 4)
        {
            TV a0 = TF.All(0f), a1 = a0, a2 = a0, a3 = a0;
            for (; i < blocks; i += 16)
            {
                a0 = terms.Take<TV, TF>(a0, i);
                a1 = terms.Take<TV, TF>(a1, i + 4);
                a2 = terms.Take<TV, TF>(a2, i + 8);
                a3 = terms.Take<TV, TF>(a3, i + 12);
            }

            TF.Store(lanes, a0);
            TF.Store(lanes + 4, a1);
            TF.Store(lanes + 8, a2);
            TF.Store(lanes + 12, a3);
        }
        else
        {
            for (int j = 0; j < 16; j++)
            {
                lanes[j] = 0f;
            }

            for (; i < blocks; i += 16)
            {
                for (int j = 0; j < 16; j++)
                {
                    lanes[j] = terms.Take<float, F1>(lanes[j], i + j);
                }
            }
        }

        for (; i < n; i++)
        {
            lanes[i & 15] = terms.Take<float, F1>(lanes[i & 15], i);
        }
    }

    // Sixteen lanes added in a fixed tree: lane j with lane j + 8, then + 4, + 2, + 1. A call of its own, compiled optimized
    // from its first call: where a sum's terms were inlined, the JIT's inlining budget was spent before it, and it was a
    // call compiled unoptimized until it tiered up.
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    internal static unsafe float Tree16(float* lanes)
    {
        float a0 = lanes[0] + lanes[8], a1 = lanes[1] + lanes[9], a2 = lanes[2] + lanes[10], a3 = lanes[3] + lanes[11];
        float a4 = lanes[4] + lanes[12], a5 = lanes[5] + lanes[13], a6 = lanes[6] + lanes[14], a7 = lanes[7] + lanes[15];
        float b0 = a0 + a4, b1 = a1 + a5, b2 = a2 + a6, b3 = a3 + a7;
        float c0 = b0 + b2, c1 = b1 + b3;
        return c0 + c1;
    }

    // Σ x[i].
    internal unsafe struct SumTerms(float* x) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Add(sum, TF.Load(x + i));
    }

    // Σ x[i]².
    internal unsafe struct SquareTerms(float* x) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Load(x + i);
            return TF.Fma(v, v, sum);
        }
    }

    // Σ a[i] · b[i].
    internal unsafe struct DotTerms(float* a, float* b) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Fma(TF.Load(a + i), TF.Load(b + i), sum);
    }

    // Σ (x[i] - mean)².
    internal unsafe struct DeviationTerms(float* x, float mean) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var d = TF.Sub(TF.Load(x + i), TF.All(mean));
            return TF.Fma(d, d, sum);
        }
    }

    // Σ (a[i] + b[i])², the sum also written to s.
    internal unsafe struct AddSquareTerms(float* a, float* b, float* s) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Add(TF.Load(a + i), TF.Load(b + i));
            TF.Store(s + i, v);
            return TF.Fma(v, v, sum);
        }
    }

    // The largest x[i] (NaN kept), in sixteen lanes as Reduce16 adds: each lane's maximum, then the lanes' in the tree.
    internal static unsafe float MaxOf<TV, TF>(float* x, nint n) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        float* lanes = stackalloc float[16];
        var terms = new MaxTerms(x);
        nint blocks = n & ~(nint)15, i = 0;
        for (int j = 0; j < 16; j++)
        {
            lanes[j] = float.NegativeInfinity;
        }

        if (TF.Count > 1)
        {
            TV a0 = TF.All(float.NegativeInfinity), a1 = a0, a2 = a0, a3 = a0;
            for (; i < blocks; i += 16)
            {
                a0 = terms.Take<TV, TF>(a0, i);
                if (TF.Count <= 8)
                {
                    a1 = terms.Take<TV, TF>(a1, i + TF.Count);
                }

                if (TF.Count == 4)
                {
                    a2 = terms.Take<TV, TF>(a2, i + 8);
                    a3 = terms.Take<TV, TF>(a3, i + 12);
                }
            }

            TF.Store(lanes, a0);
            if (TF.Count <= 8)
            {
                TF.Store(lanes + TF.Count, a1);
            }

            if (TF.Count == 4)
            {
                TF.Store(lanes + 8, a2);
                TF.Store(lanes + 12, a3);
            }
        }

        for (; i < n; i++)
        {
            lanes[i & 15] = terms.Take<float, F1>(lanes[i & 15], i);
        }

        for (int s = 8; s >= 1; s >>= 1)
        {
            for (int j = 0; j < s; j++)
            {
                lanes[j] = Fn<float, F1>.Max(lanes[j], lanes[j + s]);
            }
        }

        return lanes[0];
    }

    internal unsafe struct MaxTerms(float* x) : ITerms
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly TV Take<TV, TF>(TV sum, nint i) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Max(sum, TF.Load(x + i));
    }
}
