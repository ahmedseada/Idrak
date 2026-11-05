// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// The CPU's sums in fixed lanes (rule 70: one result on every CPU). A sum over n terms keeps LaneCount partial sums:
// term i goes to lane i mod LaneCount, each lane adds its terms in index order with the same operation (+ or FMA), and
// the lanes are added at the end in one fixed tree (lane j += lane j + 8, then + 4, + 2, + 1). A vector of the machine's
// width holds a run of lanes, so one 512-bit accumulator, two 256-bit (AVX2) or four 128-bit (NEON, SSE) keep the very
// same partial sums as the scalar path's sixteen floats: the bits of a sum do not depend on the vector width the machine
// has, on whether the runtime accelerates vectors at all (DOTNET_EnableHWIntrinsic=0), nor on where the tail starts.
internal static partial class CpuMath
{
    /// <summary>The partial sums every vectorized sum keeps (see the notes above).</summary>
    internal const int LaneCount = 16;

    [InlineArray(LaneCount)]
    internal struct FloatLanes
    {
        private float _first;
    }

    [InlineArray(LaneCount)]
    internal struct DoubleLanes
    {
        private double _first;
    }

    /// <summary>The terms of a sum: one more vector of the machine's width (the terms from i on) or one more term (i).</summary>
    internal interface ILaneSum
    {
        Vector<float> Add(Vector<float> sum, int i);

        float Add(float sum, int i);
    }

    /// <summary>Two sums over the same terms in one pass (a sum and a sum of products).</summary>
    internal interface ILanePair
    {
        void Add(ref Vector<float> first, ref Vector<float> second, int i);

        void Add(ref float first, ref float second, int i);
    }

    /// <summary>
    /// A sum in double: the vector of the machine's width from i, widened, added to two double vectors (the first half
    /// of the terms to the low one), or one term.
    /// </summary>
    internal interface ILaneDoubleSum
    {
        void Add(ref Vector<double> low, ref Vector<double> high, int i);

        double Add(double sum, int i);
    }

    // Whole vectors cover a block of LaneCount terms (4, 8 or 16 floats a vector: 4, 2 or 1 of them).
    private static bool LaneVectors => Vector.IsHardwareAccelerated && Vector<float>.Count is 4 or 8 or 16;

    /// <summary>The sum of the n terms in the fixed lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float Fold<T>(T terms, int n) where T : ILaneSum, allows ref struct
    {
        FloatLanes lanes = default;
        ref float s = ref lanes[0];
        int i = 0, w = Vector<float>.Count;
        if (LaneVectors && n >= LaneCount)
        {
            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            for (; i <= n - LaneCount; i += LaneCount)
            {
                a0 = terms.Add(a0, i);
                if (w <= 8)
                {
                    a1 = terms.Add(a1, i + w);
                }

                if (w == 4)
                {
                    a2 = terms.Add(a2, i + 8);
                    a3 = terms.Add(a3, i + 12);
                }
            }

            a0.StoreUnsafe(ref s);
            if (w <= 8)
            {
                a1.StoreUnsafe(ref s, (nuint)w);
            }

            if (w == 4)
            {
                a2.StoreUnsafe(ref s, 8);
                a3.StoreUnsafe(ref s, 12);
            }
        }

        for (; i < n; i++)
        {
            ref float lane = ref Unsafe.Add(ref s, i & (LaneCount - 1));
            lane = terms.Add(lane, i);
        }

        return Reduce(ref s);
    }

    /// <summary>The two sums of the n terms, each in the fixed lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static (float First, float Second) FoldPair<T>(T terms, int n) where T : ILanePair, allows ref struct
    {
        FloatLanes firsts = default, seconds = default;
        ref float s = ref firsts[0];
        ref float t = ref seconds[0];
        int i = 0, w = Vector<float>.Count;
        if (LaneVectors && n >= LaneCount)
        {
            Vector<float> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector<float> b0 = default, b1 = default, b2 = default, b3 = default;
            for (; i <= n - LaneCount; i += LaneCount)
            {
                terms.Add(ref a0, ref b0, i);
                if (w <= 8)
                {
                    terms.Add(ref a1, ref b1, i + w);
                }

                if (w == 4)
                {
                    terms.Add(ref a2, ref b2, i + 8);
                    terms.Add(ref a3, ref b3, i + 12);
                }
            }

            a0.StoreUnsafe(ref s);
            b0.StoreUnsafe(ref t);
            if (w <= 8)
            {
                a1.StoreUnsafe(ref s, (nuint)w);
                b1.StoreUnsafe(ref t, (nuint)w);
            }

            if (w == 4)
            {
                a2.StoreUnsafe(ref s, 8);
                a3.StoreUnsafe(ref s, 12);
                b2.StoreUnsafe(ref t, 8);
                b3.StoreUnsafe(ref t, 12);
            }
        }

        for (; i < n; i++)
        {
            terms.Add(ref Unsafe.Add(ref s, i & (LaneCount - 1)), ref Unsafe.Add(ref t, i & (LaneCount - 1)), i);
        }

        return (Reduce(ref s), Reduce(ref t));
    }

    /// <summary>The sum of the n terms in sixteen double lanes (the same lanes as <see cref="Fold{T}"/>'s, in double).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double FoldDouble<T>(T terms, int n) where T : ILaneDoubleSum, allows ref struct
    {
        DoubleLanes lanes = default;
        ref double s = ref lanes[0];
        int i = 0, w = Vector<float>.Count, h = w / 2;
        if (LaneVectors && n >= LaneCount)
        {
            Vector<double> d0 = default, d1 = default, d2 = default, d3 = default, d4 = default, d5 = default, d6 = default, d7 = default;
            for (; i <= n - LaneCount; i += LaneCount)
            {
                terms.Add(ref d0, ref d1, i);
                if (w <= 8)
                {
                    terms.Add(ref d2, ref d3, i + w);
                }

                if (w == 4)
                {
                    terms.Add(ref d4, ref d5, i + 8);
                    terms.Add(ref d6, ref d7, i + 12);
                }
            }

            d0.StoreUnsafe(ref s);
            d1.StoreUnsafe(ref s, (nuint)h);
            if (w <= 8)
            {
                d2.StoreUnsafe(ref s, (nuint)(2 * h));
                d3.StoreUnsafe(ref s, (nuint)(3 * h));
            }

            if (w == 4)
            {
                d4.StoreUnsafe(ref s, 8);
                d5.StoreUnsafe(ref s, 10);
                d6.StoreUnsafe(ref s, 12);
                d7.StoreUnsafe(ref s, 14);
            }
        }

        for (; i < n; i++)
        {
            ref double lane = ref Unsafe.Add(ref s, i & (LaneCount - 1));
            lane = terms.Add(lane, i);
        }

        return Reduce(ref s);
    }

    // The lanes' fixed tree: lane j += lane j + 8, then j + 4, j + 2 and j + 1 (four 128-bit adds and one scalar add,
    // the same operations as the scalar tree, on any machine).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Reduce(ref float s)
    {
        var low = Vector128.LoadUnsafe(ref s) + Vector128.LoadUnsafe(ref s, 8);               // lanes 0-3 (+ 8-11)
        var high = Vector128.LoadUnsafe(ref s, 4) + Vector128.LoadUnsafe(ref s, 12);          // lanes 4-7 (+ 12-15)
        var four = low + high;
        var two = four + Vector128.Shuffle(four, Vector128.Create(2, 3, 0, 1));
        return two.ToScalar() + two.GetElement(1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Reduce(ref double s)
    {
        var q0 = Vector128.LoadUnsafe(ref s) + Vector128.LoadUnsafe(ref s, 8);                // lanes 0-1 (+ 8-9)
        var q1 = Vector128.LoadUnsafe(ref s, 2) + Vector128.LoadUnsafe(ref s, 10);
        var q2 = Vector128.LoadUnsafe(ref s, 4) + Vector128.LoadUnsafe(ref s, 12);
        var q3 = Vector128.LoadUnsafe(ref s, 6) + Vector128.LoadUnsafe(ref s, 14);
        var two = q0 + q2 + (q1 + q3);                                                       // lanes 0-1 (+ 4-5), then + (2-3 (+ 6-7))
        return two.ToScalar() + two.GetElement(1);
    }

    /// <summary>
    /// <see cref="Vector{T}.Count"/> bfloat16 values from <paramref name="halves"/> widened to float (exactly: each half as a
    /// float's high bits), reading only those halves.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> WidenBFloat16(ref ushort halves)
    {
        if (Vector<float>.Count == 8)
        {
            return (Vector256.WidenLower(Vector128.LoadUnsafe(ref halves).ToVector256Unsafe()) << 16).AsSingle().AsVector();
        }

        if (Vector<float>.Count == 4)
        {
            var four = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ref Unsafe.As<ushort, byte>(ref halves))).AsUInt16();
            return (Vector128.WidenLower(four) << 16).AsSingle().AsVector();
        }

        var sixteen = Vector256.LoadUnsafe(ref halves);
        return Vector512.Create(Vector256.WidenLower(sixteen) << 16, Vector256.WidenUpper(sixteen) << 16).AsSingle().AsVector();
    }

    /// <summary><see cref="Vector{T}.Count"/> signed bytes from <paramref name="bytes"/> widened to float (exactly), reading only those bytes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> WidenSBytes(ref sbyte bytes)
    {
        if (Vector<float>.Count == 8)
        {
            var shorts = Vector128.WidenLower(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(ref Unsafe.As<sbyte, byte>(ref bytes))).AsSByte());
            return Vector256.ConvertToSingle(Vector256.WidenLower(shorts.ToVector256Unsafe())).AsVector();
        }

        if (Vector<float>.Count == 4)
        {
            var four = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<int>(ref Unsafe.As<sbyte, byte>(ref bytes))).AsSByte();
            return Vector128.ConvertToSingle(Vector128.WidenLower(Vector128.WidenLower(four))).AsVector();
        }

        var all = Vector128.LoadUnsafe(ref bytes);
        Vector128<short> low = Vector128.WidenLower(all), high = Vector128.WidenUpper(all);
        return Vector512.ConvertToSingle(Vector512.Create(Vector256.Create(Vector128.WidenLower(low), Vector128.WidenUpper(low)),
            Vector256.Create(Vector128.WidenLower(high), Vector128.WidenUpper(high)))).AsVector();
    }

    // ------------------------------------------------------------ the kernels' sums

    /// <summary>Σ x.</summary>
    internal static float Sum(ReadOnlySpan<float> x) => Fold(new SumTerms(ref MemoryMarshal.GetReference(x)), x.Length);

    /// <summary>Σ a · b, each product rounded, then added (the attention kernels' scores).</summary>
    internal static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        Fold(new DotTerms(ref MemoryMarshal.GetReference(a), ref MemoryMarshal.GetReference(b)), a.Length);

    /// <summary>Σ a · widened h (bfloat16), as <see cref="Dot(ReadOnlySpan{float}, ReadOnlySpan{float})"/> over the widened values.</summary>
    internal static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<ushort> h) =>
        Fold(new DotBFloat16Terms(ref MemoryMarshal.GetReference(a), ref MemoryMarshal.GetReference(h)), a.Length);

    /// <summary>Σ a · q (signed bytes), as <see cref="Dot(ReadOnlySpan{float}, ReadOnlySpan{float})"/> over the widened values.</summary>
    internal static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<sbyte> q) =>
        Fold(new DotSByteTerms(ref MemoryMarshal.GetReference(a), ref MemoryMarshal.GetReference(q)), a.Length);

    /// <summary>Σ a · b over n, each term one FMA onto its lane (the few-row products' dot products).</summary>
    internal static float DotFused(ref float a, ref float b, int n) => Fold(new FusedDotTerms(ref a, ref b), n);

    /// <summary>Σ a · widened h (bfloat16) over n, as <see cref="DotFused(ref float, ref float, int)"/> over the widened values.</summary>
    internal static float DotFused(ref float a, ref ushort h, int n) => Fold(new FusedDotBFloat16Terms(ref a, ref h), n);

    /// <summary>Σ (x − mean)², each term one FMA onto its lane.</summary>
    internal static float SquaredDeviation(ReadOnlySpan<float> x, float mean) => Fold(new SquaredDeviationTerms(ref MemoryMarshal.GetReference(x), mean), x.Length);

    /// <summary>Σ a and Σ a · b (the products as FMAs) in one pass.</summary>
    internal static (float Sum, float SumProduct) SumAndDot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        FoldPair(new SumAndDotTerms(ref MemoryMarshal.GetReference(a), ref MemoryMarshal.GetReference(b)), a.Length);

    /// <summary>Σ x² in double (each square exact).</summary>
    internal static double SumSquaresDouble(ReadOnlySpan<float> x) => FoldDouble(new SquaresDoubleTerms(ref MemoryMarshal.GetReference(x)), x.Length);

    /// <summary>Σ a · b in double (each product exact).</summary>
    internal static double DotDouble(ReadOnlySpan<float> a, ReadOnlySpan<float> b) =>
        FoldDouble(new DotDoubleTerms(ref MemoryMarshal.GetReference(a), ref MemoryMarshal.GetReference(b)), a.Length);

    private readonly ref struct SumTerms : ILaneSum
    {
        private readonly ref float _x;

        public SumTerms(ref float x) => _x = ref x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => sum + Vector.LoadUnsafe(ref _x, (nuint)i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => sum + Unsafe.Add(ref _x, i);
    }

    private readonly ref struct DotTerms : ILaneSum
    {
        private readonly ref float _a, _b;

        public DotTerms(ref float a, ref float b)
        {
            _a = ref a;
            _b = ref b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => sum + Vector.LoadUnsafe(ref _a, (nuint)i) * Vector.LoadUnsafe(ref _b, (nuint)i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => sum + Unsafe.Add(ref _a, i) * Unsafe.Add(ref _b, i);
    }

    private readonly ref struct DotBFloat16Terms : ILaneSum
    {
        private readonly ref float _a;
        private readonly ref ushort _h;

        public DotBFloat16Terms(ref float a, ref ushort h)
        {
            _a = ref a;
            _h = ref h;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => sum + Vector.LoadUnsafe(ref _a, (nuint)i) * WidenBFloat16(ref Unsafe.Add(ref _h, i));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => sum + Unsafe.Add(ref _a, i) * BitConverter.Int32BitsToSingle(Unsafe.Add(ref _h, i) << 16);
    }

    private readonly ref struct DotSByteTerms : ILaneSum
    {
        private readonly ref float _a;
        private readonly ref sbyte _q;

        public DotSByteTerms(ref float a, ref sbyte q)
        {
            _a = ref a;
            _q = ref q;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => sum + Vector.LoadUnsafe(ref _a, (nuint)i) * WidenSBytes(ref Unsafe.Add(ref _q, i));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => sum + Unsafe.Add(ref _a, i) * Unsafe.Add(ref _q, i);
    }

    private readonly ref struct FusedDotTerms : ILaneSum
    {
        private readonly ref float _a, _b;

        public FusedDotTerms(ref float a, ref float b)
        {
            _a = ref a;
            _b = ref b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => Vector.FusedMultiplyAdd(Vector.LoadUnsafe(ref _a, (nuint)i), Vector.LoadUnsafe(ref _b, (nuint)i), sum);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => MathF.FusedMultiplyAdd(Unsafe.Add(ref _a, i), Unsafe.Add(ref _b, i), sum);
    }

    private readonly ref struct FusedDotBFloat16Terms : ILaneSum
    {
        private readonly ref float _a;
        private readonly ref ushort _h;

        public FusedDotBFloat16Terms(ref float a, ref ushort h)
        {
            _a = ref a;
            _h = ref h;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i) => Vector.FusedMultiplyAdd(Vector.LoadUnsafe(ref _a, (nuint)i), WidenBFloat16(ref Unsafe.Add(ref _h, i)), sum);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i) => MathF.FusedMultiplyAdd(Unsafe.Add(ref _a, i), BitConverter.Int32BitsToSingle(Unsafe.Add(ref _h, i) << 16), sum);
    }

    private readonly ref struct SquaredDeviationTerms : ILaneSum
    {
        private readonly ref float _x;
        private readonly float _mean;

        public SquaredDeviationTerms(ref float x, float mean)
        {
            _x = ref x;
            _mean = mean;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector<float> Add(Vector<float> sum, int i)
        {
            var d = Vector.LoadUnsafe(ref _x, (nuint)i) - new Vector<float>(_mean);
            return Vector.FusedMultiplyAdd(d, d, sum);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float sum, int i)
        {
            float d = Unsafe.Add(ref _x, i) - _mean;
            return MathF.FusedMultiplyAdd(d, d, sum);
        }
    }

    private readonly ref struct SumAndDotTerms : ILanePair
    {
        private readonly ref float _a, _b;

        public SumAndDotTerms(ref float a, ref float b)
        {
            _a = ref a;
            _b = ref b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref Vector<float> first, ref Vector<float> second, int i)
        {
            var a = Vector.LoadUnsafe(ref _a, (nuint)i);
            first += a;
            second = Vector.FusedMultiplyAdd(a, Vector.LoadUnsafe(ref _b, (nuint)i), second);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref float first, ref float second, int i)
        {
            float a = Unsafe.Add(ref _a, i);
            first += a;
            second = MathF.FusedMultiplyAdd(a, Unsafe.Add(ref _b, i), second);
        }
    }

    private readonly ref struct SquaresDoubleTerms : ILaneDoubleSum
    {
        private readonly ref float _x;

        public SquaresDoubleTerms(ref float x) => _x = ref x;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref Vector<double> low, ref Vector<double> high, int i)
        {
            Vector.Widen(Vector.LoadUnsafe(ref _x, (nuint)i), out var a, out var b);
            low += a * a;
            high += b * b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Add(double sum, int i)
        {
            double x = Unsafe.Add(ref _x, i);
            return sum + x * x;
        }
    }

    private readonly ref struct DotDoubleTerms : ILaneDoubleSum
    {
        private readonly ref float _a, _b;

        public DotDoubleTerms(ref float a, ref float b)
        {
            _a = ref a;
            _b = ref b;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(ref Vector<double> low, ref Vector<double> high, int i)
        {
            Vector.Widen(Vector.LoadUnsafe(ref _a, (nuint)i), out var a0, out var a1);
            Vector.Widen(Vector.LoadUnsafe(ref _b, (nuint)i), out var b0, out var b1);
            low += a0 * b0;
            high += a1 * b1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Add(double sum, int i) => sum + (double)Unsafe.Add(ref _a, i) * Unsafe.Add(ref _b, i);
    }
}
