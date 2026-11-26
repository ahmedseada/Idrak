// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// The float64 lanes: operations whose inputs hold values float32 cannot (a 64-bit float, an integer of more than 24 bits:
// WideValues) compute in float64, so such values are never narrowed. Every input is read a chunk at a time as float64
// (through its type, IElementType.Read's float64 read; float32's own values exactly), each lane the same IEEE operation
// at every width (fused multiply-adds, sums in sixteen canonical lanes and whole sums in fixed chunks), the width the
// measured choice per shape (ChosenLanes, "... float64"); the result written in the output's type (float64's own, or
// float32 rounded once from the float64 result). Built for Unary (all but the exact GELU), Binary, Affine, Sum,
// SumSquares and the product; any other operation refuses such inputs naming itself (TypedInput), never narrowing them.
internal sealed unsafe partial class CpuBackend
{
    /// <summary>The float64 lanes of one width: what the float64 operations are written over.</summary>
    internal interface IDoubles<TV> where TV : unmanaged
    {
        static abstract int Count { get; }

        static abstract TV Load(double* p);

        static abstract void Store(double* p, TV v);

        static abstract TV All(double v);

        static abstract TV Add(TV a, TV b);

        static abstract TV Sub(TV a, TV b);

        static abstract TV Mul(TV a, TV b);

        static abstract TV Div(TV a, TV b);

        static abstract TV Fma(TV a, TV b, TV c);

        static abstract TV Sqrt(TV a);

        static abstract TV Less(TV a, TV b);

        static abstract TV Equal(TV a, TV b);

        static abstract TV Select(TV mask, TV a, TV b);
    }

    internal struct D512 : IDoubles<Vector512<double>>
    {
        public static int Count => 8;
        public static Vector512<double> Load(double* p) => Vector512.Load(p);
        public static void Store(double* p, Vector512<double> v) => v.Store(p);
        public static Vector512<double> All(double v) => Vector512.Create(v);
        public static Vector512<double> Add(Vector512<double> a, Vector512<double> b) => a + b;
        public static Vector512<double> Sub(Vector512<double> a, Vector512<double> b) => a - b;
        public static Vector512<double> Mul(Vector512<double> a, Vector512<double> b) => a * b;
        public static Vector512<double> Div(Vector512<double> a, Vector512<double> b) => a / b;
        public static Vector512<double> Fma(Vector512<double> a, Vector512<double> b, Vector512<double> c) => Vector512.FusedMultiplyAdd(a, b, c);
        public static Vector512<double> Sqrt(Vector512<double> a) => Vector512.Sqrt(a);
        public static Vector512<double> Less(Vector512<double> a, Vector512<double> b) => Vector512.LessThan(a, b);
        public static Vector512<double> Equal(Vector512<double> a, Vector512<double> b) => Vector512.Equals(a, b);
        public static Vector512<double> Select(Vector512<double> m, Vector512<double> a, Vector512<double> b) => Vector512.ConditionalSelect(m, a, b);
    }

    internal struct D256 : IDoubles<Vector256<double>>
    {
        public static int Count => 4;
        public static Vector256<double> Load(double* p) => Vector256.Load(p);
        public static void Store(double* p, Vector256<double> v) => v.Store(p);
        public static Vector256<double> All(double v) => Vector256.Create(v);
        public static Vector256<double> Add(Vector256<double> a, Vector256<double> b) => a + b;
        public static Vector256<double> Sub(Vector256<double> a, Vector256<double> b) => a - b;
        public static Vector256<double> Mul(Vector256<double> a, Vector256<double> b) => a * b;
        public static Vector256<double> Div(Vector256<double> a, Vector256<double> b) => a / b;

        // The CPU's fused multiply-add where it has one, else the lanes one at a time by .NET's exact fused one (the same bits).
        public static Vector256<double> Fma(Vector256<double> a, Vector256<double> b, Vector256<double> c) =>
            FusedInHardware.Double ? Vector256.FusedMultiplyAdd(a, b, c)
            : Vector256.Create(Math.FusedMultiplyAdd(a[0], b[0], c[0]), Math.FusedMultiplyAdd(a[1], b[1], c[1]), Math.FusedMultiplyAdd(a[2], b[2], c[2]),
                Math.FusedMultiplyAdd(a[3], b[3], c[3]));

        public static Vector256<double> Sqrt(Vector256<double> a) => Vector256.Sqrt(a);
        public static Vector256<double> Less(Vector256<double> a, Vector256<double> b) => Vector256.LessThan(a, b);
        public static Vector256<double> Equal(Vector256<double> a, Vector256<double> b) => Vector256.Equals(a, b);
        public static Vector256<double> Select(Vector256<double> m, Vector256<double> a, Vector256<double> b) => Vector256.ConditionalSelect(m, a, b);
    }

    internal struct D128 : IDoubles<Vector128<double>>
    {
        public static int Count => 2;
        public static Vector128<double> Load(double* p) => Vector128.Load(p);
        public static void Store(double* p, Vector128<double> v) => v.Store(p);
        public static Vector128<double> All(double v) => Vector128.Create(v);
        public static Vector128<double> Add(Vector128<double> a, Vector128<double> b) => a + b;
        public static Vector128<double> Sub(Vector128<double> a, Vector128<double> b) => a - b;
        public static Vector128<double> Mul(Vector128<double> a, Vector128<double> b) => a * b;
        public static Vector128<double> Div(Vector128<double> a, Vector128<double> b) => a / b;

        public static Vector128<double> Fma(Vector128<double> a, Vector128<double> b, Vector128<double> c) =>
            FusedInHardware.Double ? Vector128.FusedMultiplyAdd(a, b, c) : Vector128.Create(Math.FusedMultiplyAdd(a[0], b[0], c[0]), Math.FusedMultiplyAdd(a[1], b[1], c[1]));

        public static Vector128<double> Sqrt(Vector128<double> a) => Vector128.Sqrt(a);
        public static Vector128<double> Less(Vector128<double> a, Vector128<double> b) => Vector128.LessThan(a, b);
        public static Vector128<double> Equal(Vector128<double> a, Vector128<double> b) => Vector128.Equals(a, b);
        public static Vector128<double> Select(Vector128<double> m, Vector128<double> a, Vector128<double> b) => Vector128.ConditionalSelect(m, a, b);
    }

    internal struct D1 : IDoubles<double>
    {
        public static int Count => 1;
        public static double Load(double* p) => *p;
        public static void Store(double* p, double v) => *p = v;
        public static double All(double v) => v;
        public static double Add(double a, double b) => a + b;
        public static double Sub(double a, double b) => a - b;
        public static double Mul(double a, double b) => a * b;
        public static double Div(double a, double b) => a / b;
        public static double Fma(double a, double b, double c) => Math.FusedMultiplyAdd(a, b, c);
        public static double Sqrt(double a) => Math.Sqrt(a);
        public static double Less(double a, double b) => a < b ? BitConverter.Int64BitsToDouble(-1) : 0d;
        public static double Equal(double a, double b) => a == b ? BitConverter.Int64BitsToDouble(-1) : 0d;

        public static double Select(double m, double a, double b)
        {
            long mask = BitConverter.DoubleToInt64Bits(m);
            return BitConverter.Int64BitsToDouble((mask & BitConverter.DoubleToInt64Bits(a)) | (~mask & BitConverter.DoubleToInt64Bits(b)));
        }
    }

    // A float64 function of each lane, a lane at a time (the library's own elementary functions: CpuMath).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV PerLane<TV, TD>(TV x, delegate*<double, double> f) where TV : unmanaged where TD : struct, IDoubles<TV>
    {
        if (TD.Count == 1)
        {
            return TD.All(f(Unsafe.As<TV, double>(ref x)));
        }

        double* lanes = stackalloc double[TD.Count];
        TD.Store(lanes, x);
        for (int j = 0; j < TD.Count; j++)
        {
            lanes[j] = f(lanes[j]);
        }

        return TD.Load(lanes);
    }

    private static double Sigmoid64(double x) => 1d / (1d + CpuMath.ExpDouble(-x));

    private static double Silu64(double x) => x / (1d + CpuMath.ExpDouble(-x));

    private static double Gelu64(double x) => 0.5 * x * (1d + CpuMath.TanhDouble(0.7978845608028654 * Math.FusedMultiplyAdd(0.044715 * x * x, x, x)));

    private static double Sign64(double x) => x > 0 ? 1d : x < 0 ? -1d : x;

    // y = op(x) in float64.
    private static TV Unary64<TV, TD>(UnaryOp op, TV x) where TV : unmanaged where TD : struct, IDoubles<TV>
    {
        var zero = TD.All(0d);
        return op switch
        {
            UnaryOp.Relu => TD.Select(TD.Less(zero, x), x, TD.Select(TD.Equal(x, x), zero, x)),
            UnaryOp.Square => TD.Mul(x, x),
            UnaryOp.Abs => TD.Select(TD.Less(x, zero), TD.Mul(TD.All(-1d), x), TD.Select(TD.Equal(x, zero), zero, x)),
            UnaryOp.Sqrt => TD.Sqrt(x),
            UnaryOp.Sign => PerLane<TV, TD>(x, &Sign64),
            UnaryOp.Exp => PerLane<TV, TD>(x, &CpuMath.ExpDouble),
            UnaryOp.Log => PerLane<TV, TD>(x, &CpuMath.LogDouble),
            UnaryOp.Tanh => PerLane<TV, TD>(x, &CpuMath.TanhDouble),
            UnaryOp.Sin => PerLane<TV, TD>(x, &CpuMath.SinDouble),
            UnaryOp.Cos => PerLane<TV, TD>(x, &CpuMath.CosDouble),
            UnaryOp.Sigmoid => PerLane<TV, TD>(x, &Sigmoid64),
            UnaryOp.Silu => PerLane<TV, TD>(x, &Silu64),
            UnaryOp.Gelu => PerLane<TV, TD>(x, &Gelu64),
            _ => throw new NotSupportedException($"The CPU's float64 lane has no {op} (its float64 function is not built)."),
        };
    }

    private static double Cos64(double x) => CpuMath.CosDouble(x);

    private static double NegSin64(double x) => -CpuMath.SinDouble(x);

    // s (1 + x (1 - s)), s = sigmoid(x).
    private static double SiluSlope64(double x)
    {
        double s = Sigmoid64(x);
        return s * Math.FusedMultiplyAdd(x, 1d - s, 1d);
    }

    // 0.5 (1 + t) + 0.5 x (1 - t²) c (1 + 3 a x²), t = tanh(c (x + a x³)), as the float32 lanes' GeluSlope.
    private static double GeluSlope64(double x)
    {
        double x2 = x * x;
        double t = CpuMath.TanhDouble(0.7978845608028654 * Math.FusedMultiplyAdd(0.044715 * x2, x, x));
        double slope = 0.7978845608028654 * Math.FusedMultiplyAdd(3 * 0.044715, x2, 1d);
        return Math.FusedMultiplyAdd(0.5 * x * Math.FusedMultiplyAdd(-t, t, 1d), slope, 0.5 * (1d + t));
    }

    // f'(x) of op from x and y = f(x) in float64 (the float32 lanes' formulas).
    private static TV Slope64<TV, TD>(UnaryOp op, TV x, TV y) where TV : unmanaged where TD : struct, IDoubles<TV>
    {
        var zero = TD.All(0d);
        var one = TD.All(1d);
        return op switch
        {
            UnaryOp.Sigmoid => TD.Mul(y, TD.Sub(one, y)),
            UnaryOp.Tanh => TD.Fma(TD.Mul(TD.All(-1d), y), y, one),
            UnaryOp.Relu => TD.Select(TD.Less(zero, x), one, zero),
            UnaryOp.Square => TD.Add(x, x),
            UnaryOp.Abs => TD.Select(TD.Less(zero, x), one, TD.Select(TD.Less(x, zero), TD.All(-1d), zero)),
            UnaryOp.Exp => y,
            UnaryOp.Log => TD.Div(one, x),
            UnaryOp.Sqrt => TD.Div(TD.All(0.5), y),
            UnaryOp.Sin => PerLane<TV, TD>(x, &Cos64),
            UnaryOp.Cos => PerLane<TV, TD>(x, &NegSin64),
            UnaryOp.Silu => PerLane<TV, TD>(x, &SiluSlope64),
            UnaryOp.Gelu => PerLane<TV, TD>(x, &GeluSlope64),
            UnaryOp.Sign => zero,
            _ => throw new NotSupportedException($"The CPU's float64 lane has no {op} gradient (its float64 function is not built)."),
        };
    }

    // c = op(a, b) in float64 (Maximum and Minimum as Math.Max and Math.Min: NaN kept, +0 over -0 for the maximum and -0
    // under +0 for the minimum, as the float32 lanes).
    private static TV Binary64<TV, TD>(BinaryOp op, TV a, TV b) where TV : unmanaged where TD : struct, IDoubles<TV>
    {
        switch (op)
        {
            case BinaryOp.Add:
                return TD.Add(a, b);
            case BinaryOp.Sub:
                return TD.Sub(a, b);
            case BinaryOp.Mul:
                return TD.Mul(a, b);
        }

        if (op is not (BinaryOp.Maximum or BinaryOp.Minimum))
        {
            throw new ArgumentOutOfRangeException(nameof(op), op, "Not a binary function.");
        }

        // Equal values: themselves, zeros by their signs (+0 + -0 = +0 for the maximum; -((-a) + (-b)) for the minimum).
        var zero = TD.All(0d);
        bool max = op == BinaryOp.Maximum;
        var tie = max ? TD.Add(a, b) : TD.Mul(TD.All(-1d), TD.Add(TD.Mul(TD.All(-1d), a), TD.Mul(TD.All(-1d), b)));
        var equal = TD.Select(TD.Equal(a, zero), tie, a);
        var pick = max ? TD.Select(TD.Less(b, a), a, TD.Select(TD.Less(a, b), b, equal)) : TD.Select(TD.Less(a, b), a, TD.Select(TD.Less(b, a), b, equal));
        return TD.Select(TD.Equal(a, a), TD.Select(TD.Equal(b, b), pick, b), a);
    }

    // ------------------------------------------------------------------ reading and writing float64 chunks

    // Whether any of an operation's inputs holds values that need float64.
    private static bool Wide(Storage? a, Storage? b = null) => a is { Typed: true } && WideValues(a.Type) || b is { Typed: true } && WideValues(b.Type);

    [ThreadStatic]
    private static double[]?[]? t_doubles;

    // A pinned float64 buffer of the thread's (slot 0 to 7: 3 is ReadDoubles' own) holding at least `count` values.
    private static double* DoubleBuffer(int slot, long count)
    {
        var buffers = t_doubles ??= new double[8][];
        var buffer = buffers[slot];
        if (buffer is null || buffer.Length < count)
        {
            buffers[slot] = buffer = GC.AllocateUninitializedArray<double>((int)Math.Max(count, ChunkValues), pinned: true);
        }

        return (double*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buffer));
    }

    // Values [first, + count) of an input (its words at `words`) as float64 into `into`: float32's own exactly, any other
    // type through its type's float64 read (from the unit holding the first).
    private static void ReadDoubles(Storage s, float* words, long first, int count, double* into)
    {
        if (!s.Typed)
        {
            for (int i = 0; i < count; i++)
            {
                into[i] = words[first + i];
            }

            return;
        }

        var type = s.Type;
        long unit = Math.Max(1, type.Description.UnitValues), start = first / unit * unit;
        int total = (int)(first + count - start);
        var bytes = new ReadOnlySpan<byte>((byte*)words + type.ByteCount(start), (int)type.ByteCount(total));
        if (start == first)
        {
            type.Read(bytes, new Span<double>(into, count));
            return;
        }

        double* temp = DoubleBuffer(3, total);
        type.Read(bytes, new Span<double>(temp, total));
        new ReadOnlySpan<double>(temp + (first - start), count).CopyTo(new Span<double>(into, count));
    }

    // Values [first, + count) written to an output in its type: float64's own as they are, float32 rounded once.
    private static void WriteDoubles(Storage y, float* words, long first, int count, double* from, bool accumulate = false)
    {
        if (y.Typed && y.Type.Native == NativeKind.Float64)
        {
            double* to = (double*)words + first;
            for (int i = 0; i < count; i++)
            {
                to[i] = accumulate ? to[i] + from[i] : from[i];
            }

            return;
        }

        if (!NativeFloats(y))
        {
            throw new NotSupportedException($"The CPU's float64 lane writes float64 or float32 values, not {y.Type.Name}.");
        }

        float* into = words + first;
        for (int i = 0; i < count; i++)
        {
            into[i] = accumulate ? (float)(into[i] + from[i]) : (float)from[i];
        }
    }

    // ------------------------------------------------------------------ element-wise operations in float64

    private enum Map64 : byte
    {
        Unary,                 // y = f(a)
        Binary,                // y = f(a, b)
        Affine,                // y = α a + β
        Axpy,                  // y = α a + b (b the output's own values)
        MulAdd,                // y = a b + c (c the output's)
        AddScalar,             // y = α β + a (a the output's)
        UnaryBack,             // y = c f'(a, b) + d (a the input, b the output, c the gradient, d the input's gradient)
        Clamp,                 // y = min(max(a, α), β)
        ClampBack,             // y = b where α ≤ a ≤ β (+ c, the input's gradient)
        Leaky,                 // y = a where a > 0, else α a
        LeakyBack,             // y = b or α b by a's sign (+ c)
        InvSqrt,               // y = 1 / √(a + α)
        Where,                 // y = c where a ≠ 0... a the condition: y = a == 0 ? c : b
        Pow,                   // y = a^α (a value at a time, the library's power)
        PowBack,               // y = b α a^(α-1) + c (a value at a time)
    }

    // An element-wise float64 run over a chunk: y = f(a[, b, c, d]).
    private struct Map64Run(Map64 kind, int op, double* a, double* b, double* y, int count, double alpha, double beta, double* c = null, double* d = null) : ILanesRun
    {
        private double* _y = y;

        public readonly void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512: Steps64<Vector512<double>, D512>(); break;
                case Lanes.Wide256: Steps64<Vector256<double>, D256>(); break;
                case Lanes.Wide128: Steps64<Vector128<double>, D128>(); break;
                default: Steps64<double, D1>(); break;
            }
        }

        private readonly void Steps64<TV, TD>() where TV : unmanaged where TD : struct, IDoubles<TV>
        {
            int i = 0;
            if (kind is Map64.Pow or Map64.PowBack)
            {
                // The library's float64 power, a value at a time (the same at every width).
                for (; i < count; i++)
                {
                    double p = CpuMath.PowDouble(a[i], kind == Map64.Pow ? alpha : alpha - 1);
                    _y[i] = kind == Map64.Pow ? p : Math.FusedMultiplyAdd(b[i], alpha == 0 ? 0 : alpha * p, c[i]);
                }

                return;
            }

            for (; TD.Count > 1 && i + TD.Count <= count; i += TD.Count)
            {
                TD.Store(_y + i, Of<TV, TD>(i));
            }

            for (; i < count; i++)
            {
                D1.Store(_y + i, Of<double, D1>(i));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private readonly TV Of<TV, TD>(int i) where TV : unmanaged where TD : struct, IDoubles<TV>
        {
            var x = TD.Load(a + i);
            var zero = TD.All(0d);
            switch (kind)
            {
                case Map64.Unary:
                    return Unary64<TV, TD>((UnaryOp)op, x);
                case Map64.Binary:
                    return Binary64<TV, TD>((BinaryOp)op, x, TD.Load(b + i));
                case Map64.Affine:
                    return TD.Fma(TD.All(alpha), x, TD.All(beta));
                case Map64.Axpy:
                    return TD.Fma(TD.All(alpha), x, TD.Load(b + i));
                case Map64.MulAdd:
                    return TD.Fma(x, TD.Load(b + i), TD.Load(c + i));
                case Map64.AddScalar:
                    return TD.Fma(TD.All(alpha), TD.All(beta), x);
                case Map64.UnaryBack:
                    return TD.Fma(TD.Load(c + i), Slope64<TV, TD>((UnaryOp)op, x, TD.Load(b + i)), TD.Load(d + i));
                case Map64.Clamp:
                    return Binary64<TV, TD>(BinaryOp.Minimum, Binary64<TV, TD>(BinaryOp.Maximum, x, TD.All(alpha)), TD.All(beta));
                case Map64.ClampBack:
                {
                    var inside = TD.Select(TD.Less(x, TD.All(alpha)), zero, TD.Select(TD.Less(TD.All(beta), x), zero, TD.Select(TD.Equal(x, x), TD.All(1d), zero)));
                    return TD.Add(TD.Load(c + i), TD.Select(TD.Equal(inside, zero), zero, TD.Load(b + i)));
                }

                case Map64.Leaky:
                    return TD.Select(TD.Less(zero, x), x, TD.Mul(TD.All(alpha), x));
                case Map64.LeakyBack:
                {
                    var g = TD.Load(b + i);
                    return TD.Add(TD.Load(c + i), TD.Select(TD.Less(zero, x), g, TD.Mul(TD.All(alpha), g)));
                }

                case Map64.InvSqrt:
                    return TD.Div(TD.All(1d), TD.Sqrt(TD.Add(x, TD.All(alpha))));
                default:
                    return TD.Select(TD.Equal(x, zero), TD.Load(c + i), TD.Load(b + i));     // Where
            }
        }

        public void Redirect(ref Scratch scratch) => _y = (double*)scratch.Take(2L * count);
    }

    // y = f(x[, b, c, d]) over [0, n) in float64, a chunk at a time (an input that is the output itself read before it is
    // written), the result written in y's type.
    private static void Elements64(string operation, Map64 kind, int op, Storage x, Storage? b, Storage y, long n, double alpha = 0, double beta = 0,
        Storage? c = null, Storage? d = null)
    {
        fixed (float* wx = D(x), wb = b is null ? null : D(b), wc = c is null ? null : D(c), wd = d is null ? null : D(d), wy = D(y))
        {
            for (long i = 0; i < n; i += ChunkValues)
            {
                int count = (int)Math.Min(ChunkValues, n - i);
                double* pa = DoubleBuffer(0, count), pb = b is null ? null : DoubleBuffer(1, count), py = DoubleBuffer(2, count);
                double* pc = c is null ? null : DoubleBuffer(4, count), pd = d is null ? null : DoubleBuffer(5, count);
                ReadDoubles(x, wx, i, count, pa);
                if (b is not null)
                {
                    ReadDoubles(b, wb, i, count, pb);
                }

                if (c is not null)
                {
                    ReadDoubles(c, wc, i, count, pc);
                }

                if (d is not null)
                {
                    ReadDoubles(d, wd, i, count, pd);
                }

                var run = new Map64Run(kind, op, pa, pb, py, count, alpha, beta, pc, pd);
                run.Run(ChosenLanes(operation + " float64", SizeClass(count), 0, ref run));
                WriteDoubles(y, wy, i, count, py);
            }
        }
    }

    // Whether an operation's output (read where it accumulates) or inputs hold values that need float64, or the output is
    // a float64 storage of its own: then the operation computes in float64 and writes in the output's type.
    private static bool Wide64(Storage? a, Storage? b = null, Storage? c = null, Storage? d = null) => Wide(a, b) || Wide(c, d);

    // c = a + v a row (gelu: GELU of it) in float64, a row at a time, written in c's type (the float32 lanes' sum and GELU).
    private static void RowVector64(Storage a, Storage v, Storage c, int rows, int cols, bool gelu)
    {
        fixed (float* wa = D(a), wv = D(v), wc = D(c))
        {
            double* vector = DoubleBuffer(1, Math.Max(1, cols)), row = DoubleBuffer(0, Math.Max(1, cols));
            ReadDoubles(v, wv, 0, cols, vector);
            for (int r = 0; r < rows; r++)
            {
                ReadDoubles(a, wa, (long)r * cols, cols, row);
                for (int j = 0; j < cols; j++)
                {
                    double sum = row[j] + vector[j];
                    row[j] = gelu ? Gelu64(sum) : sum;
                }

                WriteDoubles(c, wc, (long)r * cols, cols, row);
            }
        }
    }

    // y[j] += Σ_r x[offset + r · ld + j] in float64, each column's rows in order from zero, added to y in its type.
    private static void ColumnSums64(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        fixed (float* wx = D(x), wy = D(y))
        {
            double* sums = DoubleBuffer(1, Math.Max(1, cols)), row = DoubleBuffer(0, Math.Max(1, cols));
            new Span<double>(sums, cols).Clear();
            for (int r = 0; r < rows; r++)
            {
                ReadDoubles(x, wx, offset + (long)r * ld, cols, row);
                for (int j = 0; j < cols; j++)
                {
                    sums[j] += row[j];
                }
            }

            WriteDoubles(y, wy, 0, cols, sums, accumulate: true);
        }
    }

    // Σ x (or Σ x²) over [0, n) in float64: chunks of SumChunk in sixteen lanes each, the chunks' sums added in order.
    private static double Sum64(Storage x, long n, bool squares)
    {
        double total = 0;
        fixed (float* wx = D(x))
        {
            for (long i = 0; i < n; i += SumChunk)
            {
                int count = (int)Math.Min(SumChunk, n - i);
                double* pa = DoubleBuffer(0, count);
                ReadDoubles(x, wx, i, count, pa);
                var run = new Sum64Run(pa, count, squares);
                run.Run(ChosenLanes(squares ? "SumSquares float64" : "Sum float64", SizeClass(count), 0, ref run));
                total += run.Result;
            }
        }

        return total;
    }

    private struct Sum64Run(double* x, int count, bool squares) : ILanesRun
    {
        public double Result;

        public void Run(Lanes lanes) => Result = lanes switch
        {
            Lanes.Wide512 => Lanes64<Vector512<double>, D512>(),
            Lanes.Wide256 => Lanes64<Vector256<double>, D256>(),
            Lanes.Wide128 => Lanes64<Vector128<double>, D128>(),
            _ => Lanes64<double, D1>(),
        };

        // Element i into lane i mod 16, each lane in order, the lanes added in a fixed tree.
        private readonly double Lanes64<TV, TD>() where TV : unmanaged where TD : struct, IDoubles<TV>
        {
            double* lanes = stackalloc double[16];
            int i = 0, blocks = count & ~15, per = 16 / TD.Count;
            if (TD.Count > 1)
            {
                TV* sums = stackalloc TV[8];
                for (int v = 0; v < per; v++)
                {
                    sums[v] = TD.All(0d);
                }

                for (; i < blocks; i += 16)
                {
                    for (int v = 0; v < per; v++)
                    {
                        var value = TD.Load(x + i + v * TD.Count);
                        sums[v] = squares ? TD.Fma(value, value, sums[v]) : TD.Add(sums[v], value);
                    }
                }

                for (int v = 0; v < per; v++)
                {
                    TD.Store(lanes + v * TD.Count, sums[v]);
                }
            }
            else
            {
                for (int j = 0; j < 16; j++)
                {
                    lanes[j] = 0;
                }
            }

            for (; i < count; i++)
            {
                double value = x[i];
                lanes[i & 15] = squares ? Math.FusedMultiplyAdd(value, value, lanes[i & 15]) : lanes[i & 15] + value;
            }

            for (int s = 8; s >= 1; s >>= 1)
            {
                for (int j = 0; j < s; j++)
                {
                    lanes[j] += lanes[j + s];
                }
            }

            return lanes[0];
        }

        public readonly void Redirect(ref Scratch scratch)
        {
        }
    }

    // ------------------------------------------------------------------ the product in float64

    // c = op(a) · op(b) + β · c in float64: each output from β · c (0 where β is 0), its terms fused in k's order. A block of
    // rows of c is held in float64 (a chunk's worth), and B read a block of its k rows at a time (a chunk's worth, through
    // its type), each block's terms taken by every row of the block before the next: the same order for each output, and
    // no operand read whole.
    private static void Product64(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        fixed (float* wa = D(a), wb = D(b), wc = D(c))
        {
            int rowsAt = Math.Max(1, ChunkValues / Math.Max(1, n)), kAt = Math.Max(1, ChunkValues / Math.Max(1, n));
            double* acc = DoubleBuffer(2, (long)Math.Min(rowsAt, m) * n);
            double* bBlock = DoubleBuffer(1, (long)Math.Min(kAt, k) * n);
            double* aBlock = DoubleBuffer(0, (long)Math.Min(rowsAt, m) * Math.Min(kAt, k));
            for (int matrix = 0; matrix < batch; matrix++)
            {
                for (int i0 = 0; i0 < m; i0 += rowsAt)
                {
                    int rows = Math.Min(rowsAt, m - i0);
                    long at = ((long)matrix * m + i0) * n;
                    if (beta == 0f)
                    {
                        new Span<double>(acc, rows * n).Clear();
                    }
                    else
                    {
                        ReadDoubles(c, wc, at, rows * n, acc);
                        for (int j = 0; j < rows * n; j++)
                        {
                            acc[j] *= beta;
                        }
                    }

                    for (int k0 = 0; k0 < k; k0 += kAt)
                    {
                        int kc = Math.Min(kAt, k - k0);
                        for (int kk = 0; kk < kc; kk++)
                        {
                            if (transB)
                            {
                                for (int j = 0; j < n; j++)
                                {
                                    ReadDoubles(b, wb, ((long)matrix * n + j) * k + k0 + kk, 1, bBlock + (long)kk * n + j);
                                }
                            }
                            else
                            {
                                ReadDoubles(b, wb, ((long)matrix * k + k0 + kk) * n, n, bBlock + (long)kk * n);
                            }
                        }

                        for (int r = 0; r < rows; r++)
                        {
                            double* aRow = aBlock + (long)r * kc;
                            if (transA)
                            {
                                for (int kk = 0; kk < kc; kk++)
                                {
                                    ReadDoubles(a, wa, ((long)matrix * k + k0 + kk) * m + i0 + r, 1, aRow + kk);
                                }
                            }
                            else
                            {
                                ReadDoubles(a, wa, ((long)matrix * m + i0 + r) * k + k0, kc, aRow);
                            }

                            var run = new RowProduct64(aRow, bBlock, acc + (long)r * n, n, kc);
                            run.Run(ChosenLanes("MatMul float64", SizeClass(n), SizeClass(kc), ref run));
                        }
                    }

                    WriteDoubles(c, wc, at, rows * n, acc);
                }
            }
        }
    }

    // row[j] += Σ_kk a[kk] · b[kk, j], each term fused in k's order.
    private struct RowProduct64(double* a, double* b, double* row, int n, int k) : ILanesRun
    {
        private double* _row = row;

        public readonly void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512: Columns<Vector512<double>, D512>(); break;
                case Lanes.Wide256: Columns<Vector256<double>, D256>(); break;
                case Lanes.Wide128: Columns<Vector128<double>, D128>(); break;
                default: Columns<double, D1>(); break;
            }
        }

        private readonly void Columns<TV, TD>() where TV : unmanaged where TD : struct, IDoubles<TV>
        {
            int j = 0;
            for (; TD.Count > 1 && j + TD.Count <= n; j += TD.Count)
            {
                var sum = TD.Load(_row + j);
                for (int kk = 0; kk < k; kk++)
                {
                    sum = TD.Fma(TD.All(a[kk]), TD.Load(b + (long)kk * n + j), sum);
                }

                TD.Store(_row + j, sum);
            }

            for (; j < n; j++)
            {
                double sum = _row[j];
                for (int kk = 0; kk < k; kk++)
                {
                    sum = Math.FusedMultiplyAdd(a[kk], b[(long)kk * n + j], sum);
                }

                _row[j] = sum;
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            var into = (double*)scratch.Take(2L * n);
            new ReadOnlySpan<double>(_row, n).CopyTo(new Span<double>(into, n));
            _row = into;
        }
    }
}
