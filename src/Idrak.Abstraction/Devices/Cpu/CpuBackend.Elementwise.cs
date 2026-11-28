// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The element-wise operations, each written once over the float lanes (IFloats) and run by the width measured for its
// shape (Elements): float32's own arithmetic, every multiply-add fused, the same bits by every width.
internal sealed unsafe partial class CpuBackend
{
    /// <summary>A function of one value, written once over the lanes.</summary>
    internal interface IFn1
    {
        static abstract TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV>;
    }

    /// <summary>A function of two values.</summary>
    internal interface IFn2
    {
        static abstract TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV>;
    }

    /// <summary>A unary function's derivative from its input and output.</summary>
    internal interface IFnBack
    {
        static abstract TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV>;
    }

    // A float function of each lane, a lane at a time (the functions computed in double and rounded once: sin, cos, pow).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static TV PerLane<TV, TF>(TV x, delegate*<float, float> f) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        if (TF.Count == 1)
        {
            return TF.All(f(Unsafe.As<TV, float>(ref x)));
        }

        float* lanes = stackalloc float[TF.Count];
        TF.Store(lanes, x);
        for (int j = 0; j < TF.Count; j++)
        {
            lanes[j] = f(lanes[j]);
        }

        return TF.Load(lanes);
    }

    private struct SigmoidFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Sigmoid(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(y, TF.Sub(TF.All(1f), y));
    }

    private struct TanhFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Tanh(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Fma(Fn<TV, TF>.Neg(y), y, TF.All(1f));
    }

    private struct ReluFn : IFn1, IFnBack
    {
        // MathF.Max(x, 0): x where it is above 0 or NaN, else +0 (-0 too).
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Select(TF.Less(TF.All(0f), x), x, TF.Select(TF.Equal(x, x), TF.All(0f), x));

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.And(TF.Less(TF.All(0f), x), TF.All(1f));
    }

    private struct SquareFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(x, x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Add(x, x);
    }

    private struct AbsFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Abs(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Select(TF.Less(TF.All(0f), x), TF.All(1f), TF.And(TF.Less(x, TF.All(0f)), TF.All(-1f)));
    }

    private struct ExpFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Exp(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => y;
    }

    private struct LogFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Log(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Div(TF.All(1f), x);
    }

    private struct GeluFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Gelu(x);

        // 0.5 (1 + t) + 0.5 x (1 - t²) c (1 + 3 a x²), t = tanh(c (x + a x³)).
        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => GeluSlope<TV, TF>(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV GeluSlope<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var x2 = TF.Mul(x, x);
        var inner = TF.Mul(TF.All(0.7978845608028654f), TF.Fma(TF.Mul(TF.All(0.044715f), x2), x, x));
        var t = Fn<TV, TF>.Tanh(inner);
        var slope = TF.Mul(TF.All(0.7978845608028654f), TF.Fma(TF.All(3f * 0.044715f), x2, TF.All(1f)));
        var half = TF.Mul(TF.All(0.5f), TF.Add(TF.All(1f), t));
        return TF.Fma(TF.Mul(TF.Mul(TF.All(0.5f), x), TF.Fma(Fn<TV, TF>.Neg(t), t, TF.All(1f))), slope, half);
    }

    private struct SqrtFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Sqrt(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Div(TF.All(0.5f), y);
    }

    private struct SinFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => PerLane<TV, TF>(x, &CpuMath.Sin);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => PerLane<TV, TF>(x, &CpuMath.Cos);
    }

    private struct CosFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => PerLane<TV, TF>(x, &CpuMath.Cos);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Neg(PerLane<TV, TF>(x, &CpuMath.Sin));
    }

    private struct SiluFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Silu(x);

        // s (1 + x (1 - s)), s = sigmoid(x).
        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => SiluSlope<TV, TF>(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV SiluSlope<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV>
    {
        var s = Fn<TV, TF>.Sigmoid(x);
        return TF.Mul(s, TF.Fma(x, TF.Sub(TF.All(1f), s), TF.All(1f)));
    }

    private struct SignFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Sign(x);

        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV> => TF.All(0f);
    }

    private struct GeluErfFn : IFn1, IFnBack
    {
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.GeluErf(x);

        // Φ(x) + x φ(x): 0.5 (1 + erf(x / √2)) + x e^(-x² / 2) / √(2π).
        public static TV Of<TV, TF>(TV x, TV y) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var cdf = TF.Mul(TF.All(0.5f), TF.Add(TF.All(1f), Fn<TV, TF>.Erf(TF.Mul(x, TF.All(0.70710678118654752f)))));
            var pdf = TF.Mul(TF.All(0.3989422804014327f), Fn<TV, TF>.Exp(TF.Mul(TF.All(-0.5f), TF.Mul(x, x))));
            return TF.Fma(x, pdf, cdf);
        }
    }

    private struct AddFn : IFn2
    {
        public static TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Add(a, b);
    }

    private struct SubFn : IFn2
    {
        public static TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Sub(a, b);
    }

    private struct MulFn : IFn2
    {
        public static TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(a, b);
    }

    private struct MaxFn : IFn2
    {
        public static TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Max(a, b);
    }

    private struct MinFn : IFn2
    {
        public static TV Of<TV, TF>(TV a, TV b) where TV : unmanaged where TF : struct, IFloats<TV> => Fn<TV, TF>.Min(a, b);
    }

    // y = f(x).
    private struct Map1<TFn>(float* x, float* y, long n) : IMap where TFn : struct, IFn1
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TFn.Of<TV, TF>(TF.Load(x + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // c = f(a, b).
    private struct Map2<TFn>(float* a, float* b, float* c, long n) : IMap where TFn : struct, IFn2
    {
        private float* _c = c;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_c + i, TFn.Of<TV, TF>(TF.Load(a + i), TF.Load(b + i)));

        public void Redirect(ref Scratch scratch) => _c = scratch.Take(n);
    }

    // dx += dy · f'(x) (f' from x and y = f(x)).
    private struct MapBack<TFn>(float* x, float* y, float* dy, float* dx, long n) : IMap where TFn : struct, IFnBack
    {
        private float* _dx = dx;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_dx + i, TF.Fma(TF.Load(dy + i), TFn.Of<TV, TF>(TF.Load(x + i), TF.Load(y + i)), TF.Load(_dx + i)));

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take(n);
    }

    // The work of an element of a unary function, for the parallel cut-over.
    private static int UnaryWork(UnaryOp op) => op switch
    {
        UnaryOp.Relu or UnaryOp.Square or UnaryOp.Abs or UnaryOp.Sign => 1,
        UnaryOp.Sqrt => 4,
        UnaryOp.Sin or UnaryOp.Cos => 64,
        _ => 24,
    };

    public override void FillKernel(Storage y, int n, float value)
    {
        fixed (float* py = V(y))
        {
            Elements("Fill", new FillMap(py, value, n), n);
        }
    }

    private struct FillMap(float* y, float value, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.All(value));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // Each element-wise operation takes its inputs through TypedInput: float32's own words as they are (one run over every
    // element), another type's values read a chunk at a time (the run a chunk at a time, its kernel indexing elements as
    // in the whole run).
    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        if (Wide(x, y))
        {
            Elements64("Unary " + op, Map64.Unary, (int)op, x, null, y, n);
            return;
        }

        fixed (float* wx = D(x), wy = D(y))
        {
            // A typed output written through its type a chunk (whole groups of its blocks) at a time (TypedInOut.Into).
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Unary");
            var ty = new TypedInOut(y, wy, 0, ChunkValues, "Unary");
            for (long i = 0, step = tx.Typed || ty.Typed ? ty.Piece(ChunkValues) : Math.Max(1, n); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Unary(op, tx.Range(i, e - i), ty.Into(i), n, i, e);
                ty.Store(i, e - i);
            }
        }
    }

    private static void Unary(UnaryOp op, float* px, float* py, long n, long first, long last)
    {
        int work = UnaryWork(op);
        switch (op)
        {
            case UnaryOp.Sigmoid: Elements("Unary Sigmoid", new Map1<SigmoidFn>(px, py, n), last, work, first); break;
            case UnaryOp.Tanh: Elements("Unary Tanh", new Map1<TanhFn>(px, py, n), last, work, first); break;
            case UnaryOp.Relu: Elements("Unary Relu", new Map1<ReluFn>(px, py, n), last, work, first); break;
            case UnaryOp.Square: Elements("Unary Square", new Map1<SquareFn>(px, py, n), last, work, first); break;
            case UnaryOp.Abs: Elements("Unary Abs", new Map1<AbsFn>(px, py, n), last, work, first); break;
            case UnaryOp.Exp: Elements("Unary Exp", new Map1<ExpFn>(px, py, n), last, work, first); break;
            case UnaryOp.Log: Elements("Unary Log", new Map1<LogFn>(px, py, n), last, work, first); break;
            case UnaryOp.Gelu: Elements("Unary Gelu", new Map1<GeluFn>(px, py, n), last, work, first); break;
            case UnaryOp.Sqrt: Elements("Unary Sqrt", new Map1<SqrtFn>(px, py, n), last, work, first); break;
            case UnaryOp.Sin: Elements("Unary Sin", new Map1<SinFn>(px, py, n), last, work, first); break;
            case UnaryOp.Cos: Elements("Unary Cos", new Map1<CosFn>(px, py, n), last, work, first); break;
            case UnaryOp.Silu: Elements("Unary Silu", new Map1<SiluFn>(px, py, n), last, work, first); break;
            case UnaryOp.Sign: Elements("Unary Sign", new Map1<SignFn>(px, py, n), last, work, first); break;
            case UnaryOp.GeluErf: Elements("Unary GeluErf", new Map1<GeluErfFn>(px, py, n), last, work, first); break;
            default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a unary function.");
        }
    }

    public override void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        if (Wide64(x, y, dy, dx))
        {
            // In float64 (an input or the gradient needing it), written in dx's type: dx += dy · f'(x, y).
            Elements64("UnaryBackward " + op, Map64.UnaryBack, (int)op, x, y, dx, n, c: dy, d: dx);
            return;
        }

        fixed (float* wx = D(x), wy = D(y), wdy = D(dy), pdx = V(dx))
        {
            TypedInput tx = new(x, wx, 0, ChunkValues, "UnaryBackward"), ty = new(y, wy, 1, ChunkValues, "UnaryBackward"), tdy = new(dy, wdy, 2, ChunkValues, "UnaryBackward");
            for (long i = 0, step = Step(n, tx.Typed || ty.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                UnaryBackward(op, tx.Range(i, e - i), ty.Range(i, e - i), tdy.Range(i, e - i), pdx, n, i, e);
            }
        }
    }

    private static void UnaryBackward(UnaryOp op, float* px, float* py, float* pdy, float* pdx, long n, long first, long last)
    {
        int work = UnaryWork(op) + 2;
        switch (op)
        {
            case UnaryOp.Sigmoid: Elements("UnaryBackward Sigmoid", new MapBack<SigmoidFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Tanh: Elements("UnaryBackward Tanh", new MapBack<TanhFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Relu: Elements("UnaryBackward Relu", new MapBack<ReluFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Square: Elements("UnaryBackward Square", new MapBack<SquareFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Abs: Elements("UnaryBackward Abs", new MapBack<AbsFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Exp: Elements("UnaryBackward Exp", new MapBack<ExpFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Log: Elements("UnaryBackward Log", new MapBack<LogFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Gelu: Elements("UnaryBackward Gelu", new MapBack<GeluFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Sqrt: Elements("UnaryBackward Sqrt", new MapBack<SqrtFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Sin: Elements("UnaryBackward Sin", new MapBack<SinFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Cos: Elements("UnaryBackward Cos", new MapBack<CosFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Silu: Elements("UnaryBackward Silu", new MapBack<SiluFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.Sign: Elements("UnaryBackward Sign", new MapBack<SignFn>(px, py, pdy, pdx, n), last, work, first); break;
            case UnaryOp.GeluErf: Elements("UnaryBackward GeluErf", new MapBack<GeluErfFn>(px, py, pdy, pdx, n), last, work, first); break;
            default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a unary function.");
        }
    }

    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        if (Wide64(a, b, c))
        {
            Elements64("Binary " + op, Map64.Binary, (int)op, a, b, c, n);
            return;
        }

        fixed (float* wa = D(a), wb = D(b), wc = D(c))
        {
            TypedInput ta = new(a, wa, 0, ChunkValues, "Binary"), tb = new(b, wb, 1, ChunkValues, "Binary");
            var tc = new TypedInOut(c, wc, 0, ChunkValues, "Binary");              // a typed output written through its type
            for (long i = 0, step = ta.Typed || tb.Typed || tc.Typed ? tc.Piece(ChunkValues) : Math.Max(1, n); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                float* pa = ta.Range(i, e - i), pb = tb.Range(i, e - i), pc = tc.Into(i);
                switch (op)
                {
                    case BinaryOp.Add: Elements("Binary Add", new Map2<AddFn>(pa, pb, pc, n), e, 1, i); break;
                    case BinaryOp.Sub: Elements("Binary Sub", new Map2<SubFn>(pa, pb, pc, n), e, 1, i); break;
                    case BinaryOp.Mul: Elements("Binary Mul", new Map2<MulFn>(pa, pb, pc, n), e, 1, i); break;
                    case BinaryOp.Maximum: Elements("Binary Maximum", new Map2<MaxFn>(pa, pb, pc, n), e, 1, i); break;
                    case BinaryOp.Minimum: Elements("Binary Minimum", new Map2<MinFn>(pa, pb, pc, n), e, 1, i); break;
                    default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a binary function.");
                }

                tc.Store(i, e - i);
            }
        }
    }

    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        fixed (float* wa = D(a), wb = D(b), wdy = D(dy), pda = da is null ? null : V(da), pdb = db is null ? null : V(db))
        {
            TypedInput ta = new(a, wa, 0, ChunkValues, "ExtremumBackward"), tb = new(b, wb, 1, ChunkValues, "ExtremumBackward"), tdy = new(dy, wdy, 2, ChunkValues, "ExtremumBackward");
            for (long i = 0, step = Step(n, ta.Typed || tb.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("ExtremumBackward", new ExtremumMap(op == BinaryOp.Maximum, ta.Range(i, e - i), tb.Range(i, e - i), tdy.Range(i, e - i), pda, pdb, n), e, 2, i);
            }
        }
    }

    // da += dy where a wins (max: a ≥ b; min: a ≤ b), db += dy elsewhere.
    private struct ExtremumMap(bool maximum, float* a, float* b, float* dy, float* da, float* db, long n) : IMap
    {
        private float* _da = da, _db = db;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV va = TF.Load(a + i), vb = TF.Load(b + i), g = TF.Load(dy + i);
            var wins = maximum ? TF.LessOrEqual(vb, va) : TF.LessOrEqual(va, vb);
            if (_da != null)
            {
                TF.Store(_da + i, TF.Add(TF.Load(_da + i), TF.And(wins, g)));
            }

            if (_db != null)
            {
                TF.Store(_db + i, TF.Add(TF.Load(_db + i), TF.Select(wins, TF.All(0f), g)));
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _da = _da == null ? null : scratch.Take(n);
            _db = _db == null ? null : scratch.Take(n);
        }
    }

    public override void PowKernel(Storage x, Storage y, int n, float exponent)
    {
        if (Wide(x, y))
        {
            Elements64("Pow", Map64.Pow, 0, x, null, y, n, exponent);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Pow");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Pow", new PowMap(tx.Range(i, e - i), py, exponent, n), e, 64, i);
            }
        }
    }

    // y = x^e: the double power rounded once (CpuMath.Pow), a lane at a time.
    private struct PowMap(float* x, float* y, float exponent, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            for (int j = 0; j < TF.Count; j++)
            {
                _y[i + j] = CpuMath.Pow(x[i + j], exponent);
            }
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        if (Wide64(x, dy, dx))
        {
            Elements64("PowBackward", Map64.PowBack, 0, x, dy, dx, n, exponent, c: dx);
            return;
        }

        fixed (float* wx = D(x), wdy = D(dy), pdx = V(dx))
        {
            TypedInput tx = new(x, wx, 0, ChunkValues, "PowBackward"), tdy = new(dy, wdy, 1, ChunkValues, "PowBackward");
            for (long i = 0, step = Step(n, tx.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("PowBackward", new PowBackMap(tx.Range(i, e - i), tdy.Range(i, e - i), pdx, exponent, n), e, 64, i);
            }
        }
    }

    // dx += dy · e · x^(e - 1).
    private struct PowBackMap(float* x, float* dy, float* dx, float exponent, long n) : IMap
    {
        private float* _dx = dx;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            for (int j = 0; j < TF.Count; j++)
            {
                float slope = exponent == 0f ? 0f : exponent * CpuMath.Pow(x[i + j], exponent - 1f);
                _dx[i + j] = MathF.FusedMultiplyAdd(dy[i + j], slope, _dx[i + j]);
            }
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take(n);
    }

    public override void ClampKernel(Storage x, Storage y, int n, float min, float max)
    {
        if (Wide(x, y))
        {
            Elements64("Clamp", Map64.Clamp, 0, x, null, y, n, min, max);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Clamp");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Clamp", new ClampMap(tx.Range(i, e - i), py, min, max, n), e, 1, i);
            }
        }
    }

    private struct ClampMap(float* x, float* y, float min, float max, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, Fn<TV, TF>.Min(Fn<TV, TF>.Max(TF.Load(x + i), TF.All(min)), TF.All(max)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        if (Wide64(x, dy, dx))
        {
            Elements64("ClampBackward", Map64.ClampBack, 0, x, dy, dx, n, min, max, c: dx);
            return;
        }

        fixed (float* wx = D(x), wdy = D(dy), pdx = V(dx))
        {
            TypedInput tx = new(x, wx, 0, ChunkValues, "ClampBackward"), tdy = new(dy, wdy, 1, ChunkValues, "ClampBackward");
            for (long i = 0, step = Step(n, tx.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("ClampBackward", new ClampBackMap(tx.Range(i, e - i), tdy.Range(i, e - i), pdx, min, max, n), e, 1, i);
            }
        }
    }

    // dx += dy where min ≤ x ≤ max.
    private struct ClampBackMap(float* x, float* dy, float* dx, float min, float max, long n) : IMap
    {
        private float* _dx = dx;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Load(x + i);
            var inside = TF.And(TF.LessOrEqual(TF.All(min), v), TF.LessOrEqual(v, TF.All(max)));
            TF.Store(_dx + i, TF.Add(TF.Load(_dx + i), TF.And(inside, TF.Load(dy + i))));
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take(n);
    }

    public override void LeakyReluKernel(Storage x, Storage y, int n, float slope)
    {
        if (Wide(x, y))
        {
            Elements64("LeakyRelu", Map64.Leaky, 0, x, null, y, n, slope);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "LeakyRelu");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("LeakyRelu", new LeakyMap(tx.Range(i, e - i), py, slope, n), e, 1, i);
            }
        }
    }

    private struct LeakyMap(float* x, float* y, float slope, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Load(x + i);
            TF.Store(_y + i, TF.Select(TF.Less(TF.All(0f), v), v, TF.Mul(TF.All(slope), v)));
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void LeakyReluBackwardKernel(Storage x, Storage dy, Storage dx, int n, float slope)
    {
        if (Wide64(x, dy, dx))
        {
            Elements64("LeakyReluBackward", Map64.LeakyBack, 0, x, dy, dx, n, slope, c: dx);
            return;
        }

        fixed (float* wx = D(x), wdy = D(dy), pdx = V(dx))
        {
            TypedInput tx = new(x, wx, 0, ChunkValues, "LeakyReluBackward"), tdy = new(dy, wdy, 1, ChunkValues, "LeakyReluBackward");
            for (long i = 0, step = Step(n, tx.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("LeakyReluBackward", new LeakyBackMap(tx.Range(i, e - i), tdy.Range(i, e - i), pdx, slope, n), e, 1, i);
            }
        }
    }

    private struct LeakyBackMap(float* x, float* dy, float* dx, float slope, long n) : IMap
    {
        private float* _dx = dx;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var g = TF.Load(dy + i);
            var d = TF.Select(TF.Less(TF.All(0f), TF.Load(x + i)), g, TF.Mul(TF.All(slope), g));
            TF.Store(_dx + i, TF.Add(TF.Load(_dx + i), d));
        }

        public void Redirect(ref Scratch scratch) => _dx = scratch.Take(n);
    }

    public override void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n)
    {
        if (Wide64(condition, a, b, y))
        {
            Elements64("Where", Map64.Where, 0, condition, a, y, n, c: b);
            return;
        }

        fixed (float* wc = D(condition), wa = D(a), wb = D(b), py = V(y))
        {
            TypedInput tc = new(condition, wc, 0, ChunkValues, "Where"), ta = new(a, wa, 1, ChunkValues, "Where"), tb = new(b, wb, 2, ChunkValues, "Where");
            for (long i = 0, step = Step(n, tc.Typed || ta.Typed || tb.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Where", new WhereMap(tc.Range(i, e - i), ta.Range(i, e - i), tb.Range(i, e - i), py, n), e, 1, i);
            }
        }
    }

    private struct WhereMap(float* c, float* a, float* b, float* y, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, TF.Select(TF.Equal(TF.Load(c + i), TF.All(0f)), TF.Load(b + i), TF.Load(a + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        fixed (float* wc = D(condition), wdy = D(dy), pda = da is null ? null : V(da), pdb = db is null ? null : V(db))
        {
            TypedInput tc = new(condition, wc, 0, ChunkValues, "WhereBackward"), tdy = new(dy, wdy, 1, ChunkValues, "WhereBackward");
            for (long i = 0, step = Step(n, tc.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("WhereBackward", new WhereBackMap(tc.Range(i, e - i), tdy.Range(i, e - i), pda, pdb, n), e, 1, i);
            }
        }
    }

    private struct WhereBackMap(float* c, float* dy, float* da, float* db, long n) : IMap
    {
        private float* _da = da, _db = db;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var zero = TF.Equal(TF.Load(c + i), TF.All(0f));
            var g = TF.Load(dy + i);
            if (_da != null)
            {
                TF.Store(_da + i, TF.Add(TF.Load(_da + i), TF.Select(zero, TF.All(0f), g)));
            }

            if (_db != null)
            {
                TF.Store(_db + i, TF.Add(TF.Load(_db + i), TF.And(zero, g)));
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _da = _da == null ? null : scratch.Take(n);
            _db = _db == null ? null : scratch.Take(n);
        }
    }

    public override void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        if (Wide(x, y))
        {
            Elements64("Affine", Map64.Affine, 0, x, null, y, n, alpha, beta);
            return;
        }

        fixed (float* wx = D(x), wy = D(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Affine");
            var ty = new TypedInOut(y, wy, 0, ChunkValues, "Affine");              // a typed output written through its type
            for (long i = 0, step = tx.Typed || ty.Typed ? ty.Piece(ChunkValues) : Math.Max(1, n); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Affine", new AffineMap(tx.Range(i, e - i), ty.Into(i), alpha, beta, n), e, 1, i);
                ty.Store(i, e - i);
            }
        }
    }

    // y = alpha · x + beta, rounded once.
    private struct AffineMap(float* x, float* y, float alpha, float beta, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(alpha), TF.Load(x + i), TF.All(beta)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        if (Wide(x, y))
        {
            // y (its own values read in float64) += alpha · x, written in y's type.
            Elements64("Axpy", Map64.Axpy, 0, x, y, y, n, alpha);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Axpy");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Axpy", new AxpyMap(tx.Range(i, e - i), py, alpha, n), e, 1, i);
            }
        }
    }

    // y += alpha · x, rounded once.
    private struct AxpyMap(float* x, float* y, float alpha, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(alpha), TF.Load(x + i), TF.Load(_y + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        if (Wide64(a, b, c))
        {
            Elements64("MulAdd", Map64.MulAdd, 0, a, b, c, n, c: c);
            return;
        }

        fixed (float* wa = D(a), wb = D(b), pc = V(c))
        {
            TypedInput ta = new(a, wa, 0, ChunkValues, "MulAdd"), tb = new(b, wb, 1, ChunkValues, "MulAdd");
            for (long i = 0, step = Step(n, ta.Typed || tb.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("MulAdd", new MulAddMap(ta.Range(i, e - i), tb.Range(i, e - i), pc, n), e, 1, i);
            }
        }
    }

    // c += a · b, rounded once.
    private struct MulAddMap(float* a, float* b, float* c, long n) : IMap
    {
        private float* _c = c;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_c + i, TF.Fma(TF.Load(a + i), TF.Load(b + i), TF.Load(_c + i)));

        public void Redirect(ref Scratch scratch) => _c = scratch.Take(n);
    }

    public override void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        if (Wide(s, y))
        {
            double value;
            fixed (float* ws = D(s))
            {
                ReadDoubles(s, ws, 0, 1, &value);
            }

            Elements64("AddBroadcastScalar", Map64.AddScalar, 0, y, null, y, n, scale, value);
            return;
        }

        fixed (float* ws = D(s), py = V(y))
        {
            var ts = new TypedInput(s, ws, 0, 1, "AddBroadcastScalar");
            Elements("AddBroadcastScalar", new AxpyScalarMap(py, scale, *ts.Range(0, 1), n), n);
        }
    }

    // y += scale · s, rounded once.
    private struct AxpyScalarMap(float* y, float scale, float s, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(scale), TF.All(s), TF.Load(_y + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void AxpyAtKernel(Storage x, Storage y, int offset, float alpha)
    {
        var py = V(y);
        fixed (float* wx = D(x))
        {
            var tx = new TypedInput(x, wx, 0, 1, "AxpyAt");
            py[offset] = MathF.FusedMultiplyAdd(alpha, *tx.Range(0, 1), py[offset]);
        }
    }

    public override void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        if (Wide(x, y))
        {
            Elements64("InvSqrt", Map64.InvSqrt, 0, x, null, y, n, eps);
            return;
        }

        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "InvSqrt");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("InvSqrt", new InvSqrtMap(tx.Range(i, e - i), py, eps, n), e, 4, i);
            }
        }
    }

    // y = 1 / √(x + eps).
    private struct InvSqrtMap(float* x, float* y, float eps, long n) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, TF.Div(TF.All(1f), TF.Sqrt(TF.Add(TF.Load(x + i), TF.All(eps)))));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // ------------------------------------------------------------------ gated activations

    public override void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind)
    {
        fixed (float* wg = D(gate), wu = D(up), py = V(y))
        {
            TypedInput tg = new(gate, wg, 0, ChunkValues, "GatedActivation"), tu = new(up, wu, 1, ChunkValues, "GatedActivation");
            for (long i = 0, step = Step(n, tg.Typed || tu.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Gated(tg.Range(i, e - i), tu.Range(i, e - i), py, n, kind, i, e);
            }
        }
    }

    // y = act(gate) · up over [first, last) of n (y may be up itself: each element read before it is written).
    private static void Gated(float* pg, float* pu, float* py, long n, int kind, long first = 0, long last = -1)
    {
        last = last < 0 ? n : last;
        switch (kind)
        {
            case 0: Elements("GatedActivation SiLU", new Map2<GatedSilu>(pg, pu, py, n), last, 24, first); break;
            case 1: Elements("GatedActivation GELU", new Map2<GatedGelu>(pg, pu, py, n), last, 24, first); break;
            case 2: Elements("GatedActivation ReLU", new Map2<GatedRelu>(pg, pu, py, n), last, 1, first); break;
            case 3: Elements("GatedActivation tanh-sigmoid", new Map2<GatedTanhSigmoid>(pg, pu, py, n), last, 48, first); break;
            default: throw new ArgumentOutOfRangeException(nameof(kind), kind, "A gated activation is 0 (SiLU), 1 (GELU), 2 (ReLU) or 3 (tanh · sigmoid).");
        }
    }

    private struct GatedSilu : IFn2
    {
        public static TV Of<TV, TF>(TV gate, TV up) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(Fn<TV, TF>.Silu(gate), up);
    }

    private struct GatedGelu : IFn2
    {
        public static TV Of<TV, TF>(TV gate, TV up) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(Fn<TV, TF>.Gelu(gate), up);
    }

    private struct GatedRelu : IFn2
    {
        public static TV Of<TV, TF>(TV gate, TV up) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(ReluFn.Of<TV, TF>(gate), up);
    }

    private struct GatedTanhSigmoid : IFn2
    {
        public static TV Of<TV, TF>(TV gate, TV up) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Mul(Fn<TV, TF>.Tanh(gate), Fn<TV, TF>.Sigmoid(up));
    }

    public override void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        if ((flags & 3) == 0)
        {
            return;
        }

        fixed (float* wg = D(gate), wu = D(up), wdy = D(dy), pdg = (flags & 1) != 0 ? V(dgate) : null, pdu = (flags & 2) != 0 ? V(dup) : null)
        {
            TypedInput tg = new(gate, wg, 0, ChunkValues, "GatedActivationBackward"), tu = new(up, wu, 1, ChunkValues, "GatedActivationBackward"),
                tdy = new(dy, wdy, 2, ChunkValues, "GatedActivationBackward");
            for (long i = 0, step = Step(n, tg.Typed || tu.Typed || tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("GatedActivationBackward", new GatedBackMap(kind, flags, tg.Range(i, e - i), tu.Range(i, e - i), tdy.Range(i, e - i), pdg, pdu, n), e, 48, i);
            }
        }
    }

    // dgate (+)= dy · up · act'(gate), dup (+)= dy · act(gate) (kind 3: up through the sigmoid).
    private struct GatedBackMap(int kind, int flags, float* gate, float* up, float* dy, float* dgate, float* dup, long n) : IMap
    {
        private float* _dg = dgate, _du = dup;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV g = TF.Load(gate + i), u = TF.Load(up + i), d = TF.Load(dy + i), act, slope;
            switch (kind)
            {
                case 0:
                    act = Fn<TV, TF>.Silu(g);
                    slope = SiluSlope<TV, TF>(g);
                    break;
                case 1:
                    act = Fn<TV, TF>.Gelu(g);
                    slope = GeluSlope<TV, TF>(g);
                    break;
                case 2:
                    act = ReluFn.Of<TV, TF>(g);
                    slope = TF.And(TF.Less(TF.All(0f), g), TF.All(1f));
                    break;
                default:
                {
                    var t = Fn<TV, TF>.Tanh(g);
                    var s = Fn<TV, TF>.Sigmoid(u);
                    act = t;
                    slope = TF.Fma(Fn<TV, TF>.Neg(t), t, TF.All(1f));
                    var upSlope = TF.Mul(s, TF.Sub(TF.All(1f), s));
                    if (_dg != null)
                    {
                        var gradient = TF.Mul(TF.Mul(d, s), slope);
                        TF.Store(_dg + i, (flags & 4) != 0 ? gradient : TF.Add(TF.Load(_dg + i), gradient));
                    }

                    if (_du != null)
                    {
                        var gradient = TF.Mul(TF.Mul(d, t), upSlope);
                        TF.Store(_du + i, (flags & 8) != 0 ? gradient : TF.Add(TF.Load(_du + i), gradient));
                    }

                    return;
                }
            }

            if (_dg != null)
            {
                var gradient = TF.Mul(TF.Mul(d, u), slope);
                TF.Store(_dg + i, (flags & 4) != 0 ? gradient : TF.Add(TF.Load(_dg + i), gradient));
            }

            if (_du != null)
            {
                var gradient = TF.Mul(d, act);
                TF.Store(_du + i, (flags & 8) != 0 ? gradient : TF.Add(TF.Load(_du + i), gradient));
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            _dg = _dg == null ? null : scratch.Take(n);
            _du = _du == null ? null : scratch.Take(n);
        }
    }

    // ------------------------------------------------------------------ dropout

    public override void DropoutKernel(Storage x, Storage y, int n, float p, uint seed)
    {
        fixed (float* wx = D(x), py = V(y))
        {
            var tx = new TypedInput(x, wx, 0, ChunkValues, "Dropout");
            for (long i = 0, step = Step(n, tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("Dropout", new DropoutMap(tx.Range(i, e - i), py, null, p, seed, n, accumulate: false), e, 8, i);
            }
        }
    }

    public override void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        fixed (float* wdy = D(dy), pdx = V(dx))
        {
            var tdy = new TypedInput(dy, wdy, 0, ChunkValues, "DropoutBackward");
            for (long i = 0, step = Step(n, tdy.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("DropoutBackward", new DropoutMap(tdy.Range(i, e - i), pdx, null, p, seed, n, accumulate: true), e, 8, i);
            }
        }
    }

    public override void AddDropoutKernel(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        fixed (float* wr = D(residual), wx = D(x), py = V(output))
        {
            TypedInput tr = new(residual, wr, 0, ChunkValues, "AddDropout"), tx = new(x, wx, 1, ChunkValues, "AddDropout");
            for (long i = 0, step = Step(n, tr.Typed || tx.Typed); i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("AddDropout", new DropoutMap(tx.Range(i, e - i), py, tr.Range(i, e - i), p, seed, n, accumulate: false), e, 8, i);
            }
        }
    }

    // y = keep(i) ? x / (1 - p) : 0 (plus the residual where given; added into y with accumulate).
    private struct DropoutMap(float* x, float* y, float* residual, float p, uint seed, long n, bool accumulate) : IMap
    {
        private float* _y = y;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            float keep = 1f - p;
            for (int j = 0; j < TF.Count; j++)
            {
                nint at = i + j;
                float v = DropoutMask.Keep(seed, (uint)at, p) ? x[at] / keep : 0f;
                if (residual != null)
                {
                    v = residual[at] + v;
                }

                _y[at] = accumulate ? _y[at] + v : v;
            }
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // ------------------------------------------------------------------ optimizers

    public override void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        // A typed gradient read through its type a chunk at a time; typed parameters and momentum read and written back
        // through their types a chunk at a time (TypedInOut).
        fixed (float* wp = D(p), wg = D(g), wv = v is null ? null : D(v))
        {
            TypedInOut tp = new(p, wp, 0, ChunkValues, "SgdStep"), tv = v is null ? default : new(v, wv, 1, ChunkValues, "SgdStep");
            var tg = new TypedInput(g, wg, 0, ChunkValues, "SgdStep");
            long step = tp.Typed || tv.Typed || tg.Typed ? Math.Min(tp.Piece(ChunkValues), tv.Piece(ChunkValues)) : Math.Max(1, n);
            for (long i = 0; i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                float* pp = tp.Range(i, e - i), pv = v is null ? null : tv.Range(i, e - i);
                Elements("SgdStep", new SgdMap(pp, tg.Range(i, e - i), pv, lr, momentum, n), e, 2, i);
                tp.Store(i, e - i);
                tv.Store(i, e - i);
            }
        }
    }

    // v = momentum · v + g; p -= lr · v (no v: p -= lr · g).
    private struct SgdMap(float* p, float* g, float* v, float lr, float momentum, long n) : IMap
    {
        private float* _p = p, _v = v;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var step = TF.Load(g + i);
            if (_v != null)
            {
                step = TF.Fma(TF.All(momentum), TF.Load(_v + i), step);
                TF.Store(_v + i, step);
            }

            TF.Store(_p + i, TF.Fma(TF.All(-lr), step, TF.Load(_p + i)));
        }

        public void Redirect(ref Scratch scratch)
        {
            _p = scratch.Take(n);
            _v = _v == null ? null : scratch.Take(n);
        }
    }

    public override void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        // A typed gradient read through its type a chunk at a time; typed parameters and moments read and written back
        // through their types a chunk at a time (TypedInOut).
        fixed (float* wp = D(p), wg = D(g), wm = D(m), wv = D(v))
        {
            TypedInOut tp = new(p, wp, 0, ChunkValues, "AdamStep"), tm = new(m, wm, 1, ChunkValues, "AdamStep"), tv = new(v, wv, 2, ChunkValues, "AdamStep");
            var tg = new TypedInput(g, wg, 0, ChunkValues, "AdamStep");
            long step = tp.Typed || tm.Typed || tv.Typed || tg.Typed ? Math.Min(tp.Piece(ChunkValues), Math.Min(tm.Piece(ChunkValues), tv.Piece(ChunkValues))) : Math.Max(1, n);
            for (long i = 0; i < n; i += step)
            {
                long e = Math.Min(n, i + step);
                Elements("AdamStep", new AdamMap(tp.Range(i, e - i), tg.Range(i, e - i), tm.Range(i, e - i), tv.Range(i, e - i), lr, beta1, beta2, eps, n), e, 8, i);
                tp.Store(i, e - i);
                tm.Store(i, e - i);
                tv.Store(i, e - i);
            }
        }
    }

    // m = β1 m + (1 − β1) g; v = β2 v + (1 − β2) g²; p −= lr · m / (√v + ε).
    private struct AdamMap(float* p, float* g, float* m, float* v, float lr, float beta1, float beta2, float eps, long n) : IMap
    {
        private float* _p = p, _m = m, _v = v;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var grad = TF.Load(g + i);
            var mean = TF.Fma(TF.All(beta1), TF.Load(_m + i), TF.Mul(TF.All(1f - beta1), grad));
            var square = TF.Fma(TF.All(beta2), TF.Load(_v + i), TF.Mul(TF.All(1f - beta2), TF.Mul(grad, grad)));
            TF.Store(_m + i, mean);
            TF.Store(_v + i, square);
            var step = TF.Div(TF.Mul(TF.All(lr), mean), TF.Add(TF.Sqrt(square), TF.All(eps)));
            TF.Store(_p + i, TF.Sub(TF.Load(_p + i), step));
        }

        public void Redirect(ref Scratch scratch)
        {
            _p = scratch.Take(n);
            _m = scratch.Take(n);
            _v = scratch.Take(n);
        }
    }

    public override void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        float s;
        fixed (float* w = D(sumSquares))
        {
            s = *new TypedInput(sumSquares, w, 0, 1, "ClipFactor").Range(0, 1);
        }

        V(factor)[0] = s > 0f ? MathF.Min(1f, maxNorm / MathF.Sqrt(s)) : 1f;
    }

    // ------------------------------------------------------------------ whole sums

    // Elements a sum's chunk takes: the chunks summed apart (each in sixteen lanes), then added in order, so a sum's bits
    // depend on neither the threads nor the width.
    private const int SumChunk = 16384;

    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        if (Wide(x))
        {
            // In the result's type: float64's own (never narrowed), or float32 rounded once.
            double sum = scale * Sum64(x, n, squares: false);
            fixed (float* wr = D(result))
            {
                WriteDoubles(result, wr, 0, 1, &sum);
            }

            return;
        }

        fixed (float* wx = D(x))
        {
            V(result)[0] = scale * ChunkedSum(new TypedInput(x, wx, 0, SumChunk, "Sum"), n, squares: false);
        }
    }

    public override void SumSquaresKernel(Storage x, Storage total, int n)
    {
        if (Wide(x))
        {
            // Added to the total in its type: float64's own (never narrowed), or float32 rounded once.
            double squares = Sum64(x, n, squares: true);
            fixed (float* wt = D(total))
            {
                WriteDoubles(total, wt, 0, 1, &squares, accumulate: true);
            }

            return;
        }

        fixed (float* wx = D(x))
        {
            V(total)[0] += ChunkedSum(new TypedInput(x, wx, 0, SumChunk, "SumSquares"), n, squares: true);
        }
    }

    // Σ x (or Σ x²) over [0, n): chunks of SumChunk in sixteen lanes each, the chunks' sums added in order (a typed x read
    // a chunk at a time: the same chunks, the same bits).
    private static float ChunkedSum(TypedInput x, long n, bool squares)
    {
        if (n <= 0)
        {
            return 0f;
        }

        int chunks = (int)((n + SumChunk - 1) / SumChunk);
        Span<float> sums = chunks <= 64 ? stackalloc float[chunks] : new float[chunks];
        fixed (float* ps = sums)
        {
            int step = x.Typed ? 1 : chunks;
            for (int c = 0; c < chunks; c += step)
            {
                int end = Math.Min(chunks, c + step);
                long at = (long)c * SumChunk;
                float* px = x.Range(at, Math.Min(n, (long)end * SumChunk) - at);
                Rows(squares ? "SumSquares" : "Sum", new ChunkSums(px, ps, n, squares), end, SumChunk, 1, c);
            }
        }

        float total = 0f;
        foreach (float s in sums)
        {
            total += s;
        }

        return total;
    }

    // Each chunk's sum (a row of this kernel is a chunk).
    private struct ChunkSums(float* x, float* sums, long n, bool squares) : IRowKernel
    {
        private float* _sums = sums;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public readonly void Row<TV, TF>(int c) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            long start = c * (long)SumChunk;
            nint length = (nint)Math.Min(SumChunk, n - start);
            if (squares)
            {
                var terms = new SquareTerms(x + start);
                _sums[c] = Reduce16<TV, TF, SquareTerms>(ref terms, length);
            }
            else
            {
                var terms = new SumTerms(x + start);
                _sums[c] = Reduce16<TV, TF, SumTerms>(ref terms, length);
            }
        }

        public void Redirect(ref Scratch scratch) => _sums = scratch.Take((n + SumChunk - 1) / SumChunk);
    }
}
