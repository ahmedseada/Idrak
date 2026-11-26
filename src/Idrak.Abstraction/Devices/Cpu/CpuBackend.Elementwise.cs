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
        public static TV Of<TV, TF>(TV x) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Select(TF.Less(x, TF.All(0f)), TF.All(0f), x);

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

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TFn.Of<TV, TF>(TF.Load(x + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // c = f(a, b).
    private struct Map2<TFn>(float* a, float* b, float* c, long n) : IMap where TFn : struct, IFn2
    {
        private float* _c = c;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_c + i, TFn.Of<TV, TF>(TF.Load(a + i), TF.Load(b + i)));

        public void Redirect(ref Scratch scratch) => _c = scratch.Take(n);
    }

    // dx += dy · f'(x) (f' from x and y = f(x)).
    private struct MapBack<TFn>(float* x, float* y, float* dy, float* dx, long n) : IMap where TFn : struct, IFnBack
    {
        private float* _dx = dx;

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

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.All(value));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        fixed (float* px = V(x), py = V(y))
        {
            int work = UnaryWork(op);
            switch (op)
            {
                case UnaryOp.Sigmoid: Elements("Unary Sigmoid", new Map1<SigmoidFn>(px, py, n), n, work); break;
                case UnaryOp.Tanh: Elements("Unary Tanh", new Map1<TanhFn>(px, py, n), n, work); break;
                case UnaryOp.Relu: Elements("Unary Relu", new Map1<ReluFn>(px, py, n), n, work); break;
                case UnaryOp.Square: Elements("Unary Square", new Map1<SquareFn>(px, py, n), n, work); break;
                case UnaryOp.Abs: Elements("Unary Abs", new Map1<AbsFn>(px, py, n), n, work); break;
                case UnaryOp.Exp: Elements("Unary Exp", new Map1<ExpFn>(px, py, n), n, work); break;
                case UnaryOp.Log: Elements("Unary Log", new Map1<LogFn>(px, py, n), n, work); break;
                case UnaryOp.Gelu: Elements("Unary Gelu", new Map1<GeluFn>(px, py, n), n, work); break;
                case UnaryOp.Sqrt: Elements("Unary Sqrt", new Map1<SqrtFn>(px, py, n), n, work); break;
                case UnaryOp.Sin: Elements("Unary Sin", new Map1<SinFn>(px, py, n), n, work); break;
                case UnaryOp.Cos: Elements("Unary Cos", new Map1<CosFn>(px, py, n), n, work); break;
                case UnaryOp.Silu: Elements("Unary Silu", new Map1<SiluFn>(px, py, n), n, work); break;
                case UnaryOp.Sign: Elements("Unary Sign", new Map1<SignFn>(px, py, n), n, work); break;
                case UnaryOp.GeluErf: Elements("Unary GeluErf", new Map1<GeluErfFn>(px, py, n), n, work); break;
                default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a unary function.");
            }
        }
    }

    public override void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        fixed (float* px = V(x), py = V(y), pdy = V(dy), pdx = V(dx))
        {
            int work = UnaryWork(op) + 2;
            switch (op)
            {
                case UnaryOp.Sigmoid: Elements("UnaryBackward Sigmoid", new MapBack<SigmoidFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Tanh: Elements("UnaryBackward Tanh", new MapBack<TanhFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Relu: Elements("UnaryBackward Relu", new MapBack<ReluFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Square: Elements("UnaryBackward Square", new MapBack<SquareFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Abs: Elements("UnaryBackward Abs", new MapBack<AbsFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Exp: Elements("UnaryBackward Exp", new MapBack<ExpFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Log: Elements("UnaryBackward Log", new MapBack<LogFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Gelu: Elements("UnaryBackward Gelu", new MapBack<GeluFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Sqrt: Elements("UnaryBackward Sqrt", new MapBack<SqrtFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Sin: Elements("UnaryBackward Sin", new MapBack<SinFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Cos: Elements("UnaryBackward Cos", new MapBack<CosFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Silu: Elements("UnaryBackward Silu", new MapBack<SiluFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.Sign: Elements("UnaryBackward Sign", new MapBack<SignFn>(px, py, pdy, pdx, n), n, work); break;
                case UnaryOp.GeluErf: Elements("UnaryBackward GeluErf", new MapBack<GeluErfFn>(px, py, pdy, pdx, n), n, work); break;
                default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a unary function.");
            }
        }
    }

    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        fixed (float* pa = V(a), pb = V(b), pc = V(c))
        {
            switch (op)
            {
                case BinaryOp.Add: Elements("Binary Add", new Map2<AddFn>(pa, pb, pc, n), n); break;
                case BinaryOp.Sub: Elements("Binary Sub", new Map2<SubFn>(pa, pb, pc, n), n); break;
                case BinaryOp.Mul: Elements("Binary Mul", new Map2<MulFn>(pa, pb, pc, n), n); break;
                case BinaryOp.Maximum: Elements("Binary Maximum", new Map2<MaxFn>(pa, pb, pc, n), n); break;
                case BinaryOp.Minimum: Elements("Binary Minimum", new Map2<MinFn>(pa, pb, pc, n), n); break;
                default: throw new ArgumentOutOfRangeException(nameof(op), op, "Not a binary function.");
            }
        }
    }

    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        fixed (float* pa = V(a), pb = V(b), pdy = V(dy), pda = da is null ? null : V(da), pdb = db is null ? null : V(db))
        {
            Elements("ExtremumBackward", new ExtremumMap(op == BinaryOp.Maximum, pa, pb, pdy, pda, pdb, n), n, 2);
        }
    }

    // da += dy where a wins (max: a ≥ b; min: a ≤ b), db += dy elsewhere.
    private struct ExtremumMap(bool maximum, float* a, float* b, float* dy, float* da, float* db, long n) : IMap
    {
        private float* _da = da, _db = db;

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
        fixed (float* px = V(x), py = V(y))
        {
            Elements("Pow", new PowMap(px, py, exponent, n), n, 64);
        }
    }

    // y = x^e: the double power rounded once (CpuMath.Pow), a lane at a time.
    private struct PowMap(float* x, float* y, float exponent, long n) : IMap
    {
        private float* _y = y;

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
        fixed (float* px = V(x), pdy = V(dy), pdx = V(dx))
        {
            Elements("PowBackward", new PowBackMap(px, pdy, pdx, exponent, n), n, 64);
        }
    }

    // dx += dy · e · x^(e - 1).
    private struct PowBackMap(float* x, float* dy, float* dx, float exponent, long n) : IMap
    {
        private float* _dx = dx;

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
        fixed (float* px = V(x), py = V(y))
        {
            Elements("Clamp", new ClampMap(px, py, min, max, n), n);
        }
    }

    private struct ClampMap(float* x, float* y, float min, float max, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, Fn<TV, TF>.Min(Fn<TV, TF>.Max(TF.Load(x + i), TF.All(min)), TF.All(max)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        fixed (float* px = V(x), pdy = V(dy), pdx = V(dx))
        {
            Elements("ClampBackward", new ClampBackMap(px, pdy, pdx, min, max, n), n);
        }
    }

    // dx += dy where min ≤ x ≤ max.
    private struct ClampBackMap(float* x, float* dy, float* dx, float min, float max, long n) : IMap
    {
        private float* _dx = dx;

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
        fixed (float* px = V(x), py = V(y))
        {
            Elements("LeakyRelu", new LeakyMap(px, py, slope, n), n);
        }
    }

    private struct LeakyMap(float* x, float* y, float slope, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var v = TF.Load(x + i);
            TF.Store(_y + i, TF.Select(TF.Less(TF.All(0f), v), v, TF.Mul(TF.All(slope), v)));
        }

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void LeakyReluBackwardKernel(Storage x, Storage dy, Storage dx, int n, float slope)
    {
        fixed (float* px = V(x), pdy = V(dy), pdx = V(dx))
        {
            Elements("LeakyReluBackward", new LeakyBackMap(px, pdy, pdx, slope, n), n);
        }
    }

    private struct LeakyBackMap(float* x, float* dy, float* dx, float slope, long n) : IMap
    {
        private float* _dx = dx;

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
        fixed (float* pc = V(condition), pa = V(a), pb = V(b), py = V(y))
        {
            Elements("Where", new WhereMap(pc, pa, pb, py, n), n);
        }
    }

    private struct WhereMap(float* c, float* a, float* b, float* y, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, TF.Select(TF.Equal(TF.Load(c + i), TF.All(0f)), TF.Load(b + i), TF.Load(a + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        fixed (float* pc = V(condition), pdy = V(dy), pda = da is null ? null : V(da), pdb = db is null ? null : V(db))
        {
            Elements("WhereBackward", new WhereBackMap(pc, pdy, pda, pdb, n), n);
        }
    }

    private struct WhereBackMap(float* c, float* dy, float* da, float* db, long n) : IMap
    {
        private float* _da = da, _db = db;

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
        fixed (float* px = V(x), py = V(y))
        {
            Elements("Affine", new AffineMap(px, py, alpha, beta, n), n);
        }
    }

    // y = alpha · x + beta, rounded once.
    private struct AffineMap(float* x, float* y, float alpha, float beta, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(alpha), TF.Load(x + i), TF.All(beta)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Elements("Axpy", new AxpyMap(px, py, alpha, n), n);
        }
    }

    // y += alpha · x, rounded once.
    private struct AxpyMap(float* x, float* y, float alpha, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(alpha), TF.Load(x + i), TF.Load(_y + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        fixed (float* pa = V(a), pb = V(b), pc = V(c))
        {
            Elements("MulAdd", new MulAddMap(pa, pb, pc, n), n);
        }
    }

    // c += a · b, rounded once.
    private struct MulAddMap(float* a, float* b, float* c, long n) : IMap
    {
        private float* _c = c;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_c + i, TF.Fma(TF.Load(a + i), TF.Load(b + i), TF.Load(_c + i)));

        public void Redirect(ref Scratch scratch) => _c = scratch.Take(n);
    }

    public override void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        fixed (float* ps = V(s), py = V(y))
        {
            Elements("AddBroadcastScalar", new AxpyScalarMap(py, scale, *ps, n), n);
        }
    }

    // y += scale · s, rounded once.
    private struct AxpyScalarMap(float* y, float scale, float s, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> => TF.Store(_y + i, TF.Fma(TF.All(scale), TF.All(s), TF.Load(_y + i)));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    public override void AxpyAtKernel(Storage x, Storage y, int offset, float alpha)
    {
        var px = V(x);
        var py = V(y);
        py[offset] = MathF.FusedMultiplyAdd(alpha, px[0], py[offset]);
    }

    public override void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        fixed (float* px = V(x), py = V(y))
        {
            Elements("InvSqrt", new InvSqrtMap(px, py, eps, n), n, 4);
        }
    }

    // y = 1 / √(x + eps).
    private struct InvSqrtMap(float* x, float* y, float eps, long n) : IMap
    {
        private float* _y = y;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV> =>
            TF.Store(_y + i, TF.Div(TF.All(1f), TF.Sqrt(TF.Add(TF.Load(x + i), TF.All(eps)))));

        public void Redirect(ref Scratch scratch) => _y = scratch.Take(n);
    }

    // ------------------------------------------------------------------ gated activations

    public override void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind)
    {
        fixed (float* pg = V(gate), pu = V(up), py = V(y))
        {
            Gated(pg, pu, py, n, kind);
        }
    }

    // y = act(gate) · up over [0, n) (y may be up itself: each element read before it is written).
    private static void Gated(float* pg, float* pu, float* py, long n, int kind)
    {
        switch (kind)
        {
            case 0: Elements("GatedActivation SiLU", new Map2<GatedSilu>(pg, pu, py, n), n, 24); break;
            case 1: Elements("GatedActivation GELU", new Map2<GatedGelu>(pg, pu, py, n), n, 24); break;
            case 2: Elements("GatedActivation ReLU", new Map2<GatedRelu>(pg, pu, py, n), n); break;
            case 3: Elements("GatedActivation tanh-sigmoid", new Map2<GatedTanhSigmoid>(pg, pu, py, n), n, 48); break;
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

        fixed (float* pg = V(gate), pu = V(up), pdy = V(dy), pdg = (flags & 1) != 0 ? V(dgate) : null, pdu = (flags & 2) != 0 ? V(dup) : null)
        {
            Elements("GatedActivationBackward", new GatedBackMap(kind, flags, pg, pu, pdy, pdg, pdu, n), n, 48);
        }
    }

    // dgate (+)= dy · up · act'(gate), dup (+)= dy · act(gate) (kind 3: up through the sigmoid).
    private struct GatedBackMap(int kind, int flags, float* gate, float* up, float* dy, float* dgate, float* dup, long n) : IMap
    {
        private float* _dg = dgate, _du = dup;

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
        fixed (float* px = V(x), py = V(y))
        {
            Elements("Dropout", new DropoutMap(px, py, null, p, seed, n, accumulate: false), n, 8);
        }
    }

    public override void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        fixed (float* pdy = V(dy), pdx = V(dx))
        {
            Elements("DropoutBackward", new DropoutMap(pdy, pdx, null, p, seed, n, accumulate: true), n, 8);
        }
    }

    public override void AddDropoutKernel(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        fixed (float* pr = V(residual), px = V(x), py = V(output))
        {
            Elements("AddDropout", new DropoutMap(px, py, pr, p, seed, n, accumulate: false), n, 8);
        }
    }

    // y = keep(i) ? x / (1 - p) : 0 (plus the residual where given; added into y with accumulate).
    private struct DropoutMap(float* x, float* y, float* residual, float p, uint seed, long n, bool accumulate) : IMap
    {
        private float* _y = y;

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
        fixed (float* pp = V(p), pg = V(g), pv = v is null ? null : V(v))
        {
            Elements("SgdStep", new SgdMap(pp, pg, pv, lr, momentum, n), n, 2);
        }
    }

    // v = momentum · v + g; p -= lr · v (no v: p -= lr · g).
    private struct SgdMap(float* p, float* g, float* v, float lr, float momentum, long n) : IMap
    {
        private float* _p = p, _v = v;

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
        fixed (float* pp = V(p), pg = V(g), pm = V(m), pv = V(v))
        {
            Elements("AdamStep", new AdamMap(pp, pg, pm, pv, lr, beta1, beta2, eps, n), n, 8);
        }
    }

    // m = β1 m + (1 − β1) g; v = β2 v + (1 − β2) g²; p −= lr · m / (√v + ε).
    private struct AdamMap(float* p, float* g, float* m, float* v, float lr, float beta1, float beta2, float eps, long n) : IMap
    {
        private float* _p = p, _m = m, _v = v;

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
        float s = V(sumSquares)[0];
        V(factor)[0] = s > 0f ? MathF.Min(1f, maxNorm / MathF.Sqrt(s)) : 1f;
    }

    // ------------------------------------------------------------------ whole sums

    // Elements a sum's chunk takes: the chunks summed apart (each in sixteen lanes), then added in order, so a sum's bits
    // depend on neither the threads nor the width.
    private const int SumChunk = 16384;

    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        fixed (float* px = V(x))
        {
            V(result)[0] = scale * ChunkedSum(px, n, squares: false);
        }
    }

    public override void SumSquaresKernel(Storage x, Storage total, int n)
    {
        fixed (float* px = V(x))
        {
            V(total)[0] += ChunkedSum(px, n, squares: true);
        }
    }

    // Σ x (or Σ x²) over [0, n): chunks of SumChunk in sixteen lanes each, the chunks' sums added in order.
    private static float ChunkedSum(float* x, long n, bool squares)
    {
        if (n <= 0)
        {
            return 0f;
        }

        int chunks = (int)((n + SumChunk - 1) / SumChunk);
        Span<float> sums = chunks <= 64 ? stackalloc float[chunks] : new float[chunks];
        fixed (float* ps = sums)
        {
            Rows(squares ? "SumSquares" : "Sum", new ChunkSums(x, ps, n, squares), chunks, SumChunk, 1);
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
