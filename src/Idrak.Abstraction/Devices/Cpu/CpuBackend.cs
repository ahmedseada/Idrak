// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

internal sealed class CpuStorage(CpuBackend backend, float[] data, int length) : Storage(backend, length)
{
    public float[] Data = data;                                         // null while evicted (Backend.Evict)
}

/// <summary>
/// CPU implementation: pooled managed arrays, <see cref="Vector{T}"/> SIMD kernels, and
/// <see cref="Parallel"/> for large tensors. Small tensors stay on the calling thread so
/// tiny networks are not dominated by scheduling overhead.
/// </summary>
internal sealed partial class CpuBackend : Backend
{
    public static readonly CpuBackend Instance = new();

    /// <summary>Element-wise work (≈ element operations) from which kernels split across cores: measured, see <see cref="CpuTuning.ParallelElements"/>.</summary>
    internal static int ParallelThreshold => CpuTuning.ParallelElements;

    private readonly Dictionary<int, Stack<float[]>> _pool = [];
    private readonly MemoryAccountant _memory = new(() => ComputeResources.CpuMemoryLimit, "the CPU");

    private CpuBackend()
    {
    }

    // The CPU's own limits, all from its numerical contract (CpuTuning): its packed products stream the weights for up to
    // CpuTuning.FewRows rows before expanding them once for the register-tiled product (the split decides the int4
    // rounding), and its attention kernels (which take any head size) run up to the head sizes from which the layers have
    // always used the [t, t] weights instead (the split decides the summation order, so the reference keeps it).
    public override BackendCapabilities Capabilities { get; } = new()
    {
        FewRows = CpuTuning.FewRows,
        DecodeAttentionHeadDim = CpuTuning.DecodeAttentionHeads,
        TiledAttentionHeadDim = CpuTuning.TiledAttentionHeads,
        MatrixUnits = false,
        MatrixUnitAttentionHeadDim = static _ => false,
        FusedKernels = false,
        Profiling = false,
    };

    public override string Kind => "cpu";

    public override string Name => $"CPU ({Environment.ProcessorCount} threads, {System.Numerics.Vector<float>.Count}-wide SIMD)";

    public override string? TensorCoresUnavailable() => "the CPU computes matrix products in float32";

    public override Storage Allocate(int length, bool zeroed)
    {
        long bytes = (long)length * sizeof(float);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes); // throws when over the in-use limit
        float[]? data = null;
        lock (_pool)
        {
            if (_pool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                data = bucket.Pop();
                _memory.Reused(bytes);
            }
        }

        if (data is null)
        {
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }

            data = zeroed ? new float[length] : GC.AllocateUninitializedArray<float>(length);
            _memory.Allocated(bytes);
        }
        else if (zeroed)
        {
            Array.Clear(data);
        }

        return new CpuStorage(this, data, length);
    }

    private protected override void Detach(Storage storage) => ((CpuStorage)storage).Data = null!;

    private protected override void Attach(Storage storage, Storage fresh) => ((CpuStorage)storage).Data = ((CpuStorage)fresh).Data;

    public override void Return(Storage storage)
    {
        var data = ((CpuStorage)storage).Data;
        lock (_pool)
        {
            if (!_pool.TryGetValue(data.Length, out var bucket))
            {
                _pool[data.Length] = bucket = new Stack<float[]>();
            }

            bucket.Push(data);
            _memory.Returned((long)data.Length * sizeof(float));
        }
    }

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    public override void ReleaseCachedMemory()
    {
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                _memory.Freed((long)length * sizeof(float) * bucket.Count);
                bucket.Clear();
            }
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(D(destination));

    public override void Download(Storage source, Span<float> destination) => D(source).AsSpan(0, destination.Length).CopyTo(destination);

    public override void FillKernel(Storage y, int n, float value) => D(y).AsSpan(0, n).Fill(value);

    public override void Copy(Storage x, Storage y, int n) => D(x).AsSpan(0, n).CopyTo(D(y));

    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        switch (op)
        {
            case UnaryOp.Sigmoid: Run(new SigmoidLoop(D(x), D(y)), n); break;
            case UnaryOp.Tanh: Run(new TanhLoop(D(x), D(y)), n); break;
            case UnaryOp.Relu: Run(new ReluLoop(D(x), D(y)), n); break;
            case UnaryOp.Square: Run(new SquareLoop(D(x), D(y)), n); break;
            case UnaryOp.Abs: Run(new AbsLoop(D(x), D(y)), n); break;
            case UnaryOp.Exp: Run(new ExpLoop(D(x), D(y)), n); break;
            case UnaryOp.Log: Run(new LogLoop(D(x), D(y)), n); break;
            case UnaryOp.Gelu: Run(new GeluLoop(D(x), D(y)), n); break;
            case UnaryOp.Sqrt or UnaryOp.Sin or UnaryOp.Cos or UnaryOp.Silu or UnaryOp.Sign: Run(new MathLoop(op, D(x), D(y)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        switch (op)
        {
            case UnaryOp.Sigmoid: Run(new SigmoidBackwardLoop(D(y), D(dy), D(dx)), n); break;
            case UnaryOp.Tanh: Run(new TanhBackwardLoop(D(y), D(dy), D(dx)), n); break;
            case UnaryOp.Relu: Run(new ReluBackwardLoop(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Square: Run(new SquareBackwardLoop(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Abs: Run(new AbsBackwardLoop(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Exp: Run(new MulAddLoop(D(dy), D(y), D(dx)), n); break;
            case UnaryOp.Log: Run(new LogBackwardLoop(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Gelu: Run(new GeluBackwardLoop(D(x), D(dy), D(dx)), n); break;
            case UnaryOp.Sqrt or UnaryOp.Sin or UnaryOp.Cos or UnaryOp.Silu or UnaryOp.Sign: Run(new MathBackwardLoop(op, D(x), D(y), D(dy), D(dx)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        switch (op)
        {
            case BinaryOp.Add: Run(new AddLoop(D(a), D(b), D(c)), n); break;
            case BinaryOp.Sub: Run(new SubLoop(D(a), D(b), D(c)), n); break;
            case BinaryOp.Mul: Run(new MulLoop(D(a), D(b), D(c)), n); break;
            case BinaryOp.Maximum or BinaryOp.Minimum: Run(new ExtremumLoop(op == BinaryOp.Minimum, D(a), D(b), D(c)), n); break;
            default: throw new ArgumentOutOfRangeException(nameof(op));
        }
    }

    public override void AffineKernel(Storage x, Storage y, int n, float alpha, float beta) => Run(new AffineLoop(D(x), D(y), alpha, beta), n);

    public override void AxpyKernel(Storage x, Storage y, int n, float alpha) => Run(new AxpyLoop(D(x), D(y), alpha), n);

    public override void MulAddKernel(Storage a, Storage b, Storage c, int n) => Run(new MulAddLoop(D(a), D(b), D(c)), n);

    public override void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        float[] av = D(a), vv = D(v), cv = D(c);
        var bias = vv.AsSpan(0, cols);
        for (int r = 0; r < rows; r++)
        {
            int o = r * cols;
            AddLoop.Apply(av.AsSpan(o, cols), bias, cv.AsSpan(o, cols));
        }
    }

    public override void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        float[] xv = D(x);
        var acc = D(y).AsSpan(0, cols);
        for (int r = 0; r < rows; r++)
        {
            AddLoop.Apply(acc, xv.AsSpan(r * cols, cols), acc);
        }
    }

    // y[j] += Σ_r x[offset + r·ld + j]. The base method's fallback calls this one, so the CPU must implement it.
    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        float[] xv = D(x);
        var acc = D(y).AsSpan(0, cols);
        for (int r = 0; r < rows; r++)
        {
            AddLoop.Apply(acc, xv.AsSpan((int)(offset + (long)r * ld), cols), acc);
        }
    }

    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        float[] xv = D(x);
        double total;
        if (n < CpuTuning.ReductionElements || !ComputeResources.AllowParallel)
        {
            total = SumSpan(xv.AsSpan(0, n));
        }
        else
        {
            // Chunks from the numerical contract (never a timing), so the sum's order is the same on every run as before;
            // they run on the threads once the measured cut-over says so.
            int chunks = ReductionChunkCount(n);
            int size = ChunkSize(n, chunks);
            var partials = new double[chunks];
            void Chunk(int c)
            {
                int start = c * size;
                partials[c] = SumSpan(xv.AsSpan(start, Math.Max(0, Math.Min(size, n - start))));
            }

            if (CpuTuning.SplitElements(n))
            {
                Parallel.For(0, chunks, ComputeResources.ParallelOptions, Chunk);
            }
            else
            {
                for (int c = 0; c < chunks; c++)
                {
                    Chunk(c);
                }
            }

            total = partials.Sum();
        }

        D(result)[0] = (float)(total * scale);
    }

    public override void AxpyAtKernel(Storage x, Storage y, int offset, float alpha) => D(y)[offset] += alpha * D(x)[0];

    public override void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        float value = D(s)[0] * scale;
        Run(new AffineLoop(D(y), D(y), 1f, value), n);
    }


    public override void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Run(new AxpyLoop(D(g), D(p), -lr), n);
        }
        else
        {
            Run(new SgdMomentumLoop(D(p), D(g), D(v), lr, momentum), n);
        }
    }

    public override void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps) =>
        Run(new AdamLoop(D(p), D(g), D(m), D(v), lr, beta1, beta2, eps), n);

    public override bool FusedAdamWKernel(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        float factor = 1f;
        if (maxNorm > 0f)
        {
            double sum = 0;
            foreach (var t in tensors)
            {
                sum += SumSquaresParallel(D(t.G), t.N);
            }

            factor = sum > 0 ? MathF.Min(1f, maxNorm / MathF.Sqrt((float)sum)) : 1f;
        }

        // SIMD on all cores; each element is computed with the same operations in the same order as the scalar update.
        foreach (var t in tensors)
        {
            Run(new FusedAdamKernel(D(t.P), D(t.G), D(t.M), D(t.V), factor, lr, decay, beta1, beta2, eps, zeroGradients), t.N);
        }

        return true;
    }

    // p = p·decay − lr·m / (√v + ε) with m and v updated from the gradient times the clipping factor (zeroed after when asked).
    private readonly struct FusedAdamKernel(float[] p, float[] g, float[] m, float[] v, float factor, float lr, float decay, float beta1,
        float beta2, float eps, bool zero) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ps = p.AsSpan(start, end - start);
            var gs = g.AsSpan(start, end - start);
            var ms = m.AsSpan(start, end - start);
            var vs = v.AsSpan(start, end - start);
            var pv = MemoryMarshal.Cast<float, Vector<float>>(ps);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var mv = MemoryMarshal.Cast<float, Vector<float>>(ms);
            var vv = MemoryMarshal.Cast<float, Vector<float>>(vs);
            float c1 = 1f - beta1, c2 = 1f - beta2;
            Vector<float> vf = new(factor), b1 = new(beta1), b2 = new(beta2), vc1 = new(c1), vc2 = new(c2), ve = new(eps), vlr = new(lr), vd = new(decay);
            for (int i = 0; i < pv.Length; i++)
            {
                var grad = gv[i] * vf;
                var mom = Vector.FusedMultiplyAdd(b1, mv[i], vc1 * grad);
                var vel = Vector.FusedMultiplyAdd(b2, vv[i], vc2 * grad * grad);
                mv[i] = mom;
                vv[i] = vel;
                pv[i] = pv[i] * vd - vlr * mom / (Vector.SquareRoot(vel) + ve);
            }

            for (int i = pv.Length * Vector<float>.Count; i < ps.Length; i++)
            {
                float grad = gs[i] * factor;
                float mom = MathF.FusedMultiplyAdd(beta1, ms[i], c1 * grad);
                float vel = MathF.FusedMultiplyAdd(beta2, vs[i], c2 * grad * grad);
                ms[i] = mom;
                vs[i] = vel;
                ps[i] = ps[i] * decay - lr * mom / (MathF.Sqrt(vel) + eps);
            }

            if (zero)
            {
                gs.Clear();
            }
        }
    }

    // Σx² of the first n values in double precision: SIMD per chunk, chunks on all cores for large inputs.
    private static double SumSquaresParallel(float[] x, int n)
    {
        if (n < CpuTuning.ReductionElements || !ComputeResources.AllowParallel)
        {
            return SumSquaresSpan(x.AsSpan(0, n));
        }

        // Chunks from the numerical contract, threads from the measured cut-over (as in Sum).
        int chunks = ReductionChunkCount(n);
        int size = ChunkSize(n, chunks);
        var partials = new double[chunks];
        void Chunk(int c)
        {
            int start = c * size;
            partials[c] = SumSquaresSpan(x.AsSpan(start, Math.Max(0, Math.Min(size, n - start))));
        }

        if (CpuTuning.SplitElements(n))
        {
            Parallel.For(0, chunks, ComputeResources.ParallelOptions, Chunk);
        }
        else
        {
            for (int c = 0; c < chunks; c++)
            {
                Chunk(c);
            }
        }

        double total = 0;
        foreach (double partial in partials)
        {
            total += partial;                                              // in chunk order: the same total on every run
        }

        return total;
    }

    private static double SumSquaresSpan(ReadOnlySpan<float> x)
    {
        var lanes = MemoryMarshal.Cast<float, Vector<float>>(x);
        Vector<double> low = Vector<double>.Zero, high = Vector<double>.Zero;
        foreach (var lane in lanes)
        {
            Vector.Widen(lane, out var a, out var b);
            low += a * a;
            high += b * b;
        }

        double sum = Vector.Sum(low) + Vector.Sum(high);
        for (int i = lanes.Length * Vector<float>.Count; i < x.Length; i++)
        {
            sum += (double)x[i] * x[i];
        }

        return sum;
    }

    public override void PackBFloat16Kernel(Storage x, Storage packed, int n)
    {
        float[] values = D(x), words = D(packed);
        var bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(words.AsSpan());
        static uint Round(float v)
        {
            uint u = BitConverter.SingleToUInt32Bits(v);
            return float.IsNaN(v) ? 0x7FC0u : (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
        }

        for (int w = 0; w < (n + 1) / 2; w++)
        {
            uint low = Round(values[2 * w]), high = 2 * w + 1 < n ? Round(values[2 * w + 1]) : 0u;
            bits[w] = low | high << 16;
        }
    }

    public override void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        float sum = D(sumSquares)[0];
        D(factor)[0] = sum > 0f ? MathF.Min(1f, maxNorm / MathF.Sqrt(sum)) : 1f;
    }

    public override void SumSquaresKernel(Storage x, Storage total, int n)
    {
        D(total)[0] += (float)SumSquaresParallel(D(x), n);
    }

    public override void AdamStep8BitKernel(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay)
    {
        float[] ps = D(p), gs = D(g), ms = D(m), vs = D(v), scales = D(absMax), codes = D(map);
        const int B = EightBitMoments.BlockSize;
        int blocks = (n + B - 1) / B;
        Parallel.For(0, blocks, ComputeResources.ParallelOptions, block =>
        {
            var mb = MemoryMarshal.AsBytes(ms.AsSpan());
            var vb = MemoryMarshal.AsBytes(vs.AsSpan());
            var signedMap = codes.AsSpan(0, 256);
            var unsignedMap = codes.AsSpan(256, 256);
            int start = block * B, end = Math.Min(n, start + B);
            Span<float> mNew = stackalloc float[B], vNew = stackalloc float[B];
            float mScale = scales[block], vScale = scales[blocks + block], mMax = 0f, vMax = 0f;
            for (int i = start; i < end; i++)
            {
                float grad = gs[i] * gradientScale;
                float mom = MathF.FusedMultiplyAdd(beta1, signedMap[mb[i]] * mScale, (1f - beta1) * grad);
                float vel = MathF.FusedMultiplyAdd(beta2, unsignedMap[vb[i]] * vScale, grad * grad * (1f - beta2));
                ps[i] = ps[i] * decay - mom / (MathF.Sqrt(vel) + eps) * lr;
                mNew[i - start] = mom;
                vNew[i - start] = vel;
                mMax = MathF.Max(mMax, MathF.Abs(mom));
                vMax = MathF.Max(vMax, vel);
            }

            for (int i = start; i < end; i++)
            {
                mb[i] = EightBitMoments.Nearest(signedMap, mMax > 0f ? mNew[i - start] / mMax : 0f);
                vb[i] = EightBitMoments.Nearest(unsignedMap, vMax > 0f ? vNew[i - start] / vMax : 0f);
            }

            scales[block] = mMax;
            scales[blocks + block] = vMax;
        });
    }

    public override void DropoutKernel(Storage x, Storage y, int n, float p, uint seed) => Run(new DropoutLoop(D(x), D(y), p, seed, accumulate: false), n);

    public override void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed) => Run(new DropoutLoop(D(dy), D(dx), p, seed, accumulate: true), n);

    public override void Synchronize()
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[] D(Storage s) => ((CpuStorage)s).Data;

    private static double SumSpan(ReadOnlySpan<float> x)
    {
        var acc = Vector<float>.Zero;
        int i = 0;
        var xv = MemoryMarshal.Cast<float, Vector<float>>(x);
        foreach (var v in xv)
        {
            acc += v;
        }

        i = xv.Length * Vector<float>.Count;
        double total = Vector.Sum(acc);
        for (; i < x.Length; i++)
        {
            total += x[i];
        }

        return total;
    }

    // About two chunks per thread, each at least a quarter of the cut-over (a chunk smaller than that costs more to start
    // than to run). Reductions take the contract's value (their chunks set the summation order), the rest the measured one.
    private static int ChunkCount(int n, int cutover) => Math.Max(1, Math.Min(ComputeResources.MaxCpuThreads * 2, n / Math.Max(1, cutover / 4)));

    private static int ReductionChunkCount(int n) => ChunkCount(n, CpuTuning.ReductionElements);

    private static int ChunkSize(int n, int chunks)
    {
        int size = (n + chunks - 1) / chunks;
        int w = Vector<float>.Count;
        return (size + w - 1) / w * w;
    }

    /// <summary>Runs a range kernel inline for small inputs and in parallel chunks for large ones.</summary>
    internal static void Run<TKernel>(TKernel kernel, int n)
        where TKernel : struct, IRangeKernel
    {
        if (!CpuTuning.SplitElements(n))
        {
            kernel.Execute(0, n);
            return;
        }

        int chunks = ChunkCount(n, ParallelThreshold);
        int size = ChunkSize(n, chunks);
        Parallel.For(0, chunks, ComputeResources.ParallelOptions, c =>
        {
            int start = c * size;
            kernel.Execute(start, Math.Min(start + size, n));
        });
    }

    internal interface IRangeKernel
    {
        void Execute(int start, int end);
    }

    // Each kernel processes [start, end): a SIMD main loop over Vector<float> lanes, then a scalar tail.

    private readonly struct SigmoidLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var one = Vector<float>.One;
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = one / (one + Vector.Exp(-xv[i]));
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = 1f / (1f + MathF.Exp(-xs[i]));
            }
        }
    }

    private readonly struct TanhLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var one = Vector<float>.One;
            var two = new Vector<float>(2f);
            for (int i = 0; i < xv.Length; i++)
            {
                // tanh(x) = 1 - 2 / (e^(2x) + 1); saturates cleanly to +-1 when e^(2x) overflows or underflows.
                yv[i] = one - two / (Vector.Exp(xv[i] * two) + one);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Tanh(xs[i]);
            }
        }
    }

    private readonly struct ReluLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Max(xv[i], Vector<float>.Zero);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Max(xs[i], 0f);
            }
        }
    }

    private readonly struct SquareLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end) => MulLoop.Apply(x.AsSpan(start, end - start), x.AsSpan(start, end - start), y.AsSpan(start, end - start));
    }

    private readonly struct AbsLoop(float[] x, float[] y) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.Abs(xv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.Abs(xs[i]);
            }
        }
    }

    private readonly struct DropoutLoop(float[] x, float[] y, float p, uint seed, bool accumulate) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            float scale = 1f / (1f - p);
            int w = Vector<uint>.Count;
            int i = start;
            if (end - start >= w)
            {
                // Vectorized DropoutMask.Keep: the same MurmurHash3 finalizer on w consecutive indices at once.
                Span<uint> lanes = stackalloc uint[w];
                for (int l = 0; l < w; l++)
                {
                    lanes[l] = (uint)l;
                }

                var laneOffsets = new Vector<uint>(lanes);
                var vseed = new Vector<uint>(seed);
                var vp = new Vector<float>(p);
                var vscale = new Vector<float>(scale);
                var inv24 = new Vector<float>(1f / 16777216f);
                for (; i + w <= end; i += w)
                {
                    var h = (new Vector<uint>((uint)i) + laneOffsets) * new Vector<uint>(0x9E3779B9u) ^ vseed;
                    h ^= Vector.ShiftRightLogical(h, 16);
                    h *= new Vector<uint>(0x85EBCA6Bu);
                    h ^= Vector.ShiftRightLogical(h, 13);
                    h *= new Vector<uint>(0xC2B2AE35u);
                    h ^= Vector.ShiftRightLogical(h, 16);
                    var u = Vector.ConvertToSingle(Vector.ShiftRightLogical(h, 8)) * inv24;
                    var keep = Vector.GreaterThanOrEqual(u, vp);
                    var xv = Vector.LoadUnsafe(ref x[i]);
                    var v = Vector.ConditionalSelect(keep, xv * vscale, Vector<float>.Zero);
                    ref float target = ref y[i];
                    (accumulate ? Vector.LoadUnsafe(ref target) + v : v).StoreUnsafe(ref target);
                }
            }

            for (; i < end; i++)
            {
                float v = DropoutMask.Keep(seed, (uint)i, p) ? x[i] * scale : 0f;
                y[i] = accumulate ? y[i] + v : v;
            }
        }
    }

    private readonly struct AbsBackwardLoop(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            for (int i = 0; i < xv.Length; i++)
            {
                var positive = Vector.ConditionalSelect(Vector.GreaterThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
                var negative = Vector.ConditionalSelect(Vector.LessThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
                dv[i] += positive - negative;
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ds[i] += xs[i] > 0f ? gs[i] : xs[i] < 0f ? -gs[i] : 0f;
            }
        }
    }

    private readonly struct SigmoidBackwardLoop(float[] y, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ys = y.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var one = Vector<float>.One;
            for (int i = 0; i < yv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(gv[i], yv[i] * (one - yv[i]), dv[i]);
            }

            for (int i = yv.Length * Vector<float>.Count; i < ys.Length; i++)
            {
                ds[i] += gs[i] * ys[i] * (1f - ys[i]);
            }
        }
    }

    private readonly struct TanhBackwardLoop(float[] y, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ys = y.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var one = Vector<float>.One;
            for (int i = 0; i < yv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(gv[i], one - yv[i] * yv[i], dv[i]);
            }

            for (int i = yv.Length * Vector<float>.Count; i < ys.Length; i++)
            {
                ds[i] += gs[i] * (1f - ys[i] * ys[i]);
            }
        }
    }

    private readonly struct ReluBackwardLoop(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            for (int i = 0; i < xv.Length; i++)
            {
                dv[i] += Vector.ConditionalSelect(Vector.GreaterThan(xv[i], Vector<float>.Zero), gv[i], Vector<float>.Zero);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                if (xs[i] > 0f)
                {
                    ds[i] += gs[i];
                }
            }
        }
    }

    private readonly struct SquareBackwardLoop(float[] x, float[] dy, float[] dx) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var gs = dy.AsSpan(start, end - start);
            var ds = dx.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var dv = MemoryMarshal.Cast<float, Vector<float>>(ds);
            var two = new Vector<float>(2f);
            for (int i = 0; i < xv.Length; i++)
            {
                dv[i] = Vector.FusedMultiplyAdd(two * xv[i], gv[i], dv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ds[i] += 2f * xs[i] * gs[i];
            }
        }
    }

    private readonly struct AddLoop(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end) => Apply(a.AsSpan(start, end - start), b.AsSpan(start, end - start), c.AsSpan(start, end - start));

        public static void Apply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
        {
            var av = MemoryMarshal.Cast<float, Vector<float>>(a);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(c);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] + bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < a.Length; i++)
            {
                c[i] = a[i] + b[i];
            }
        }
    }

    private readonly struct SubLoop(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var @as = a.AsSpan(start, end - start);
            var bs = b.AsSpan(start, end - start);
            var cs = c.AsSpan(start, end - start);
            var av = MemoryMarshal.Cast<float, Vector<float>>(@as);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(bs);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(cs);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] - bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < @as.Length; i++)
            {
                cs[i] = @as[i] - bs[i];
            }
        }
    }

    private readonly struct MulLoop(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end) => Apply(a.AsSpan(start, end - start), b.AsSpan(start, end - start), c.AsSpan(start, end - start));

        public static void Apply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c)
        {
            var av = MemoryMarshal.Cast<float, Vector<float>>(a);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(b);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(c);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = av[i] * bv[i];
            }

            for (int i = av.Length * Vector<float>.Count; i < a.Length; i++)
            {
                c[i] = a[i] * b[i];
            }
        }
    }

    private readonly struct AffineLoop(float[] x, float[] y, float alpha, float beta) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var va = new Vector<float>(alpha);
            var vb = new Vector<float>(beta);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.FusedMultiplyAdd(xv[i], va, vb);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.FusedMultiplyAdd(xs[i], alpha, beta);
            }
        }
    }

    internal readonly struct AxpyLoop(float[] x, float[] y, float alpha) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var xs = x.AsSpan(start, end - start);
            var ys = y.AsSpan(start, end - start);
            var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
            var va = new Vector<float>(alpha);
            for (int i = 0; i < xv.Length; i++)
            {
                yv[i] = Vector.FusedMultiplyAdd(xv[i], va, yv[i]);
            }

            for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)
            {
                ys[i] = MathF.FusedMultiplyAdd(xs[i], alpha, ys[i]);
            }
        }
    }

    internal readonly struct MulAddLoop(float[] a, float[] b, float[] c) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var @as = a.AsSpan(start, end - start);
            var bs = b.AsSpan(start, end - start);
            var cs = c.AsSpan(start, end - start);
            var av = MemoryMarshal.Cast<float, Vector<float>>(@as);
            var bv = MemoryMarshal.Cast<float, Vector<float>>(bs);
            var cv = MemoryMarshal.Cast<float, Vector<float>>(cs);
            for (int i = 0; i < av.Length; i++)
            {
                cv[i] = Vector.FusedMultiplyAdd(av[i], bv[i], cv[i]);
            }

            for (int i = av.Length * Vector<float>.Count; i < @as.Length; i++)
            {
                cs[i] = MathF.FusedMultiplyAdd(@as[i], bs[i], cs[i]);
            }
        }
    }

    private readonly struct SgdMomentumLoop(float[] p, float[] g, float[] v, float lr, float momentum) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ps = p.AsSpan(start, end - start);
            var gs = g.AsSpan(start, end - start);
            var vs = v.AsSpan(start, end - start);
            var pv = MemoryMarshal.Cast<float, Vector<float>>(ps);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var vv = MemoryMarshal.Cast<float, Vector<float>>(vs);
            var mu = new Vector<float>(momentum);
            var nlr = new Vector<float>(-lr);
            for (int i = 0; i < pv.Length; i++)
            {
                var vel = Vector.FusedMultiplyAdd(mu, vv[i], gv[i]);
                vv[i] = vel;
                pv[i] = Vector.FusedMultiplyAdd(nlr, vel, pv[i]);
            }

            for (int i = pv.Length * Vector<float>.Count; i < ps.Length; i++)
            {
                float vel = momentum * vs[i] + gs[i];
                vs[i] = vel;
                ps[i] -= lr * vel;
            }
        }
    }

    private readonly struct AdamLoop(float[] p, float[] g, float[] m, float[] v, float lr, float beta1, float beta2, float eps) : IRangeKernel
    {
        public void Execute(int start, int end)
        {
            var ps = p.AsSpan(start, end - start);
            var gs = g.AsSpan(start, end - start);
            var ms = m.AsSpan(start, end - start);
            var vs = v.AsSpan(start, end - start);
            var pv = MemoryMarshal.Cast<float, Vector<float>>(ps);
            var gv = MemoryMarshal.Cast<float, Vector<float>>(gs);
            var mv = MemoryMarshal.Cast<float, Vector<float>>(ms);
            var vv = MemoryMarshal.Cast<float, Vector<float>>(vs);
            var b1 = new Vector<float>(beta1);
            var b2 = new Vector<float>(beta2);
            var c1 = new Vector<float>(1f - beta1);
            var c2 = new Vector<float>(1f - beta2);
            var ve = new Vector<float>(eps);
            var vlr = new Vector<float>(lr);
            for (int i = 0; i < pv.Length; i++)
            {
                var grad = gv[i];
                var mom = Vector.FusedMultiplyAdd(b1, mv[i], c1 * grad);
                var vel = Vector.FusedMultiplyAdd(b2, vv[i], c2 * grad * grad);
                mv[i] = mom;
                vv[i] = vel;
                pv[i] -= vlr * mom / (Vector.SquareRoot(vel) + ve);
            }

            for (int i = pv.Length * Vector<float>.Count; i < ps.Length; i++)
            {
                float grad = gs[i];
                float mom = beta1 * ms[i] + (1f - beta1) * grad;
                float vel = beta2 * vs[i] + (1f - beta2) * grad * grad;
                ms[i] = mom;
                vs[i] = vel;
                ps[i] -= lr * mom / (MathF.Sqrt(vel) + eps);
            }
        }
    }
}
