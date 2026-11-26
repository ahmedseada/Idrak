// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices.Cpu;

// The operations that take views (plan 37, V1b) and the fused AdamW pass. A view operand is read through its layout into
// a pooled block of its own elements (the same values in row-major order: the operation's bits on the copies made with
// views off), the block given back when the operation ends; a contiguous operand from its first element is read in place.
internal sealed unsafe partial class CpuBackend
{
    // The operations whose inputs the CPU reads as views (BackendCapabilities.StridedInputs).
    private static HashSet<Operation> StridedInputSet() =>
    [
        Ops.StridedCopy, Ops.BatchedMatMul, Ops.BatchedMatMulStrided, Ops.AttentionTiledStrided, Ops.AttentionTiledStridedBackward, Ops.KeyValueWriteStrided,
    ];

    // The attention passes that write their rows token-major (AttentionVariant.Merged): every causal pass here.
    public override bool MergesAttention(Operation operation, int dim, AttentionVariant variant) =>
        operation == Ops.AttentionTiled || operation == Ops.AttentionDecode || operation == Ops.AttentionTiledStrided;

    // The operand's elements in its layout's row-major order: itself where it is contiguous from its first element, else
    // a pooled block of them (released by the caller when it differs from the operand).
    private Storage ViewElements(Storage s, TensorLayout layout)
    {
        if (layout.IsContiguousFromStart)
        {
            return s;
        }

        var block = Allocate(checked((int)layout.Elements), zeroed: false);
        StridedCopyKernel(s, layout.Offset, layout.Shape, layout.Strides, block);
        return block;
    }

    private static void Done(Storage block, Storage operand)
    {
        if (!ReferenceEquals(block, operand))
        {
            block.Release();
        }
    }

    // The layout as `batch` planes of `rows` rows of `cols` values with its last dimension's step 1: the steps between
    // planes and between rows (dimensions of 1 left out, neighbours that run on merged, a merged run split back where the
    // sizes divide it); false where it is not so.
    private static bool Planes(TensorLayout layout, int batch, int rows, int cols, out long planeStep, out long rowStep)
    {
        planeStep = rowStep = 0;
        var shape = layout.Shape;
        var strides = layout.Strides;
        Span<long> size = stackalloc long[shape.Length + 1], step = stackalloc long[shape.Length + 1];
        int count = 0;
        for (int d = 0; d < shape.Length; d++)
        {
            if (shape[d] == 1)
            {
                continue;
            }

            if (count > 0 && step[count - 1] == strides[d] * shape[d])
            {
                size[count - 1] *= shape[d];
                step[count - 1] = strides[d];
            }
            else
            {
                size[count] = shape[d];
                step[count] = strides[d];
                count++;
            }
        }

        // Peels `length` values of step `unit` (any step where unit < 0) off the innermost run.
        bool Peel(long length, long unit, out long taken, Span<long> sizes, Span<long> steps, ref int n)
        {
            taken = 0;
            if (length == 1)
            {
                return true;
            }

            if (n == 0 || sizes[n - 1] % length != 0 || unit >= 0 && steps[n - 1] != unit)
            {
                return false;
            }

            taken = steps[n - 1];
            if (sizes[n - 1] == length)
            {
                n--;
            }
            else
            {
                sizes[n - 1] /= length;
                steps[n - 1] *= length;
            }

            return true;
        }

        if (!Peel(cols, 1, out _, size, step, ref count) || !Peel(rows, -1, out rowStep, size, step, ref count) || rows > 1 && rowStep <= 0)
        {
            return false;
        }

        if (batch > 1)
        {
            if (count != 1 || size[0] != batch)
            {
                return false;
            }

            planeStep = step[0];
            return true;
        }

        return count == 0;
    }

    public override void BatchedMatMulStridedKernel(Storage a, TensorLayout aLayout, Storage b, TensorLayout bLayout, Storage c, int batch, int m, int n, int k,
        bool transA, bool transB, float beta)
    {
        // Planes the product reads where they lie (a matrix step and a row step each): no copy.
        if (NativeFloats(a) && NativeFloats(b) && Planes(aLayout, batch, transA ? k : m, transA ? m : k, out long sa, out long la)
            && Planes(bLayout, batch, transB ? n : k, transB ? k : n, out long sb, out long lb) && batch > 0 && m > 0 && n > 0)
        {
            fixed (float* pa = D(a), pb = D(b), pc = V(c))
            {
                Product(pa + aLayout.Offset, pb + bLayout.Offset, null, pc, batch, m, n, k, transA, transB, beta, la, null, 0, batch > 1 ? sa : -1, batch > 1 ? sb : -1, lb);
            }

            return;
        }

        if (Wide(a, b) || batch <= 0 || m <= 0 || n <= 0 || k <= 0)
        {
            // Values float32 cannot hold (the float64 product reads its operands as their own elements), or nothing to sum.
            Storage ea = ViewElements(a, aLayout), eb = ViewElements(b, bLayout);
            try
            {
                BatchedMatMulKernel(ea, eb, c, batch, m, n, k, transA, transB, beta);
            }
            finally
            {
                Done(ea, a);
                Done(eb, b);
            }

            return;
        }

        GatheredProduct(a, aLayout, b, bLayout, c, batch, m, n, k, transA, transB, beta);
    }

    // The product of views of any strides and types, never a copy of either: op(B) a block of k rows at a time and op(A) a
    // block of rows of that block's columns, each gathered from where its values lie (GatherView) into the thread's working
    // rows, the product over each block with β for the first block of k and 1 after it. Each output takes its terms in k's
    // order from β · c (a fused multiply-add each, the blocks one after another), so its bits are those of the product over
    // the whole operands.
    private static void GatheredProduct(Storage a, TensorLayout aLayout, Storage b, TensorLayout bLayout, Storage c, int batch, int m, int n, int k, bool transA,
        bool transB, float beta)
    {
        fixed (float* wa = D(a), wb = D(b), pc = V(c))
        {
            TypedRanges ra = new(a, 0, "BatchedMatMulStrided"), rb = new(b, 1, "BatchedMatMulStrided");
            int kc = (int)Math.Clamp(ChunkValues / Math.Max(1, n), 1, k);
            int rows = (int)Math.Clamp(ChunkValues / kc / TileRows * TileRows, TileRows, Math.Max(TileRows, m));
            float* blockB = WorkRow(0, (long)kc * n), blockA = WorkRow(1, (long)rows * kc);
            for (int matrix = 0; matrix < batch; matrix++)
            {
                long aBase = (long)matrix * m * k, bBase = (long)matrix * k * n;
                for (int k0 = 0; k0 < k; k0 += kc)
                {
                    int depth = Math.Min(kc, k - k0);
                    // op(B)'s rows [k0, k0 + depth), packed [depth, n] (B transposed: [n, depth], read transposed).
                    if (transB)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            GatherView(rb, wb, bLayout, bBase + (long)j * k + k0, depth, blockB + (long)j * depth);
                        }
                    }
                    else
                    {
                        GatherView(rb, wb, bLayout, bBase + (long)k0 * n, (long)depth * n, blockB);
                    }

                    for (int i0 = 0; i0 < m; i0 += rows)
                    {
                        int taken = Math.Min(rows, m - i0);
                        // op(A)'s rows [i0, i0 + taken) of the block's columns, packed [taken, depth] (A transposed: [depth, taken]).
                        if (transA)
                        {
                            for (int kk = 0; kk < depth; kk++)
                            {
                                GatherView(ra, wa, aLayout, aBase + (long)(k0 + kk) * m + i0, taken, blockA + (long)kk * taken);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < taken; i++)
                            {
                                GatherView(ra, wa, aLayout, aBase + (long)(i0 + i) * k + k0, depth, blockA + (long)i * depth);
                            }
                        }

                        Product(blockA, blockB, null, pc + ((long)matrix * m + i0) * n, 1, taken, n, depth, transA, transB, k0 == 0 ? beta : 1f,
                            transA ? taken : depth, ldb: transB ? depth : n);
                    }
                }
            }
        }
    }

    public override void AttentionTiledStridedKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values, TensorLayout valuesLayout,
        Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        // Heads whose rows lie as planes (a step between heads and between rows, the last step 1): read where they lie, a
        // typed operand's rows through its type (RowsIn).
        if (Planes(qLayout, heads, rowsPerHead, dim, out long qh, out long qr)
            && Planes(keysLayout, heads, capacity, dim, out long kh, out long kr) && Planes(valuesLayout, heads, capacity, dim, out long vh, out long vr))
        {
            var at = new HeadSteps(heads > 1 ? qh : 0, rowsPerHead > 1 ? qr : dim, heads > 1 ? kh : 0, capacity > 1 ? kr : dim, heads > 1 ? vh : 0, capacity > 1 ? vr : dim);
            Causal(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant, "AttentionTiled",
                (qLayout.Offset, keysLayout.Offset, valuesLayout.Offset, at));
            return;
        }

        // Any other layout: each row gathered from where its values lie (RowsIn's view mode), never a copy of the view.
        Causal(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant, "AttentionTiled", null, (qLayout, keysLayout, valuesLayout));
    }

    public override void AttentionTiledStridedBackwardKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values,
        TensorLayout valuesLayout, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default) =>
        TiledBackward(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, rowsPerHead, steps, capacity, dim, scale, variant, qLayout, keysLayout,
            valuesLayout);

    // A source view's rows (its elements in their row-major order, [heads, steps, dim]) gathered from where they lie a chunk
    // at a time and written as KeyValueWrite writes them: never a copy of the whole view.
    public override void KeyValueWriteStridedKernel(Storage source, TensorLayout sourceLayout, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (sourceLayout.IsContiguousFromStart)
        {
            KeyValueWriteKernel(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        long start;
        fixed (float* pp = D(position))
        {
            start = IndicesOf(position, pp)[0];
        }

        long count = Math.Clamp(capacity - start, 0, steps) * (long)dim;
        if (count <= 0)
        {
            return;
        }

        fixed (float* ws = D(source), wc = D(cache))
        {
            var reader = new TypedRanges(source, 0, "KeyValueWriteStrided");
            var cacheType = NativeFloats(cache) ? null : cache.Type;
            long unit = cacheType is null ? 1 : Math.Max(1, cacheType.Description.UnitValues), piece = Math.Max(unit, ChunkValues / unit * unit);
            float* gathered = WorkRow(0, piece);
            for (int h = 0; h < heads; h++)
            {
                long from = (long)h * steps * dim, to = (h * (long)capacity + start) * dim;
                if (cacheType is not null && (to % unit != 0 || count % unit != 0))
                {
                    throw new NotSupportedException($"The CPU's KeyValueWrite writes a cache of {cacheType.Name} whole groups of {unit} values: rows of {dim} from row {start} are not.");
                }

                for (long c = 0; c < count; c += piece)
                {
                    int length = (int)Math.Min(piece, count - c);
                    GatherView(reader, ws, sourceLayout, from + c, length, gathered);
                    var values = new ReadOnlySpan<float>(gathered, length);
                    if (cacheType is null)
                    {
                        values.CopyTo(new Span<float>(wc + to + c, length));
                    }
                    else
                    {
                        cacheType.Write(values, new Span<byte>((byte*)wc + cacheType.ByteCount(to + c), (int)cacheType.ByteCount(length)));
                    }
                }
            }
        }
    }

    public override bool FusedAdamWKernel(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        // The clipping factor of the gradients' global norm: each tensor's Σ g² (SumSquares' chunks), added in order.
        float factor = 1f;
        if (maxNorm > 0f)
        {
            float total = 0f;
            foreach (var t in tensors)
            {
                fixed (float* pg = D(t.G))
                {
                    total += ChunkedSum(new TypedInput(t.G, pg, 0, SumChunk, "FusedAdamW"), t.N, squares: true);
                }
            }

            factor = total > 0f ? MathF.Min(1f, maxNorm / MathF.Sqrt(total)) : 1f;
        }

        // Typed parameters, gradients (zeroed in place where asked) and moments read and written back through their types a
        // chunk at a time (TypedInOut); float32's own in one run.
        foreach (var t in tensors)
        {
            fixed (float* wp = D(t.P), wg = D(t.G), wm = D(t.M), wv = D(t.V))
            {
                TypedInOut tp = new(t.P, wp, 0, ChunkValues, "FusedAdamW"), tg = new(t.G, wg, 1, ChunkValues, "FusedAdamW"), tm = new(t.M, wm, 2, ChunkValues, "FusedAdamW"),
                    tv = new(t.V, wv, 3, ChunkValues, "FusedAdamW");
                long step = tp.Typed || tg.Typed || tm.Typed || tv.Typed
                    ? Math.Min(Math.Min(tp.Piece(ChunkValues), tg.Piece(ChunkValues)), Math.Min(tm.Piece(ChunkValues), tv.Piece(ChunkValues)))
                    : Math.Max(1, t.N);
                for (long i = 0; i < t.N; i += step)
                {
                    long e = Math.Min(t.N, i + step);
                    Elements("FusedAdamW", new AdamWMap(tp.Range(i, e - i), tg.Range(i, e - i), tm.Range(i, e - i), tv.Range(i, e - i), lr, decay, beta1, beta2, eps, factor,
                        zeroGradients, t.N), e, 8, i);
                    tp.Store(i, e - i);
                    if (zeroGradients)
                    {
                        tg.Store(i, e - i);
                    }

                    tm.Store(i, e - i);
                    tv.Store(i, e - i);
                }
            }
        }

        return true;
    }

    // g' = factor · g; m, v as AdamStep; p = p · decay − lr · m / (√v + ε); g zeroed where asked.
    private struct AdamWMap(float* p, float* g, float* m, float* v, float lr, float decay, float beta1, float beta2, float eps, float factor, bool zero, long n) : IMap
    {
        private float* _p = p, _g = g, _m = m, _v = v;

        public readonly void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            var grad = TF.Mul(TF.Load(_g + i), TF.All(factor));
            var mean = TF.Fma(TF.All(beta1), TF.Load(_m + i), TF.Mul(TF.All(1f - beta1), grad));
            var square = TF.Fma(TF.All(beta2), TF.Load(_v + i), TF.Mul(TF.All(1f - beta2), TF.Mul(grad, grad)));
            TF.Store(_m + i, mean);
            TF.Store(_v + i, square);
            var step = TF.Div(TF.Mul(TF.All(lr), mean), TF.Add(TF.Sqrt(square), TF.All(eps)));
            TF.Store(_p + i, TF.Sub(TF.Mul(TF.Load(_p + i), TF.All(decay)), step));
            if (zero)
            {
                TF.Store(_g + i, TF.All(0f));
            }
        }

        public void Redirect(ref Scratch scratch)
        {
            // Every operand is written: measured on copies of them.
            _p = Copied(ref scratch, _p, n);
            _g = Copied(ref scratch, _g, n);
            _m = Copied(ref scratch, _m, n);
            _v = Copied(ref scratch, _v, n);
        }
    }

    // A scratch copy of n floats.
    private static float* Copied(ref Scratch scratch, float* from, long n)
    {
        float* to = scratch.Take(n);
        new ReadOnlySpan<float>(from, (int)n).CopyTo(new Span<float>(to, (int)n));
        return to;
    }
}
