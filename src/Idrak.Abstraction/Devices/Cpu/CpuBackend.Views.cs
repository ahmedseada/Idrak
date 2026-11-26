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
    }

    public override void AttentionTiledStridedKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values, TensorLayout valuesLayout,
        Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        Storage eq = ViewElements(q, qLayout), ek = ViewElements(keys, keysLayout), ev = ViewElements(values, valuesLayout);
        try
        {
            AttentionTiledKernel(eq, ek, ev, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
        finally
        {
            Done(eq, q);
            Done(ek, keys);
            Done(ev, values);
        }
    }

    public override void AttentionTiledStridedBackwardKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values,
        TensorLayout valuesLayout, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        Storage eq = ViewElements(q, qLayout), ek = ViewElements(keys, keysLayout), ev = ViewElements(values, valuesLayout);
        try
        {
            AttentionTiledBackwardKernel(eq, ek, ev, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, rowsPerHead, steps, capacity, dim, scale, variant);
        }
        finally
        {
            Done(eq, q);
            Done(ek, keys);
            Done(ev, values);
        }
    }

    public override void KeyValueWriteStridedKernel(Storage source, TensorLayout sourceLayout, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        var es = ViewElements(source, sourceLayout);
        try
        {
            KeyValueWriteKernel(es, cache, position, heads, steps, capacity, dim);
        }
        finally
        {
            Done(es, source);
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
                fixed (float* pg = V(t.G))
                {
                    total += ChunkedSum(new TypedInput(t.G, pg, 0, SumChunk, "FusedAdamW"), t.N, squares: true);
                }
            }

            factor = total > 0f ? MathF.Min(1f, maxNorm / MathF.Sqrt(total)) : 1f;
        }

        foreach (var t in tensors)
        {
            fixed (float* pp = V(t.P), pg = V(t.G), pm = V(t.M), pv = V(t.V))
            {
                Elements("FusedAdamW", new AdamWMap(pp, pg, pm, pv, lr, decay, beta1, beta2, eps, factor, zeroGradients, t.N), t.N, 8);
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
