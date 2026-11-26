// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// Weights stored as their file lays them out on the CPU: the product by the described lane (CpuBackend.Lanes.cs), which
// reads any type from its description alone and writes no weight's value anywhere.
internal sealed partial class CpuBackend
{
    // x [m, k] times its columns' scales into `into`: value (i, c) times scales[c / block], a row at a time, the scales of a
    // run of `block` columns as one vector, over the threads where x is large.
    private static void ScaleColumns(float[] x, float[] scales, float[] into, int m, int k, int block)
    {
        void Row(int i)
        {
            var row = x.AsSpan(i * k, k);
            var to = into.AsSpan(i * k, k);
            for (int c = 0; c < k; c += block)
            {
                int count = Math.Min(block, k - c), j = 0;
                float s = scales[c / block];
                var from = row.Slice(c, count);
                var into1 = to.Slice(c, count);
                var sv = System.Numerics.Vector<float>.One * s;
                for (; j <= count - System.Numerics.Vector<float>.Count; j += System.Numerics.Vector<float>.Count)
                {
                    (new System.Numerics.Vector<float>(from[j..]) * sv).CopyTo(into1[j..]);
                }

                for (; j < count; j++)
                {
                    into1[j] = from[j] * s;
                }
            }
        }

        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, Row);
            return;
        }

        for (int i = 0; i < m; i++)
        {
            Row(i);
        }
    }

    public override void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            D(y).AsSpan(0, m * n).Clear();
            return;
        }

        // Scales on x's columns (StoredScales.OnColumns: one block of rows over every output, each scale a column or a run of
        // columns all outputs share): x times them once a call, then the product with no scales, so no weight's value is
        // scaled anywhere (m · k multiplies, not n · k).
        if (zeros is null && scales is not null && layout.OnColumns(n, k))
        {
            var scaled = Allocate(m * k, zeroed: false);
            try
            {
                ScaleColumns(D(x), D(scales), D(scaled), m, k, layout.BlockColumns);
                if (StoredRouted(scaled, weight, null, null, y, m, n, k, default))
                {
                    return;
                }
            }
            finally
            {
                scaled.Release();
            }
        }

        if (StoredRouted(x, weight, scales, zeros, y, m, n, k, layout))
        {
            return;
        }

        throw new NotSupportedException(weight.Description is null
            ? "The CPU multiplies stored weights by their type's description; this weight's storage has none."
            : $"The CPU's described lane does not read this weight: rows of {k} values of its description, its scales laid out as {layout}{(zeros is null ? "" : ", with zero points")}{(weight is MappedStorage ? ", read from a file's mapping" : "")}.");
    }
}
