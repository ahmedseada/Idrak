// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Backends;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Backends.Vulkan;
using Idrak.Layers;

// Matrix units on Vulkan (cooperative matrices, VulkanBackend.Matrix.cs): the shape is chosen from what a device reports,
// a device without them runs exactly the kernels it ran before, and the cooperative-matrix products match the CPU within
// float32 error, element by element relative to the sum of the magnitudes it adds (so rows and columns of very
// different sizes are each checked). On a device with cooperative matrices the products run on them; on one without
// (a CPU driver), they run emulated (everything but the matrix operations as on such a device) in several shapes and
// subgroup counts, wherever the device has 16-bit floats in shaders.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanMatrixGroup =
    [
        ("vulkan matrix units: the cooperative-matrix shape comes from what the device reports (float16 operands, float32 sums, subgroup scope, a shape the kernels tile; the most multiply-adds per operation)", VulkanMatrixShapeChoice),
        ("vulkan matrix units: without cooperative matrices nothing changes (no matrix kernel is a candidate or runs; asking for one runs the float32 kernels, matching the CPU)", VulkanMatrixAbsent),
        ("vulkan matrix units: the cooperative-matrix products match the CPU within float32 error (float32 every transpose, batches, beta; int8, int4, bfloat16 prompts; partial blocks; rows and columns from 1e-12 to 1e12 and past 65504; the device's matrices, or emulated in several shapes and subgroup counts)", VulkanMatrixProducts),
    ];

    private static VkCooperativeMatrixProperties MatrixProperties(uint m, uint n, uint k, uint a = 0, uint b = 0, uint c = 1, uint result = 1, uint scope = 3) =>
        new() { MSize = m, NSize = n, KSize = k, AType = a, BType = b, CType = c, ResultType = result, Scope = scope };

    private static void VulkanMatrixShapeChoice(Device device)
    {
        _ = device;
        // A list of the kind one device reports (16-bit and 8-bit types, several shapes): the float16 → float32 shape with the most work.
        var chosen = VulkanBackend.ChooseMatrixShape([MatrixProperties(16, 8, 8), MatrixProperties(16, 8, 16), MatrixProperties(16, 16, 16, a: 3, b: 3, c: 5, result: 5),
            MatrixProperties(16, 16, 16), MatrixProperties(16, 16, 16, c: 0, result: 0)]);
        Check(chosen == new VulkanKernels.CoopShape(16, 16, 16), $"chose {chosen}");
        // Another kind (one narrow shape): taken as reported.
        Check(VulkanBackend.ChooseMatrixShape([MatrixProperties(8, 16, 16)]) == new VulkanKernels.CoopShape(8, 16, 16), "8 × 16 × 16");
        // Ties go to the first reported.
        Check(VulkanBackend.ChooseMatrixShape([MatrixProperties(16, 8, 16), MatrixProperties(8, 16, 16)]) == new VulkanKernels.CoopShape(16, 8, 16), "tie");
        // Nothing usable: other scopes, float16 sums only, integers only, shapes the kernels do not tile.
        Check(VulkanBackend.ChooseMatrixShape([MatrixProperties(16, 16, 16, scope: 2), MatrixProperties(16, 16, 16, c: 0, result: 0),
            MatrixProperties(16, 16, 32, a: 3, b: 3, c: 5, result: 5), MatrixProperties(16, 16, 4), MatrixProperties(64, 64, 16)]) is null, "nothing usable");
        Check(VulkanBackend.ChooseMatrixShape([]) is null, "empty list");
    }

    private static void VulkanMatrixAbsent(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        var own = (VulkanBackend)device.Backend;
        Console.WriteLine($"    ({device}: {own.DescribeMatrixUnits()})");
        var saved = VulkanBackend.MatrixUnitsOverride;
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-matrix-absent-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        (VulkanBackend.MatrixUnitsOverride, VulkanBackend.TuningCacheFile) = (false, cache);
        try
        {
            WithBackend(device.Ordinal, mapped: true, backend =>
            {
                Check(backend.MatrixShape is null && backend.DescribeMatrixUnits() == "no cooperative matrices", backend.DescribeMatrixUnits());
                foreach (int? kernel in new int?[] { 3, null })
                {
                    VulkanBackend.MatMulKernel = kernel;
                    VulkanBackend.PackedPromptKernel = kernel;
                    MatrixProductCases(backend, $"without cooperative matrices, kernel {kernel?.ToString() ?? "measured"}", full: false, expectMatrix: false);
                }
            });
        }
        finally
        {
            (VulkanBackend.MatrixUnitsOverride, VulkanBackend.TuningCacheFile) = (saved, savedCache);
            File.Delete(cache);
            VulkanBackend.MatMulKernel = null;
            VulkanBackend.PackedPromptKernel = null;
        }
    }

    private static void VulkanMatrixProducts(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false")
        {
            Console.WriteLine("    (IDRAK_VULKAN_KERNELS=0: kernels off, skipped)");
            return;
        }

        string cache = Path.Combine(Path.GetTempPath(), $"idrak-matrix-tuning-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        VulkanBackend.TuningCacheFile = cache;
        try
        {
            var own = (VulkanBackend)device.Backend;
            if (own.MatrixShape is { } native && !own.MatrixUnitsEmulated)
            {
                // The device's own cooperative matrices: forced, then measured (any choice must match).
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    Console.WriteLine($"    ({backend.DescribeMatrixUnits()})");
                    Check(backend.MatrixShape == native, $"shape {backend.MatrixShape} on a second backend, {native} on the first");
                    VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = 3;
                    MatrixProductCases(backend, $"{device} {native}", full: true, expectMatrix: true);
                    VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = null;
                    MatrixProductCases(backend, $"{device} {native}, measured", full: false, expectMatrix: false);
                });
                return;
            }

            // Emulated: shapes with one, two and four tiles per side of a subgroup block, at the default width (four
            // subgroups: one round), half of it (two rounds) and twice it (half the subgroups idle in the products).
            int subgroup = (int)own.Facts.SubgroupSize;
            var cases = new (VulkanKernels.CoopShape Shape, int? Width)[]
            {
                (new(16, 16, 16), null), (new(16, 16, 16), 2 * subgroup), (new(8, 8, 8), 8 * subgroup), (new(16, 8, 16), null), (new(8, 32, 8), null), (new(32, 16, 16), 4 * subgroup), (new(16, 16, 32), null),
            };
            bool ran = false;
            foreach (var (shape, width) in cases)
            {
                VulkanBackend.EmulatedMatrixUnits = shape;
                VulkanBackend.CoopWidthOverride = width is int w && w >= 16 && w <= VulkanKernels.MaxWidth ? w : null;
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    if (backend.MatrixShape is null || !backend.MatrixUnitsEmulated)
                    {
                        return;                                                // no 16-bit floats, or the subgroup size does not divide the shape
                    }

                    ran = true;
                    VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = 3;
                    // A depth of 32 (K = 32) needs about 36 KiB of workgroup memory: on a device with less, the float32
                    // kernels run instead (checked too).
                    bool fits = shape.K <= 16 || backend.Limits.SharedBytes >= 36 << 10;
                    MatrixProductCases(backend, $"emulated {shape}, width {backend.CoopWidth}, subgroup {subgroup}{(fits ? "" : ", no room: float32 kernels")}",
                        full: width is null && shape.M == 16 && shape.N == 16, expectMatrix: fits);
                });
            }

            if (!ran)
            {
                Console.WriteLine($"    ({device}: no 16-bit floats in shaders, or subgroups of {subgroup}: the emulation cannot run; the kernels are validated by spirv-val only)");
            }
        }
        finally
        {
            VulkanBackend.EmulatedMatrixUnits = null;
            VulkanBackend.CoopWidthOverride = null;
            VulkanBackend.MatMulKernel = null;
            VulkanBackend.PackedPromptKernel = null;
            VulkanBackend.TuningCacheFile = savedCache;
            File.Delete(cache);
        }
    }

    // The products through `backend` against the CPU: float32 (every transpose, batches, beta) and the packed prompts
    // (int8, int4, bfloat16), on partial blocks and on rows and columns of very different magnitudes. With
    // `expectMatrix`, a cooperative-matrix kernel must have run each of them; without it, none may run when the backend
    // has no cooperative matrices.
    private static void MatrixProductCases(VulkanBackend backend, string label, bool full, bool expectMatrix)
    {
        var random = new Random(41);
        float[] R(int n, float lo = -1f, float hi = 1f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];

        // Values whose rows (or columns, every `stride` values) span 1e-12 to 1e12 and some past 65504, some all zero.
        float[] Spread(int lines, int length, bool alongRows)
        {
            float[] factors = [1e-12f, 1f, 3e-5f, 0f, 1e12f, 7e4f, 1e-3f, 250f];
            var values = R(lines * length);
            for (int i = 0; i < values.Length; i++)
            {
                int line = alongRows ? i / length : i % lines;
                values[i] *= factors[line % factors.Length];
            }

            return values;
        }

        // bfloat16 words from floats (low half first), rounded to nearest even, columns scaled by `columnScale`.
        float[] BFloat16Words(int k, int n, Func<int, float> columnScale)
        {
            int words = (n + 1) / 2;
            var packed = new float[k * words];
            for (int r = 0; r < k; r++)
            {
                for (int w = 0; w < words; w++)
                {
                    uint Half(int col) => col < n ? Round((2 * random.NextSingle() - 1) * columnScale(col)) : 0;
                    packed[r * words + w] = BitConverter.Int32BitsToSingle((int)(Half(2 * w) | Half(2 * w + 1) << 16));
                }
            }

            return packed;

            static uint Round(float v)
            {
                uint bits = BitConverter.SingleToUInt32Bits(v);
                return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
            }
        }

        // float32 products.
        var matShapes = full
            ? new[] { (1, 64, 64, 64), (2, 70, 130, 33), (3, 129, 65, 200), (1, 5, 7, 3), (1, 100, 1, 40), (1, 1, 90, 77), (2, 64, 64, 0) }
            : [(2, 70, 130, 33), (1, 5, 7, 3)];
        foreach (var (batch, m, n, k) in matShapes)
        {
            foreach (var (ta, tb) in new[] { (false, false), (true, false), (false, true), (true, true) })
            {
                foreach (float beta in new[] { 0f, 0.5f })
                {
                    float[] a = R(batch * m * k), b = R(batch * k * n), c = R(batch * m * n);
                    MatrixCase(backend, $"{label}: float32 {batch}×{m}×{n}×{k}, transA {ta}, transB {tb}, beta {beta}", [a, b, c], 2,
                        (be, s) => be.BatchedMatMul(s[0], s[1], s[2], batch, m, n, k, ta, tb, beta), AbsProduct(a, b, batch, m, n, k, ta, tb, beta, c), expectMatrix);
                }
            }
        }

        // Rows of a and columns of b of very different magnitudes (each output checked against its own terms).
        foreach (var (ta, tb) in new[] { (false, false), (true, true) })
        {
            const int M = 72, N = 80, K = 70;
            float[] a = ta ? Spread(M, K, alongRows: false) : Spread(M, K, alongRows: true);
            float[] b = tb ? Spread(N, K, alongRows: true) : Spread(N, K, alongRows: false);
            MatrixCase(backend, $"{label}: float32 spread magnitudes, transA {ta}, transB {tb}", [a, b, new float[M * N]], 2,
                (be, s) => be.BatchedMatMul(s[0], s[1], s[2], 1, M, N, K, ta, tb, 0f), AbsProduct(a, b, 1, M, N, K, ta, tb, 0f, null), expectMatrix);
        }

        // Packed prompts.
        var packedShapes = full ? new[] { (9, 64, 96), (33, 100, 65), (70, 33, 7), (130, 300, 200), (64, 257, 130) } : [(33, 100, 65)];
        foreach (var (m, k, n) in packedShapes)
        {
            int words8 = (n + 3) / 4, words4 = (n + 7) / 8;
            float[] x = R(m * k);
            float[] q8 = Bits(k * words8), s8 = R(n, 0.001f, 0.02f);
            float[] q4 = Bits(k * words4), s4 = R((k + 31) / 32 * words4 * 8, 0.01f, 0.2f);
            float[] bf = BFloat16Words(k, n, _ => 1f);
            PackedMatrixCase(backend, $"{label}: int8 {m}×{k}→{n}", PackedFormat.Int8, x, q8, s8, m, n, k, expectMatrix);
            PackedMatrixCase(backend, $"{label}: int4 {m}×{k}→{n}", PackedFormat.Int4, x, q4, s4, m, n, k, expectMatrix);
            PackedMatrixCase(backend, $"{label}: bfloat16 {m}×{k}→{n}", PackedFormat.BFloat16, x, bf, null, m, n, k, expectMatrix);
        }

        {
            const int M = 72, K = 70, N = 81;
            float[] x = Spread(M, K, alongRows: true);
            float[] bf = BFloat16Words(K, N, col => new[] { 1e-12f, 1f, 1e9f, 9e4f, 2e-6f }[col % 5]);
            PackedMatrixCase(backend, $"{label}: bfloat16 spread magnitudes", PackedFormat.BFloat16, x, bf, null, M, N, K, expectMatrix);
        }
    }

    // Σ_k |op(a)[i, kk]| · |op(b)[kk, j]| (+ |beta · c|) per output, in double: what each output's float32 error is relative to.
    private static double[] AbsProduct(float[] a, float[] b, int batch, int m, int n, int k, bool ta, bool tb, float beta, float[]? c)
    {
        var bound = new double[batch * m * n];
        for (int bi = 0; bi < batch; bi++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int kk = 0; kk < k; kk++)
                    {
                        float av = a[bi * m * k + (ta ? kk * m + i : i * k + kk)], bv = b[bi * k * n + (tb ? j * k + kk : kk * n + j)];
                        sum += Math.Abs((double)av * bv);
                    }

                    int o = bi * m * n + i * n + j;
                    bound[o] = sum + (c is null ? 0 : Math.Abs((double)beta * c[o]));
                }
            }
        }

        return bound;
    }

    private static void PackedMatrixCase(VulkanBackend backend, string what, PackedFormat format, float[] x, float[] packed, float[]? scales, int m, int n, int k, bool expectMatrix)
    {
        // The weights as the CPU expands them, for the bound.
        var cpu = CpuBackend.Instance;
        var w = new float[k * n];
        var sw = cpu.Allocate(k * n, false);
        var sq = cpu.Allocate(packed.Length, false);
        var ss = scales is null ? null : cpu.Allocate(scales.Length, false);
        try
        {
            cpu.Upload(packed, sq);
            if (ss is not null)
            {
                cpu.Upload(scales!, ss);
            }

            switch (format)
            {
                case PackedFormat.Int8:
                    cpu.Int8Dequantize(sq, ss!, sw, k, n);
                    break;
                case PackedFormat.Int4:
                    cpu.Int4Dequantize(sq, ss!, sw, k, n);
                    break;
                default:
                    cpu.BFloat16Dequantize(sq, sw, k, n);
                    break;
            }

            cpu.Download(sw, w);
        }
        finally
        {
            sw.Release();
            sq.Release();
            ss?.Release();
        }

        float[][] inputs = scales is null ? [x, packed, new float[m * n]] : [x, packed, scales, new float[m * n]];
        int output = inputs.Length - 1;
        MatrixCase(backend, what, inputs, output, (b, s) =>
        {
            if (b is CpuBackend)
            {
                switch (format)
                {
                    case PackedFormat.Int8:
                        b.Int8MatMul(s[0], s[1], s[2], s[3], m, n, k);
                        break;
                    case PackedFormat.Int4:
                        b.Int4MatMul(s[0], s[1], s[2], s[3], m, n, k);
                        break;
                    default:
                        b.BFloat16MatMul(s[0], s[1], s[2], m, n, k);
                        break;
                }
            }
            else if (!b.PackedMatMulLarge(format, s[0], s[1], scales is null ? null : s[2], s[output], m, n, k))
            {
                // Expanding the weights (measured faster, or the only path): as the tensor calls it.
                var expanded = b.Allocate(k * n, false);
                switch (format)
                {
                    case PackedFormat.Int8:
                        b.Int8Dequantize(s[1], s[2], expanded, k, n);
                        break;
                    case PackedFormat.Int4:
                        b.Int4Dequantize(s[1], s[2], expanded, k, n);
                        break;
                    default:
                        b.BFloat16Dequantize(s[1], expanded, k, n);
                        break;
                }

                b.MatMul(s[0], expanded, s[output], m, n, k, false, false, 0f);
                expanded.Release();
            }
        }, AbsProduct(x, w, 1, m, n, k, false, false, 0f, null), expectMatrix);
    }

    // Runs `op` on the CPU and on `backend` (no host fallback) over copies of `inputs`, and checks storage `output`: every
    // element within 1e-4 of its bound (the sum of the magnitudes it adds) of the CPU's. With `expectMatrix` a
    // cooperative-matrix kernel must have run; without cooperative matrices on the backend, none may.
    private static void MatrixCase(VulkanBackend backend, string what, float[][] inputs, int output, Action<Backend, Storage[]> op, double[] bound, bool expectMatrix)
    {
        var expected = RunMatrixOn(CpuBackend.Instance, inputs, output, op);
        var operations = new ConcurrentDictionary<string, long>();
        var kernels = new ConcurrentDictionary<string, long>();
        (backend.HostCallsByOperation, backend.DispatchesByKernel) = (operations, kernels);
        float[] actual;
        try
        {
            actual = RunMatrixOn(backend, inputs, output, op);
        }
        finally
        {
            (backend.HostCallsByOperation, backend.DispatchesByKernel) = (null, null);
        }

        bool matrixRan = kernels.Keys.Any(name => name.StartsWith("coop_", StringComparison.Ordinal));
        Check(operations.IsEmpty, $"{what}: host fallback for {string.Join(", ", operations.Keys)}");
        Check(!expectMatrix || matrixRan, $"{what}: no cooperative-matrix kernel ran (ran {string.Join(", ", kernels.Keys)})");
        Check(backend.MatrixShape is not null || !matrixRan, $"{what}: a cooperative-matrix kernel ran without cooperative matrices");
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs((double)expected[i] - actual[i]);
            if (!(error <= 1e-4 * bound[i] + 1e-37))
            {
                throw new Exception($"{what}: element {i} is {actual[i]}, expected {expected[i]} (bound {bound[i]:G3})");
            }
        }

        static float[] RunMatrixOn(Backend b, float[][] inputs, int output, Action<Backend, Storage[]> op)
        {
            var storages = inputs.Select(d => { var s = b.Allocate(d.Length, false); b.Upload(d, s); return s; }).ToArray();
            try
            {
                op(b, storages);
                var result = new float[inputs[output].Length];
                b.Download(storages[output], result);
                return result;
            }
            finally
            {
                foreach (var s in storages)
                {
                    s.Release();
                }
            }
        }
    }
}
