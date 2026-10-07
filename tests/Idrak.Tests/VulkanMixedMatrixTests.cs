// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;

// Reduced-precision matrix products on Vulkan (MixedPrecision, VulkanBackend.Matrix.cs): with MixedPrecision.BFloat16
// the single-pass cooperative-matrix kernels round each operand once (to bfloat16, or to a 16-bit float after the
// power-of-two scaling) and sum in float32. Each output must match the product of the rounded operands (summed in
// double, within float32 summation error) and stay within bfloat16's error bound of the float32 product (2^-7 of the sum
// of the magnitudes it adds), which is what the CUDA tensor-core tests check too. On a device with cooperative matrices
// the device's own kernel runs; on one without (a CPU driver), the kernels run emulated, bfloat16 and 16-bit float
// operands both. Without MixedPrecision (float32) no reduced-precision kernel runs and the products match the CPU in
// float32.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanMixedMatrixGroup =
    [
        ("vulkan matrix units: the bfloat16 matrix shape comes from what the device reports (bfloat16 operands, float32 sums, subgroup scope; the most multiply-adds per operation)", VulkanBFloat16ShapeChoice),
        ("vulkan matrix units: reduced precision (MixedPrecision bfloat16): one product per step, operands rounded once (bfloat16, or 16-bit floats after scaling), float32 sums; matches the rounded operands' product and stays within bfloat16 error of float32 (every transpose, batches, beta, partial blocks, spread magnitudes; int8, int4, bfloat16 prompts)", VulkanMixedProducts),
        ("vulkan matrix units: without MixedPrecision nothing changes (float32 mode: no reduced-precision kernel is a candidate or runs, asking for one runs the float32 kernels; without cooperative matrices mixed mode matches the CPU in float32)", VulkanMixedAbsent),
    ];

    private static void VulkanBFloat16ShapeChoice(Device device)
    {
        _ = device;
        const uint BF16 = VulkanDriver.ComponentBFloat16;
        // A list of the kind one device reports: float16, bfloat16 and 8-bit shapes; the bfloat16 → float32 one with the most work.
        VkCooperativeMatrixProperties[] reported = [MatrixProperties(16, 16, 16), MatrixProperties(16, 8, 16, a: BF16, b: BF16), MatrixProperties(16, 16, 16, a: BF16, b: BF16),
            MatrixProperties(16, 16, 16, a: BF16, b: BF16, c: BF16, result: BF16), MatrixProperties(16, 16, 32, a: 1000491002, b: 1000491002)];
        Check(VulkanBackend.ChooseMatrixShape(reported, BF16) == new VulkanKernels.CoopShape(16, 16, 16), $"bfloat16: {VulkanBackend.ChooseMatrixShape(reported, BF16)}");
        Check(VulkanBackend.ChooseMatrixShape(reported) == new VulkanKernels.CoopShape(16, 16, 16), "float16 unchanged");
        // bfloat16 sums only, mixed operand types, other scopes: none.
        Check(VulkanBackend.ChooseMatrixShape([MatrixProperties(16, 16, 16, a: BF16, b: BF16, c: BF16, result: BF16), MatrixProperties(16, 16, 16, a: BF16, b: 0),
            MatrixProperties(16, 16, 16, a: BF16, b: BF16, scope: 2), MatrixProperties(16, 16, 16)], BF16) is null, "nothing usable");
    }

    private static void VulkanMixedProducts(Device device)
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

        string cache = Path.Combine(Path.GetTempPath(), $"idrak-mixed-tuning-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        VulkanBackend.TuningCacheFile = cache;
        try
        {
            using var precision = MixedPrecision.BFloat16();
            var own = (VulkanBackend)device.Backend;
            if (own.MatrixShape is { } native && !own.MatrixUnitsEmulated)
            {
                // The device's own cooperative matrices: its operand type, then 16-bit floats, forced; then measured.
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    Console.WriteLine($"    ({backend.DescribeMatrixUnits()})");
                    var types = backend.BFloat16MatrixShape is null
                        ? new[] { VulkanKernels.CoopPrecision.Float16 }
                        : [VulkanKernels.CoopPrecision.BFloat16, VulkanKernels.CoopPrecision.Float16];
                    foreach (var type in types)
                    {
                        VulkanBackend.MixedOperandsOverride = type;
                        VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = 4;
                        MixedProductCases(backend, $"{device} {native}, {type} operands", type, full: true, expectMixed: true);
                    }

                    VulkanBackend.MixedOperandsOverride = null;
                    VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = null;
                    MixedProductCases(backend, $"{device} {native}, measured", null, full: false, expectMixed: false);
                });
                return;
            }

            // Emulated: both operand types, in a few shapes and subgroup counts (as VulkanMatrixProducts).
            int subgroup = (int)own.Facts.SubgroupSize;
            var cases = new (VulkanKernels.CoopShape Shape, int? Width, VulkanKernels.CoopPrecision Type)[]
            {
                (new(16, 16, 16), null, VulkanKernels.CoopPrecision.BFloat16), (new(16, 16, 16), null, VulkanKernels.CoopPrecision.Float16),
                (new(16, 8, 16), 2 * subgroup, VulkanKernels.CoopPrecision.BFloat16), (new(8, 32, 8), 8 * subgroup, VulkanKernels.CoopPrecision.Float16),
                (new(16, 16, 32), null, VulkanKernels.CoopPrecision.BFloat16),
            };
            bool ran = false;
            foreach (var (shape, width, type) in cases)
            {
                VulkanBackend.EmulatedMatrixUnits = shape;
                VulkanBackend.EmulatedBFloat16 = type == VulkanKernels.CoopPrecision.BFloat16;
                VulkanBackend.MixedOperandsOverride = type;
                VulkanBackend.CoopWidthOverride = width is int w && w >= 16 && w <= VulkanKernels.MaxWidth ? w : null;
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    if (backend.MatrixShape is null || !backend.MatrixUnitsEmulated)
                    {
                        return;                                                // no 16-bit floats, or the subgroup size does not divide the shape
                    }

                    ran = true;
                    bool fits = shape.K <= 16 || backend.Limits.SharedBytes >= 36 << 10;
                    VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = 4;
                    string label = $"emulated {shape}, {type} operands, width {backend.CoopWidth}, subgroup {subgroup}";
                    MixedProductCases(backend, label, type, full: width is null && shape == new VulkanKernels.CoopShape(16, 16, 16), expectMixed: fits);
                    if (width is null && shape == new VulkanKernels.CoopShape(16, 16, 16))
                    {
                        // Measured: whichever wins (float32 or the reduced-precision kernel), within bfloat16 error of float32.
                        VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = null;
                        MixedProductCases(backend, $"{label}, measured", type, full: false, expectMixed: false);
                    }
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
            VulkanBackend.EmulatedBFloat16 = false;
            VulkanBackend.MixedOperandsOverride = null;
            VulkanBackend.CoopWidthOverride = null;
            VulkanBackend.MatMulKernel = null;
            VulkanBackend.PackedPromptKernel = null;
            VulkanBackend.TuningCacheFile = savedCache;
            File.Delete(cache);
        }
    }

    private static void VulkanMixedAbsent(Device device)
    {
        if (device.Type != DeviceType.Vulkan)
        {
            return;
        }

        var own = (VulkanBackend)device.Backend;
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-mixed-absent-{Environment.ProcessId}.tsv");
        string? savedCache = VulkanBackend.TuningCacheFile;
        VulkanBackend.TuningCacheFile = cache;
        try
        {
            // Float32 mode on a backend with cooperative matrices (the device's, or emulated with bfloat16 too): forced
            // or measured, no reduced-precision kernel runs and every product matches the CPU in float32.
            bool emulate = own.MatrixShape is null || own.MatrixUnitsEmulated;
            if (emulate)
            {
                (VulkanBackend.EmulatedMatrixUnits, VulkanBackend.EmulatedBFloat16) = (new VulkanKernels.CoopShape(16, 16, 16), true);
            }

            using (MixedPrecision.Use(MatMulPrecision.Float32))
            {
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    foreach (int? kernel in new int?[] { 4, null })
                    {
                        VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = kernel;
                        MixedProductCases(backend, $"float32 mode, kernel {kernel?.ToString() ?? "measured"}", null, full: false, expectMixed: false, float32: true);
                    }
                });
            }

            // Mixed mode without cooperative matrices: the float32 kernels, matching the CPU in float32.
            (VulkanBackend.EmulatedMatrixUnits, VulkanBackend.EmulatedBFloat16) = (null, false);
            var saved = VulkanBackend.MatrixUnitsOverride;
            VulkanBackend.MatrixUnitsOverride = false;
            try
            {
                using var precision = MixedPrecision.BFloat16();
                WithBackend(device.Ordinal, mapped: true, backend =>
                {
                    foreach (int? kernel in new int?[] { 4, null })
                    {
                        VulkanBackend.MatMulKernel = VulkanBackend.PackedPromptKernel = kernel;
                        MixedProductCases(backend, $"mixed mode without cooperative matrices, kernel {kernel?.ToString() ?? "measured"}", null, full: false, expectMixed: false, float32: true);
                    }
                });
            }
            finally
            {
                VulkanBackend.MatrixUnitsOverride = saved;
            }
        }
        finally
        {
            VulkanBackend.EmulatedMatrixUnits = null;
            VulkanBackend.EmulatedBFloat16 = false;
            VulkanBackend.MatMulKernel = null;
            VulkanBackend.PackedPromptKernel = null;
            VulkanBackend.TuningCacheFile = savedCache;
            File.Delete(cache);
        }
    }

    // The products through `backend` (float32 every transpose, batches, beta, partial blocks, spread magnitudes; int8,
    // int4 and bfloat16 prompts). `type`: the operand rounding of the reduced-precision kernel (null: whichever ran, read
    // from its name). With `expectMixed` a reduced-precision kernel must have run each product; with `float32` none may
    // run and every output must match the CPU within float32 error.
    private static void MixedProductCases(VulkanBackend backend, string label, VulkanKernels.CoopPrecision? type, bool full, bool expectMixed, bool float32 = false)
    {
        var random = new Random(43);
        float[] R(int n, float lo = -1f, float hi = 1f) => [.. Enumerable.Range(0, n).Select(_ => lo + (hi - lo) * random.NextSingle())];
        float[] Bits(int n) => [.. Enumerable.Range(0, n).Select(_ => BitConverter.Int32BitsToSingle(random.Next() ^ (random.Next() << 16)))];
        float[] Spread(int lines, int length, bool alongRows)
        {
            float[] factors = [1e-12f, 1f, 3e-5f, 0f, 1e12f, 7e4f, 1e-3f, 250f];
            var values = R(lines * length);
            for (int i = 0; i < values.Length; i++)
            {
                values[i] *= factors[(alongRows ? i / length : i % lines) % factors.Length];
            }

            return values;
        }

        var shapes = full
            ? new[] { (1, 64, 64, 64), (2, 70, 130, 33), (3, 129, 65, 200), (1, 5, 7, 3), (1, 100, 1, 40), (2, 64, 64, 0) }
            : [(2, 70, 130, 33), (1, 5, 7, 3)];
        foreach (var (batch, m, n, k) in shapes)
        {
            foreach (var (ta, tb) in new[] { (false, false), (true, false), (false, true), (true, true) })
            {
                foreach (float beta in new[] { 0f, 0.5f })
                {
                    float[] a = R(batch * m * k), b = R(batch * k * n), c = R(batch * m * n);
                    MixedCase(backend, $"{label}: float32 {batch}×{m}×{n}×{k}, transA {ta}, transB {tb}, beta {beta}", [a, b, c], 2,
                        (be, s) => be.BatchedMatMul(s[0], s[1], s[2], batch, m, n, k, ta, tb, beta),
                        t => RoundedProduct(a, b, batch, m, n, k, ta, tb, beta, c, t, roundB: true), type, expectMixed, float32);
                }
            }
        }

        foreach (var (ta, tb) in new[] { (false, false), (true, true) })
        {
            const int M = 72, N = 80, K = 70;
            float[] a = ta ? Spread(M, K, alongRows: false) : Spread(M, K, alongRows: true);
            float[] b = tb ? Spread(N, K, alongRows: true) : Spread(N, K, alongRows: false);
            MixedCase(backend, $"{label}: float32 spread magnitudes, transA {ta}, transB {tb}", [a, b, new float[M * N]], 2,
                (be, s) => be.BatchedMatMul(s[0], s[1], s[2], 1, M, N, K, ta, tb, 0f),
                t => RoundedProduct(a, b, 1, M, N, K, ta, tb, 0f, null, t, roundB: true), type, expectMixed, float32);
        }

        var packedShapes = full ? new[] { (9, 64, 96), (70, 33, 7), (130, 300, 200) } : [(33, 100, 65)];
        foreach (var (m, k, n) in packedShapes)
        {
            int words8 = (n + 3) / 4, words4 = (n + 7) / 8, words16 = (n + 1) / 2;
            float[] x = R(m * k);
            var formats = new (PackedFormat Format, float[] Packed, float[]? Scales)[]
            {
                (PackedFormat.Int8, Bits(k * words8), R(n, 0.001f, 0.02f)),
                (PackedFormat.Int4, Bits(k * words4), R((k + 31) / 32 * words4 * 8, 0.01f, 0.2f)),
                (PackedFormat.BFloat16, [.. Enumerable.Range(0, k * words16).Select(_ => BitConverter.Int32BitsToSingle((int)((uint)BitConverter.SingleToInt32Bits(R(1)[0]) >> 16
                    | (uint)BitConverter.SingleToInt32Bits(R(1)[0]) & 0xFFFF0000u)))], null),
            };
            foreach (var (format, packed, scales) in formats)
            {
                var w = DequantizeOnCpu(format, packed, scales, k, n);
                float[][] inputs = scales is null ? [x, packed, new float[m * n]] : [x, packed, scales, new float[m * n]];
                int output = inputs.Length - 1;
                // An int8 weight is its code (exact in either type) times a column scale applied to the sum: not rounded.
                MixedCase(backend, $"{label}: {format} {m}×{k}→{n}", inputs, output, (be, s) => PackedOn(be, format, s, scales is not null, m, n, k),
                    t => RoundedProduct(x, w, 1, m, n, k, false, false, 0f, null, t, roundB: format != PackedFormat.Int8), type, expectMixed, float32);
            }
        }
    }

    // y = x · w for a packed weight as the tensors run it (the expanding path where the backend measured it faster).
    private static void PackedOn(Backend b, PackedFormat format, Storage[] s, bool scaled, int m, int n, int k)
    {
        int output = scaled ? 3 : 2;
        if (b is CpuBackend || !b.PackedMatMulLarge(format, s[0], s[1], scaled ? s[2] : null, s[output], m, n, k))
        {
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
    }

    // The weights as the CPU expands them.
    private static float[] DequantizeOnCpu(PackedFormat format, float[] packed, float[]? scales, int k, int n)
    {
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
            return w;
        }
        finally
        {
            sw.Release();
            sq.Release();
            ss?.Release();
        }
    }

    /// <summary>
    /// op(a) · op(b) (+ beta · c) with the operands rounded as a reduced-precision kernel of <paramref name="type"/>
    /// rounds them (op(b) only with <paramref name="roundB"/>), summed in double: the sums, the sums of the rounded terms'
    /// magnitudes, and the sums of the unrounded terms' magnitudes (the bound of bfloat16's error). BFloat16: every operand
    /// to the nearest bfloat16, ties to even. Float16: every row of op(a) and column of op(b) multiplied by the power of
    /// two that brings its largest magnitude into [2^14, 2^15), each value rounded to 11 significant bits (ties to even)
    /// and converted to a 16-bit float, the sum divided by the two powers.
    /// </summary>
    internal static (double[] Sum, double[] Rounded, double[] Exact) RoundedProduct(float[] a, float[] b, int batch, int m, int n, int k, bool ta, bool tb, float beta,
        float[]? c, VulkanKernels.CoopPrecision type, bool roundB)
    {
        var (sum, rounded, exact) = (new double[batch * m * n], new double[batch * m * n], new double[batch * m * n]);
        for (int bi = 0; bi < batch; bi++)
        {
            var (opA, opB) = (new float[m * k], new float[k * n]);
            for (int i = 0; i < m; i++)
            {
                for (int q = 0; q < k; q++)
                {
                    opA[i * k + q] = a[bi * m * k + (ta ? q * m + i : i * k + q)];
                }
            }

            for (int q = 0; q < k; q++)
            {
                for (int j = 0; j < n; j++)
                {
                    opB[q * n + j] = b[bi * k * n + (tb ? j * k + q : q * n + j)];
                }
            }

            var (ra, rb) = ((float[])opA.Clone(), (float[])opB.Clone());
            var (rowScale, colScale) = (Enumerable.Repeat(1.0, m).ToArray(), Enumerable.Repeat(1.0, n).ToArray());
            if (type == VulkanKernels.CoopPrecision.BFloat16)
            {
                ra = [.. ra.Select(RoundBFloat16)];
                rb = roundB ? [.. rb.Select(RoundBFloat16)] : rb;
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    int shift = LineShift(Enumerable.Range(0, k).Select(q => opA[i * k + q]));
                    rowScale[i] = Math.ScaleB(1.0, -shift);
                    for (int q = 0; q < k; q++)
                    {
                        ra[i * k + q] = RoundHalf(opA[i * k + q] * Power2(shift));
                    }
                }

                for (int j = 0; j < n && roundB; j++)
                {
                    int shift = LineShift(Enumerable.Range(0, k).Select(q => opB[q * n + j]));
                    colScale[j] = Math.ScaleB(1.0, -shift);
                    for (int q = 0; q < k; q++)
                    {
                        rb[q * n + j] = RoundHalf(opB[q * n + j] * Power2(shift));
                    }
                }
            }

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double s = 0, r = 0, e = 0;
                    for (int q = 0; q < k; q++)
                    {
                        double term = (double)ra[i * k + q] * rb[q * n + j] * rowScale[i] * colScale[j];
                        s += term;
                        r += Math.Abs(term);
                        e += Math.Abs((double)opA[i * k + q] * opB[q * n + j]);
                    }

                    int o = bi * m * n + i * n + j;
                    double extra = c is null ? 0 : (double)beta * c[o];
                    (sum[o], rounded[o], exact[o]) = (s + extra, r + Math.Abs(extra), e + Math.Abs(extra));
                }
            }
        }

        return (sum, rounded, exact);

        static float Power2(int shift) => BitConverter.Int32BitsToSingle((shift + 127) << 23);

        // The kernels' shift: max · 2^s in [2^14, 2^15) (from the exponent bits; zeros and infinities clamped).
        static int LineShift(IEnumerable<float> line)
        {
            float best = line.Select(MathF.Abs).DefaultIfEmpty(0f).Max();
            int exponent = (BitConverter.SingleToInt32Bits(best) >> 23) & 0xFF;
            return Math.Clamp(141 - exponent, -126, 126);
        }

        static float RoundHalf(float x)
        {
            uint bits = BitConverter.SingleToUInt32Bits(x);
            bits = (bits + 0xFFFu + ((bits >> 13) & 1u)) & 0xFFFFE000u;
            return (float)(Half)BitConverter.UInt32BitsToSingle(bits);
        }
    }

    // Runs `op` on the CPU and on `backend` over copies of `inputs` and checks storage `output`: with a reduced-precision
    // kernel run (its rounding from `type`, else its name), every element within 1e-4 of its rounded terms' magnitudes of
    // the rounded operands' product and within 2^-7 of its terms' magnitudes of the CPU's float32 product; without one,
    // within 1e-4 of the CPU's (float32).
    private static void MixedCase(VulkanBackend backend, string what, float[][] inputs, int output, Action<Backend, Storage[]> op,
        Func<VulkanKernels.CoopPrecision, (double[] Sum, double[] Rounded, double[] Exact)> reference, VulkanKernels.CoopPrecision? type, bool expectMixed, bool float32)
    {
        var expected = RunOn(CpuBackend.Instance, inputs, output, op);
        _ = RunOn(backend, inputs, output, op);                                 // measures the shape first: the kernels recorded below are the choice
        var operations = new ConcurrentDictionary<string, long>();
        var kernels = new ConcurrentDictionary<string, long>();
        CountHostCalls(backend, operations);
        backend.DispatchesByKernel = kernels;
        float[] actual;
        try
        {
            actual = RunOn(backend, inputs, output, op);
        }
        finally
        {
            CountHostCalls(backend, null);
            backend.DispatchesByKernel = null;
        }

        string? mixed = kernels.Keys.FirstOrDefault(name => name.Contains("_round_", StringComparison.Ordinal));
        Check(operations.IsEmpty, $"{what}: host fallback for {string.Join(", ", operations.Keys)}");
        Check(!expectMixed || mixed is not null, $"{what}: no reduced-precision kernel ran (ran {string.Join(", ", kernels.Keys)})");
        Check(!float32 || mixed is null, $"{what}: a reduced-precision kernel ran ({mixed})");
        if (mixed is null)
        {
            var (_, _, magnitudes) = reference(VulkanKernels.CoopPrecision.BFloat16);
            for (int i = 0; i < expected.Length; i++)
            {
                double error = Math.Abs((double)expected[i] - actual[i]);
                if (!(error <= 1e-4 * magnitudes[i] + 1e-37))
                {
                    throw new Exception($"{what}: element {i} is {actual[i]}, expected {expected[i]} (float32; bound {magnitudes[i]:G3})");
                }
            }

            return;
        }

        var rounding = mixed.Contains("_round_bf16", StringComparison.Ordinal) ? VulkanKernels.CoopPrecision.BFloat16 : VulkanKernels.CoopPrecision.Float16;
        Check(type is null || type == rounding, $"{what}: {mixed} ran, expected {type} operands");
        var (sum, rounded, exact) = reference(rounding);
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(sum[i] - actual[i]);
            if (!(error <= 1e-4 * rounded[i] + 1e-37))
            {
                throw new Exception($"{what}: element {i} is {actual[i]}, expected {sum[i]} from the rounded operands (bound {rounded[i]:G3}; {mixed})");
            }

            double contract = Math.Abs((double)expected[i] - actual[i]);
            if (!(contract <= Math.ScaleB(exact[i], -7) + 1e-37))
            {
                throw new Exception($"{what}: element {i} is {actual[i]}, float32 {expected[i]}: off by more than bfloat16 error (bound {exact[i]:G3}; {mixed})");
            }
        }

        static float[] RunOn(Backend b, float[][] inputs, int output, Action<Backend, Storage[]> op)
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
