// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Layers;

// The CPU's tiling, blocking and threading choices come from what the machine reports (cores, vector width and
// registers, caches) or are measured on it; none of them may change a result, since the CPU is the reference.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CpuTuningGroup =
    [
        ("cpu tuning: the formulas give the former constants on a 32 KiB L1 / 1 MiB L2 / 8-wide machine and scale with the reported caches and width", CpuTuningFormulas),
        ("cpu tuning: measured cut-overs, every tiled kernel, every vector path setting, register rows, k chunks and the streaming loop give the results of the fallback formulas, bit for bit", CpuTunedChoicesMatch),
        ("cpu tuning: measured cut-overs are kept per machine, runtime and thread count and read back; the machine's report parses", CpuTuningCache),
        ("cpu tuning: the CPU's memory is the system's (/proc/meminfo, the lowest cgroup v1 or v2 limit above the process with its usage less inactive file cache) where the runtime's figure is its heap budget (Mono), held to the runtime's heap limit where it has one (CoreCLR: unchanged), the runtime's where the system reports none; both listed, labelled", CpuSystemMemory),
        ("cpu tuning: the cores are named by their groups from /proc/cpuinfo and cpufreq (part numbers and highest frequencies, raw; the system's name where given; x86's model name as before)", CpuCoreGroups),
        ("cpu tuning: the vector paths (IDRAK_CPU_VECTORS auto, native, portable, scalar) choose the kernels the runtime's report allows; portable 128-bit vectors where it reports no NEON; lane shifts equal their multiplies; the description says what runs", CpuVectorPathsChosen),
        ("cpu tuning: the k-split product (ComputeResources.CpuProductKChunk, off by default) splits only where the product's row and column blocks cannot occupy the threads (never on one thread, never a prompt's 512 x 1024 x 1024 or 512 x 1024 x 3072): there it sums its chunks in chunk order (every transpose and beta, the chunks' own products added in order, close to the unsplit product), elsewhere the unsplit bits", CpuProductSplitK),
        ("cpu tuning: one result on every CPU (plan 37, P0 and P0b): exp, log, the sigmoids, GELU, softmax, log-softmax, sin, cos and pow with their gradients, layer norm, batch statistics over short runs in chunks, the gated activations, few-row and tiled products, the lane sums and dot products (float, bfloat16, int8, double), the sums past one chunk, the sampler's weights and the host's double functions (PortableMath) hash to the pinned bits (the same with DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0, DOTNET_EnableHWIntrinsic=0 and IDRAK_CPU_VECTORS=scalar, portable or native)", CpuOneResult),
        ("cpu tuning: the portable math (plan 37, P0b): the CPU's float sin and cos lanes give their scalar form's bits and are within an ulp of MathF's over [-1e4, 1e4]; PortableMath's sin, cos, tan, atan, atan2, tanh, pow, log2 and log10 are within a few ulp of Math's, with its special cases; integer powers, log2 of powers of two and log10 of powers of ten exact", CpuPortableMath),
    ];

    internal static void CpuPortableMath(Device device)
    {
        if (device.Backend is not CpuBackend || !FirstRun(nameof(CpuPortableMath)))
        {
            return;
        }

        // Float sin and cos: lanes against the scalar form (bits) and MathF (an ulp of the result or 1e-7 absolute near zeros).
        int width = Vector<float>.Count;
        var inputs = new List<float> { 0f, -0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e-30f, MathF.PI / 2, MathF.PI, 1e6f };
        for (float v = -1e4f; v <= 1e4f; v += 0.7319f)
        {
            inputs.Add(v);
        }

        for (float v = -7f; v <= 7f; v += 1e-3f)
        {
            inputs.Add(v);
        }

        while (inputs.Count % 16 != 0)
        {
            inputs.Add(0.25f);
        }

        float[] values = [.. inputs], sines = new float[values.Length], cosines = new float[values.Length];
        for (int i = 0; i < values.Length; i += width)
        {
            CpuMath.Sin(new Vector<float>(values.AsSpan(i, width))).CopyTo(sines.AsSpan(i, width));
            CpuMath.Cos(new Vector<float>(values.AsSpan(i, width))).CopyTo(cosines.AsSpan(i, width));
        }

        int differ = 0;
        double worst = 0;
        float worstAt = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float x = values[i];
            differ += BitConverter.SingleToInt32Bits(CpuMath.Sin(x)) != BitConverter.SingleToInt32Bits(sines[i]) ? 1 : 0;
            differ += BitConverter.SingleToInt32Bits(CpuMath.Cos(x)) != BitConverter.SingleToInt32Bits(cosines[i]) ? 1 : 0;
            if (!float.IsFinite(x))
            {
                Check(float.IsNaN(sines[i]) && float.IsNaN(cosines[i]), $"sin and cos of {x} are NaN, got {sines[i]}, {cosines[i]}");
                continue;
            }

            foreach (var (got, expected) in new[] { (sines[i], Math.Sin(x)), (cosines[i], Math.Cos(x)) })
            {
                float magnitude = MathF.Abs((float)expected);
                double ulps = Math.Abs(got - expected) / Math.Max(MathF.BitIncrement(magnitude) - magnitude, float.Epsilon);   // in the result's ulps
                if (ulps > worst)
                {
                    (worst, worstAt) = (ulps, x);
                }
            }
        }

        Check(differ == 0, $"the float sin and cos lanes give their scalar form's bits: {differ} of {2 * values.Length} differ");
        Check(worst <= 1.0, $"the float sin and cos are within an ulp of the true value: {worst:F2} ulp at x = {worstAt}");

        // The double functions against the runtime's (a few ulp), and their special cases.
        double sinWorst = 0, tanWorst = 0, atanWorst = 0, tanhWorst = 0, powWorst = 0, logWorst = 0;
        static double Relative(double got, double expected) => expected == got ? 0 : Math.Abs(got - expected) / Math.Max(Math.Abs(expected), 1e-300);
        for (double v = -1e5; v <= 1e5; v += 13.37)
        {
            sinWorst = Math.Max(sinWorst, Math.Max(Math.Abs(PortableMath.Sin(v) - Math.Sin(v)), Math.Abs(PortableMath.Cos(v) - Math.Cos(v))));
        }

        for (double v = -1.5; v <= 1.5; v += 1e-4)
        {
            tanWorst = Math.Max(tanWorst, Relative(PortableMath.Tan(v), Math.Tan(v)));
            atanWorst = Math.Max(atanWorst, Math.Max(Relative(PortableMath.Atan(v * 40), Math.Atan(v * 40)), Relative(PortableMath.Atan2(v, 1 - v), Math.Atan2(v, 1 - v))));
            tanhWorst = Math.Max(tanhWorst, Relative(PortableMath.Tanh(v * 9), Math.Tanh(v * 9)));
            powWorst = Math.Max(powWorst, Relative(PortableMath.Pow(Math.Abs(v) * 7 + 0.01, v * 20), Math.Pow(Math.Abs(v) * 7 + 0.01, v * 20)));
            logWorst = Math.Max(logWorst, Math.Max(Relative(PortableMath.Log2(Math.Abs(v) * 1e3 + 1e-9), Math.Log2(Math.Abs(v) * 1e3 + 1e-9)),
                Relative(PortableMath.Log10(Math.Abs(v) * 1e3 + 1e-9), Math.Log10(Math.Abs(v) * 1e3 + 1e-9))));
        }

        Check(sinWorst <= 3e-16 && tanWorst <= 1e-15 && atanWorst <= 1e-15 && tanhWorst <= 1e-14 && powWorst <= 2e-13 && logWorst <= 1e-15,
            $"the double functions are close to the runtime's: sin and cos {sinWorst:E2} absolute, tan {tanWorst:E2}, atan and atan2 {atanWorst:E2}, tanh {tanhWorst:E2}, pow {powWorst:E2}, log2 and log10 {logWorst:E2}");
        bool exact = Enumerable.Range(-1074, 2098).All(e => PortableMath.Log2(Math.ScaleB(1, e)) == e)
                     && Enumerable.Range(0, 23).All(e => PortableMath.Log10(Math.Pow(10, e)) == e && PortableMath.Pow(10, e) == Math.Pow(10, e))
                     && PortableMath.Pow(1024, 3) == 1073741824 && PortableMath.Pow(2, -3) == 0.125 && PortableMath.Pow(-2, 3) == -8;
        Check(exact, "log2 of every power of two, log10 and pow of the powers of ten a double holds, and integer powers are exact");
        Check(PortableMath.Pow(0, -1) == double.PositiveInfinity && PortableMath.Pow(-0.0, -1) == double.NegativeInfinity && double.IsNaN(PortableMath.Pow(-2, 0.5))
              && PortableMath.Pow(double.NaN, 0) == 1 && PortableMath.Pow(1, double.NaN) == 1 && PortableMath.Pow(0.5, double.PositiveInfinity) == 0
              && PortableMath.Pow(double.NegativeInfinity, 3) == double.NegativeInfinity && PortableMath.Atan2(0, -1) == Math.PI && PortableMath.Atan2(-0.0, 1) == 0
              && double.IsNegative(PortableMath.Atan2(-0.0, 1)) && PortableMath.Atan2(1, 0) == Math.PI / 2 && PortableMath.Atan(double.PositiveInfinity) == Math.PI / 2
              && PortableMath.Tanh(-30) == -1 && double.IsNaN(PortableMath.Sin(double.PositiveInfinity)) && PortableMath.Sin(-0.0) == 0 && double.IsNegative(PortableMath.Sin(-0.0))
              && PortableMath.Cos(0) == 1 && PortableMathF.Pow(2f, 10f) == 1024f && PortableMathF.Pow(-8f, 1f / 3) is float.NaN,
            "the special cases are Math's: pow at 0, ±∞ and NaN, atan2's quadrants and signed zeros, atan and tanh at the ends, sin and cos at ∞ and ±0");
    }

    // The pinned bits of CpuOneResult's outputs (see there): any change to the CPU's arithmetic changes them, on purpose only.
    private const string CpuOneResultBits = "4DADF671C4E39E78";

    internal static void CpuOneResult(Device device)
    {
        if (device.Backend is not CpuBackend || !FirstRun(nameof(CpuOneResult)))
        {
            return;
        }

        var random = new Random(23);
        float[] Values(int count, float low, float high) => [.. Enumerable.Range(0, count).Select(_ => low + random.NextSingle() * (high - low))];
        var outputs = new List<(string What, float[] Values)>();
        const int Rows = 3, Cols = 1037;                                    // a tail past every vector width and the 16 lanes
        using (var scope = new TensorScope())
        {
            float[] xs = Values(Rows * Cols, -9f, 9f), gs = Values(Rows * Cols, -1f, 1f);
            var g = Tensor.From(gs, [Rows, Cols], device);
            foreach (var (name, op) in new (string, Func<Tensor, Tensor>)[]
                     {
                         ("exp", t => t.Exp()), ("log", t => (t.Square() + 1e-3f).Log()), ("sigmoid", t => t.Sigmoid()), ("silu", t => t.Silu()),
                         ("gelu", t => t.Gelu()), ("gelu erf", t => t.GeluErf()), ("tanh", t => t.Tanh()), ("softmax", t => t.Softmax()),
                         ("log-softmax", t => t.LogSoftmax()), ("sin", t => t.Sin()), ("cos", t => t.Cos()), ("pow 2.5", t => (t.Square() + 0.5f).Pow(2.5f)),
                     })
            {
                var x = Tensor.From(xs, [Rows, Cols], device, requiresGrad: true);
                var y = op(x);
                outputs.Add((name, y.ToArray()));
                (y * g).Sum().Backward();
                outputs.Add((name + "'s gradient", x.Grad!.ToArray()));
            }

            var gamma = Tensor.From(Values(Cols, 0.5f, 1.5f), [Cols], device, requiresGrad: true);
            var beta = Tensor.From(Values(Cols, -0.5f, 0.5f), [Cols], device, requiresGrad: true);
            var normed = Tensor.From(xs, [Rows, Cols], device, requiresGrad: true);
            var norm = normed.LayerNormTrain(gamma, beta, 1e-5f);
            outputs.Add(("layer norm", norm.ToArray()));
            (norm * g).Sum().Backward();
            outputs.Add(("layer norm's input gradient", normed.Grad!.ToArray()));
            outputs.Add(("layer norm's gamma gradient", gamma.Grad!.ToArray()));

            var gate = Tensor.From(xs, [Rows, Cols], device);
            foreach (var activation in new[] { Idrak.Abstraction.Generation.FeedForwardActivation.Silu, Idrak.Abstraction.Generation.FeedForwardActivation.Gelu })
            {
                outputs.Add(($"gated {activation}", Tensor.GatedActivation(gate, g, activation).ToArray()));
            }

            // Few rows against B stored [n, k] (the lane sums), and a tiled product (one FMA chain an element), beta 1.
            var a = Tensor.From(Values(2 * Cols, -1f, 1f), [2, Cols], device);
            var b = Tensor.From(Values(67 * Cols, -1f, 1f), [67, Cols], device);
            outputs.Add(("few-row product", a.MatMul(b, transposeB: true).ToArray()));
            float[] ta = Values(37 * 129, -1f, 1f), tb = Values(129 * 45, -1f, 1f), tc = Values(37 * 45, -1f, 1f);
            CpuMatMul.Multiply(ta, tb, tc, 37, 45, 129, false, false, 1f);
            outputs.Add(("tiled product, beta 1", tc));

            float[] big = Values(70_001, -1f, 1f);                            // past one reduction chunk
            outputs.Add(("sum past one chunk", [Tensor.From(big, [big.Length], device).Sum().Item()]));

            // Batch statistics over short runs (inner 2 < 16): three chunks of rows, their sums added in chunk order.
            const int Outer = 12_000, Groups = 7, Inner = 2;
            var batch = Tensor.From(Values(Outer * Groups * Inner, -3f, 3f), [Outer, Groups, Inner], device, requiresGrad: true);
            var normalized = batch.Normalize(Outer, Groups, Inner, 1e-5f, out var batchMean, out var batchVariance);
            outputs.Add(("batch norm", normalized.ToArray()));
            outputs.Add(("batch norm's mean and variance", [.. batchMean.ToArray(), .. batchVariance.ToArray()]));
            (normalized * Tensor.From(Values(Outer * Groups * Inner, -1f, 1f), [Outer, Groups, Inner], device)).Sum().Backward();
            outputs.Add(("batch norm's gradient", batch.Grad!.ToArray()));
        }

        // The host's double functions (PortableMath), as two floats a double.
        double[] arguments = [.. Enumerable.Range(0, 300).Select(i => -40.0 + i * 0.2771)];
        double[] portable = [.. arguments.Select(PortableMath.Sin), .. arguments.Select(PortableMath.Cos), .. arguments.Select(PortableMath.Tan),
            .. arguments.Select(PortableMath.Atan), .. arguments.Select(v => PortableMath.Atan2(v, 3.5 - v)), .. arguments.Select(v => PortableMath.Tanh(v / 8)),
            .. arguments.Select(v => PortableMath.Pow(Math.Abs(v) + 0.1, v / 7)), .. arguments.Select(v => PortableMath.Log2(Math.Abs(v) + 1e-3)),
            .. arguments.Select(v => PortableMath.Log10(Math.Abs(v) + 1e-3)), .. arguments.Select(v => PortableMath.Exp(v)), .. arguments.Select(v => PortableMath.Log(Math.Abs(v) + 1e-3))];
        outputs.Add(("portable double math", System.Runtime.InteropServices.MemoryMarshal.Cast<double, float>(portable.AsSpan()).ToArray()));

        float[] p = Values(Cols, -2f, 2f), q = Values(Cols, -2f, 2f);
        ushort[] halves = [.. q.Select(v => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16))];
        sbyte[] bytes = [.. q.Select(v => (sbyte)(v * 60))];
        outputs.Add(("lane sums", [CpuMath.Sum(p), CpuMath.Dot(p, q), CpuMath.Dot(p, halves), CpuMath.Dot(p, bytes),
            CpuMath.DotFused(ref p[0], ref q[0], Cols), CpuMath.DotFused(ref p[0], ref halves[0], Cols), CpuMath.SquaredDeviation(p, 0.25f),
            CpuMath.SumAndDot(p, q).Sum, CpuMath.SumAndDot(p, q).SumProduct, (float)CpuMath.SumSquaresDouble(p), (float)CpuMath.DotDouble(p, q)]));
        double[] doubles = [.. Enumerable.Range(0, 400).Select(i => -700.0 + i * 3.5123)];
        outputs.Add(("double exp and log", [.. doubles.Select(v => (float)CpuMath.ExpDouble(v / 4)), .. doubles.Select(v => (float)CpuMath.LogDouble(Math.Abs(v) + 1e-3))]));

        var weights = new float[Cols];
        float[]? candidateScores = null, candidateWeights = null;
        try
        {
            float total = Idrak.Abstraction.Generation.SamplingWeights.Compute(p, weights, 0.7f, 40, 0.9f, 0.05f, new float[40], ref candidateScores, ref candidateWeights);
            outputs.Add(("sampler weights", [.. weights, total]));
        }
        finally
        {
            foreach (var rented in new[] { candidateScores, candidateWeights })
            {
                if (rented is not null)
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(rented);
                }
            }
        }

        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        foreach (var (what, values) in outputs)
        {
            var bits = System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan());
            hash.AppendData(bits);
            Console.WriteLine($"    {what}: bits {Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bits))[..16]}");
        }

        string all = Convert.ToHexString(hash.GetHashAndReset())[..16];
        Console.WriteLine($"    all: bits {all} ({System.Numerics.Vector<float>.Count}-wide Vector<float>, accelerated {System.Numerics.Vector.IsHardwareAccelerated}, " +
                          $"512-bit {System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated}, IDRAK_CPU_VECTORS {ComputeResources.CpuVectors})");
        Check(all == CpuOneResultBits, $"the CPU's outputs hash to {all}, pinned {CpuOneResultBits}: the same bits are expected on every CPU and vector path");
    }

    internal static void CpuProductSplitK(Device device)
    {
        if (device.Backend is not CpuBackend || !FirstRun(nameof(CpuProductSplitK)))
        {
            return;
        }

        Check(ComputeResources.CpuProductKChunk == 0 || Environment.GetEnvironmentVariable("IDRAK_CPU_SPLIT_K") is { Length: > 0 }, "off unless IDRAK_CPU_SPLIT_K sets it");

        // The gate from the shape and the threads: a prompt's products fill any core count's threads; a token's do not.
        foreach (int threads in new[] { 2, 8, 64, 512 })
        {
            Check(!CpuMatMul.SplitsK(512, 1024, 1024, 256, threads) && !CpuMatMul.SplitsK(512, 1024, 3072, 1024, threads),
                $"a 512-token prompt's products are not split on {threads} threads");
            Check(!CpuMatMul.SplitsK(1, 1024, 3072, 0, threads) && !CpuMatMul.SplitsK(1, 1024, 1024, 1024, threads), $"off at chunk 0 and where k does not pass the chunk ({threads} threads)");
        }

        Check(CpuMatMul.SplitsK(1, 1024, 3072, 1024, 512) && !CpuMatMul.SplitsK(1, 1024, 3072, 1024, 1), "a token's product is split where its column blocks leave threads idle, never on one thread");
        var random = new Random(17);
        int chunkBefore = ComputeResources.CpuProductKChunk, threadsBefore = ComputeResources.MaxCpuThreads;
        try
        {
            ComputeResources.CpuProductKChunk = 256;
            ComputeResources.MaxCpuThreads = Environment.ProcessorCount;
            Check(!CpuMatMul.SplitsK(512, 1024, 1024), "512 x 1024 x 1024 in chunks of 256 is not split on this machine's threads");
            foreach (var (m, n, k, chunk) in new[] { (3, 5, 37, 8), (17, 33, 1000, 128), (64, 48, 4099, 1024), (9, 1, 300, 299) })
            {
                foreach (bool transA in new[] { false, true })
                {
                    foreach (bool transB in new[] { false, true })
                    {
                        foreach (float beta in new[] { 0f, 1f })
                        {
                            float[] a = [.. Enumerable.Range(0, m * k).Select(_ => random.NextSingle() - 0.5f)];
                            float[] b = [.. Enumerable.Range(0, k * n).Select(_ => random.NextSingle() - 0.5f)];
                            float[] c0 = [.. Enumerable.Range(0, m * n).Select(_ => random.NextSingle())];
                            float[] Run(int chunkLength, int threads)
                            {
                                ComputeResources.CpuProductKChunk = chunkLength;
                                ComputeResources.MaxCpuThreads = threads;
                                var c = (float[])c0.Clone();
                                CpuMatMul.Multiply(a, b, c, m, n, k, transA, transB, beta);
                                return c;
                            }

                            string what = $"{m}x{n}x{k} in chunks of {chunk}, transA {transA}, transB {transB}, beta {beta}";
                            var whole = Run(0, Environment.ProcessorCount);
                            var split = Run(chunk, Environment.ProcessorCount);
                            Check(Run(chunk, 1).SequenceEqual(whole), $"{what}: on one thread never split (the unsplit bits)");
                            if (!CpuMatMul.SplitsK(m, n, k, chunk, Environment.ProcessorCount))
                            {
                                Check(split.SequenceEqual(whole), $"{what}: its blocks occupy the {Environment.ProcessorCount} threads, so not split (the unsplit bits)");
                                continue;
                            }

                            // The order: each chunk's own product (a product of its k), added in chunk order, then + beta · C.
                            var expected = new float[m * n];
                            for (int k0 = 0; k0 < k; k0 += chunk)
                            {
                                int kc = Math.Min(chunk, k - k0);
                                // The chunk's slices in A's and B's own layouts ([m, kc] or [kc, m]; [kc, n] or [n, kc]).
                                float[] ac = new float[m * kc], bc = new float[kc * n], part = new float[m * n];
                                for (int i = 0; i < m; i++)
                                {
                                    for (int p = 0; p < kc; p++)
                                    {
                                        if (transA)
                                        {
                                            ac[p * m + i] = a[(k0 + p) * m + i];
                                        }
                                        else
                                        {
                                            ac[i * kc + p] = a[i * k + k0 + p];
                                        }
                                    }
                                }

                                for (int p = 0; p < kc; p++)
                                {
                                    for (int j = 0; j < n; j++)
                                    {
                                        if (transB)
                                        {
                                            bc[j * kc + p] = b[j * k + k0 + p];
                                        }
                                        else
                                        {
                                            bc[p * n + j] = b[(k0 + p) * n + j];
                                        }
                                    }
                                }

                                ComputeResources.CpuProductKChunk = 0;
                                CpuMatMul.Multiply(ac, bc, part, m, n, kc, transA, transB, 0f);
                                for (int i = 0; i < m * n; i++)
                                {
                                    expected[i] = k0 == 0 ? part[i] : expected[i] + part[i];
                                }
                            }

                            for (int i = 0; i < m * n; i++)
                            {
                                expected[i] = beta == 0f ? expected[i] : expected[i] + beta * c0[i];
                            }

                            Check(split.SequenceEqual(expected), $"{what}: the chunks' products added in chunk order");
                            AssertClose(whole, split, 1e-4f, $"{what}: close to the unsplit product");
                        }
                    }
                }
            }
        }
        finally
        {
            ComputeResources.CpuProductKChunk = chunkBefore;
            ComputeResources.MaxCpuThreads = threadsBefore;
        }
    }

    internal static void CpuTuningFormulas(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        var reference = CpuInfo.With(32 << 10, 1 << 20) with { VectorFloats = 8, VectorRegisters = 16, CacheLine = 64 };
        Check(CpuTuning.ParallelElementsFormula(reference) == 1 << 16, "element-wise cut-over 65,536 (the former ParallelThreshold)");
        Check(CpuTuning.ParallelFlopsFormula(reference) == 1L << 17, "product cut-over 131,072 (the former ParallelWork)");
        Check(CpuTuning.TileBytes(reference) == 256 << 10, "A tile 256 KiB (the former constant)");
        // Tile grids on 4 threads (AVX2 kernel 6 × 16): A tile 256 KiB, at most 16 row blocks (the former constants), the
        // at least two tiles per thread and whole rounds (a multiple of the threads).
        Check(CpuTuning.MaxRowBlocks(reference) == 16, "at most 16 row blocks per tile (the former constant)");
        Check(CpuTuning.TileGrid(reference, 512, 512, 512, 6, 16, 4) == (96, 256), "512³: 6 × 2 tiles of 96 × 256 (the former grid)");
        Check(CpuTuning.TileGrid(reference, 16, 4096, 1024, 6, 16, 4) == (60, 512), "16 × 4096 × 1024: one row tile, 8 column tiles");
        Check(CpuTuning.TileGrid(reference, 1024, 1024, 1024, 6, 16, 4) == (60, 512), "1024³: 18 × 2 tiles of 60 × 512 (36: whole rounds on 4 threads)");
        Check(CpuTuning.TileGrid(reference, 512, 2048, 128, 6, 16, 4) == (96, 1024), "512 × 2048 × 128: 6 × 2 tiles (the former grid)");
        Check(CpuTuning.TileGrid(reference, 6, 64, 64, 6, 16, 4).Columns == 32, "too few panels for whole rounds: two panels per tile");
        Check(CpuTuning.TileGrid(reference, 512, 512, 512, 6, 16, 1).Columns >= 2 * 16, "one thread: at least two panels per tile");
        var sixteen = reference with { L2 = 512 << 10, LogicalProcessors = 16 };
        var twentyFour = reference with { L1 = 48 << 10, L2 = 2 << 20, LogicalProcessors = 24 };
        Console.WriteLine($"    512³ grids: 16 threads / 512 KiB L2 {CpuTuning.TileGrid(sixteen, 512, 512, 512, 6, 16, 16)}, "
            + $"24 threads / 48 KiB L1, 2 MiB L2 {CpuTuning.TileGrid(twentyFour, 512, 512, 512, 6, 16, 24)}");
        Check(CpuTuning.TransposeTile(reference) == 32, "transpose tile 32 (the former constant)");
        Check(CpuTuning.MinColumns(reference) == 64, "column block 64 (the former SmallColumns, Int8MinBlock, chunk)");
        Check(CpuTuning.TransposedInPlace == 16 && CpuTuning.FewRows == 8 && CpuTuning.ReductionElements == 1 << 16
            && CpuTuning.ColumnSplitRows == 4 && CpuTuning.ColumnSplitColumns == 128 && CpuTuning.ColumnSplitWork == 1L << 17,
            "the numerical contract (which summation order a shape gets) is today's on every machine");
        Check(CpuTuning.RegisterRowsFormula(reference) == 4 && CpuTuning.RegisterRowsFormula(reference with { VectorRegisters = 32 }) == 8, "register rows from the register count");
        Check(CpuTuning.KChunkFormula(reference) == 8, "k chunk: half a 64-byte line of floats");

        // Other machines: the values follow the reported caches and widths.
        var smallL2 = reference with { L2 = 512 << 10 };
        Check(CpuTuning.ParallelElementsFormula(smallL2) == 1 << 15 && CpuTuning.TileBytes(smallL2) == 128 << 10, "512 KiB L2: half the cut-over and tile");
        var bigL1 = reference with { L1 = 128 << 10, L2 = 4 << 20 };
        Check(CpuTuning.TransposeTile(bigL1) == 64 && CpuTuning.TileBytes(bigL1) == 1 << 20, "128 KiB L1 / 4 MiB L2: transpose tile 64, A tile 1 MiB");
        var neon = reference with { VectorFloats = 4, VectorRegisters = 32 };
        Check(CpuTuning.MinColumns(neon) == 32 && CpuTuning.RegisterRowsFormula(neon) == 8, "4-wide vectors with 32 registers: column block 32, 8 rows in registers");
        Check(CpuTuning.KernelShape(TiledKernel.Avx512) == (8, 32) && CpuTuning.KernelShape(TiledKernel.Avx2) == (6, 16) && CpuTuning.KernelShape(TiledKernel.Neon) == (8, 8), "kernel shapes");
        Check(device.Backend.Capabilities.FewRows == CpuTuning.FewRows && device.Backend.Capabilities.TiledAttentionHeadDim == CpuTuning.TiledAttentionHeads
            && device.Backend.Capabilities.DecodeAttentionHeadDim == CpuTuning.DecodeAttentionHeads && CpuTuning.TiledAttentionHeads == 128 && CpuTuning.DecodeAttentionHeads == 256,
            "the CPU's capabilities are its own numerical contract (few-row split 8, attention kernels up to heads of 256 / 128 as before)");
        Console.WriteLine($"    {CpuTuning.Describe().Replace("\n", "\n    ")}");
    }

    internal static void CpuTunedChoicesMatch(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        var r = new Random(12);
        float[] Values(int n, bool zeros = false) => [.. Enumerable.Range(0, n).Select(i => zeros && i % 7 == 3 ? 0f : r.NextSingle() * 2 - 1)];
        var shapes = new (int M, int N, int K, bool TransA, bool TransB)[]
        {
            (1, 1000, 300, false, false), (3, 777, 129, false, false), (4, 512, 64, false, false), (5, 300, 70, false, false),
            (6, 257, 33, false, false), (7, 200, 100, false, false), (8, 640, 96, false, false), (13, 333, 77, true, false),
            (40, 130, 260, false, false), (96, 96, 96, false, false), (129, 70, 300, false, false), (64, 64, 1000, false, false),
            (2, 900, 128, false, true), (16, 300, 64, false, true), (17, 300, 64, true, true), (50, 129, 31, false, true),
        };
        var data = shapes.Select(s => (A: Values(s.M * s.K, zeros: true), B: Values(s.K * s.N), C: Values(s.M * s.N))).ToArray();
        var elements = Values(300_001);
        var other = Values(300_001);
        var weights = Values(320 * 1000);
        var rows = Values(9 * 320, zeros: true);
        using var weightTensor = Tensor.From(weights, [320, 1000], device);
        using var bf16 = BFloat16Weight.Convert(weightTensor);
        using var int8 = Int8Weight.Quantize(weightTensor);
        using var int4 = Int4Weight.Quantize(weightTensor);
        // A ragged shape for the packed few-row kernels: k past a 32-row scale group, columns past whole panels and words.
        var raggedWeights = Values(77 * 333);
        var raggedRows = Values(9 * 77, zeros: true);
        using var raggedTensor = Tensor.From(raggedWeights, [77, 333], device);
        using var raggedInt8 = Int8Weight.Quantize(raggedTensor);
        using var raggedInt4 = Int4Weight.Quantize(raggedTensor);

        float[][] Run()
        {
            var results = new List<float[]>();
            for (int i = 0; i < shapes.Length; i++)
            {
                var (m, n, k, ta, tb) = shapes[i];
                foreach (float beta in new[] { 0f, 0.5f })
                {
                    var c = (float[])data[i].C.Clone();
                    CpuMatMul.Multiply(data[i].A, data[i].B, c, m, n, k, ta, tb, beta);
                    results.Add(c);
                }
            }

            using (Autograd.NoGrad())
            {
                using var a = Tensor.From(elements, [elements.Length], device);
                using var b = Tensor.From(other, [other.Length], device);
                using var sum = a + b;
                using var product = sum * a;
                using var total = product.Sum();
                results.Add(product.ToArray());
                results.Add(total.ToArray());
                foreach (int m in new[] { 1, 2, 3, 4, 5, 7, 8, 9 })
                {
                    using var x = Tensor.From(rows.AsSpan(0, m * 320), [m, 320], device);
                    using var y16 = x.MatMulBFloat16(bf16);
                    using var y8 = x.MatMulInt8(int8);
                    using var y4 = x.MatMulInt4(int4);
                    results.Add(y16.ToArray());
                    results.Add(y8.ToArray());
                    results.Add(y4.ToArray());
                    using var ragged = Tensor.From(raggedRows.AsSpan(0, m * 77), [m, 77], device);
                    using var ragged8 = ragged.MatMulInt8(raggedInt8);
                    using var ragged4 = ragged.MatMulInt4(raggedInt4);
                    results.Add(ragged8.ToArray());
                    results.Add(ragged4.ToArray());
                }
            }

            return [.. results];
        }

        int totalIndex = 2 * shapes.Length + 1;                                // the Sum: chunks follow the thread count (as before)
        void Same(float[][] expected, float[][] actual, string what, bool sameThreads = true)
        {
            Check(expected.Length == actual.Length, $"{what}: {actual.Length} results, expected {expected.Length}");
            for (int i = 0; i < expected.Length; i++)
            {
                if (!sameThreads && i == totalIndex)
                {
                    // Other chunks add the 300,001 terms in another order: rounding differs by up to about √n float32
                    // steps (on a 6-core ARM64 phone 2.7e-6 of the total; on x64 machines it stayed below 1e-6).
                    AssertClose(expected[i], actual[i], MathF.Sqrt(elements.Length) * MathF.Pow(2, -23), $"{what}: the sum (summed in chunks per thread count, as before)");
                    continue;
                }

                for (int j = 0; j < expected[i].Length; j++)
                {
                    if (BitConverter.SingleToInt32Bits(expected[i][j]) != BitConverter.SingleToInt32Bits(actual[i][j]))
                    {
                        throw new Exception($"{what}: result {i}, element {j} is {actual[i][j]}, the fallback gives {expected[i][j]}");
                    }
                }
            }
        }

        bool autotune = CpuTuning.Autotune;
        var (elementsOverride, flopsOverride) = (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride);
        try
        {
            CpuTuning.Autotune = false;                                         // the fallback: formulas only
            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (null, null);
            var fallback = Run();

            CpuTuning.Autotune = true;                                          // measured (or read from the cache file)
            Same(fallback, Run(), "measured cut-overs");

            foreach (var (e, f) in new[] { (1, 1L), (int.MaxValue, long.MaxValue), (4096, 4096L) })
            {
                (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (e, f);
                Same(fallback, Run(), $"cut-overs {e} / {f}");
            }

            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (1, 1L);
            foreach (var kernel in Enum.GetValues<TiledKernel>())
            {
                CpuTuning.TiledKernelOverride = kernel;
                Same(fallback, Run(), $"tiled kernel {kernel}");
            }

            CpuTuning.TiledKernelOverride = null;
            foreach (int registerRows in new[] { 4, 8 })
            {
                foreach (int chunk in new[] { 1, 3, 16, 1000 })
                {
                    CpuTuning.RegisterRowsOverride = registerRows;
                    CpuTuning.KChunkOverride = chunk;
                    Same(fallback, Run(), $"{registerRows} register rows, k chunk {chunk}");
                }
            }

            var vectors = ComputeResources.CpuVectors;
            try
            {
                foreach (var paths in Enum.GetValues<CpuVectorPaths>())
                {
                    ComputeResources.CpuVectors = paths;
                    Same(fallback, Run(), $"vector paths {paths} (tiled kernel {CpuTuning.Kernel})");
                }
            }
            finally
            {
                ComputeResources.CpuVectors = vectors;
            }

            CpuTuning.WidePanelsOverride = !CpuTuning.WidePanels;
            Same(fallback, Run(), "512-bit panels toggled");
            CpuTuning.WidePanelsOverride = null;
            CpuTuning.StreamingFewRows = true;
            Same(fallback, Run(), "streaming few-row loop");
            CpuTuning.StreamingFewRows = false;

            int threads = ComputeResources.MaxCpuThreads;
            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (null, null);
            try
            {
                ComputeResources.MaxCpuThreads = 1;
                Same(fallback, Run(), "one thread", sameThreads: false);
            }
            finally
            {
                ComputeResources.MaxCpuThreads = threads;
            }
        }
        finally
        {
            CpuTuning.Autotune = autotune;
            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (elementsOverride, flopsOverride);
            CpuTuning.TiledKernelOverride = null;
            CpuTuning.RegisterRowsOverride = null;
            CpuTuning.KChunkOverride = null;
            CpuTuning.WidePanelsOverride = null;
            CpuTuning.StreamingFewRows = false;
        }
    }

    internal static void CpuTuningCache(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        Check(CpuInfo.ParseSize("32K") == 32 << 10 && CpuInfo.ParseSize("1024K") == 1 << 20 && CpuInfo.ParseSize("33 MiB") == 33L << 20
            && CpuInfo.ParseSize("") == 0 && CpuInfo.ParseSize("12Q") == 0, "cache sizes as Linux reports them");
        Check(CpuInfo.CountList("0-3,8,10-11") == 7 && CpuInfo.CountList("5") == 1, "processor lists as Linux reports them");
        var machine = CpuInfo.Current;
        Check(machine.LogicalProcessors == Environment.ProcessorCount && machine.PhysicalCores >= 1 && machine.PhysicalCores <= machine.LogicalProcessors
            && machine.VectorFloats == Vector<float>.Count && machine.L1 > 0 && machine.L2 > 0 && machine.CacheLine > 0, $"the machine's report: {machine}");

        var saved = new[] { "IDRAK_CACHE", "IDRAK_TUNING_CACHE", "IDRAK_CPU_TUNING_FILE" }.Select(v => (v, Environment.GetEnvironmentVariable(v))).ToArray();
        try
        {
            foreach (var (name, _) in saved)
            {
                Environment.SetEnvironmentVariable(name, null);
            }

            string root = Path.Combine(Path.GetTempPath(), "idrak-cache-root");
            Environment.SetEnvironmentVariable("IDRAK_CACHE", root);
            Check(CpuTuning.DefaultCacheFile() == Path.Combine(root, "cpu", "tuning.tsv"), "kept under IDRAK_CACHE/cpu");
            Environment.SetEnvironmentVariable("IDRAK_CACHE", null);
            Check(CpuTuning.DefaultCacheFile() == Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak", "cpu", "tuning.tsv"),
                "kept under ~/.cache/idrak/cpu by default");
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", "0");
            Check(CpuTuning.DefaultCacheFile() is null, "IDRAK_TUNING_CACHE=0: not kept");
        }
        finally
        {
            foreach (var (name, value) in saved)
            {
                Environment.SetEnvironmentVariable(name, value);
            }
        }

        string? file = CpuTuning.CacheFile;
        bool autotune = CpuTuning.Autotune;
        var (elementsOverride, flopsOverride) = (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride);
        string path = Path.Combine(Path.GetTempPath(), $"idrak-cpu-tuning-{Environment.ProcessId}.txt");
        try
        {
            CpuTuning.CacheFile = path;
            CpuTuning.Autotune = true;
            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (null, null);
            File.Delete(path);
            CpuTuning.Forget();
            var cut = CpuTuning.CurrentCutovers;
            int measured = cut.Elements;
            long measuredFlops = cut.Flops;
            string key = CpuTuning.CacheKey(ComputeResources.MaxCpuThreads);
            Console.WriteLine($"    cut-overs {measured:N0} / {measuredFlops:N0} ({cut.Source})");
            Check(measured == CpuTuning.ParallelElements && measuredFlops == CpuTuning.ParallelFlops, "the cut-overs in use are the ones measured");
            if (ComputeResources.AllowParallel && cut.Source == "measured")
            {
                Check(File.Exists(path) && File.ReadAllLines(path).Any(l => l.StartsWith(key + "\t", StringComparison.Ordinal)), "the measurement is kept in the cache file");
                Check(CpuTuning.Load(key) == new CpuTuning.Cutovers(measured, measuredFlops, "measured earlier on this machine"), "read back as measured");
            }

            File.AppendAllLines(path, ["garbage", "v1|other machine\t1\t2", key + "\tnot-a-number\t5"]);
            CpuTuning.Save(key, new CpuTuning.Cutovers(12_345, 67_890, "measured"));
            Check(File.ReadAllLines(path).Count(l => l.StartsWith(key + "\t", StringComparison.Ordinal)) == 1, "one line per key (the older one replaced)");
            Check(CpuTuning.Load(key) is { Elements: 12_345, Flops: 67_890 }, "the replaced value is read");
            CpuTuning.Forget();
            if (ComputeResources.AllowParallel)
            {
                Check(CpuTuning.ParallelElements == 12_345 && CpuTuning.ParallelFlops == 67_890, "a new process (forgotten measurements) uses the kept values without measuring");
            }

            CpuTuning.CacheFile = null;                                         // IDRAK_TUNING_CACHE=0
            Check(CpuTuning.Load(key) is null, "no cache file: nothing kept");
            CpuTuning.Save(key, new CpuTuning.Cutovers(1, 1, "measured"));
            CpuTuning.CacheFile = path;
            Check(CpuTuning.Load(key) is { Elements: 12_345 }, "persistence off: nothing written");
        }
        finally
        {
            CpuTuning.CacheFile = file;
            CpuTuning.Autotune = autotune;
            (CpuTuning.ParallelElementsOverride, CpuTuning.ParallelFlopsOverride) = (elementsOverride, flopsOverride);
            CpuTuning.Forget();
            File.Delete(path);
        }
    }
}
