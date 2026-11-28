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
        ("cpu tuning: the formulas (the element-wise cut-over, the transpose tile) give the former constants on a 32 KiB L1 / 1 MiB L2 / 8-wide machine and scale with the reported caches", CpuTuningFormulas),
        ("cpu tuning: cut-overs, every vector path setting and one thread give the results of the fallback formulas, bit for bit", CpuTunedChoicesMatch),
        ("cpu tuning: the machine's report parses", CpuTuningCache),
        ("cpu tuning: the CPU's memory is the system's (/proc/meminfo, the lowest cgroup v1 or v2 limit above the process with its usage less inactive file cache) where the runtime's figure is its heap budget (Mono), held to the runtime's heap limit where it has one (CoreCLR: unchanged), the runtime's where the system reports none; both listed, labelled", CpuSystemMemory),
        ("cpu tuning: the cores are named by their groups from /proc/cpuinfo and cpufreq (part numbers and highest frequencies, raw; the system's name where given; x86's model name as before)", CpuCoreGroups),
        ("cpu tuning: the measured choices are timed alike (every candidate warmed once, then the same rounds in the same order) and kept per machine (a file of its own, its identity the processor, threads and instruction sets) under the vector paths setting: a scalar choice never taken under auto, nor auto's under scalar; IDRAK_TUNING_CACHE=0 keeps none", CpuChoicesKept),
        ("cpu tuning: the vector paths (IDRAK_CPU_VECTORS auto, native, portable, scalar) parse; portable 128-bit vectors where the runtime reports no NEON; the description says what runs", CpuVectorPathsChosen),
        ("cpu tuning: one result on every CPU (plan 37, P0 and P0b): exp, log, the sigmoids, GELU, softmax, log-softmax, sin, cos and pow with their gradients, layer norm, batch statistics over short runs in chunks, the gated activations, few-row products, the sums past one chunk, the sampler's weights and the host's double functions (PortableMath) hash to the pinned bits (the same with DOTNET_EnableAVX512=0, DOTNET_EnableAVX2=0, DOTNET_EnableHWIntrinsic=0 and IDRAK_CPU_VECTORS=scalar, portable or native)", CpuOneResult),
        ("cpu tuning: the portable math (plan 37, P0b): the CPU's float sin and cos lanes give their scalar form's bits and are within an ulp of MathF's over [-1e4, 1e4]; PortableMath's sin, cos, tan, atan, atan2, tanh, pow, log2 and log10 are within a few ulp of Math's, with its special cases; integer powers, log2 of powers of two and log10 of powers of ten exact", CpuPortableMath),
    ];

    // RoundTimes' runs counted per candidate.
    private struct CountedRuns : CpuChoices.ICandidates
    {
        public int[] Runs;

        public readonly void Run(int index) => Runs[index]++;
    }

    internal static void CpuChoicesKept(Device device)
    {
        if (device.Backend is not CpuBackend || !FirstRun(nameof(CpuChoicesKept)))
        {
            return;
        }

        var counted = new CountedRuns { Runs = new int[3] };
        var times = CpuChoices.RoundTimes(ref counted, 3);
        Check(times.Length == 3 && counted.Runs[0] == counted.Runs[1] && counted.Runs[1] == counted.Runs[2] && counted.Runs[0] >= 10,
            $"every candidate run alike: {string.Join(", ", counted.Runs)} runs");

        string folder = Path.Combine(Path.GetTempPath(), "idrak-cpu-choices-" + Guid.NewGuid().ToString("N"));
        string? setting = Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE");
        var vectors = ComputeResources.CpuVectors;
        bool autotune = CpuTuning.Autotune;
        try
        {
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", folder);
            CpuTuning.Autotune = true;
            string file = CpuChoices.FilePath ?? throw new Exception("no file under a folder setting");
            Check(file.StartsWith(Path.Combine(folder, "cpu"), StringComparison.Ordinal), $"the machine's file {file} lies in the folder set");
            string[] Lines() => File.Exists(file) ? File.ReadAllLines(file) : [];
            void Measure(CpuVectorPaths paths)
            {
                ComputeResources.CpuVectors = paths;
                CpuBackend.ForgetChoices();
                using var x = Tensor.From([.. Enumerable.Range(0, 3001).Select(i => i * 0.001f)], [3001], device);
                using var y = x.Exp();
                GC.KeepAlive(y.ToArray());
            }

            Measure(CpuVectorPaths.Automatic);
            var lines = Lines();
            Check(lines.Length >= 3 && lines[1] == CpuChoices.Identity, $"the file names the machine ({CpuChoices.Identity}): {string.Join(" | ", lines.Take(2))}");
            Check(lines.Any(l => l.StartsWith($"lanes Unary Exp 2^12x2^0 Automatic t{ComputeResources.MaxCpuThreads}\t", StringComparison.Ordinal)), "auto's choice kept under auto");
            Measure(CpuVectorPaths.Scalar);
            lines = Lines();
            Check(ProductRoutes.Choices.TryGetValue("cpu: Unary Exp 2^12x2^0", out var chosen) && chosen.StartsWith("one float at a time (the only candidate)", StringComparison.Ordinal),
                $"under scalar one float at a time, the only candidate (nothing measured): {chosen}");
            Check(!lines.Any(l => l.Contains(" Scalar ", StringComparison.Ordinal) && l.Contains("Exp", StringComparison.Ordinal)), "nothing measured under scalar is kept");
            Check(lines.Any(l => l.StartsWith($"lanes Unary Exp 2^12x2^0 Automatic t{ComputeResources.MaxCpuThreads}\t", StringComparison.Ordinal)), "auto's choice still there");

            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", "0");
            Check(CpuChoices.FilePath is null, "IDRAK_TUNING_CACHE=0 keeps no file");
        }
        finally
        {
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", setting);
            ComputeResources.CpuVectors = vectors;
            CpuTuning.Autotune = autotune;
            CpuBackend.ForgetChoices();
            try
            {
                Directory.Delete(folder, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }

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
    private const string CpuOneResultBits = "21953D092DF51A40";

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

            // Few rows against B stored [n, k] (the lane sums).
            var a = Tensor.From(Values(2 * Cols, -1f, 1f), [2, Cols], device);
            var b = Tensor.From(Values(67 * Cols, -1f, 1f), [67, Cols], device);
            outputs.Add(("few-row product", a.MatMul(b, transposeB: true).ToArray()));

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

        float[] p = Values(Cols, -2f, 2f);
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

    internal static void CpuTuningFormulas(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        var reference = CpuInfo.With(32 << 10, 1 << 20) with { VectorFloats = 8, VectorRegisters = 16, CacheLine = 64 };
        Check(CpuTuning.ParallelElementsFormula(reference) == 1 << 16, "element-wise cut-over 65,536 (the former ParallelThreshold)");
        Check(CpuTuning.TransposeTile(reference) == 32, "transpose tile 32 (the former constant)");

        // Other machines: the values follow the reported caches.
        var smallL2 = reference with { L2 = 512 << 10 };
        Check(CpuTuning.ParallelElementsFormula(smallL2) == 1 << 15, "512 KiB L2: half the cut-over");
        var bigL1 = reference with { L1 = 128 << 10, L2 = 4 << 20 };
        Check(CpuTuning.TransposeTile(bigL1) == 64, "128 KiB L1 / 4 MiB L2: transpose tile 64");
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
        var elements = Values(300_001);
        var other = Values(300_001);

        float[][] Run()
        {
            var results = new List<float[]>();
            using (Autograd.NoGrad())
            {
                using var a = Tensor.From(elements, [elements.Length], device);
                using var b = Tensor.From(other, [other.Length], device);
                using var sum = a + b;
                using var product = sum * a;
                using var total = product.Sum();
                results.Add(product.ToArray());
                results.Add(total.ToArray());
            }

            return [.. results];
        }

        int totalIndex = 1;                                                   // the Sum: chunks follow the thread count (as before)
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
        var elementsOverride = CpuTuning.ParallelElementsOverride;
        try
        {
            CpuTuning.Autotune = false;                                         // the fallback: formulas only
            CpuTuning.ParallelElementsOverride = null;
            var fallback = Run();

            CpuTuning.Autotune = true;
            Same(fallback, Run(), "measured cut-overs");

            foreach (int e in new[] { 1, int.MaxValue, 4096 })
            {
                CpuTuning.ParallelElementsOverride = e;
                Same(fallback, Run(), $"cut-over {e}");
            }

            CpuTuning.ParallelElementsOverride = 1;
            var vectors = ComputeResources.CpuVectors;
            try
            {
                foreach (var paths in Enum.GetValues<CpuVectorPaths>())
                {
                    ComputeResources.CpuVectors = paths;
                    Same(fallback, Run(), $"vector paths {paths}");
                }
            }
            finally
            {
                ComputeResources.CpuVectors = vectors;
            }

            int threads = ComputeResources.MaxCpuThreads;
            CpuTuning.ParallelElementsOverride = null;
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
            CpuTuning.ParallelElementsOverride = elementsOverride;
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
    }
}
