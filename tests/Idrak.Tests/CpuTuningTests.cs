// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Layers;

// The CPU's tiling, blocking and threading choices come from what the machine reports (cores, vector width and
// registers, caches) or are measured on it; none of them may change a result, since the CPU is the reference.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CpuTuningGroup =
    [
        ("cpu tuning: the formulas give the former constants on a 32 KiB L1 / 1 MiB L2 / 8-wide machine and scale with the reported caches and width", CpuTuningFormulas),
        ("cpu tuning: measured cut-overs, every tiled kernel, register rows, k chunks and the streaming loop give the results of the fallback formulas, bit for bit", CpuTunedChoicesMatch),
        ("cpu tuning: measured cut-overs are kept per machine, runtime and thread count and read back; the machine's report parses", CpuTuningCache),
    ];

    private static void CpuTuningFormulas(Device device)
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

    private static void CpuTunedChoicesMatch(Device device)
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

    private static void CpuTuningCache(Device device)
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
