// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's rows kernels in GCN (GcnKernels.Rows), each run by the GCN interpreter and proven against the CPU (plan 35, G3b).
// Every kernel is assembled for gfx90c and gfx908, run over rows of 1 column, fewer than a wave, between a wave and the
// block, more than the block and 2,333 (a multiple of nothing), several rows a run, with special values; compared to
// the CPU backend within the device-against-CPU tolerance (OnBoth's); run again on another grid and another wave order
// to give the same bits (each row's sums in a fixed tree); and the interpreter reports no missing wait, race or
// access out of range (a report fails the run). With IDRAK_GCN_PRINTED naming a folder, each kernel's printed text
// (physical registers, LLVM's syntax) is written there for the development check against llvm-mc.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnRowsGroup =
    [
        ("gcn kernels: rows: every row kernel assembles for gfx90c and gfx908 with its block of 256 declared, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnRowsAssembled),
        ("gcn kernels: rows: rms_norm_f32 (y and inv) and rms_norm_affine_f32 equal the CPU's over rows of 1, 37, 200, 1,000 and 2,333 columns (large magnitudes in a row), the same bits on two grids and wave orders", GcnRowsRmsNorm),
        ("gcn kernels: rows: layernorm_fused_f32 and layernorm_train_f32 (y and the stats) equal the CPU's over rows of 1 to 2,333 columns (large magnitudes in a row), the same bits on two grids and wave orders", GcnRowsLayerNorm),
        ("gcn kernels: rows: layernorm_bwd_f32 (dx added to), layernorm_bwd_params_f32 with one chunk (into dgamma and dbeta, a null one skipped) and with three (partials folded by fold_chunks_f32) equal the CPU's gradients, the same bits on two grids and wave orders", GcnRowsLayerNormBackward),
        ("gcn kernels: rows: softmax_f32 and log-softmax (in place too) equal the CPU's over rows of 1 to 2,333 columns, with large magnitudes, -inf entries and an all -inf row (NaN on both), the same bits on two grids and wave orders", GcnRowsSoftmax),
        ("gcn kernels: rows: scale_mask_softmax_f32 equals the CPU's with a mask of 3 rows (-inf entries, an all-masked row: NaN on both; the mask row stepped with the grid's stride) and without a mask, the same bits on two grids and wave orders", GcnRowsScaleMaskSoftmax),
        ("gcn kernels: rows: log_softmax_topk_f32 equals the CPU's picks (values and columns; ties to the smaller column; -inf and NaN never picked; -inf and -1 once none is left, k past the row too), the same bits on two grids and wave orders", GcnRowsLogSoftmaxTopK),
    ];

    private static readonly GcnIsaVersion[] RowsIsas = [GcnIsaVersion.Parse("gfx90c"), GcnIsaVersion.Parse("gfx908")];

    private static readonly int[] RowsLengths = [1, 37, 200, 1000, 2333];

    // OnBoth's device-against-CPU tolerance (relative, at least 1 in magnitude).
    private const float RowsTolerance = 1e-3f;

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> rowKernels = [];

    // The kernel assembled for isa, written into a code object and read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) RowKernel(string name, GcnIsaVersion isa)
    {
        lock (rowKernels)
        {
            if (rowKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = GcnKernels.Rows.Single(k => k.Name == name);
            var target = GcnTargets.For(isa);
            using var asm = new GcnAssembler(target, isa);
            var code = asm.AssembleKernel(source.Text);
            string printed = asm.PrintedText;
            byte[] again = AssembleGcn(target, isa, printed);
            Check(again.AsSpan().SequenceEqual(code.Code.Span), $"{name} ({isa}): its printed text reads back to other bytes:\n{printed}");
            if (Environment.GetEnvironmentVariable("IDRAK_GCN_PRINTED") is { Length: > 0 } folder)
            {
                Directory.CreateDirectory(folder);
                File.WriteAllText(Path.Combine(folder, $"{name}.{isa}.s"), printed);
                File.WriteAllBytes(Path.Combine(folder, $"{name}.{isa}.bin"), code.Code.ToArray());
            }

            var image = GcnCodeObjectReader.Read(GcnCodeObject.Write(target, isa, 5, [code]), name);
            return rowKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch of a row kernel: its arguments as 64-bit values (addresses, 32-bit values, float bits) in order.
    private static void RowDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y) grid, ulong seed, params ulong[] values)
    {
        var (code, image) = RowKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = (grid.X, grid.Y, 1), WorkgroupSize = (GcnKernels.RowThreads, 1, 1), Arguments = arguments, Seed = seed,
        });
    }

    private static ulong Bits(float value) => BitConverter.SingleToUInt32Bits(value);

    // The buffers on the CPU backend after op.
    private static float[][] RowsOnCpu(float[][] inputs, Action<Backend, Storage[]> op)
    {
        var cpu = CpuBackend.Instance;
        var storages = inputs.Select(d => { var s = cpu.Allocate(Math.Max(1, d.Length), false); cpu.Upload(d, s); return s; }).ToArray();
        try
        {
            op(cpu, storages);
            return [.. storages.Select((s, i) => { var r = new float[inputs[i].Length]; cpu.Download(s, r); return r; })];
        }
        finally
        {
            foreach (var s in storages)
            {
                s.Release();
            }
        }
    }

    // The buffers mapped into a fresh memory, run by `run` (given the memory and the buffers' addresses), read back.
    private static float[][] RowsOnGcn(float[][] inputs, Action<GcnMemory, ulong[]> run)
    {
        var memory = new GcnMemory();
        ulong[] addresses = [.. inputs.Select((d, i) => memory.Map<float>($"buffer {i}", d.Length == 0 ? [0f] : d))];
        run(memory, addresses);
        return [.. inputs.Select((d, i) => memory.Read<float>(addresses[i], d.Length))];
    }

    // The largest error per storage against the CPU's (relative, at least 1 in magnitude): NaN only where the CPU has NaN,
    // an infinity only where it has the same; every other element within RowsTolerance.
    private static float RowsCompare(float[] expected, float[] actual, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values, the CPU's {expected.Length}");
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            if (float.IsNaN(e) || float.IsNaN(a) || float.IsInfinity(e) || float.IsInfinity(a))
            {
                Check(float.IsNaN(e) ? float.IsNaN(a) : e == a, $"{what}: element {i} is {a:R}, the CPU's {e:R}");
                continue;
            }

            float error = MathF.Abs(e - a) / MathF.Max(1f, MathF.Abs(e));
            Check(error <= RowsTolerance, $"{what}: element {i} is {a:R}, the CPU's {e:R} (off by {error:G3} of it)");
            worst = MathF.Max(worst, error);
        }

        return worst;
    }

    private static void RowsSameBits(float[][] first, float[][] second, string what)
    {
        for (int b = 0; b < first.Length; b++)
        {
            Check(MemoryMarshal.AsBytes(first[b].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(second[b].AsSpan())),
                $"{what}: storage {b} differs between two grids and wave orders (the sums' order must be fixed)");
        }
    }

    // Runs `gcn` twice (grid of min(rows, 2) blocks with seed 1, then `rows` blocks with seed 7: another stride and wave
    // order), checks the same bits, compares the outputs named by `outputs` with the CPU's, and keeps the worst error.
    private static void RowsCase(string what, float[][] inputs, int rows, int[] outputs, Action<Backend, Storage[]> cpu,
        Action<GcnMemory, ulong[], (int X, int Y), ulong> gcn, Dictionary<string, float> worst, string kernel)
    {
        float[][] expected = RowsOnCpu(inputs, cpu);
        float[][] first = RowsOnGcn(inputs, (m, a) => gcn(m, a, (Math.Min(rows, 2), 1), 1));
        float[][] second = RowsOnGcn(inputs, (m, a) => gcn(m, a, (rows, 1), 7));
        RowsSameBits(first, second, what);
        foreach (int o in outputs)
        {
            float error = RowsCompare(expected[o], first[o], $"{what}, storage {o}");
            worst[kernel] = MathF.Max(worst.GetValueOrDefault(kernel), error);
        }
    }

    private static void RowsReport(Dictionary<string, float> worst)
    {
        foreach (var (kernel, error) in worst)
        {
            Console.WriteLine($"    {kernel}: largest error against the CPU {error:G3}");
        }
    }

    private static float[] RowsRandom(Random random, int n, float scale, float shift = 0f)
    {
        var values = new float[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1) * scale + shift;
        }

        return values;
    }

    internal static void GcnRowsAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Rows.Select(k => k.Name).ToArray();
        Check(names.Length == 10 && names.Distinct().Count() == names.Length, $"10 row kernels, each once: {string.Join(", ", names)}");
        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(GcnKernels.Rows.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            foreach (var isa in RowsIsas)
            {
                var (code, _) = RowKernel(name, isa);
                Check(code is { RequiredWorkgroupSize: (GcnKernels.RowThreads, 1, 1), MaxFlatWorkgroupSize: GcnKernels.RowThreads, WaveSize: 64 },
                    $"{name} ({isa}): declares the block of {GcnKernels.RowThreads} (required {code.RequiredWorkgroupSize}, max flat {code.MaxFlatWorkgroupSize}, wave {code.WaveSize})");
                Check(code.LdsBytes <= 4096, $"{name} ({isa}): {code.LdsBytes} bytes of LDS, at most two areas of 2,048");
                Console.WriteLine($"    {name} ({isa}): {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
            }
        }
    }

    internal static void GcnRowsRmsNorm(Device device)
    {
        _ = device;
        const int Rows = 3;
        const float Eps = 1e-6f, Offset = 0.5f;
        var worst = new Dictionary<string, float>();
        var random = new Random(31);
        foreach (int cols in RowsLengths)
        {
            float[] x = RowsRandom(random, Rows * cols, 3f);
            for (int j = 0; j < cols; j++)
            {
                x[cols + j] *= 3e16f;                                       // large magnitudes: x² near 1e34
            }

            float[] gain = RowsRandom(random, cols, 1f);
            foreach (var isa in RowsIsas)
            {
                RowsCase($"rms_norm_f32 ({isa}), {Rows} x {cols}", [x, new float[Rows * cols], new float[Rows]], Rows, [1, 2],
                    (b, s) => b.RmsNormKernel(s[0], s[1], s[2], Rows, cols, Eps),
                    (m, a, grid, seed) => RowDispatch("rms_norm_f32", isa, m, grid, seed, a[0], a[1], a[2], (ulong)cols, Bits(Eps), Rows),
                    worst, "rms_norm_f32");
                RowsCase($"rms_norm_affine_f32 ({isa}), {Rows} x {cols}", [x, gain, new float[Rows * cols]], Rows, [2],
                    (b, s) => b.RmsNormAffineKernel(s[0], s[1], s[2], Rows, cols, Eps, Offset),
                    (m, a, grid, seed) => RowDispatch("rms_norm_affine_f32", isa, m, grid, seed, a[0], a[1], a[2], (ulong)cols, Bits(Eps), Bits(Offset), Rows),
                    worst, "rms_norm_affine_f32");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnRowsLayerNorm(Device device)
    {
        _ = device;
        const int Rows = 3;
        const float Eps = 1e-5f;
        var worst = new Dictionary<string, float>();
        var random = new Random(32);
        foreach (int cols in RowsLengths)
        {
            float[] x = RowsRandom(random, Rows * cols, 2f, 0.5f);
            for (int j = 0; j < cols; j++)
            {
                x[2 * cols + j] = x[2 * cols + j] * 1e6f + 3e6f;            // large magnitudes with a spread to match
            }

            float[] gamma = RowsRandom(random, cols, 1f, 1f), beta = RowsRandom(random, cols, 0.5f);
            foreach (var isa in RowsIsas)
            {
                RowsCase($"layernorm_fused_f32 ({isa}), {Rows} x {cols}", [x, gamma, beta, new float[Rows * cols]], Rows, [3],
                    (b, s) => b.LayerNormFusedKernel(s[0], s[1], s[2], s[3], Rows, cols, Eps),
                    (m, a, grid, seed) => RowDispatch("layernorm_fused_f32", isa, m, grid, seed, a[0], a[1], a[2], a[3], (ulong)cols, Bits(Eps), Rows),
                    worst, "layernorm_fused_f32");
                RowsCase($"layernorm_train_f32 ({isa}), {Rows} x {cols}", [x, gamma, beta, new float[Rows * cols], new float[2 * Rows]], Rows, [3, 4],
                    (b, s) => b.LayerNormTrainKernel(s[0], s[1], s[2], s[3], s[4], Rows, cols, Eps),
                    (m, a, grid, seed) => RowDispatch("layernorm_train_f32", isa, m, grid, seed, a[0], a[1], a[2], a[3], a[4], (ulong)cols, Bits(Eps), Rows),
                    worst, "layernorm_train_f32");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnRowsLayerNormBackward(Device device)
    {
        _ = device;
        const float Eps = 1e-5f;
        const int Chunk = 64;                                               // CUDA's LayerNormBackwardKernel's rows a chunk
        var worst = new Dictionary<string, float>();
        var random = new Random(33);
        foreach (int cols in RowsLengths)
        {
            foreach (int rows in (int[])[3, 40, 150])
            {
                float[] x = RowsRandom(random, rows * cols, 2f, 0.25f), gamma = RowsRandom(random, cols, 1f, 1f), dy = RowsRandom(random, rows * cols, 1f);
                float[] stats = RowsOnCpu([x, gamma, new float[cols], new float[rows * cols], new float[2 * rows]],
                    (b, s) => b.LayerNormTrainKernel(s[0], s[1], s[2], s[3], s[4], rows, cols, Eps))[4];
                float[] dx = RowsRandom(random, rows * cols, 0.5f), dgamma = RowsRandom(random, cols, 0.5f), dbeta = RowsRandom(random, cols, 0.5f);
                foreach (var isa in RowsIsas)
                {
                    if (rows == 3)
                    {
                        RowsCase($"layernorm_bwd_f32 ({isa}), {rows} x {cols}", [x, gamma, dy, stats, dx], rows, [4],
                            (b, s) => b.LayerNormBackwardKernel(s[0], s[1], s[2], s[3], s[4], null, null, rows, cols),
                            (m, a, grid, seed) => RowDispatch("layernorm_bwd_f32", isa, m, grid, seed, a[0], a[1], a[2], a[3], a[4], (ulong)cols, (ulong)rows),
                            worst, "layernorm_bwd_f32");
                        continue;
                    }

                    // The parameters' gradients as CUDA's LayerNormBackwardKernel launches them: chunks of 64 rows, partials
                    // folded in chunk order when there are several. With one chunk, also dgamma alone (dbeta's pointer 0).
                    int chunks = (rows + Chunk - 1) / Chunk, blocks = (cols + GcnKernels.RowThreads - 1) / GcnKernels.RowThreads;
                    var partials = new float[2 * chunks * cols];
                    void Gradients(GcnMemory m, ulong[] a, (int X, int Y) grid, ulong seed, bool both)
                    {
                        // Fewer column blocks than columns on the first grid (a grid-stride loop), every one on the second.
                        (int X, int Y) shape = (grid.X == blocks ? blocks : Math.Max(1, Math.Min(grid.X, blocks - 1)), chunks);
                        RowDispatch("layernorm_bwd_params_f32", isa, m, shape, seed, a[0], a[1], a[2], a[3], both ? a[4] : 0, a[5], (ulong)rows, (ulong)cols, Chunk);
                        if (chunks > 1)
                        {
                            (int X, int Y) fold = (grid.X == blocks ? blocks : 1, 1);
                            RowDispatch("fold_chunks_f32", isa, m, fold, seed, a[5], a[3], (ulong)cols, (ulong)chunks, (ulong)cols);
                            RowDispatch("fold_chunks_f32", isa, m, fold, seed, a[5] + (ulong)(4L * chunks * cols), a[4], (ulong)cols, (ulong)chunks, (ulong)cols);
                        }
                    }

                    string what = $"layernorm_bwd_params_f32{(chunks > 1 ? " and fold_chunks_f32" : "")} ({isa}), {rows} x {cols}, {chunks} chunks";
                    RowsCase(what, [x, dy, stats, dgamma, dbeta, partials], blocks, [3, 4],
                        (b, s) => b.LayerNormBackwardKernel(s[0], s[0], s[1], s[2], null, s[3], s[4], rows, cols),   // gamma: unused for these
                        (m, a, grid, seed) => Gradients(m, a, grid, seed, both: true),
                        worst, chunks > 1 ? "layernorm_bwd_params_f32 + fold_chunks_f32" : "layernorm_bwd_params_f32");
                    if (chunks == 1)
                    {
                        RowsCase(what + ", dgamma alone", [x, dy, stats, dgamma, dbeta, partials], blocks, [3, 4],
                            (b, s) => b.LayerNormBackwardKernel(s[0], s[0], s[1], s[2], null, s[3], null, rows, cols),
                            (m, a, grid, seed) => Gradients(m, a, grid, seed, both: false),
                            worst, "layernorm_bwd_params_f32");
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnRowsSoftmax(Device device)
    {
        _ = device;
        const int Rows = 4;
        var worst = new Dictionary<string, float>();
        var random = new Random(34);
        foreach (int cols in RowsLengths)
        {
            float[] x = RowsRandom(random, Rows * cols, 4f);
            for (int j = 0; j < cols; j++)
            {
                x[cols + j] *= 7500f;                                        // ±3e4: most exponentials underflow
                if (j % 3 == 1)
                {
                    x[2 * cols + j] = float.NegativeInfinity;
                }

                x[3 * cols + j] = float.NegativeInfinity;                    // all -inf: NaN on the CPU and here
            }

            foreach (var isa in RowsIsas)
            {
                foreach (bool log in (bool[])[false, true])
                {
                    string what = $"softmax_f32{(log ? " (log)" : "")} ({isa}), {Rows} x {cols}";
                    RowsCase(what, [x, new float[Rows * cols]], Rows, [1],
                        (b, s) => b.SoftmaxKernel(s[0], s[1], Rows, cols, log),
                        (m, a, grid, seed) => RowDispatch("softmax_f32", isa, m, grid, seed, a[0], a[1], (ulong)cols, log ? 1UL : 0UL, Rows),
                        worst, log ? "softmax_f32 (log)" : "softmax_f32");
                }

                RowsCase($"softmax_f32 (log, in place) ({isa}), {Rows} x {cols}", [x], Rows, [0],
                    (b, s) => b.SoftmaxKernel(s[0], s[0], Rows, cols, true),
                    (m, a, grid, seed) => RowDispatch("softmax_f32", isa, m, grid, seed, a[0], a[0], (ulong)cols, 1UL, Rows),
                    worst, "softmax_f32 (log)");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnRowsScaleMaskSoftmax(Device device)
    {
        _ = device;
        const int Rows = 7, MaskRows = 3;
        const float Scale = 0.125f;
        var worst = new Dictionary<string, float>();
        var random = new Random(35);
        foreach (int cols in RowsLengths)
        {
            float[] x = RowsRandom(random, Rows * cols, 8f);
            for (int j = 0; j < cols; j++)
            {
                x[2 * cols + j] *= 4000f;                                    // large magnitudes
            }

            float[] mask = RowsRandom(random, MaskRows * cols, 1f);
            for (int j = 0; j < cols; j++)
            {
                if (j % 4 == 2)
                {
                    mask[j] = float.NegativeInfinity;
                }

                mask[cols + j] = float.NegativeInfinity;                     // mask row 1: rows 1 and 4 all masked (NaN)
            }

            foreach (var isa in RowsIsas)
            {
                RowsCase($"scale_mask_softmax_f32 ({isa}), {Rows} x {cols}, a mask of {MaskRows} rows", [x, mask, new float[Rows * cols]], Rows, [2],
                    (b, s) => b.ScaleMaskSoftmaxKernel(s[0], s[1], s[2], Rows, cols, MaskRows, Scale),
                    (m, a, grid, seed) => RowDispatch("scale_mask_softmax_f32", isa, m, grid, seed, a[0], a[1], a[2], (ulong)cols, MaskRows, Bits(Scale), 1UL, Rows),
                    worst, "scale_mask_softmax_f32");
                RowsCase($"scale_mask_softmax_f32 ({isa}), {Rows} x {cols}, no mask", [x, new float[Rows * cols]], Rows, [1],
                    (b, s) => b.ScaleMaskSoftmaxKernel(s[0], null, s[1], Rows, cols, 1, Scale),
                    (m, a, grid, seed) => RowDispatch("scale_mask_softmax_f32", isa, m, grid, seed, a[0], a[0], a[1], (ulong)cols, 1UL, Bits(Scale), 0UL, Rows),
                    worst, "scale_mask_softmax_f32");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnRowsLogSoftmaxTopK(Device device)
    {
        _ = device;
        const int Rows = 6, K = 5;
        var worst = new Dictionary<string, float>();
        var random = new Random(36);
        foreach (int cols in RowsLengths)
        {
            float[] x = RowsRandom(random, Rows * cols, 3f);
            for (int j = 0; j < cols; j++)
            {
                x[cols + j] = MathF.Round(x[cols + j] * 2f) / 2f;           // ties: equal values go to the smaller column first
                if (j % 5 == 3)
                {
                    x[2 * cols + j] = float.NegativeInfinity;                // never picked
                }

                x[4 * cols + j] = float.NegativeInfinity;                    // all -inf: no candidate
                x[5 * cols + j] *= 1e4f;                                     // large magnitudes
            }

            x[3 * cols + cols / 2] = float.NaN;                              // a NaN: no candidate in its row
            foreach (var isa in RowsIsas)
            {
                RowsCase($"log_softmax_topk_f32 ({isa}), {Rows} x {cols}, k = {K}", [x, new float[Rows * K], new float[Rows * K]], Rows, [1, 2],
                    (b, s) => b.LogSoftmaxTopKKernel(s[0], s[1], s[2], Rows, cols, K),
                    (m, a, grid, seed) => RowDispatch("log_softmax_topk_f32", isa, m, grid, seed, a[0], a[1], a[2], (ulong)cols, K, Rows),
                    worst, "log_softmax_topk_f32");
            }
        }

        RowsReport(worst);
    }
}
