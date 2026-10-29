// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;
using GcnWeights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;

// HIP's products in GCN (GcnKernels.Products), each run by the GCN interpreter and proven against the CPU (plan 35, G4a).
// Every kernel is assembled for gfx90c and gfx906 and run over shapes that are not tile multiples (1, a tile less one,
// a few tiles and a remainder), every transposition, beta zero and not, batches, with and without a bias; the packed
// few-row products for 1 to 8 rows (fewer than their kernel's too), odd n over more than one block of words, k in one
// split and in several (added by int8_gemv_finish_f32); compared to the CPU backend within the device-against-CPU
// tolerance (OnBoth's, 1e-3 of max(1, |expected|)), the expansions bit for bit; run a second time with another wave
// order (and, where the grid is free, another grid) to give the same bits; and the interpreter reports no missing wait,
// race or access out of range (a report fails the run). With IDRAK_GCN_PRINTED naming a folder, each kernel's printed
// text (physical registers, LLVM's syntax) is written there for the development check against llvm-mc.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnProductsGroup =
    [
        ("gcn kernels: products: every product, packed few-row product and expansion kernel assembles for gfx90c and gfx906 with CUDA's argument list and its declared block, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnProductsAssembled),
        ("gcn kernels: products: matmul_f32 equals the CPU's product (then alpha, beta and a bias) for nn, nt, tn and tt over 1 x 1 x 1, a tile less one (63 x 63 x 15), a few tiles and a remainder (70 x 130 x 37), a long k and batches of 3; beta 0 never reads c (a NaN there stays out); the same bits on a second run", GcnProductsMatMul),
        ("gcn kernels: products: gemv_nn_f32 and gemv_nt_f32 equal the CPU's over 1 to 11 rows (two row groups), odd n, k of 1 to 300 (both kernels' steps and tails), batches and beta; the same bits on a second run", GcnProductsGemv),
        ("gcn kernels: products: int8_gemv_r1/_r2/_r4_f32 and int8_gemv_f32 equal the CPU's Int8MatMul for 1, 2, 3, 5 and 8 rows, n = 301 (two blocks of words), k = 70 in one split and in three (int8_gemv_finish_f32, scaled); the same bits on a second run and finish grid", _ => GcnProductsPacked(GcnWeights.Int8)),
        ("gcn kernels: products: int4_gemv_r1/_r2/_r4_f32 and int4_gemv_f32 equal the CPU's Int4MatMul for 1, 2, 3, 5 and 8 rows, n = 523, k = 96 (three groups) in one split and in two of 64 rows; the same bits on a second run and finish grid", _ => GcnProductsPacked(GcnWeights.Int4)),
        ("gcn kernels: products: bf16_gemv_r1/_r2/_r4_f32 and bf16_gemv_f32 equal the CPU's BFloat16MatMul for 1, 2, 3, 5 and 8 rows, n = 131, k = 70 in one split and in three; the same bits on a second run and finish grid", _ => GcnProductsPacked(GcnWeights.BFloat16)),
        ("gcn kernels: products: f16_gemv_r1/_r2/_r4_f32 and f16_gemv_f32 equal the CPU's Float16MatMul for 1, 2, 3, 5 and 8 rows, n = 131, k = 70 in one split and in three; the same bits on a second run and finish grid", _ => GcnProductsPacked(GcnWeights.Float16)),
        ("gcn kernels: products: int8_matmul_f32 (many rows) equals the CPU's Int8MatMul over 11 x 75 x 37 and 1 x 1 x 1, the same bits on another grid", GcnProductsInt8MatMul),
        ("gcn kernels: products: int8_dequant_f32, int4_dequant_f32, bf16_dequant_f32, f16_dequant_f32 (NaN payloads, infinities, subnormals) and bf16_pack_f32 (ties to even, NaN, overflow, an odd count) equal the CPU's bit for bit, on two grids", GcnProductsExpansions),
    ];

    private static readonly GcnIsaVersion[] ProductIsas = [GcnIsaVersion.Parse("gfx90c"), GcnIsaVersion.Parse("gfx906")];

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> productKernels = [];

    // The kernel assembled for isa, its printed text checked to read back to the same bytes, written into a code object
    // and read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) ProductKernel(string name, GcnIsaVersion isa)
    {
        lock (productKernels)
        {
            if (productKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = GcnKernels.Products.Single(k => k.Name == name);
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
            return productKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch: the arguments packed by `pack` into the kernel's argument bytes (the Pack* helper's length first).
    private static void ProductDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y, int Z) grid, int block, ulong seed, Action<byte[]> pack)
    {
        var (code, image) = ProductKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        pack(arguments);
        GcnInterpreter.Run(image, memory, new GcnDispatch { Workgroups = grid, WorkgroupSize = (block, 1, 1), Arguments = arguments, Seed = seed });
    }

    // Runs `gcn` twice (the second with another seed, so another wave order, and second = true, which may pick another
    // grid), checks the same bits, and compares the outputs with `expected`'s (exact: bit for bit), keeping the worst error.
    private static void ProductsCase(string what, float[][] inputs, int[] outputs, float[][] expected, Action<GcnMemory, ulong[], bool, ulong> gcn,
        Dictionary<string, float> worst, string kernel, bool exact = false)
    {
        float[][] first = RowsOnGcn(inputs, (m, a) => gcn(m, a, false, 1));
        float[][] second = RowsOnGcn(inputs, (m, a) => gcn(m, a, true, 7));
        RowsSameBits(first, second, what);
        foreach (int o in outputs)
        {
            if (exact)
            {
                var want = MemoryMarshal.Cast<float, uint>(expected[o]);
                var have = MemoryMarshal.Cast<float, uint>(first[o]);
                Check(want.Length == have.Length, $"{what}, storage {o}: {have.Length} values, the CPU's {want.Length}");
                for (int i = 0; i < want.Length; i++)
                {
                    Check(want[i] == have[i], $"{what}, storage {o}: element {i} is 0x{have[i]:x8}, the CPU's 0x{want[i]:x8}");
                }

                worst[kernel] = worst.GetValueOrDefault(kernel);
                continue;
            }

            float error = RowsCompare(expected[o], first[o], $"{what}, storage {o}");
            worst[kernel] = MathF.Max(worst.GetValueOrDefault(kernel), error);
        }
    }

    private static float[] FromBits(uint[] bits) => MemoryMarshal.Cast<uint, float>(bits.AsSpan()).ToArray();

    internal static void GcnProductsAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Products.Select(k => k.Name).ToArray();
        Check(names.Length == 26 && names.Distinct().Count() == names.Length, $"26 product kernels, each once: {string.Join(", ", names)}");
        foreach (var format in (GcnWeights[])[GcnWeights.Int8, GcnWeights.Int4, GcnWeights.BFloat16, GcnWeights.Float16])
        {
            for (int m = 1; m <= GcnKernels.GemvRows; m++)
            {
                string kernel = GcnKernels.PackedGemvKernel(format, m);
                int rows = m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8;
                string prefix = format switch { GcnWeights.Int8 => "int8", GcnWeights.Int4 => "int4", GcnWeights.BFloat16 => "bf16", _ => "f16" };
                Check(kernel == (rows == 8 ? $"{prefix}_gemv_f32" : $"{prefix}_gemv_r{rows}_f32") && names.Contains(kernel), $"{format}, {m} rows: {kernel}");
            }
        }

        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(GcnKernels.Products.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            var (arguments, block) = ProductShape(name);
            foreach (var isa in ProductIsas)
            {
                var (code, _) = ProductKernel(name, isa);
                Check(code.Arguments.SequenceEqual(arguments) && code is { WaveSize: 64 } && code.RequiredWorkgroupSize == (block, 1, 1) && code.MaxFlatWorkgroupSize == block,
                    $"{name} ({isa}): arguments {string.Join(", ", code.Arguments)}, block {code.RequiredWorkgroupSize} (max flat {code.MaxFlatWorkgroupSize}), wave {code.WaveSize}");
                Check(code.LdsBytes <= 16384, $"{name} ({isa}): {code.LdsBytes} bytes of LDS");
                Console.WriteLine($"    {name} ({isa}): {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
            }
        }
    }

    // Each kernel's argument list and block.
    private static (IReadOnlyList<GcnArgument> Arguments, int Block) ProductShape(string name) => name switch
    {
        "matmul_f32" => (GcnKernels.MatMulArguments, GcnKernels.MatMulThreads),
        "gemv_nn_f32" or "gemv_nt_f32" => (GcnKernels.GemvArguments, GcnKernels.GemvThreads),
        "int8_gemv_finish_f32" => (GcnKernels.GemvFinishArguments, GcnKernels.ExpansionBlock),
        "int8_matmul_f32" => (GcnKernels.Int8MatMulArguments, GcnKernels.ExpansionBlock),
        "int8_dequant_f32" or "int4_dequant_f32" => (GcnKernels.ScaledDequantArguments, GcnKernels.ExpansionBlock),
        "bf16_dequant_f32" or "f16_dequant_f32" => (GcnKernels.HalfDequantArguments, GcnKernels.ExpansionBlock),
        "bf16_pack_f32" => (GcnKernels.PackBFloat16Arguments, GcnKernels.ExpansionBlock),
        _ => (GcnKernels.PackedGemvArguments, GcnKernels.PackedGemvWords),
    };

    // ---- matmul_f32 ------------------------------------------------------------------------------------------------

    private sealed record MatMulCase(int M, int N, int K, int Batch, bool TransA, bool TransB, float Beta, float Alpha, bool Bias);

    internal static void GcnProductsMatMul(Device device)
    {
        _ = device;
        MatMulCase[] cases =
        [
            new(1, 1, 1, 1, false, false, 0f, 1f, false),
            new(1, 1, 1, 1, true, true, 0.5f, 2f, true),
            new(63, 63, 15, 1, false, false, 0f, 1f, false),
            new(63, 63, 15, 1, false, true, 0.75f, -0.5f, true),
            new(63, 63, 15, 1, true, false, 0f, 1f, true),
            new(63, 63, 15, 1, true, true, 0.25f, 1f, false),
            new(70, 130, 37, 1, false, false, 0.75f, -0.5f, true),
            new(70, 130, 37, 1, false, true, 0f, 1f, false),
            new(70, 130, 37, 1, true, false, 0.75f, 1f, false),
            new(70, 130, 37, 1, true, true, -0f, -0.5f, true),
            new(20, 33, 50, 3, false, false, 0.5f, 1f, true),
            new(20, 33, 50, 3, true, true, 0f, 1f, false),
            new(5, 7, 300, 2, false, true, 0f, 1f, false),
        ];
        var worst = new Dictionary<string, float>();
        var random = new Random(41);
        foreach (var c in cases)
        {
            int mk = c.M * c.K, kn = c.K * c.N, mn = c.M * c.N;
            float[] a = RowsRandom(random, c.Batch * mk, 1f), b = RowsRandom(random, c.Batch * kn, 1f), old = RowsRandom(random, c.Batch * mn, 2f);
            float[] bias = RowsRandom(random, c.N, 1f);
            old[0] = float.NaN;                                                  // read only when beta is not ±0
            float[] product = RowsOnCpu([a, b, new float[c.Batch * mn]], (cpu, s) => cpu.BatchedMatMulKernel(s[0], s[1], s[2], c.Batch, c.M, c.N, c.K, c.TransA, c.TransB, 0f))[2];
            float[] expected = new float[product.Length];
            for (int i = 0; i < expected.Length; i++)
            {
                double value = (double)c.Alpha * product[i] + (c.Beta == 0f ? 0.0 : (double)c.Beta * old[i]) + (c.Bias ? bias[i % c.N] : 0.0);
                expected[i] = (float)value;
            }

            string what = $"matmul_f32 {(c.TransA ? 't' : 'n')}{(c.TransB ? 't' : 'n')} {c.M}x{c.N}x{c.K}, batch {c.Batch}, alpha {c.Alpha}, beta {c.Beta}{(c.Bias ? ", bias" : "")}";
            foreach (var isa in ProductIsas)
            {
                ProductsCase($"{what} ({isa})", [a, b, old, bias], [2], [a, b, expected, bias],
                    (memory, at, _, seed) => ProductDispatch("matmul_f32", isa, memory,
                        ((c.N + GcnKernels.MatMulTile - 1) / GcnKernels.MatMulTile, (c.M + GcnKernels.MatMulTile - 1) / GcnKernels.MatMulTile, c.Batch), GcnKernels.MatMulThreads, seed,
                        args => GcnKernels.PackMatMul(args.AsSpan(0, GcnKernels.MatMulArgumentBytes), at[0], at[1], at[2], (uint)c.M, (uint)c.N, (uint)c.K, c.TransA, c.TransB,
                            c.Beta, (uint)mk, (uint)kn, (uint)mn, c.Alpha, c.Bias ? at[3] : 0UL)),
                    worst, "matmul_f32");
            }
        }

        RowsReport(worst);
    }

    // ---- gemv_nn_f32, gemv_nt_f32 ----------------------------------------------------------------------------------

    internal static void GcnProductsGemv(Device device)
    {
        _ = device;
        (int M, int N, int K, int Batch, float Beta)[] cases = [(1, 1, 1, 1, 0f), (1, 70, 45, 1, 0.5f), (3, 133, 300, 1, 0f), (8, 64, 17, 2, 0.75f), (11, 65, 64, 1, 0.5f)];
        var worst = new Dictionary<string, float>();
        var random = new Random(42);
        foreach (var (m, n, k, batch, beta) in cases)
        {
            foreach (bool transB in (bool[])[false, true])
            {
                int mk = m * k, kn = k * n, mn = m * n;
                float[] a = RowsRandom(random, batch * mk, 1f), b = RowsRandom(random, batch * kn, 1f), old = RowsRandom(random, batch * mn, 2f);
                old[0] = float.NaN;
                float[] product = RowsOnCpu([a, b, new float[batch * mn]], (cpu, s) => cpu.BatchedMatMulKernel(s[0], s[1], s[2], batch, m, n, k, false, transB, 0f))[2];
                float[] expected = [.. product.Select((p, i) => (float)(p + (beta == 0f ? 0.0 : (double)beta * old[i])))];
                string kernel = transB ? "gemv_nt_f32" : "gemv_nn_f32";
                (int, int, int) grid = transB
                    ? ((n + GcnKernels.GemvNtColumns - 1) / GcnKernels.GemvNtColumns, (m + GcnKernels.GemvRows - 1) / GcnKernels.GemvRows, batch)
                    : ((n + GcnKernels.GemvNnColumns - 1) / GcnKernels.GemvNnColumns, (m + GcnKernels.GemvRows - 1) / GcnKernels.GemvRows, batch);
                foreach (var isa in ProductIsas)
                {
                    ProductsCase($"{kernel} {m}x{n}x{k}, batch {batch}, beta {beta} ({isa})", [a, b, old], [2], [a, b, expected],
                        (memory, at, _, seed) => ProductDispatch(kernel, isa, memory, grid, GcnKernels.GemvThreads, seed,
                            args => GcnKernels.PackGemv(args.AsSpan(0, GcnKernels.GemvArgumentBytes), at[0], at[1], at[2], (uint)m, (uint)n, (uint)k, beta, (ulong)mk, (ulong)kn, (ulong)mn)),
                        worst, kernel);
                }
            }
        }

        RowsReport(worst);
    }

    // ---- The packed few-row products -------------------------------------------------------------------------------

    // Random packed weights [k, n] of `format` as the CPU backend reads them, and their scales (int8: n; int4: a row of
    // 8 · words per group of 32 rows; none for the half-width formats).
    private static (float[] Packed, float[] Scales) ProductsWeights(GcnWeights format, Random random, int k, int n)
    {
        int cpw = GcnKernels.ColumnsPerWord(format), words = (n + cpw - 1) / cpw;
        var bits = new uint[k * words];
        for (int r = 0; r < k; r++)
        {
            for (int j = 0; j < n; j++)
            {
                uint value = format switch
                {
                    GcnWeights.Int8 => (byte)(sbyte)random.Next(-127, 128),
                    GcnWeights.Int4 => (uint)random.Next(-8, 8) & 0xF,
                    GcnWeights.BFloat16 => BitConverter.SingleToUInt32Bits((float)(random.NextDouble() * 2 - 1)) >> 16,
                    _ => BitConverter.HalfToUInt16Bits((Half)(random.NextDouble() * 2 - 1)),
                };
                int shift = 32 / cpw * (j % cpw);
                bits[r * words + j / cpw] |= value << shift;
            }
        }

        float[] scales = format switch
        {
            GcnWeights.Int8 => RowsRandom(random, n, 0.02f, 0.03f),
            GcnWeights.Int4 => RowsRandom(random, (k + GcnKernels.Int4Group - 1) / GcnKernels.Int4Group * 8 * words, 0.05f, 0.1f),
            _ => [0f],
        };
        return (FromBits(bits), scales);
    }

    internal static void GcnProductsPacked(GcnWeights format)
    {
        int n = format switch { GcnWeights.Int8 => 301, GcnWeights.Int4 => 523, _ => 131 };
        int k = format == GcnWeights.Int4 ? 96 : 70, splitChunk = format == GcnWeights.Int4 ? 64 : 32;
        int cpw = GcnKernels.ColumnsPerWord(format), words = (n + cpw - 1) / cpw;
        bool scaled = format == GcnWeights.Int8;
        var worst = new Dictionary<string, float>();
        var random = new Random(43 + (int)format);
        var (packed, scales) = ProductsWeights(format, random, k, n);
        foreach (int m in (int[])[1, 2, 3, 8, 5])
        {
            string kernel = GcnKernels.PackedGemvKernel(format, m);
            float[] x = RowsRandom(random, m * k, 1f);
            float[] expected = RowsOnCpu([x, packed, scales, new float[m * n]], (cpu, s) =>
            {
                switch (format)
                {
                    case GcnWeights.Int8:
                        cpu.Int8MatMulKernel(s[0], s[1], s[2], s[3], m, n, k);
                        break;
                    case GcnWeights.Int4:
                        cpu.Int4MatMulKernel(s[0], s[1], s[2], s[3], m, n, k);
                        break;
                    case GcnWeights.BFloat16:
                        cpu.BFloat16MatMulKernel(s[0], s[1], s[3], m, n, k);
                        break;
                    default:
                        cpu.Float16MatMulKernel(s[0], s[1], s[3], m, n, k);
                        break;
                }
            })[3];
            foreach (int chunk in (int[])[k, splitChunk])
            {
                int splits = (k + chunk - 1) / chunk;
                float[] y = new float[m * n], part = new float[splits * m * n];
                foreach (var isa in ProductIsas)
                {
                    void Run(GcnMemory memory, ulong[] at, bool second, ulong seed)
                    {
                        ProductDispatch(kernel, isa, memory, ((words + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords, splits, 1), GcnKernels.PackedGemvWords, seed,
                            args => GcnKernels.PackPackedGemv(args.AsSpan(0, GcnKernels.PackedGemvArgumentBytes), at[0], at[1], format == GcnWeights.BFloat16 || format == GcnWeights.Float16 ? 0UL : at[2],
                                at[3], at[4], (uint)m, (uint)n, (uint)k, (uint)words, (uint)chunk, (uint)splits));
                        if (splits > 1)
                        {
                            // One block (the grid-stride loop) on the first run, every element's on the second.
                            int blocks = second ? (m * n + GcnKernels.ExpansionBlock - 1) / GcnKernels.ExpansionBlock : 1;
                            ProductDispatch("int8_gemv_finish_f32", isa, memory, (blocks, 1, 1), GcnKernels.ExpansionBlock, seed,
                                args => GcnKernels.PackGemvFinish(args.AsSpan(0, GcnKernels.GemvFinishArgumentBytes), at[4], scaled ? at[2] : 0UL, at[3], (uint)n, (uint)(m * n), (uint)splits));
                        }
                    }

                    string what = $"{kernel}, {m} rows, {k} x {n}, {splits} split{(splits > 1 ? "s and int8_gemv_finish_f32" : "")} ({isa})";
                    ProductsCase(what, [x, packed, scales, y, part], [3], [x, packed, scales, expected, part], Run, worst,
                        splits > 1 ? $"{kernel} + int8_gemv_finish_f32" : kernel);
                }
            }
        }

        RowsReport(worst);
    }

    // ---- int8_matmul_f32 -------------------------------------------------------------------------------------------

    internal static void GcnProductsInt8MatMul(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(44);
        foreach (var (m, n, k) in ((int, int, int)[])[(11, 75, 37), (1, 1, 1)])
        {
            var (packed, scales) = ProductsWeights(GcnWeights.Int8, random, k, n);
            float[] x = RowsRandom(random, m * k, 1f);
            float[] expected = RowsOnCpu([x, packed, scales, new float[m * n]], (cpu, s) => cpu.Int8MatMulKernel(s[0], s[1], s[2], s[3], m, n, k))[3];
            int threads = m * ((n + 3) / 4);
            foreach (var isa in ProductIsas)
            {
                ProductsCase($"int8_matmul_f32 {m}x{n}x{k} ({isa})", [x, packed, scales, new float[m * n]], [3], [x, packed, scales, expected],
                    (memory, at, second, seed) => ProductDispatch("int8_matmul_f32", isa, memory,
                        (second ? (threads + GcnKernels.ExpansionBlock - 1) / GcnKernels.ExpansionBlock : 1, 1, 1), GcnKernels.ExpansionBlock, seed,
                        args => GcnKernels.PackInt8MatMul(args.AsSpan(0, GcnKernels.Int8MatMulArgumentBytes), at[0], at[1], at[2], at[3], (uint)m, (uint)n, (uint)k)),
                    worst, "int8_matmul_f32");
            }
        }

        RowsReport(worst);
    }

    // ---- The expansions --------------------------------------------------------------------------------------------

    // Halves a float16 test mixes in: NaN payloads (quiet and signalling, both signs), infinities, subnormals, zeros, the
    // largest finite value.
    private static readonly ushort[] ProductsHalfSpecials = [0x7C01, 0xFE55, 0x7E00, 0x7C00, 0xFC00, 0x0001, 0x83FF, 0x0200, 0x8000, 0x0000, 0x7BFF, 0xFBFF, 0x3C00];

    // Floats bf16_pack_f32 is checked on: ties (to even, both ways), NaN payloads, infinities, values that round to
    // infinity, subnormals, zeros.
    private static readonly uint[] ProductsPackSpecials =
        [0x3F808000, 0x3F818000, 0x3F807FFF, 0x7FC00001, 0xFF812345, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x00000001, 0x80008000, 0x80000000, 0x00000000, 0x3F800000];

    internal static void GcnProductsExpansions(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(45);
        foreach (var (k, n) in ((int, int)[])[(37, 75), (1, 1), (3, 130)])
        {
            foreach (var format in (GcnWeights[])[GcnWeights.Int8, GcnWeights.Int4, GcnWeights.BFloat16, GcnWeights.Float16])
            {
                int cpw = GcnKernels.ColumnsPerWord(format), words = (n + cpw - 1) / cpw;
                var (packed, scales) = ProductsWeights(format, random, k, n);
                if (format is GcnWeights.Float16 or GcnWeights.BFloat16)
                {
                    // Special values in some halves: as float16 their own bits, as bfloat16 the float32 whose high half they are.
                    var bits = MemoryMarshal.Cast<float, ushort>(packed.AsSpan());
                    for (int i = 0; i < bits.Length && i < 3 * ProductsHalfSpecials.Length; i += 3)
                    {
                        ushort special = ProductsHalfSpecials[i / 3];
                        bits[i] = format == GcnWeights.Float16 ? special : (ushort)(special == 0x7C01 ? 0x7F81 : special == 0xFE55 ? 0xFFC5 : special);
                    }
                }

                string kernel = GcnKernels.DequantKernel(format);
                float[] expected = RowsOnCpu([packed, scales, new float[k * n]], (cpu, s) =>
                {
                    switch (format)
                    {
                        case GcnWeights.Int8:
                            cpu.Int8DequantizeKernel(s[0], s[1], s[2], k, n);
                            break;
                        case GcnWeights.Int4:
                            cpu.Int4DequantizeKernel(s[0], s[1], s[2], k, n);
                            break;
                        case GcnWeights.BFloat16:
                            cpu.BFloat16DequantizeKernel(s[0], s[2], k, n);
                            break;
                        default:
                            cpu.Float16DequantizeKernel(s[0], s[2], k, n);
                            break;
                    }
                })[2];
                int bytes = format is GcnWeights.Int8 or GcnWeights.Int4 ? GcnKernels.ScaledDequantArgumentBytes : GcnKernels.HalfDequantArgumentBytes;
                foreach (var isa in ProductIsas)
                {
                    ProductsCase($"{kernel} {k} x {n} ({isa})", [packed, scales, new float[k * n]], [2], [packed, scales, expected],
                        (memory, at, second, seed) => ProductDispatch(kernel, isa, memory,
                            (second ? (k * words + GcnKernels.ExpansionBlock - 1) / GcnKernels.ExpansionBlock : 1, 1, 1), GcnKernels.ExpansionBlock, seed,
                            args => GcnKernels.PackDequant(args.AsSpan(0, bytes), format, at[0], at[1], at[2], (uint)k, (uint)n)),
                        worst, kernel, exact: true);
                }
            }
        }

        foreach (int count in (int[])[1, 75, 600])
        {
            float[] x = RowsRandom(random, count, 3f);
            for (int i = 0; i < count && i < 2 * ProductsPackSpecials.Length; i += 2)
            {
                x[i] = BitConverter.UInt32BitsToSingle(ProductsPackSpecials[i / 2]);
            }

            int words = (count + 1) / 2;
            float[] expected = RowsOnCpu([x, new float[words]], (cpu, s) => cpu.PackBFloat16Kernel(s[0], s[1], count))[1];
            foreach (var isa in ProductIsas)
            {
                ProductsCase($"bf16_pack_f32, {count} floats ({isa})", [x, new float[words]], [1], [x, expected],
                    (memory, at, second, seed) => ProductDispatch("bf16_pack_f32", isa, memory,
                        (second ? (words + GcnKernels.ExpansionBlock - 1) / GcnKernels.ExpansionBlock : 1, 1, 1), GcnKernels.ExpansionBlock, seed,
                        args => GcnKernels.PackBFloat16Pack(args.AsSpan(0, GcnKernels.PackBFloat16ArgumentBytes), at[0], at[1], (uint)count)),
                    worst, "bf16_pack_f32", exact: true);
            }
        }

        RowsReport(worst);
    }
}
