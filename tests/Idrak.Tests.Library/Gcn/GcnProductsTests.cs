// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's products in GCN (GcnKernels.Products), each run by the GCN interpreter and proven against the CPU (plan 35, G4a).
// Every kernel is assembled for gfx90c, gfx906, gfx1030 and gfx1100 and run over shapes that are not tile multiples (1, a tile less one,
// a few tiles and a remainder), every transposition, beta zero and not, batches, with and without a bias; compared to
// the CPU backend within the device-against-CPU tolerance (OnBoth's, 1e-3 of max(1, |expected|)); run a second time
// with another wave order to give the same bits; and the interpreter reports no missing wait, race or access out of
// range (a report fails the run). With IDRAK_GCN_PRINTED naming a folder, each kernel's printed text (physical
// registers, LLVM's syntax) is written there for the development check against llvm-mc.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnProductsGroup =
    [
        ("gcn kernels: products: every product kernel assembles for gfx90c, gfx906, gfx1030 and gfx1100 with CUDA's argument list and its declared block, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnProductsAssembled),
        ("gcn kernels: products: matmul_f32 equals the CPU's product (then alpha, beta and a bias) for nn, nt, tn and tt over 1 x 1 x 1, a tile less one (63 x 63 x 15), a few tiles and a remainder (70 x 130 x 37), a long k and batches of 3; beta 0 never reads c (a NaN there stays out); the same bits on a second run", GcnProductsMatMul),
        ("gcn kernels: products: gemv_nn_f32 and gemv_nt_f32 equal the CPU's over 1 to 11 rows (two row groups), odd n, k of 1 to 300 (both kernels' steps and tails), batches and beta; the same bits on a second run", GcnProductsGemv),
    ];

    private static readonly GcnIsaVersion[] ProductIsas = [GcnIsaVersion.Parse("gfx90c"), GcnIsaVersion.Parse("gfx906"), GcnIsaVersion.Parse("gfx1030"), GcnIsaVersion.Parse("gfx1100")];

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
            var target = KernelTarget(isa);
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
        GcnInterpreter.Run(image, memory, new GcnDispatch { Workgroups = grid, WorkgroupSize = (block, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1 });
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

    internal static void GcnProductsAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Products.Select(k => k.Name).ToArray();
        Check(names.Length == 4 && names.Distinct().Count() == names.Length, $"4 product kernels (matmul_f32, gemv_nn_f32, gemv_nt_f32, gemv_multi_f32), each once: {string.Join(", ", names)}");

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
        "gemv_multi_f32" => (GcnKernels.GemvMultiArguments, GcnKernels.GemvThreads),
        _ => (GcnKernels.GemvArguments, GcnKernels.GemvThreads),                      // gemv_nn_f32, gemv_nt_f32
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
}
