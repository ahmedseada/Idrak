// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's attention and shape kernels in GCN (GcnKernels.Attention, GcnKernels.Shapes), each run by the GCN interpreter and
// proven against the CPU (plan 35, G4b). Every kernel is assembled for gfx90c and gfx906 and run over shapes that are
// multiples of nothing: head sizes of 37 and 64 (and 300, past one slot of the block), caches of 1 position to a few
// tiles of 256, grouped heads, tables of ranges with an empty one, windows and soft-caps, one split and several merged,
// each cache format; compared to the CPU backend within OnBoth's tolerance (the attention, which sums in another order)
// or bit for bit (the cache writes, masks, gathers, permutes and copies); run again on another grid and another wave
// order to give the same bits; and the interpreter reports no missing wait, race or access out of range (a report fails
// the run). With IDRAK_GCN_PRINTED naming a folder, each kernel's printed text (physical registers, LLVM's syntax) is
// written there for the development check against llvm-mc.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnAttentionGroup =
    [
        ("gcn kernels: attention: every attention and shape kernel assembles for gfx90c and gfx906 with its block of 256 declared, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnAttentionAssembled),
        ("gcn kernels: attention: attention_decode_f32 equals the CPU's decoding and tiled attention over a cache (1 to 300 positions, heads of 37, 64 and 300, rows of several steps, a window, a soft-cap, the log-sum-exp; one split, and two merged by attention_merge_f32), the same bits on two grids and wave orders", GcnAttentionDecode),
        ("gcn kernels: attention: attention_decode_bf16 and attention_decode_int8 equal the CPU's attention over bfloat16 and int8 caches (odd heads, several steps, a window, a soft-cap, two splits merged), the same bits on two grids and wave orders", GcnAttentionPacked),
        ("gcn kernels: attention: attention_tiled_r2/r4/r8_f32 (a block's rows sharing each key and value tile) equal attention_decode_f32 bit for bit without a window and the CPU's tiled attention over a cache with a window and a soft-cap (heads of 37, 64 and 256, rows of a head not a multiple of the block's, the log-sum-exp), the same bits on two grids and wave orders", GcnAttentionTiled),
        ("gcn kernels: attention: attention_spans_f32 equals the CPU's attention over key ranges (grouped heads, two tables, an empty range: zeros and a log-sum-exp of -inf; a soft-cap; two splits merged) and its strided form over packed rows (two batches), the same bits on two grids and wave orders", GcnAttentionSpans),
        ("gcn kernels: attention: attn_scores_int8 and attn_context_int8 equal the CPU's scores and weighted values over an int8 cache (an odd head), the same bits on two grids and wave orders", GcnAttentionInt8Products),
        ("gcn kernels: shapes: kv_write_f32, kv_write_bf16 (an odd head, a NaN kept quiet) and kv_write_int8 (an odd head, a zero row) write the CPU's cache bit for bit, decoder_mask_f32 its mask, the same bits on two grids and wave orders", GcnShapesCache),
        ("gcn kernels: shapes: gather_f32, gather_bf16_f32 and gather_bf16_cols_f32 (odd widths, the last row) equal the CPU's gathers bit for bit, the same bits on two grids and wave orders", GcnShapesGather),
        ("gcn kernels: shapes: permute_f32 (ranks 1, 4 and 6; adding) and copy2d_f32 (pitches, adding) equal the CPU's bit for bit, the same bits on two grids and wave orders", GcnShapesPermute),
    ];

    private static readonly GcnIsaVersion[] AttentionIsas = [GcnIsaVersion.Parse("gfx90c"), GcnIsaVersion.Parse("gfx906")];

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> attentionKernels = [];

    private static IEnumerable<GcnKernelSource> AttentionAndShapes => GcnKernels.Attention.Concat(GcnKernels.Shapes);

    // The kernel assembled for isa, written into a code object and read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) AttentionKernel(string name, GcnIsaVersion isa)
    {
        lock (attentionKernels)
        {
            if (attentionKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = AttentionAndShapes.Single(k => k.Name == name);
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
            return attentionKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch of an attention or shape kernel: its arguments as 64-bit values (addresses, 32-bit values, float bits) in order.
    private static void AttentionDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y) grid, ulong seed, params ulong[] values)
    {
        var (code, image) = AttentionKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = (grid.X, grid.Y, 1), WorkgroupSize = (GcnKernels.RowThreads, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1,
        });
    }

    // A split attention launch merged both ways (plan 35, G6b): by attention_merge_f32 in a second launch, and by each
    // row's last split block (its completion counters mapped here, `rows` of them): run on every row's block with the
    // blocks in three orders (another block last each time), the outputs equal the second launch's bit for bit, and
    // every counter is left at zero (a second run, the replay, finds them so and gives the same bits).
    private static void AttentionMergedAlike(string what, float[][] inputs, int[] outputs, int rows, Action<GcnMemory, ulong[], (int X, int Y), ulong, ulong> gcn)
    {
        float[][] launched = RowsOnGcn(inputs, (m, a) => gcn(m, a, (rows, 1), 7, 0UL));
        foreach (ulong seed in (ulong[])[3, 11, 29])
        {
            float[] counters = [];
            float[][] merged = RowsOnGcn(inputs, (m, a) =>
            {
                ulong at = m.Map<float>("counters", new float[rows]);
                gcn(m, a, (rows, 1), seed, at);
                gcn(m, a, (rows, 1), seed + 1, at);
                counters = m.Read<float>(at, rows);
            });
            foreach (int o in outputs)
            {
                Check(MemoryMarshal.AsBytes(launched[o].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(merged[o].AsSpan())),
                    $"{what}: storage {o} merged by the last split blocks (blocks in order {seed}) differs from attention_merge_f32's bit for bit");
            }

            Check(counters.All(v => BitConverter.SingleToUInt32Bits(v) == 0), $"{what}: the last split blocks left a counter that is not zero (blocks in order {seed})");
        }
    }

    // Blocks of 256 for n elements of a one-element-a-work-item kernel.
    private static int ShapeGrid(long n) => (int)Math.Max(1, (n + 255) / 256);

    // As RowsCase, but every output compared bit for bit with the CPU's.
    private static void ShapesCase(string what, float[][] inputs, int blocks, int[] outputs, Action<Backend, Storage[]> cpu,
        Action<GcnMemory, ulong[], (int X, int Y), ulong> gcn)
    {
        float[][] expected = RowsOnCpu(inputs, cpu);
        float[][] first = RowsOnGcn(inputs, (m, a) => gcn(m, a, (Math.Min(blocks, 2), 1), 1));
        float[][] second = RowsOnGcn(inputs, (m, a) => gcn(m, a, (blocks, 1), 7));
        RowsSameBits(first, second, what);
        foreach (int o in outputs)
        {
            var e = MemoryMarshal.Cast<float, uint>(expected[o].AsSpan());
            var g = MemoryMarshal.Cast<float, uint>(first[o].AsSpan());
            for (int i = 0; i < e.Length; i++)
            {
                Check(e[i] == g[i], $"{what}, storage {o}: element {i} is 0x{g[i]:X8}, the CPU's 0x{e[i]:X8}");
            }
        }
    }

    // A bfloat16 value's bits (the CPU's rounding).
    private static ushort Half(float value) => Idrak.Abstraction.Devices.BFloat16Bits.Round(value);

    // Rows of bfloat16 halves packed two a word (low half first), each row padded to whole words.
    private static float[] PackHalves(float[] values, int rows, int dim)
    {
        int words = (dim + 1) / 2;
        var packed = new uint[rows * words];
        for (int r = 0; r < rows; r++)
        {
            for (int d = 0; d < dim; d++)
            {
                packed[r * words + d / 2] |= (uint)Half(values[r * dim + d]) << (16 * (d % 2));
            }
        }

        return MemoryMarshal.Cast<uint, float>(packed.AsSpan()).ToArray();
    }

    // Rows of int8 values packed four a word (byte d % 4 of word d / 4), each row padded to whole words.
    private static float[] PackBytes(Random random, int rows, int dim)
    {
        int words = (dim + 3) / 4;
        var bytes = new sbyte[rows * words * 4];
        for (int r = 0; r < rows; r++)
        {
            for (int d = 0; d < dim; d++)
            {
                bytes[r * words * 4 + d] = (sbyte)random.Next(-100, 101);
            }
        }

        return MemoryMarshal.Cast<sbyte, float>(bytes.AsSpan()).ToArray();
    }

    // The splits a test asks for: chunk 0 is one split over the whole cache.
    private static (int Splits, int Chunk) TestSplits(int positions, int chunk) =>
        chunk == 0 ? (1, Math.Max(positions, 1)) : ((positions + chunk - 1) / chunk, chunk);

    internal static void GcnAttentionAssembled(Device device)
    {
        _ = device;
        var names = AttentionAndShapes.Select(k => k.Name).ToArray();
        Check(names.Length == 19 && names.Distinct().Count() == names.Length, $"19 attention and shape kernels, each once: {string.Join(", ", names)}");
        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(AttentionAndShapes.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            foreach (var isa in AttentionIsas)
            {
                var (code, _) = AttentionKernel(name, isa);
                Check(code is { RequiredWorkgroupSize: (256, 1, 1), MaxFlatWorkgroupSize: 256, WaveSize: 64 },
                    $"{name} ({isa}): declares the block of 256 (required {code.RequiredWorkgroupSize}, max flat {code.MaxFlatWorkgroupSize}, wave {code.WaveSize})");
                bool rows = name is "attention_decode_f32" or "attention_decode_bf16" or "attention_decode_int8" or "attention_spans_f32";
                int tiled = name.StartsWith("attention_tiled_r", StringComparison.Ordinal) ? name["attention_tiled_r".Length] - '0' : 0;
                int lds = rows ? GcnKernels.AttentionLdsBytes : tiled > 0 ? GcnKernels.AttentionTiledLdsBytes(tiled) : 0;
                Check(code.LdsBytes == lds && name == (tiled > 0 ? GcnKernels.AttentionTiledKernel(tiled) : name),
                    $"{name} ({isa}): {code.LdsBytes} bytes of LDS ({lds}: {(rows ? "the reductions, a tile's weights and the query" : tiled > 0 ? "a row's reductions, tile weights and query" : "none")})");
                Console.WriteLine($"    {name} ({isa}): {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
            }
        }
    }

    internal static void GcnAttentionDecode(Device device)
    {
        _ = device;
        const int Heads = 3;
        const float Scale = 0.125f;
        var worst = new Dictionary<string, float>();
        var random = new Random(41);
        (int Rph, int Steps, int Capacity, int Position, int Dim, int Window, float Softcap, int Chunk, bool Lse)[] cases =
        [
            (1, 1, 1, 0, 37, 0, 0f, 0, false),                                    // one position
            (2, 1, 300, 299, 64, 0, 0f, 0, true),                                 // two tiles, one split, the log-sum-exp
            (2, 1, 300, 290, 64, 0, 0f, 256, false),                              // two splits merged
            (5, 5, 40, 7, 37, 4, 0f, 0, true),                                    // rows of five steps, a window of 4
            (3, 3, 70, 30, 24, 0, 2f, 0, true),                                   // a soft-cap
            (1, 1, 9, 8, 300, 0, 0f, 0, false),                                   // a head past one slot of the block
        ];
        foreach (var (rph, steps, capacity, position, dim, window, softcap, chunk, lse) in cases)
        {
            int rows = Heads * rph;
            float[] q = RowsRandom(random, rows * dim, 2f), keys = RowsRandom(random, Heads * capacity * dim, 2f), values = RowsRandom(random, Heads * capacity * dim, 2f);
            var (splits, size) = TestSplits(capacity, chunk);
            var variant = new AttentionVariant(window, softcap);
            foreach (var isa in AttentionIsas)
            {
                string what = $"attention_decode_f32 ({isa}), {Heads} heads x {rph} rows, cache {capacity} at {position}, head {dim}, window {window}, cap {softcap}, {splits} splits";
                RowsCase(what, [q, keys, values, [position], new float[rows * dim], new float[rows]], rows, lse ? [4, 5] : [4],
                    (b, s) => b.AttentionTiledKernel(s[0], s[1], s[2], s[3], s[4], lse ? s[5] : null, Heads, rph, steps, capacity, dim, Scale, variant),
                    (m, a, grid, seed) => Launch(m, a, grid, seed, 0UL),
                    worst, splits > 1 ? "attention_decode_f32 + attention_merge_f32" : "attention_decode_f32");
                if (splits > 1)
                {
                    AttentionMergedAlike(what, [q, keys, values, [position], new float[rows * dim], new float[rows]], lse ? [4, 5] : [4], rows, Launch);
                }

                // counters: 0 merges the splits by attention_merge_f32; else each row's last split block merges them.
                void Launch(GcnMemory m, ulong[] a, (int X, int Y) grid, ulong seed, ulong counters)
                {
                    ulong part = splits > 1 ? m.Map<float>("part", new float[rows * splits * (dim + 2)]) : 0UL, logSumExp = lse ? a[5] : 0UL;
                    AttentionDispatch("attention_decode_f32", isa, m, (grid.X, splits), seed, a[0], a[1], a[2], a[3], a[4], logSumExp, part, counters,
                        (ulong)rph, (ulong)steps, (ulong)capacity, (ulong)dim, Bits(Scale), (ulong)window, Bits(softcap), (ulong)size, (ulong)splits, (ulong)rows);
                    if (splits > 1 && counters == 0)
                    {
                        AttentionDispatch("attention_merge_f32", isa, m, (grid.X, 1), seed, part, a[4], logSumExp, (ulong)splits, (ulong)dim, 1UL, (ulong)rows, 0UL, 0UL,
                            (ulong)dim, (ulong)rows);
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnAttentionTiled(Device device)
    {
        _ = device;
        const int Heads = 2;
        const float Scale = 0.125f;
        var worst = new Dictionary<string, float>();
        var random = new Random(44);

        // Rows of a head (steps), the cache, the head size, a window and a soft-cap; the rows not a multiple of a block's.
        (int Steps, int Capacity, int Dim, int Window, float Softcap)[] cases =
        [
            (13, 13, 37, 0, 0f),                                                  // a prompt over itself, causal
            (21, 300, 64, 0, 0f),                                                 // a longer cache: a row past 256 keys
            (19, 300, 256, 0, 0f),                                                // the largest head
            (11, 40, 24, 5, 0f),                                                  // a window
            (9, 30, 37, 0, 2f),                                                   // a soft-cap
        ];
        foreach (var (steps, capacity, dim, window, softcap) in cases)
        {
            int rows = Heads * steps;
            float[] q = RowsRandom(random, rows * dim, 2f), keys = RowsRandom(random, Heads * capacity * dim, 2f), values = RowsRandom(random, Heads * capacity * dim, 2f);
            float position = capacity - steps;
            var variant = new AttentionVariant(window, softcap);
            float[][] inputs = [q, keys, values, [position], new float[rows * dim], new float[rows]];
            foreach (int perBlock in GcnKernels.AttentionTiledRows)
            {
                string kernel = GcnKernels.AttentionTiledKernel(perBlock);
                int blocksPerHead = (steps + perBlock - 1) / perBlock, blocks = Heads * blocksPerHead;
                foreach (var isa in AttentionIsas)
                {
                    void Tiled(GcnMemory m, ulong[] a, (int X, int Y) grid, ulong seed) =>
                        AttentionDispatch(kernel, isa, m, (grid.X, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], (ulong)steps, (ulong)steps, (ulong)capacity, (ulong)dim,
                            Bits(Scale), (ulong)window, Bits(softcap), (ulong)blocksPerHead, (ulong)blocks);
                    string what = $"{kernel} ({isa}), {Heads} heads x {steps} rows, cache {capacity} at {position}, head {dim}, window {window}, cap {softcap}";
                    RowsCase(what, inputs, blocks, [4, 5],
                        (b, s) => b.AttentionTiledKernel(s[0], s[1], s[2], s[3], s[4], s[5], Heads, steps, steps, capacity, dim, Scale, variant), Tiled, worst, kernel);
                    if (window == 0)
                    {
                        float[][] tiled = RowsOnGcn(inputs, (m, a) => Tiled(m, a, (blocks, 1), 7));
                        float[][] decoded = RowsOnGcn(inputs, (m, a) => AttentionDispatch("attention_decode_f32", isa, m, (rows, 1), 7, a[0], a[1], a[2], a[3], a[4], a[5], 0UL, 0UL,
                            (ulong)steps, (ulong)steps, (ulong)capacity, (ulong)dim, Bits(Scale), 0UL, Bits(softcap), (ulong)capacity, 1UL, (ulong)rows));
                        foreach (int o in (int[])[4, 5])
                        {
                            Check(MemoryMarshal.AsBytes(tiled[o].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(decoded[o].AsSpan())),
                                $"{what}: storage {o} differs from attention_decode_f32's (a block a row) bit for bit");
                        }
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnAttentionPacked(Device device)
    {
        _ = device;
        const int Heads = 2;
        const float Scale = 0.2f;
        var worst = new Dictionary<string, float>();
        var random = new Random(42);
        (int Rph, int Steps, int Capacity, int Position, int Dim, int Window, float Softcap, int Chunk)[] cases =
        [
            (1, 1, 5, 4, 37, 0, 0f, 0),
            (4, 2, 300, 295, 37, 0, 0f, 256),                                    // two splits merged
            (3, 3, 60, 20, 30, 7, 3f, 0),                                         // a window and a soft-cap
        ];
        foreach (var (rph, steps, capacity, position, dim, window, softcap, chunk) in cases)
        {
            int rows = Heads * rph;
            float[] q = RowsRandom(random, rows * dim, 2f);
            var (splits, size) = TestSplits(capacity, chunk);
            var variant = new AttentionVariant(window, softcap);
            foreach (var isa in AttentionIsas)
            {
                void Run(GcnMemory m, ulong[] a, (int X, int Y) grid, ulong seed, string kernel, int pointers, int words, ulong counters = 0)
                {
                    ulong part = splits > 1 ? m.Map<float>("part", new float[rows * splits * (dim + 2)]) : 0UL;
                    ulong[] values = [.. a.Take(pointers), 0UL, part, counters, (ulong)rph, (ulong)steps, (ulong)capacity, (ulong)dim, Bits(Scale), (ulong)words,
                        (ulong)window, Bits(softcap), (ulong)size, (ulong)splits, (ulong)rows];
                    AttentionDispatch(kernel, isa, m, (grid.X, splits), seed, values);
                    if (splits > 1 && counters == 0)
                    {
                        AttentionDispatch("attention_merge_f32", isa, m, (grid.X, 1), seed, part, a[pointers - 1], 0UL, (ulong)splits, (ulong)dim, 1UL, (ulong)rows, 0UL, 0UL,
                            (ulong)dim, (ulong)rows);
                    }
                }

                string shape = $"{Heads} heads x {rph} rows, cache {capacity} at {position}, head {dim}, window {window}, cap {softcap}, {splits} splits";
                float[] keys = PackHalves(RowsRandom(random, Heads * capacity * dim, 2f), Heads * capacity, dim);
                float[] values = PackHalves(RowsRandom(random, Heads * capacity * dim, 2f), Heads * capacity, dim);
                RowsCase($"attention_decode_bf16 ({isa}), {shape}", [q, keys, values, [position], new float[rows * dim]], rows, [4],
                    (b, s) => b.AttentionBFloat16Kernel(s[0], s[1], s[2], s[3], s[4], Heads, rph, steps, capacity, dim, Scale, tiled: false, variant),
                    (m, a, grid, seed) => Run(m, a, grid, seed, "attention_decode_bf16", 5, (dim + 1) / 2),
                    worst, splits > 1 ? "attention_decode_bf16 + attention_merge_f32" : "attention_decode_bf16");
                if (splits > 1)
                {
                    AttentionMergedAlike($"attention_decode_bf16 ({isa}), {shape}", [q, keys, values, [position], new float[rows * dim]], [4], rows,
                        (m, a, grid, seed, counters) => Run(m, a, grid, seed, "attention_decode_bf16", 5, (dim + 1) / 2, counters));
                }

                float[] keys8 = PackBytes(random, Heads * capacity, dim), values8 = PackBytes(random, Heads * capacity, dim);
                float[] keyScales = RowsRandom(random, Heads * capacity, 0.01f, 0.02f), valueScales = RowsRandom(random, Heads * capacity, 0.01f, 0.02f);
                RowsCase($"attention_decode_int8 ({isa}), {shape}", [q, keys8, values8, keyScales, valueScales, [position], new float[rows * dim]], rows, [6],
                    (b, s) => b.AttentionInt8Kernel(s[0], s[1], s[2], s[3], s[4], s[5], s[6], Heads, rph, steps, capacity, dim, Scale, tiled: false, variant),
                    (m, a, grid, seed) => Run(m, a, grid, seed, "attention_decode_int8", 7, (dim + 3) / 4),
                    worst, splits > 1 ? "attention_decode_int8 + attention_merge_f32" : "attention_decode_int8");
                if (splits > 1)
                {
                    AttentionMergedAlike($"attention_decode_int8 ({isa}), {shape}", [q, keys8, values8, keyScales, valueScales, [position], new float[rows * dim]], [6], rows,
                        (m, a, grid, seed, counters) => Run(m, a, grid, seed, "attention_decode_int8", 7, (dim + 3) / 4, counters));
                }

            }
        }

        RowsReport(worst);
    }

    internal static void GcnAttentionSpans(Device device)
    {
        _ = device;
        const float Scale = 0.15f;
        var worst = new Dictionary<string, float>();
        var random = new Random(43);

        // Grouped heads (4 query heads over 2 key/value heads), two tables of ranges (2 heads each).
        const int Heads = 4, KvHeads = 2, PerTable = 2;
        (int Rows, int KeyRows, int Dim, float Softcap, int Chunk)[] cases = [(9, 37, 37, 0f, 0), (6, 300, 64, 0f, 256), (5, 20, 24, 1.5f, 0)];
        foreach (var (rows, keyRows, dim, softcap, chunk) in cases)
        {
            int all = Heads * rows;
            float[] q = RowsRandom(random, all * dim, 2f), keys = RowsRandom(random, KvHeads * keyRows * dim, 2f), values = RowsRandom(random, KvHeads * keyRows * dim, 2f);
            float[] starts = [.. Enumerable.Range(0, 2 * rows).Select(i => (float)(i % 3 == 2 ? -4 : i % 3))];
            float[] ends = [.. Enumerable.Range(0, 2 * rows).Select(i => (float)(keyRows + 3 - i))];
            ends[1] = 1f;                                                         // row 1 of the first table: keys 1 to 1, empty
            var (splits, size) = TestSplits(keyRows, chunk);
            foreach (var isa in AttentionIsas)
            {
                string what = $"attention_spans_f32 ({isa}), {Heads} heads over {KvHeads}, {rows} rows over {keyRows} keys, head {dim}, cap {softcap}, {splits} splits";
                RowsCase(what, [q, keys, values, starts, ends, new float[all * dim], new float[all]], all, [5, 6],
                    (b, s) => b.AttentionSpansKernel(s[0], s[1], s[2], s[3], s[4], s[5], s[6], Heads, KvHeads, PerTable, rows, keyRows, dim, Scale, new AttentionVariant(0, softcap)),
                    (m, a, grid, seed) => Launch(m, a, grid, seed, 0UL),
                    worst, splits > 1 ? "attention_spans_f32 + attention_merge_f32" : "attention_spans_f32");
                if (splits > 1)
                {
                    AttentionMergedAlike(what, [q, keys, values, starts, ends, new float[all * dim], new float[all]], [5, 6], all, Launch);
                }

                void Launch(GcnMemory m, ulong[] a, (int X, int Y) grid, ulong seed, ulong counters)
                {
                    ulong part = splits > 1 ? m.Map<float>("part", new float[all * splits * (dim + 2)]) : 0UL;
                    AttentionDispatch("attention_spans_f32", isa, m, (grid.X, splits), seed, a[0], a[1], a[2], a[3], a[4], a[5], a[6], part, counters,
                        Heads, Heads / KvHeads, PerTable, (ulong)rows, (ulong)keyRows, (ulong)dim, Bits(Scale), Bits(softcap), (ulong)size, (ulong)splits,
                        0UL, (ulong)(rows * dim), (ulong)dim, 0UL, (ulong)(keyRows * dim), (ulong)dim, 0UL, (ulong)(rows * dim), (ulong)dim, (ulong)all);
                    if (splits > 1 && counters == 0)
                    {
                        AttentionDispatch("attention_merge_f32", isa, m, (grid.X, 1), seed, part, a[5], a[6], (ulong)splits, (ulong)dim, Heads, (ulong)rows, 0UL,
                            (ulong)(rows * dim), (ulong)dim, (ulong)all);
                    }
                }
            }
        }

        // The strided form: two batches of packed rows [q heads | k heads | v heads], read in place.
        {
            const int Batch = 2, SHeads = 2, SKv = 1, SPerTable = 1, Rows = 7, Dim = 24;
            int row = (SHeads + 2 * SKv) * Dim, all = Batch * SHeads * Rows;
            float[] packed = RowsRandom(random, Batch * Rows * row, 2f);
            float[] starts = [.. Enumerable.Range(0, Batch * SHeads * Rows).Select(i => (float)Math.Max(0, i % Rows - 3))];
            float[] ends = [.. Enumerable.Range(0, Batch * SHeads * Rows).Select(i => (float)(i % Rows + 1))];
            foreach (var isa in AttentionIsas)
            {
                RowsCase($"attention_spans_f32 strided ({isa}), {Batch} batches of {SHeads} heads over {SKv}, {Rows} rows, head {Dim}", [packed, starts, ends, new float[all * Dim]], all, [3],
                    (b, s) => b.AttentionSpansStridedKernel(s[0], 0, s[0], SHeads * Dim, s[0], (SHeads + SKv) * Dim, row, row, s[1], s[2], s[3], Batch, SHeads, SKv, SPerTable, Rows, Dim, Scale),
                    (m, a, grid, seed) => AttentionDispatch("attention_spans_f32", isa, m, (grid.X, 1), seed, a[0], a[0] + (ulong)(4 * SHeads * Dim),
                        a[0] + (ulong)(4 * (SHeads + SKv) * Dim), a[1], a[2], a[3], 0UL, 0UL, 0UL, SHeads, SHeads / SKv, SPerTable, Rows, Rows, Dim, Bits(Scale), 0UL, Rows, 1UL,
                        (ulong)(Rows * row), Dim, (ulong)row, (ulong)(Rows * row), Dim, (ulong)row, (ulong)(Rows * SHeads * Dim), Dim, (ulong)(SHeads * Dim), (ulong)all),
                    worst, "attention_spans_f32 (strided)");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnAttentionInt8Products(Device device)
    {
        _ = device;
        const int Rows = 2, Steps = 3, Capacity = 37, Dim = 37;
        var worst = new Dictionary<string, float>();
        var random = new Random(44);
        int words = (Dim + 3) / 4;
        float[] cache = PackBytes(random, Rows * Capacity, Dim), scales = RowsRandom(random, Rows * Capacity, 0.01f, 0.02f);
        float[] q = RowsRandom(random, Rows * Steps * Dim, 2f), weights = RowsRandom(random, Rows * Steps * Capacity, 0.5f, 0.5f);
        foreach (var isa in AttentionIsas)
        {
            int n = Rows * Steps * Capacity;
            RowsCase($"attn_scores_int8 ({isa}), {Rows} x {Steps} x {Capacity}, head {Dim}", [q, cache, scales, new float[n]], ShapeGrid(n), [3],
                (b, s) => b.AttentionScoresInt8Kernel(s[0], s[1], s[2], s[3], Rows, Steps, Capacity, Dim),
                (m, a, grid, seed) => AttentionDispatch("attn_scores_int8", isa, m, (grid.X, 1), seed, a[0], a[1], a[2], a[3], Steps, Capacity, Dim, (ulong)words, (ulong)n),
                worst, "attn_scores_int8");
            int contexts = Rows * Steps * words;
            RowsCase($"attn_context_int8 ({isa}), {Rows} x {Steps} x {Capacity}, head {Dim}", [weights, cache, scales, new float[Rows * Steps * Dim]], ShapeGrid(contexts), [3],
                (b, s) => b.AttentionContextInt8Kernel(s[0], s[1], s[2], s[3], Rows, Steps, Capacity, Dim),
                (m, a, grid, seed) => AttentionDispatch("attn_context_int8", isa, m, (grid.X, 1), seed, a[0], a[1], a[2], a[3], Steps, Capacity, Dim, (ulong)words,
                    (ulong)contexts),
                worst, "attn_context_int8");
        }

        RowsReport(worst);
    }

    internal static void GcnShapesCache(Device device)
    {
        _ = device;
        const int Heads = 3, Steps = 2, Capacity = 9, Dim = 37, Position = 4;
        var random = new Random(45);
        float[] source = RowsRandom(random, Heads * Steps * Dim, 3f);
        source[5] = float.NaN;                                                    // kept quiet in bfloat16
        source[Dim + 7] = 1.00390625f;                                            // a tie: to even
        for (int d = 0; d < Dim; d++)
        {
            source[(Heads * Steps - 1) * Dim + d] = 0f;                          // the last row all zeros: scale 0, bytes 0
        }

        int words2 = (Dim + 1) / 2, words4 = (Dim + 3) / 4;
        foreach (var isa in AttentionIsas)
        {
            int n = Heads * Steps * Dim;
            ShapesCase($"kv_write_f32 ({isa})", [source, [Position], RowsRandom(random, Heads * Capacity * Dim, 1f)], ShapeGrid(n), [2],
                (b, s) => b.KeyValueWriteKernel(s[0], s[2], s[1], Heads, Steps, Capacity, Dim),
                (m, a, grid, seed) => AttentionDispatch("kv_write_f32", isa, m, (grid.X, 1), seed, a[0], a[2], a[1], Steps * Dim, Capacity * Dim, Dim, (ulong)n));
            int n2 = Heads * Steps * words2;
            ShapesCase($"kv_write_bf16 ({isa})", [source, [Position], RowsRandom(random, Heads * Capacity * words2, 1f)], ShapeGrid(n2), [2],
                (b, s) => b.KeyValueWriteBFloat16Kernel(s[0], s[2], s[1], Heads, Steps, Capacity, Dim),
                (m, a, grid, seed) => AttentionDispatch("kv_write_bf16", isa, m, (grid.X, 1), seed, a[0], a[2], a[1], Steps, Capacity, Dim, (ulong)words2, (ulong)n2));
            float[] clean = [.. source.Select(v => float.IsNaN(v) ? 0.5f : v)];
            ShapesCase($"kv_write_int8 ({isa})", [clean, [Position], RowsRandom(random, Heads * Capacity * words4, 1f), RowsRandom(random, Heads * Capacity, 1f)],
                ShapeGrid(Heads * Steps), [2, 3],
                (b, s) => b.KeyValueWriteInt8Kernel(s[0], s[2], s[3], s[1], Heads, Steps, Capacity, Dim),
                (m, a, grid, seed) => AttentionDispatch("kv_write_int8", isa, m, (grid.X, 1), seed, a[0], a[2], a[3], a[1], Steps, Capacity, Dim, (ulong)words4,
                    Heads * Steps));
            const int MaskRows = 3, MaskCapacity = 50;
            ShapesCase($"decoder_mask_f32 ({isa})", [[6f], new float[MaskRows * MaskCapacity]], ShapeGrid(MaskRows * MaskCapacity), [1],
                (b, s) => b.DecoderMaskKernel(s[0], s[1], MaskRows, MaskCapacity),
                (m, a, grid, seed) => AttentionDispatch("decoder_mask_f32", isa, m, (grid.X, 1), seed, a[0], a[1], MaskCapacity, MaskRows * MaskCapacity));
        }
    }

    internal static void GcnShapesGather(Device device)
    {
        _ = device;
        const int Vocabulary = 37, Dim = 13, Count = 5;
        var random = new Random(46);
        float[] ids = [0f, 5f, 36f, 2f, 36f];
        float[] table = RowsRandom(random, Vocabulary * Dim, 2f);
        float[] rowsTable = PackHalves(RowsRandom(random, Vocabulary * Dim, 2f), Vocabulary, Dim);
        float[] columnsTable = PackHalves(RowsRandom(random, Dim * Vocabulary, 2f), Dim, Vocabulary);
        int n = Count * Dim;
        foreach (var isa in AttentionIsas)
        {
            ShapesCase($"gather_f32 ({isa})", [table, ids, new float[n]], ShapeGrid(n), [2],
                (b, s) => b.GatherKernel(s[0], s[1], s[2], Count, Dim, Vocabulary),
                (m, a, grid, seed) => AttentionDispatch("gather_f32", isa, m, (grid.X, 1), seed, a[0], a[1], a[2], Dim, Vocabulary - 1, (ulong)n));
            ShapesCase($"gather_bf16_f32 ({isa})", [rowsTable, ids, new float[n]], ShapeGrid(n), [2],
                (b, s) => b.GatherBFloat16Kernel(s[0], s[1], s[2], Count, Dim, Vocabulary),
                (m, a, grid, seed) => AttentionDispatch("gather_bf16_f32", isa, m, (grid.X, 1), seed, a[0], a[1], a[2], Dim, Vocabulary - 1, (Dim + 1) / 2, (ulong)n));
            ShapesCase($"gather_bf16_cols_f32 ({isa})", [columnsTable, ids, new float[n]], ShapeGrid(n), [2],
                (b, s) => b.GatherBFloat16ColumnsKernel(s[0], s[1], s[2], Count, Dim, Vocabulary),
                (m, a, grid, seed) => AttentionDispatch("gather_bf16_cols_f32", isa, m, (grid.X, 1), seed, a[0], a[1], a[2], Dim, Vocabulary - 1, (Vocabulary + 1) / 2,
                    (ulong)n));
        }
    }

    internal static void GcnShapesPermute(Device device)
    {
        _ = device;
        var random = new Random(47);
        (int[] Shape, int[] Strides, int Inputs)[] permutes =
        [
            ([2, 4, 5, 3], [60, 5, 1, 20], 120),
            ([7], [3], 21),
            ([2, 1, 3, 2, 2, 3], [36, 0, 1, 18, 3, 6], 72),
        ];
        foreach (var isa in AttentionIsas)
        {
            foreach (var (shape, strides, inputs) in permutes)
            {
                int n = shape.Aggregate(1, (p, s) => p * s);
                foreach (bool accumulate in (bool[])[false, true])
                {
                    ShapesCase($"permute_f32 ({isa}), [{string.Join(", ", shape)}], adding {accumulate}", [RowsRandom(random, inputs, 2f), RowsRandom(random, n, 1f)], ShapeGrid(n), [1],
                        (b, s) => b.PermuteKernel(s[0], s[1], shape, strides, accumulate),
                        (m, a, grid, seed) =>
                        {
                            var values = new List<ulong> { a[0], a[1] };
                            int pad = GcnKernels.ShapePermuteRank - shape.Length;
                            values.AddRange(Enumerable.Range(0, GcnKernels.ShapePermuteRank).Select(d => d < pad ? 1UL : (ulong)shape[d - pad]));
                            values.AddRange(Enumerable.Range(0, GcnKernels.ShapePermuteRank).Select(d => d < pad ? 0UL : (ulong)strides[d - pad]));
                            values.Add(accumulate ? 1UL : 0UL);
                            values.Add((ulong)n);
                            AttentionDispatch("permute_f32", isa, m, (grid.X, 1), seed, [.. values]);
                        });
                }
            }

            foreach (bool accumulate in (bool[])[false, true])
            {
                const int Rows = 6, Cols = 9, SrcOffset = 5, SrcStride = 13, DstOffset = 3, DstStride = 20;
                ShapesCase($"copy2d_f32 ({isa}), adding {accumulate}", [RowsRandom(random, 200, 2f), RowsRandom(random, 200, 1f)], ShapeGrid(Rows * Cols), [1],
                    (b, s) => b.Copy2D(s[0], SrcOffset, SrcStride, s[1], DstOffset, DstStride, Rows, Cols, accumulate),
                    (m, a, grid, seed) => AttentionDispatch("copy2d_f32", isa, m, (grid.X, 1), seed, a[0], a[1], SrcOffset, SrcStride, DstOffset, DstStride, Cols,
                        accumulate ? 1UL : 0UL, Rows * Cols));
            }
        }
    }
}
