// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;
using GcnWeights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;
using Variant = Idrak.Gpu.Hip.GcnKernels.PackedGemvVariant;

// HIP's decoding kernels in GCN (GcnKernels.Decoding, and the element-wise kernels G5a added to GcnKernels.Pointwise),
// each run by the GCN interpreter and proven against the CPU (plan 35, G5a; its rest: the sampler, the heads' norm and
// rotation, gemv_multi_f32). Every kernel is assembled for gfx90c and
// gfx906 (its printed text reading back to its bytes; with IDRAK_GCN_PRINTED naming a folder, written there for the
// development check against llvm-mc), run over shapes that are not multiples of anything (fewer rows than a kernel's,
// k in one split and in several, widths over more than one block, special values), compared to the CPU backend within
// the device-against-CPU tolerance (OnBoth's, 1e-3 of max(1, |expected|); NaN and infinities where the CPU has them),
// run a second time with another wave order and grid to give the same bits, and the interpreter reports no missing
// wait, race or access out of range (a report fails the run). Each test prints each kernel's largest error.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnDecodingGroup =
    [
        ("gcn kernels: decoding: every decoding kernel and every element-wise kernel G5a added assembles for gfx90c and gfx906 with its arguments and its declared block, within its register budget, its printed text reading back to its bytes; GcnKernels.All lists each once", GcnDecodingAssembled),
        ("gcn kernels: decoding: the gated products {int8,int4,bf16}_gemv_{silu,gelu}[_rR]_f32 equal the CPU's gated activation then packed product for 1, 3 and 8 rows, in one split and in several (added by int8_gemv_finish_f32, or by each column range's last split block: the same bits in any order of the blocks, the counters left at zero); the same bits on a second run", GcnDecodingGated),
        ("gcn kernels: decoding: the multi-output products {int8,int4,bf16}_gemv_multi[_rR]_f32 equal the CPU's products (with a bias) for one and three products of different widths, 1 and 5 rows, one split (written whole) and several, and a gate/up pair with act(gate) · up for SiLU, GELU and ReLU, finished by gemv_multi_finish_f32 or by the launch's last blocks (the same bits, the counters left at zero); the same bits on a second run", GcnDecodingMulti),
        ("gcn kernels: decoding: a packed product finished by gemv_addnorm_f32, and {int8,int4,bf16}_gemv_addnorm[_rR]_f32 finished by its last block, equal the CPU's product, the residual addition and RMS norm, with y's own norm first (the sandwich block) and without, in one split and in several; the one launch the two launches' bit for bit in any order of the blocks; the same bits on a second run", GcnDecodingAddNorm),
        ("gcn kernels: decoding: attention_rows_f32 and attention_rows_bf16 equal the CPU's AttentionRows (each row from its sequence's start; padding rows of zeros; a window and a soft-cap; splits merged by attention_merge_f32 or by each row's last split block, the same bits), and segment_spans_f32 then attention_spans_f32 equal the CPU's AttentionSegmented (with the log-sum-exp, a window and a soft-cap); the same bits on a second run", GcnDecodingAttention),
        ("gcn kernels: decoding: pow_f32 (every exponent kind, negative bases, zeros, infinities, NaN), clamp_f32, leaky_relu_f32, where_f32 (±0 and NaN conditions), inv_sqrt_f32, bias_gelu_f32, add_scalar_f32 and the four gated activations equal the CPU's; the same bits on two grids", GcnDecodingPointwise),
        ("gcn kernels: decoding: argmax_f32 (ties, all -inf, a NaN first), class_match_f32 (one column and many), add_rms_norm_affine_f32, rms_norm_rope_f32 (interleaved or not, dimensions past the rotated ones) and penalize_rows_f32 (repeated ids, ids out of the vocabulary, a wrapped history) equal the CPU's over rows of 1 to 2,333 columns; the same bits on two grids", GcnDecodingRows),
        ("gcn kernels: decoding: rope_f32 (interleaved or not, either sign), one_hot_f32 and history_push_f32 equal the CPU's, the same bits on two grids", GcnDecodingShapes),
        ("gcn kernels: decoding: rms_norm_rope2_f32 (queries and keys, interleaved or not, every dimension rotated or some) and norm_rope_heads_f32 (norms or none, rotation or none, values copied; no cache, a float32 cache with padded rows and a bfloat16 cache at a position) equal the CPU's composed operations; the same bits on two grids", GcnDecodingNormRope),
        ("gcn kernels: decoding: gemv_multi_f32 equals the CPU's float32 products of one input (MatMulMany: biases on some) for one and three products of different widths and 1 and 5 rows; the same bits on a second run", GcnDecodingFloatMulti),
        ("gcn kernels: decoding: the sampler (sample_rows_f32; topk_candidates_f32 then sample_candidates_f32; topk_greedy_f32 then sample_candidates_greedy_f32, or topk_greedy_f32 alone, its last block of a row sampling it) draws the CPU's tokens from its seeded stream over two steps (temperature, top-k, top-p, min-p; tied scores, +∞, NaN and all -∞ rows; ties overflowing a candidate block): the same tokens and alternatives, probabilities and entropy within the tolerance; the same bits on two grids; the one launch's the two launches' bit for bit", GcnDecodingSampler),
        ("gcn kernels: decoding: counts: the launches and the interpreter's instructions, memory instructions, waits and barriers per element of each fused decoding kernel finished by its last block against its second launch (a split product, the residual and norm, a gate/up pair, a split attention, a greedy sampler), printed as G6b's evidence", GcnDecodingCounts),
    ];

    private static readonly Dictionary<(string, GcnIsaVersion), (GcnKernelCode Code, GcnKernelImage Image)> decodingKernels = [];

    // The element-wise kernels G5a added (GcnKernels.Pointwise), tested with the decoding ones.
    private static readonly string[] DecodingPointwiseNames =
        ["pow_f32", "clamp_f32", "leaky_relu_f32", "where_f32", "inv_sqrt_f32", "bias_gelu_f32", "add_scalar_f32", .. GcnKernels.GatedKinds];

    // Any kernel of the set assembled for isa, its printed text checked to read back to the same bytes, written into a
    // code object and read back (once a process).
    private static (GcnKernelCode Code, GcnKernelImage Image) DecodingKernel(string name, GcnIsaVersion isa)
    {
        lock (decodingKernels)
        {
            if (decodingKernels.TryGetValue((name, isa), out var known))
            {
                return known;
            }

            var source = GcnKernels.All.Single(k => k.Name == name);
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
            return decodingKernels[(name, isa)] = (code, image);
        }
    }

    // One dispatch of any kernel of the set over `grid` blocks of its declared block, its arguments as 64-bit values in order.
    private static GcnRunResult DecodingDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y, int Z) grid, ulong seed, params ulong[] values)
    {
        var (code, image) = DecodingKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        GcnKernels.PackArguments(code.Arguments, arguments, values);
        int block = code.RequiredWorkgroupSize?.X ?? code.MaxFlatWorkgroupSize;
        return GcnInterpreter.Run(image, memory, new GcnDispatch
        {
            Workgroups = grid, WorkgroupSize = (block, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1,
        });
    }

    // A row kernel's dispatch as RowsCase gives it (x × y blocks).
    private static void DecodingRowDispatch(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y) grid, ulong seed, params ulong[] values) =>
        DecodingDispatch(name, isa, memory, (grid.X, grid.Y, 1), seed, values);

    // Blocks of 256 for n work-items: one on the first run (the grid-stride loop), every one on the second.
    private static int DecodingBlocks(long n, bool second) => second ? (int)Math.Max(1, (n + 255) / 256) : 1;

    internal static void GcnDecodingAssembled(Device device)
    {
        _ = device;
        var names = GcnKernels.Decoding.Select(k => k.Name).ToArray();
        int fused = GcnKernels.FusedGemvFormats.Count * 4 * GcnKernels.PackedGemvRowCounts.Length;
        Check(names.Length == fused + 20 && names.Distinct().Count() == names.Length, $"{fused + 20} decoding kernels, each once: {string.Join(", ", names)}");
        foreach (var format in GcnKernels.FusedGemvFormats)
        {
            foreach (var variant in (Variant[])[Variant.Silu, Variant.Gelu, Variant.Multi, Variant.AddNorm])
            {
                for (int m = 1; m <= GcnKernels.GemvRows; m++)
                {
                    string kernel = GcnKernels.FusedGemvKernel(format, variant, m);
                    int rows = m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : 8;
                    string prefix = format switch { GcnWeights.Int8 => "int8", GcnWeights.Int4 => "int4", _ => "bf16" };
                    string kind = variant.ToString().ToLowerInvariant();
                    Check(kernel == (rows == 8 ? $"{prefix}_gemv_{kind}_f32" : $"{prefix}_gemv_{kind}_r{rows}_f32") && names.Contains(kernel), $"{format} {variant}, {m} rows: {kernel}");
                }
            }
        }

        Check(GcnKernels.PowFlags(3f) == 1 && GcnKernels.PowFlags(1.5f) == 2 && GcnKernels.PowFlags(float.NaN) == 4 && GcnKernels.PowFlags(0f) == 8 && GcnKernels.PowFlags(1f) == 17
              && GcnKernels.PowFlags(2f) == 32 && GcnKernels.PowFlags(-1f) == 1 && GcnKernels.PowFlags(float.PositiveInfinity) == 0, "pow_f32's exponent flags are CUDA's");
        foreach (string name in names.Concat(DecodingPointwiseNames))
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(GcnKernels.All.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            foreach (var isa in ProductIsas)
            {
                var (code, _) = DecodingKernel(name, isa);
                int block = code.RequiredWorkgroupSize?.X ?? 0;
                Check(block is 256 or GcnKernels.PackedGemvWords && code.MaxFlatWorkgroupSize == block && code.WaveSize == 64,
                    $"{name} ({isa}): block {code.RequiredWorkgroupSize} (max flat {code.MaxFlatWorkgroupSize}), wave {code.WaveSize}");
                Check(code.LdsBytes <= 8192, $"{name} ({isa}): {code.LdsBytes} bytes of LDS");
                if (name.Contains("_gemv_", StringComparison.Ordinal))
                {
                    var (expected, bytes) = name.Contains("_multi", StringComparison.Ordinal) ? (GcnKernels.MultiGemvArguments, GcnKernels.MultiGemvArgumentBytes)
                        : name.Contains("_addnorm", StringComparison.Ordinal) ? (GcnKernels.AddNormGemvArguments, GcnKernels.AddNormGemvArgumentBytes)
                        : (GcnKernels.GatedGemvArguments, GcnKernels.GatedGemvArgumentBytes);
                    Check(code.Arguments.SequenceEqual(expected) && GcnCodeObject.KernargBytes(code) >= bytes, $"{name} ({isa}): the arguments {string.Join(", ", code.Arguments)}");
                }

                Console.WriteLine($"    {name} ({isa}): {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
            }
        }

        foreach (var (name, list) in new (string, IReadOnlyList<GcnArgument>)[]
                 {
                     ("gemv_multi_finish_f32", GcnKernels.MultiFinishArguments), ("segment_spans_f32", GcnKernels.SegmentSpansArguments), ("rope_f32", GcnKernels.RopeArguments),
                     ("one_hot_f32", GcnKernels.OneHotArguments), ("history_push_f32", GcnKernels.HistoryPushArguments),
                     ("rms_norm_rope_f32", GcnKernels.NormRopeArguments(GcnKernels.NormRopeForm.Single)), ("rms_norm_rope2_f32", GcnKernels.NormRopeArguments(GcnKernels.NormRopeForm.Pair)),
                     ("norm_rope_heads_f32", GcnKernels.NormRopeArguments(GcnKernels.NormRopeForm.Heads)), ("topk_candidates_f32", GcnKernels.CandidateArguments(greedy: false)),
                     ("topk_greedy_f32", GcnKernels.CandidateArguments(greedy: true)), ("sample_rows_f32", GcnKernels.SamplerArguments(GcnKernels.SamplerForm.Rows)),
                     ("sample_candidates_f32", GcnKernels.SamplerArguments(GcnKernels.SamplerForm.Candidates)),
                     ("sample_candidates_greedy_f32", GcnKernels.SamplerArguments(GcnKernels.SamplerForm.Greedy)), ("gemv_multi_f32", GcnKernels.GemvMultiArguments),
                 })
        {
            var code = DecodingKernel(name, ProductIsas[0]).Code;
            Check(code.Arguments.SequenceEqual(list) && GcnCodeObject.KernargBytes(code) >= list[^1].Offset + list[^1].Size, $"{name}: its arguments as listed ({string.Join(", ", code.Arguments)})");
        }

        // The kernels' names and their CUDA counterparts' argument names, in order.
        Check(GcnKernels.SamplerArguments(GcnKernels.SamplerForm.Rows).Select(a => a.Name).SequenceEqual(
                  ["logits", "ids", "stats", "step", "vocab", "rowstride", "rowoffset", "invt", "topk", "topp", "logminp", "seed", "rows", "n"]),
            "sample_rows_f32: CUDA's arguments, ln(min-p) where CUDA has min-p");
        Check(GcnKernels.NormRopeArguments(GcnKernels.NormRopeForm.Heads).Count == 28 && GcnKernels.NormRopeArguments(GcnKernels.NormRopeForm.Pair).Count == 21,
            "norm_rope_heads_f32 and rms_norm_rope2_f32 take CUDA's 28 and 21 arguments");
        var settings = GcnKernels.SamplerSettings.Of(5, 5, 0, 0f, -3, 0.9f, 0.05f, 9u, 2);
        Check(settings.InverseTemperature == 1f / 1e-3f && settings.TopK == 0 && settings.LogMinP == MathF.Log(0.05f)
              && float.IsNegativeInfinity(GcnKernels.SamplerSettings.Of(5, 5, 0, 1f, 1, 1f, 0f, 9u, 2).LogMinP)
              && float.IsNegativeInfinity(GcnKernels.SamplerSettings.Of(5, 5, 0, 1f, 1, 1f, float.NaN, 9u, 2).LogMinP),
            "the sampler's settings: the CPU's inverse temperature, a negative top-k as none, the CPU's ln(min-p) (-inf without min-p)");
    }

    // The CPU's product of `format`'s weights (int8: scales per column; int4: per group; bfloat16: none).
    private static void DecodingCpuProduct(Backend cpu, GcnWeights format, Storage x, Storage packed, Storage scales, Storage y, int m, int n, int k)
    {
        switch (format)
        {
            case GcnWeights.Int8:
                cpu.Int8MatMulKernel(x, packed, scales, y, m, n, k);
                break;
            case GcnWeights.Int4:
                cpu.Int4MatMulKernel(x, packed, scales, y, m, n, k);
                break;
            default:
                cpu.BFloat16MatMulKernel(x, packed, y, m, n, k);
                break;
        }
    }

    // The test shapes of a format: n, k and the chunk of k the split cases take (int4's on whole groups).
    private static (int N, int K, int Chunk) DecodingShape(GcnWeights format) => format switch
    {
        GcnWeights.Int8 => (133, 70, 32),
        GcnWeights.Int4 => (135, 96, 64),
        _ => (131, 70, 32),
    };

    // Both finishes of a split sum or a fused kernel's tail (plan 35, G6b): a second launch (lastBlock false) and the
    // launch's last block (true, its completion counters at inputs[counters]), each within the tolerance of the CPU and
    // the same bits on two grids and block orders (ProductsCase); then the last block's form in three more orders of the
    // blocks (so a different block finishes), each equal to the second launch's form bit for bit on every output and
    // leaving every counter at zero (the next launch, and a replay, finds them so).
    private static void FinishCase(string what, float[][] inputs, int[] outputs, float[][] expected, Action<GcnMemory, ulong[], bool, ulong, bool> gcn, int counters,
        Dictionary<string, float> worst, string launchKernel, string blockKernel)
    {
        ProductsCase($"{what}, a second launch", inputs, outputs, expected, (m, a, second, seed) => gcn(m, a, second, seed, false), worst, launchKernel);
        ProductsCase($"{what}, the last block", inputs, outputs, expected, (m, a, second, seed) => gcn(m, a, second, seed, true), worst, blockKernel);
        float[][] launched = RowsOnGcn(inputs, (m, a) => gcn(m, a, true, 7, false));
        foreach (ulong seed in (ulong[])[3, 11, 29])
        {
            float[][] finished = RowsOnGcn(inputs, (m, a) =>
            {
                gcn(m, a, true, seed, true);
                gcn(m, a, true, seed + 1, true);                                 // a replay finds the counters at zero
            });
            foreach (int o in outputs)
            {
                Check(MemoryMarshal.AsBytes(launched[o].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(finished[o].AsSpan())),
                    $"{what}: storage {o} by the last block (blocks in order {seed}) differs from the second launch's bit for bit");
            }

            Check(finished[counters].All(v => BitConverter.SingleToUInt32Bits(v) == 0), $"{what}: the last blocks left a counter that is not zero (blocks in order {seed})");
        }
    }

    internal static void GcnDecodingGated(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(51);
        foreach (var format in GcnKernels.FusedGemvFormats)
        {
            var (n, k, splitChunk) = DecodingShape(format);
            int words = (n + GcnKernels.ColumnsPerWord(format) - 1) / GcnKernels.ColumnsPerWord(format);
            int columnBlocks = (words + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
            bool scaled = format != GcnWeights.BFloat16;
            var (packed, scales) = ProductsWeights(format, random, k, n);
            foreach (int activation in (int[])[0, 1])
            {
                foreach (int m in (int[])[1, 3, 8])
                {
                    string kernel = GcnKernels.FusedGemvKernel(format, activation == 0 ? Variant.Silu : Variant.Gelu, m);
                    float[] gate = RowsRandom(random, m * k, 2f), up = RowsRandom(random, m * k, 1f);
                    float[] expected = RowsOnCpu([gate, up, packed, scales, new float[m * k], new float[m * n]], (cpu, s) =>
                    {
                        cpu.GatedActivationKernel(s[0], s[1], s[4], m * k, activation);
                        DecodingCpuProduct(cpu, format, s[4], s[2], s[3], s[5], m, n, k);
                    })[5];
                    foreach (int chunk in (int[])[k, splitChunk])
                    {
                        int splits = (k + chunk - 1) / chunk;
                        foreach (var isa in ProductIsas)
                        {
                            // Inputs: gate, up, packed, scales, y, part, counters.
                            void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                            {
                                DecodingDispatch(kernel, isa, memory, (columnBlocks, splits, 1), seed,
                                    at[0], at[2], scaled ? at[3] : 0UL, at[4], at[5], (ulong)m, (ulong)n, (ulong)k, (ulong)words, (ulong)chunk, (ulong)splits,
                                    lastBlock ? at[6] : 0UL, at[1]);
                                if (splits > 1 && !lastBlock)
                                {
                                    DecodingDispatch("int8_gemv_finish_f32", isa, memory, (DecodingBlocks(m * n, second), 1, 1), seed,
                                        at[5], format == GcnWeights.Int8 ? at[3] : 0UL, at[4], (ulong)n, (ulong)(m * n), (ulong)splits, (ulong)(m * n));
                                }
                            }

                            float[][] inputs = [gate, up, packed, scales, new float[m * n], new float[splits * m * n], new float[columnBlocks + 1]];
                            float[][] want = [.. inputs];
                            want[4] = expected;
                            string what = $"{kernel}, {m} rows, {k} x {n}, {splits} split(s) ({isa})";
                            if (splits > 1)
                            {
                                FinishCase(what, inputs, [4], want, Run, 6, worst, $"{kernel} + int8_gemv_finish_f32", $"{kernel} (last block)");
                            }
                            else
                            {
                                ProductsCase(what, inputs, [4], want, (mem, at, second, seed) => Run(mem, at, second, seed, false), worst, kernel);
                            }
                        }
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingMulti(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(52);
        foreach (var format in GcnKernels.FusedGemvFormats)
        {
            var (_, k, splitChunk) = DecodingShape(format);
            int cpw = GcnKernels.ColumnsPerWord(format);
            bool scaled = format != GcnWeights.BFloat16;
            int[] widths = [130, 67, 200];
            var weights = widths.Select(w => ProductsWeights(format, random, k, w)).ToArray();
            float[] bias = RowsRandom(random, widths[0], 1f);
            foreach (int m in (int[])[1, 5])
            {
                string kernel = GcnKernels.FusedGemvKernel(format, Variant.Multi, m);
                float[] x = RowsRandom(random, m * k, 1f);
                float[][] expected = new float[3][];
                for (int j = 0; j < 3; j++)
                {
                    int width = widths[j];
                    expected[j] = RowsOnCpu([x, weights[j].Packed, weights[j].Scales, new float[m * width], bias], (cpu, s) =>
                    {
                        DecodingCpuProduct(cpu, format, s[0], s[1], s[2], s[3], m, width, k);
                        if (j == 0)
                        {
                            cpu.AddRowVectorKernel(s[3], s[4], s[3], m, width);
                        }
                    })[3];
                }

                foreach (int count in (int[])[3, 1])
                {
                    int nmax = widths.Take(count).Max(), blocks = ((nmax + cpw - 1) / cpw + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
                    foreach (int chunk in (int[])[k, splitChunk])
                    {
                        int splits = (k + chunk - 1) / chunk;
                        foreach (var isa in ProductIsas)
                        {
                            // Inputs: x, q0, s0, q1, s1, q2, s2, bias, y0, y1, y2, part, counters.
                            void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                            {
                                var values = new List<ulong> { at[0], splits > 1 ? at[11] : 0UL, (ulong)m, (ulong)k, (ulong)chunk, (ulong)splits, lastBlock ? at[12] : 0UL, (ulong)blocks, (ulong)nmax };
                                for (int j = 0; j < 3; j++)
                                {
                                    values.AddRange([at[1 + 2 * j], scaled ? at[2 + 2 * j] : 0UL, at[8 + j], j == 0 ? at[7] : 0UL, j < count ? (ulong)widths[j] : 0UL]);
                                }

                                values.AddRange([0UL, 0UL]);
                                DecodingDispatch(kernel, isa, memory, (blocks, splits, count), seed, [.. values]);
                                if (splits > 1 && !lastBlock)
                                {
                                    var finish = new List<ulong> { at[11], 0UL, (ulong)splits, (ulong)m, (ulong)nmax, 0UL };
                                    for (int j = 0; j < 3; j++)
                                    {
                                        finish.AddRange([format == GcnWeights.Int8 ? at[2 + 2 * j] : 0UL, at[8 + j], j == 0 ? at[7] : 0UL, j < count ? (ulong)widths[j] : 0UL]);
                                    }

                                    DecodingDispatch("gemv_multi_finish_f32", isa, memory, (DecodingBlocks(m * nmax, second), count, 1), seed, [.. finish]);
                                }
                            }

                            float[][] inputs =
                            [
                                x, weights[0].Packed, weights[0].Scales, weights[1].Packed, weights[1].Scales, weights[2].Packed, weights[2].Scales, bias,
                                new float[m * widths[0]], new float[m * widths[1]], new float[m * widths[2]], new float[count * splits * m * nmax], new float[3 * blocks],
                            ];
                            float[][] want = [.. inputs];
                            for (int j = 0; j < count; j++)
                            {
                                want[8 + j] = expected[j];
                            }

                            string what = $"{kernel}, {count} product(s), {m} rows, k {k}, {splits} split(s) ({isa})";
                            int[] outputs = [.. Enumerable.Range(8, count)];
                            if (splits > 1)
                            {
                                FinishCase(what, inputs, outputs, want, Run, 12, worst, $"{kernel} + gemv_multi_finish_f32", $"{kernel} (last block)");
                            }
                            else
                            {
                                ProductsCase(what, inputs, outputs, want, (mem, at, second, seed) => Run(mem, at, second, seed, false), worst, kernel);
                            }
                        }
                    }
                }

                // A gate/up pair of equal widths: hidden = act(gate) · up, by gemv_multi_finish_f32 or the pair's last block.
                int pairWidth = 67, pairBlocks = ((pairWidth + cpw - 1) / cpw + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
                var gateWeights = ProductsWeights(format, random, k, pairWidth);
                foreach (int activation in (int[])[0, 1, 2])
                {
                    float[][] pairExpected = RowsOnCpu([x, gateWeights.Packed, gateWeights.Scales, weights[1].Packed, weights[1].Scales, new float[m * pairWidth],
                        new float[m * pairWidth], new float[m * pairWidth]], (cpu, s) =>
                    {
                        DecodingCpuProduct(cpu, format, s[0], s[1], s[2], s[5], m, pairWidth, k);
                        DecodingCpuProduct(cpu, format, s[0], s[3], s[4], s[6], m, pairWidth, k);
                        cpu.GatedActivationKernel(s[5], s[6], s[7], m * pairWidth, activation);
                    });
                    foreach (int chunk in (int[])[k, splitChunk])
                    {
                        int splits = (k + chunk - 1) / chunk;
                        foreach (var isa in ProductIsas)
                        {
                            // Inputs: x, gate q, gate s, up q, up s, gate y, up y, hidden, part, counters.
                            void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                            {
                                bool parted = splits > 1 || !lastBlock;
                                var values = new List<ulong> { at[0], parted ? at[8] : 0UL, (ulong)m, (ulong)k, (ulong)chunk, (ulong)splits, lastBlock ? at[9] : 0UL, (ulong)pairBlocks, (ulong)pairWidth };
                                var finish = new List<ulong> { at[8], at[7], (ulong)splits, (ulong)m, (ulong)pairWidth, (ulong)(activation + 1) };
                                for (int j = 0; j < 3; j++)
                                {
                                    int p = Math.Min(j, 1);
                                    values.AddRange([at[1 + 2 * p], scaled ? at[2 + 2 * p] : 0UL, at[5 + p], 0UL, j < 2 ? (ulong)pairWidth : 0UL]);
                                    finish.AddRange([format == GcnWeights.Int8 ? at[2 + 2 * p] : 0UL, at[5 + p], 0UL, j < 2 ? (ulong)pairWidth : 0UL]);
                                }

                                values.AddRange([lastBlock ? (ulong)(activation + 1) : 0UL, lastBlock ? at[7] : 0UL]);
                                DecodingDispatch(kernel, isa, memory, (pairBlocks, splits, 2), seed, [.. values]);
                                if (!lastBlock)
                                {
                                    DecodingDispatch("gemv_multi_finish_f32", isa, memory, (DecodingBlocks(m * pairWidth, second), 1, 1), seed, [.. finish]);
                                }
                            }

                            float[][] inputs =
                            [
                                x, gateWeights.Packed, gateWeights.Scales, weights[1].Packed, weights[1].Scales, new float[m * pairWidth], new float[m * pairWidth],
                                new float[m * pairWidth], new float[2 * splits * m * pairWidth], new float[3 * pairBlocks],
                            ];
                            float[][] want = [.. inputs];
                            (want[5], want[6], want[7]) = (pairExpected[5], pairExpected[6], pairExpected[7]);
                            FinishCase($"{kernel}, a gate/up pair (activation {activation}), {m} rows, {splits} split(s) ({isa})", inputs, [5, 6, 7], want, Run, 9, worst,
                                $"{kernel} + gemv_multi_finish_f32 (pair)", $"{kernel} (pair, last block)");
                        }
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingAddNorm(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(53);
        const float Eps = 1e-5f, Offset = 1f, YEps = 1e-6f, YOffset = 0f;
        foreach (var format in GcnKernels.FusedGemvFormats)
        {
            var (n, k, splitChunk) = DecodingShape(format);
            int words = (n + GcnKernels.ColumnsPerWord(format) - 1) / GcnKernels.ColumnsPerWord(format);
            int columnBlocks = (words + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords;
            bool scaled = format != GcnWeights.BFloat16;
            var (packed, scales) = ProductsWeights(format, random, k, n);
            float[] gain = RowsRandom(random, n, 0.5f), yGain = RowsRandom(random, n, 0.5f, 1f);
            foreach (int m in (int[])[1, 4])
            {
                string kernel = GcnKernels.PackedGemvKernel(format, m), fused = GcnKernels.FusedGemvKernel(format, Variant.AddNorm, m);
                float[] x = RowsRandom(random, m * k, 1f), residual = RowsRandom(random, m * n, 2f);
                foreach (bool sandwich in (bool[])[false, true])
                {
                    // Inputs: x, packed, scales, residual, gain, y gain, y, sum, normalized, part, counters.
                    float[][] Inputs(int splits) => [x, packed, scales, residual, gain, yGain, new float[m * n], new float[m * n], new float[m * n], new float[splits * m * n],
                        new float[columnBlocks + 1]];
                    float[][] expected = RowsOnCpu(Inputs(1), (cpu, s) =>
                    {
                        DecodingCpuProduct(cpu, format, s[0], s[1], s[2], s[6], m, n, k);
                        if (sandwich)
                        {
                            cpu.RmsNormAffineKernel(s[6], s[5], s[6], m, n, YEps, YOffset);
                        }

                        cpu.AddRmsNormAffineKernel(s[3], s[6], s[7], s[4], s[8], m, n, Eps, Offset);
                    });
                    foreach (int chunk in (int[])[k, splitChunk])
                    {
                        int splits = (k + chunk - 1) / chunk;
                        foreach (var isa in ProductIsas)
                        {
                            void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                            {
                                ulong part = splits > 1 ? at[9] : at[6];
                                if (lastBlock)
                                {
                                    DecodingDispatch(fused, isa, memory, (columnBlocks, splits, 1), seed,
                                        at[0], at[1], scaled ? at[2] : 0UL, at[6], part, (ulong)m, (ulong)n, (ulong)k, (ulong)words, (ulong)chunk, (ulong)splits, at[10],
                                        at[3], at[7], at[4], at[8], sandwich ? at[5] : 0UL, Bits(Eps), Bits(Offset), Bits(YEps), Bits(YOffset), sandwich ? 1UL : 0UL);
                                    return;
                                }

                                DecodingDispatch(kernel, isa, memory, (columnBlocks, splits, 1), seed,
                                    at[0], at[1], scaled ? at[2] : 0UL, at[6], part, (ulong)m, (ulong)n, (ulong)k, (ulong)words, (ulong)chunk, (ulong)splits, 0UL);
                                DecodingDispatch("gemv_addnorm_f32", isa, memory, (second ? m : 1, 1, 1), seed,
                                    part, splits > 1 && format == GcnWeights.Int8 ? at[2] : 0UL, at[6], at[3], at[7], at[4], at[8], sandwich ? at[5] : 0UL,
                                    (ulong)n, (ulong)splits, (ulong)(m * n), Bits(Eps), Bits(Offset), Bits(YEps), Bits(YOffset), sandwich ? 1UL : 0UL, (ulong)m);
                            }

                            var inputs = Inputs(splits);
                            float[][] want = [.. inputs];
                            (want[6], want[7], want[8]) = (expected[6], expected[7], expected[8]);
                            FinishCase($"{kernel}, the residual and norm{(sandwich ? " (y normalized first)" : "")}, {m} rows, {k} x {n}, {splits} split(s) ({isa})", inputs,
                                [6, 7, 8], want, Run, 10, worst, $"{kernel} + gemv_addnorm_f32", fused);
                        }
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingAttention(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(54);

        // Rows of different lengths over a cache: 4 heads, 2 a packed row, 3 rows a head (steps 3), each row from its start.
        const int Heads = 4, PerRow = 2, Rph = 3, Steps = 3, Dim = 37;
        foreach (var (capacity, position, splitChunk) in ((int, int, int)[])[(40, 30, 0), (300, 290, GcnKernels.AttentionTile)])
        {
            int rows = Heads * Rph, words = (Dim + 1) / 2;
            float[] q = RowsRandom(random, rows * Dim, 1f), keys = RowsRandom(random, Heads * capacity * Dim, 1f), values = RowsRandom(random, Heads * capacity * Dim, 1f);
            float[] starts = [0f, 5f, capacity, 2f, position + 1, position + 3];          // (heads / 2) · steps; some rows see nothing (padding)
            float[] bf16Keys = RowsOnCpu([keys, new float[Heads * capacity * words]], (cpu, s) => cpu.KeyValueWriteBFloat16Kernel(s[0], s[1], cpu.Allocate(1, true), Heads, capacity, capacity, Dim))[1];
            float[] bf16Values = RowsOnCpu([values, new float[Heads * capacity * words]], (cpu, s) => cpu.KeyValueWriteBFloat16Kernel(s[0], s[1], cpu.Allocate(1, true), Heads, capacity, capacity, Dim))[1];
            foreach (var variant in (AttentionVariant[])[default, new(9, 4f)])
            {
                foreach (bool bf16 in (bool[])[false, true])
                {
                    string kernel = bf16 ? "attention_rows_bf16" : "attention_rows_f32";
                    float[] k = bf16 ? bf16Keys : keys, v = bf16 ? bf16Values : values;
                    float[] expected = RowsOnCpu([q, k, v, [position], new float[rows * Dim], starts], (cpu, s) =>
                    {
                        if (bf16)
                        {
                            cpu.AttentionRowsBFloat16(s[0], s[1], s[2], s[3], s[4], s[5], Heads, PerRow, Rph, Steps, capacity, Dim, 0.15f, variant);
                        }
                        else
                        {
                            cpu.AttentionRows(s[0], s[1], s[2], s[3], s[4], s[5], Heads, PerRow, Rph, Steps, capacity, Dim, 0.15f, variant);
                        }
                    })[4];
                    var (splits, chunk) = splitChunk == 0 ? (1, capacity) : ((capacity + splitChunk - 1) / splitChunk, splitChunk);
                    foreach (var isa in ProductIsas)
                    {
                        // Inputs: q, keys, values, position, y, starts, part, counters.
                        void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                        {
                            var list = new List<ulong>
                            {
                                at[0], at[1], at[2], at[3], at[4], 0UL, splits > 1 ? at[6] : 0UL, lastBlock ? at[7] : 0UL, at[5], Rph, Steps, (ulong)capacity, Dim, Bits(0.15f),
                            };
                            if (bf16)
                            {
                                list.Add((ulong)words);
                            }

                            list.AddRange([(ulong)Math.Max(0, variant.Window), Bits(variant.Softcap), (ulong)chunk, (ulong)splits, PerRow, (ulong)rows]);
                            DecodingDispatch(kernel, isa, memory, (second ? rows : 2, splits, 1), seed, [.. list]);
                            if (splits > 1 && !lastBlock)
                            {
                                DecodingDispatch("attention_merge_f32", isa, memory, (second ? rows : 2, 1, 1), seed,
                                    at[6], at[4], 0UL, (ulong)splits, Dim, 1, (ulong)rows, 0, 0, Dim, (ulong)rows);
                            }
                        }

                        float[][] inputs = [q, k, v, [position], new float[rows * Dim], starts, new float[rows * splits * (Dim + 2)], new float[rows]];
                        float[][] want = [.. inputs];
                        want[4] = expected;
                        string what = $"{kernel}, {capacity} positions, window {variant.Window}, soft-cap {variant.Softcap}, {splits} split(s) ({isa})";
                        if (splits > 1)
                        {
                            FinishCase(what, inputs, [4], want, Run, 7, worst, $"{kernel} + attention_merge_f32", $"{kernel} (last block)");
                        }
                        else
                        {
                            ProductsCase(what, inputs, [4], want, (mem, at, second, seed) => Run(mem, at, second, seed, false), worst, kernel);
                        }
                    }
                }
            }
        }

        // Packed sequences: 4 heads, 2 a packed row (two tables of starts), 9 positions, head 24.
        const int SHeads = 4, SPerRow = 2, SSteps = 9, SDim = 24;
        float[] sq = RowsRandom(random, SHeads * SSteps * SDim, 1f), sk = RowsRandom(random, SHeads * SSteps * SDim, 1f), sv = RowsRandom(random, SHeads * SSteps * SDim, 1f);
        float[] segmentStarts = [0, 0, 0, 3, 3, 3, 3, 7, 7, 0, 0, 2, 2, 2, 2, 2, 2, 2];
        float[] segmentEnds = [3, 3, 3, 7, 7, 7, 7, 9, 9, 2, 2, 9, 9, 9, 9, 9, 9, 9];
        int tables = SHeads / SPerRow, sRows = SHeads * SSteps;
        foreach (var variant in (AttentionVariant[])[default, new(3, 5f)])
        {
            float[][] expected = RowsOnCpu([sq, sk, sv, new float[sRows * SDim], new float[sRows], segmentStarts, segmentEnds], (cpu, s) =>
                cpu.AttentionSegmented(s[0], s[1], s[2], s[3], s[4], s[5], s[6], SHeads, SPerRow, SSteps, SSteps, SDim, 0.2f, variant));
            foreach (var isa in ProductIsas)
            {
                // Inputs: q, keys, values, y, lse, starts, ends (unused), begins, range ends.
                void Run(GcnMemory memory, ulong[] at, bool second, ulong seed)
                {
                    DecodingDispatch("segment_spans_f32", isa, memory, (DecodingBlocks(tables * SSteps, second), 1, 1), seed,
                        at[5], at[7], at[8], SSteps, SSteps, (ulong)Math.Max(0, variant.Window), (ulong)(tables * SSteps));
                    DecodingDispatch("attention_spans_f32", isa, memory, (second ? sRows : 2, 1, 1), seed,
                        at[0], at[1], at[2], at[7], at[8], at[3], at[4], 0UL, 0UL, SHeads, 1, SPerRow, SSteps, SSteps, SDim, Bits(0.2f), Bits(variant.Softcap), SSteps, 1,
                        0, SSteps * SDim, SDim, 0, SSteps * SDim, SDim, 0, SSteps * SDim, SDim, (ulong)sRows);
                }

                float[][] inputs = [sq, sk, sv, new float[sRows * SDim], new float[sRows], segmentStarts, segmentEnds, new float[tables * SSteps], new float[tables * SSteps]];
                float[][] want = [.. inputs];
                (want[3], want[4]) = (expected[3], expected[4]);
                ProductsCase($"segment_spans_f32 + attention_spans_f32, window {variant.Window}, soft-cap {variant.Softcap} ({isa})", inputs, [3, 4], want, Run, worst,
                    "segment_spans_f32 + attention_spans_f32");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingPointwise(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(55);
        const int N = 1037, Cols = 37, Bias = Cols * 28;
        float[] x = RowsRandom(random, N, 3f), y0 = RowsRandom(random, N, 1f);
        float[] specials = [0f, -0f, 1f, -1f, -2f, float.PositiveInfinity, float.NegativeInfinity, 0.5f, -0.5f, 4f];
        float[] withSpecials = [.. x];
        specials.CopyTo(withSpecials, 3);
        float[] positive = RowsRandom(random, N, 2f, 2.01f), a = RowsRandom(random, N, 2f), b = RowsRandom(random, N, 2f), up = RowsRandom(random, N, 2f);
        float[] condition = [.. RowsRandom(random, N, 1f).Select((v, i) => i % 5 == 0 ? 0f : i % 7 == 0 ? -0f : i % 11 == 0 ? float.NaN : v)];
        float[] bias = RowsRandom(random, Cols, 1f);

        // Each case: its kernel, its inputs (the output last), the CPU's operation and the kernel's arguments from the addresses.
        void Case(string kernel, string what, float[][] inputs, int count, Action<Backend, Storage[]> cpu, Func<ulong[], ulong[]> values)
        {
            float[] expected = RowsOnCpu(inputs, cpu)[^1];
            foreach (var isa in ProductIsas)
            {
                float[][] want = [.. inputs];
                want[^1] = expected;
                ProductsCase($"{kernel}{what} ({isa})", inputs, [inputs.Length - 1], want,
                    (memory, at, second, seed) => DecodingDispatch(kernel, isa, memory, (second ? (count + 1023) / 1024 : 1, 1, 1), seed, values(at)), worst, kernel);
            }
        }

        foreach (float p in (float[])[1.5f, 2f, 3f, -1f, 0f, 0.5f, 1f, -2f, float.NaN])
        {
            Case("pow_f32", $", exponent {p}", [withSpecials, new float[N]], N, (cpu, s) => cpu.PowKernel(s[0], s[1], N, p),
                at => [at[0], at[1], Bits(p), GcnKernels.PowFlags(p), N]);
        }

        Case("clamp_f32", "", [x, new float[N]], N, (cpu, s) => cpu.ClampKernel(s[0], s[1], N, -0.5f, 0.75f), at => [at[0], at[1], Bits(-0.5f), Bits(0.75f), N]);
        Case("leaky_relu_f32", "", [x, new float[N]], N, (cpu, s) => cpu.LeakyReluKernel(s[0], s[1], N, 0.1f), at => [at[0], at[1], Bits(0.1f), N]);
        Case("where_f32", "", [condition, a, b, new float[N]], N, (cpu, s) => cpu.WhereKernel(s[0], s[1], s[2], s[3], N), at => [at[0], at[1], at[2], at[3], N]);
        Case("inv_sqrt_f32", "", [positive, new float[N]], N, (cpu, s) => cpu.InvSqrtKernel(s[0], s[1], N, 1e-5f), at => [at[0], at[1], Bits(1e-5f), N]);
        Case("bias_gelu_f32", "", [x[..Bias], bias, new float[Bias]], Bias, (cpu, s) => cpu.BiasGeluKernel(s[0], s[1], s[2], Bias, Cols),
            at => [at[0], at[1], at[2], Cols, Bias]);
        Case("add_scalar_f32", "", [[0.3f], y0], N, (cpu, s) => cpu.AddBroadcastScalarKernel(s[0], s[1], N, -1.5f), at => [at[0], at[1], Bits(-1.5f), N]);
        for (int kind = 0; kind < GcnKernels.GatedKinds.Length; kind++)
        {
            int activation = kind;
            Case(GcnKernels.GatedKinds[kind], "", [x, up, new float[N]], N, (cpu, s) => cpu.GatedActivationKernel(s[0], s[1], s[2], N, activation), at => [at[0], at[1], at[2], N]);
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingRows(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(56);

        // The arg-max and class matches: a random row, a row with ties, an all -inf row, a NaN first.
        foreach (int cols in RowsLengths)
        {
            const int Rows = 4;
            float[] x = RowsRandom(random, Rows * cols, 5f);
            if (cols > 9)
            {
                x[cols + 5] = x[cols + 9] = 100f;
            }

            Array.Fill(x, float.NegativeInfinity, 2 * cols, cols);
            x[3 * cols] = float.NaN;
            foreach (var isa in RowsIsas)
            {
                RowsCase($"argmax_f32 {Rows}x{cols} ({isa})", [x, new float[Rows]], Rows, [1], (cpu, s) => cpu.ArgMaxKernel(s[0], s[1], Rows, cols),
                    (m, at, grid, seed) => DecodingRowDispatch("argmax_f32", isa, m, grid, seed, at[0], at[1], (ulong)cols, Rows), worst, "argmax_f32");
            }

            float[] targets = RowsRandom(random, Rows * cols, 5f);
            if (cols > 1)
            {
                for (int r = 0; r < Rows; r += 2)
                {
                    int best = 0;
                    for (int j = 1; j < cols; j++)
                    {
                        best = x[r * cols + j] > x[r * cols + best] ? j : best;
                    }

                    targets[r * cols + best] = 50f;                            // these rows match
                }
            }

            foreach (var isa in RowsIsas)
            {
                RowsCase($"class_match_f32 {Rows}x{cols} ({isa})", [x, targets, new float[Rows]], Rows, [2], (cpu, s) => cpu.ClassMatchKernel(s[0], s[1], s[2], Rows, cols, 0.4f),
                    (m, at, grid, seed) => DecodingRowDispatch("class_match_f32", isa, m, grid, seed, at[0], at[1], at[2], (ulong)cols, Bits(0.4f), Rows), worst, "class_match_f32");
            }
        }

        // The residual addition with its norm.
        foreach (int cols in RowsLengths)
        {
            const int Rows = 3;
            float[] a = RowsRandom(random, Rows * cols, 3f), b = RowsRandom(random, Rows * cols, 1f), gain = RowsRandom(random, cols, 0.5f);
            foreach (var isa in RowsIsas)
            {
                RowsCase($"add_rms_norm_affine_f32 {Rows}x{cols} ({isa})", [a, b, new float[Rows * cols], gain, new float[Rows * cols]], Rows, [2, 4],
                    (cpu, s) => cpu.AddRmsNormAffineKernel(s[0], s[1], s[2], s[3], s[4], Rows, cols, 1e-6f, 1f),
                    (m, at, grid, seed) => DecodingRowDispatch("add_rms_norm_affine_f32", isa, m, grid, seed, at[0], at[1], at[2], at[3], at[4], (ulong)cols, Bits(1e-6f), Bits(1f), Rows),
                    worst, "add_rms_norm_affine_f32");
            }
        }

        // The norm and the rotary embedding: 2 steps of 3 heads, heads of 40 with 16 or 20 pairs rotated.
        {
            const int Heads = 3, Steps = 2, Cols = 40, Rows = Heads * Steps;
            float[] x = RowsRandom(random, Rows * Cols, 2f), gain = RowsRandom(random, Cols, 0.5f), positions = [5f, 7f];
            foreach (int half in (int[])[16, 20])
            {
                float[] cos = RowsRandom(random, 10 * half, 1f), sin = RowsRandom(random, 10 * half, 1f);
                foreach (bool interleaved in (bool[])[false, true])
                {
                    foreach (var isa in RowsIsas)
                    {
                        RowsCase($"rms_norm_rope_f32 {Rows}x{Cols}, {half} pairs{(interleaved ? ", interleaved" : "")} ({isa})", [x, gain, cos, sin, positions, new float[Rows * Cols]], Rows,
                            [5], (cpu, s) => cpu.RmsNormRopeKernel(s[0], s[1], s[2], s[3], s[4], s[5], Rows, Cols, 1e-6f, 0f, Heads, Steps, half, interleaved),
                            (m, at, grid, seed) => DecodingRowDispatch("rms_norm_rope_f32", isa, m, grid, seed, at[0], at[1], at[2], at[3], at[4], at[5], Cols, Bits(1e-6f), Bits(0f),
                                Heads, Steps, (ulong)half, interleaved ? 1UL : 0UL, Rows),
                            worst, "rms_norm_rope_f32");
                    }
                }
            }
        }

        // Repetition penalties: two rows of 300 tokens (a row stride of 310 from 5), a window of 6 of a history of 8 that
        // wrapped (length 11), repeated ids and ids out of the vocabulary.
        {
            const int Rows = 2, Vocabulary = 300, Stride = 310, Offset = 5, Capacity = 8, LastN = 6;
            float[] logits = RowsRandom(random, Rows * Stride, 4f), history = [12, 40, 299, 12, 350, 7, 40, 12, 3, -1, 3, 3, 250, 0, 9, 3];
            foreach (var isa in RowsIsas)
            {
                RowsCase($"penalize_rows_f32 {Rows}x{Vocabulary} ({isa})", [logits, new float[Rows * Vocabulary], history, [11f]], Rows, [1],
                    (cpu, s) => cpu.PenalizeRowsKernel(s[0], s[1], s[2], s[3], Rows, Vocabulary, Stride, Offset, Capacity, LastN, 1.3f, 0.2f, 0.1f),
                    (m, at, grid, seed) => DecodingRowDispatch("penalize_rows_f32", isa, m, grid, seed, at[0], at[1], at[2], at[3], Vocabulary, Stride, Offset, Capacity, LastN, Bits(1.3f),
                        Bits(0.2f), Bits(0.1f), Rows, Rows),
                    worst, "penalize_rows_f32");
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingShapes(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(57);
        const int Heads = 3, Steps = 2, Dim = 40, Rows = Heads * Steps;
        float[] x = RowsRandom(random, Rows * Dim, 2f), positions = [5f, 7f];
        foreach (int half in (int[])[16, 20])
        {
            float[] cos = RowsRandom(random, 10 * half, 1f), sin = RowsRandom(random, 10 * half, 1f);
            foreach (bool interleaved in (bool[])[false, true])
            {
                foreach (float sign in (float[])[1f, -1f])
                {
                    int n = Rows * half;
                    float[] expected = RowsOnCpu([x, (float[])x.Clone(), cos, sin, positions], (cpu, s) =>
                        cpu.RopeKernel(s[0], s[1], s[2], s[3], s[4], Rows, Heads, Steps, Dim, half, interleaved, sign))[1];
                    foreach (var isa in ProductIsas)
                    {
                        float[][] inputs = [x, (float[])x.Clone(), cos, sin, positions];
                        float[][] want = [.. inputs];
                        want[1] = expected;
                        ProductsCase($"rope_f32, {half} pairs{(interleaved ? ", interleaved" : "")}, sign {sign} ({isa})", inputs, [1], want,
                            (memory, at, second, seed) => DecodingDispatch("rope_f32", isa, memory, (DecodingBlocks(n, second), 1, 1), seed,
                                at[0], at[1], at[2], at[3], at[4], Heads, Steps, Dim, (ulong)half, interleaved ? 1UL : 0UL, Bits(sign), (ulong)n),
                            worst, "rope_f32");
                    }
                }
            }
        }

        {
            const int Count = 7, Classes = 13;
            float[] indices = [0f, 12f, 5f, 5f, 3f, 11f, 1f];
            float[] expected = RowsOnCpu([indices, new float[Count * Classes]], (cpu, s) => cpu.OneHotKernel(s[0], s[1], Count, Classes))[1];
            foreach (var isa in ProductIsas)
            {
                ProductsCase($"one_hot_f32 {Count}x{Classes} ({isa})", [indices, new float[Count * Classes]], [1], [indices, expected],
                    (memory, at, second, seed) => DecodingDispatch("one_hot_f32", isa, memory, (DecodingBlocks(Count * Classes, second), 1, 1), seed,
                        at[0], at[1], Classes, Classes - 1, Count * Classes),
                    worst, "one_hot_f32", exact: true);
            }
        }

        {
            const int HRows = 3, Capacity = 5;
            float[] ids = [17f, 4f, 99f], history = RowsRandom(random, HRows * Capacity, 9f);
            float[] expected = RowsOnCpu([ids, (float[])history.Clone(), [12f]], (cpu, s) => cpu.HistoryPushKernel(s[0], s[1], s[2], HRows, Capacity))[1];
            foreach (var isa in ProductIsas)
            {
                ProductsCase($"history_push_f32 {HRows} rows of {Capacity} ({isa})", [ids, (float[])history.Clone(), [12f]], [1], [ids, expected, [12f]],
                    (memory, at, second, seed) => DecodingDispatch("history_push_f32", isa, memory, (1, 1, 1), seed, at[0], at[1], at[2], Capacity, HRows, HRows),
                    worst, "history_push_f32", exact: true);
            }
        }

        RowsReport(worst);
    }

    // One dispatch of any kernel of the set, its arguments packed by `pack` (a Pack* helper, given exactly the declared
    // arguments' bytes; the kernel argument segment after them stays zero).
    private static GcnRunResult DecodingPacked(string name, GcnIsaVersion isa, GcnMemory memory, (int X, int Y, int Z) grid, ulong seed, Action<byte[]> pack)
    {
        var (code, image) = DecodingKernel(name, isa);
        var arguments = new byte[GcnCodeObject.KernargBytes(code)];
        var exact = new byte[code.Arguments[^1].Offset + code.Arguments[^1].Size];
        pack(exact);
        exact.CopyTo(arguments, 0);
        int block = code.RequiredWorkgroupSize?.X ?? code.MaxFlatWorkgroupSize;
        return GcnInterpreter.Run(image, memory, new GcnDispatch { Workgroups = grid, WorkgroupSize = (block, 1, 1), Arguments = arguments, Seed = seed, ShuffleGroups = seed != 1 });
    }

    internal static void GcnDecodingNormRope(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(58);

        // rms_norm_rope2_f32: 2 steps of 4 query heads and 2 key heads of 40, 16 pairs rotated (8 dimensions normalized
        // alone) or all 20, interleaved or not.
        {
            const int Steps = 2, Heads = 4, KvHeads = 2, Cols = 40, Rows1 = Steps * Heads, Rows2 = Steps * KvHeads;
            float[] x = RowsRandom(random, Rows1 * Cols, 2f), x2 = RowsRandom(random, Rows2 * Cols, 3f), gain = RowsRandom(random, Cols, 0.5f), gain2 = RowsRandom(random, Cols, 0.5f);
            float[] positions = [5f, 7f];
            foreach (int half in (int[])[16, 20])
            {
                float[] cos = RowsRandom(random, 10 * half, 1f), sin = RowsRandom(random, 10 * half, 1f);
                foreach (bool interleaved in (bool[])[false, true])
                {
                    foreach (var isa in RowsIsas)
                    {
                        RowsCase($"rms_norm_rope2_f32 {Rows1}+{Rows2}x{Cols}, {half} pairs{(interleaved ? ", interleaved" : "")} ({isa})",
                            [x, gain, cos, sin, positions, new float[Rows1 * Cols], x2, gain2, new float[Rows2 * Cols]], Rows1 + Rows2, [5, 8],
                            (cpu, s) => cpu.RmsNormRopePairKernel(s[0], s[1], s[5], Rows1, 1e-6f, 0f, Heads, s[6], s[7], s[8], Rows2, 1e-5f, 1f, KvHeads, s[2], s[3], s[4], Cols,
                                Steps, half, interleaved),
                            (m, at, grid, seed) => DecodingPacked("rms_norm_rope2_f32", isa, m, (grid.X, grid.Y, 1), seed, bytes => GcnKernels.PackRmsNormRopePair(bytes, at[0], at[1],
                                at[5], Rows1, 1e-6f, 0f, Heads, at[6], at[7], at[8], Rows2, 1e-5f, 1f, KvHeads, at[2], at[3], at[4], Cols, Steps, (uint)half, interleaved)),
                            worst, "rms_norm_rope2_f32");
                    }
                }
            }
        }

        // norm_rope_heads_f32: 2 sequences of 3 steps, 4 query heads over 2 key/value heads of 40.
        {
            const int Batch = 2, Steps = 3, Heads = 4, KvHeads = 2, Cols = 40, Start = 5, Capacity = 12;
            const float EpsQ = 1e-5f, OffsetQ = 0.5f, EpsK = 1e-6f, OffsetK = 0f;
            int rows1 = Batch * Steps * Heads, rows2 = Batch * Steps * KvHeads;
            float[] q = RowsRandom(random, rows1 * Cols, 2f), k = RowsRandom(random, rows2 * Cols, 3f), v = RowsRandom(random, rows2 * Cols, 1f);
            float[] gainQ = RowsRandom(random, Cols, 0.5f), gainK = RowsRandom(random, Cols, 0.5f), positions = [4f, 6f, 9f];
            foreach (var (norm, half, interleaved) in ((bool, int, bool)[])[(true, 16, false), (true, 20, true), (false, 16, true), (true, 0, false), (false, 0, false)])
            {
                float[] cos = RowsRandom(random, 10 * Math.Max(1, half), 1f), sin = RowsRandom(random, 10 * Math.Max(1, half), 1f);
                foreach (var (cache, bf16, stride) in ((bool, bool, int)[])[(false, false, Cols), (true, false, Cols + 4), (true, true, Cols)])
                {
                    int capacity = cache ? Capacity : Steps, cacheWords = bf16 ? stride / 2 : stride;
                    float[] yq = RowsRandom(random, rows1 * Cols, 1f), yk = RowsRandom(random, Batch * KvHeads * capacity * cacheWords, 1f), yv = RowsRandom(random, yk.Length, 1f);

                    // The CPU's steps: each tensor normalized and rotated (or one of them, or neither), then placed.
                    float[] Transformed(float[] x, float[] gain, int heads, bool normalized, bool rotated, float eps, float offset)
                    {
                        int rows = x.Length / Cols;
                        return RowsOnCpu([x, gain, cos, sin, positions, (float[])x.Clone()], (cpu, s) =>
                        {
                            if (normalized && rotated)
                            {
                                cpu.RmsNormRopeKernel(s[0], s[1], s[2], s[3], s[4], s[5], rows, Cols, eps, offset, heads, Steps, half, interleaved);
                            }
                            else if (normalized)
                            {
                                cpu.RmsNormAffineKernel(s[0], s[1], s[5], rows, Cols, eps, offset);
                            }
                            else if (rotated)
                            {
                                cpu.RopeKernel(s[0], s[5], s[2], s[3], s[4], rows, heads, Steps, Cols, half, interleaved, 1f);
                            }
                        })[5];
                    }

                    float[] tq = Transformed(q, gainQ, Heads, norm, half > 0, EpsQ, OffsetQ), tk = Transformed(k, gainK, KvHeads, norm, half > 0, EpsK, OffsetK);
                    float[] wantQ = (float[])yq.Clone(), wantK = (float[])yk.Clone(), wantV = (float[])yv.Clone();
                    var halvesK = MemoryMarshal.Cast<float, ushort>(wantK.AsSpan());
                    for (int b = 0; b < Batch; b++)
                    {
                        for (int s = 0; s < Steps; s++)
                        {
                            for (int h = 0; h < Heads; h++)
                            {
                                Array.Copy(tq, ((b * Steps + s) * Heads + h) * Cols, wantQ, ((b * Heads + h) * Steps + s) * Cols, Cols);
                            }

                            for (int h = 0; h < KvHeads; h++)
                            {
                                int from = ((b * Steps + s) * KvHeads + h) * Cols, row = (b * KvHeads + h) * capacity + (cache ? Start : 0) + s;
                                for (int c = 0; c < Cols; c++)
                                {
                                    if (bf16)
                                    {
                                        halvesK[row * stride + c] = Idrak.Abstraction.Devices.BFloat16Bits.Round(tk[from + c]);
                                        MemoryMarshal.Cast<float, ushort>(wantV.AsSpan())[row * stride + c] = Idrak.Abstraction.Devices.BFloat16Bits.Round(v[from + c]);
                                    }
                                    else
                                    {
                                        wantK[row * stride + c] = tk[from + c];
                                        wantV[row * stride + c] = v[from + c];
                                    }
                                }
                            }
                        }
                    }

                    // Inputs: q, gain q, k, gain k, v, cos, sin, positions, position, yq, yk, yv.
                    string what = $"norm_rope_heads_f32, {(norm ? "norms" : "no norm")}, {half} pairs{(interleaved ? " interleaved" : "")}, "
                                  + (cache ? $"a {(bf16 ? "bfloat16" : "float32")} cache of rows of {stride} at {Start}" : "no cache");
                    foreach (var isa in RowsIsas)
                    {
                        float[][] inputs = [q, gainQ, k, gainK, v, cos, sin, positions, [Start], yq, yk, yv];
                        void Run(GcnMemory memory, ulong[] at, bool second, ulong seed) =>
                            DecodingPacked("norm_rope_heads_f32", isa, memory, (second ? rows1 + 2 * rows2 : 2, 1, 1), seed, bytes => GcnKernels.PackNormRopeHeads(bytes, at[0],
                                norm ? at[1] : 0UL, at[2], norm ? at[3] : 0UL, at[4], half > 0 ? at[5] : 0UL, half > 0 ? at[6] : 0UL, half > 0 ? at[7] : 0UL, cache ? at[8] : 0UL,
                                at[9], at[10], at[11], Cols, EpsQ, OffsetQ, Heads, EpsK, OffsetK, KvHeads, Steps, (uint)half, interleaved, (uint)rows1, (uint)rows2,
                                (norm ? 1u : 0u) | (bf16 ? 2u : 0u) | (cache ? 4u : 0u), (uint)capacity, (uint)stride));
                        float[][] first = RowsOnGcn(inputs, (m, a) => Run(m, a, false, 1));
                        float[][] again = RowsOnGcn(inputs, (m, a) => Run(m, a, true, 7));
                        RowsSameBits(first, again, $"{what} ({isa})");
                        float error = RowsCompare(wantQ, first[9], $"{what} ({isa}): queries");
                        if (bf16)
                        {
                            // bfloat16 halves as floats: a value rounded on either side of a tie may differ by one bfloat16 step.
                            foreach (var (want, have, part) in new[] { (wantK, first[10], "keys"), (wantV, first[11], "values") })
                            {
                                var w = MemoryMarshal.Cast<float, ushort>(want.AsSpan());
                                var h = MemoryMarshal.Cast<float, ushort>(have.AsSpan());
                                for (int i = 0; i < w.Length; i++)
                                {
                                    float e = BitConverter.UInt32BitsToSingle((uint)w[i] << 16), g = BitConverter.UInt32BitsToSingle((uint)h[i] << 16);
                                    Check(w[i] == h[i] || MathF.Abs(e - g) <= 1e-2f * MathF.Max(1f, MathF.Abs(e)), $"{what} ({isa}): {part}' half {i} is {g}, the CPU's {e}");
                                }
                            }
                        }
                        else
                        {
                            error = MathF.Max(error, RowsCompare(wantK, first[10], $"{what} ({isa}): keys"));
                            Check(MemoryMarshal.AsBytes(wantV.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(first[11].AsSpan())), $"{what} ({isa}): the values copied as they are");
                        }

                        worst["norm_rope_heads_f32"] = MathF.Max(worst.GetValueOrDefault("norm_rope_heads_f32"), error);
                    }
                }
            }
        }

        RowsReport(worst);
    }

    internal static void GcnDecodingFloatMulti(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(59);
        const int K = 70;
        int[] widths = [130, 67, 200];
        float[][] weights = [.. widths.Select(w => RowsRandom(random, K * w, 0.5f))];
        float[] bias0 = RowsRandom(random, widths[0], 1f), bias2 = RowsRandom(random, widths[2], 1f);
        foreach (int m in (int[])[1, 5])
        {
            float[] x = RowsRandom(random, m * K, 1f);
            foreach (int count in (int[])[3, 1])
            {
                // Inputs: x, w0, w1, w2, bias0, bias2, y0, y1, y2.
                float[][] inputs = [x, weights[0], weights[1], weights[2], bias0, bias2, new float[m * widths[0]], new float[m * widths[1]], new float[m * widths[2]]];
                float[][] expected = RowsOnCpu(inputs, (cpu, s) =>
                {
                    (Storage Weight, Storage? Bias, Storage Output, int Columns)[] products = [(s[1], s[4], s[6], widths[0]), (s[2], null, s[7], widths[1]), (s[3], s[5], s[8], widths[2])];
                    cpu.MatMulManyKernel(s[0], m, K, products.AsSpan(0, count));
                });
                int widest = widths.Take(count).Max();
                foreach (var isa in ProductIsas)
                {
                    void Run(GcnMemory memory, ulong[] at, bool second, ulong seed) =>
                        DecodingPacked("gemv_multi_f32", isa, memory, ((widest + GcnKernels.GemvNnColumns - 1) / GcnKernels.GemvNnColumns, count, 1), seed,
                            bytes => GcnKernels.PackGemvMulti(bytes, at[0], (uint)m, K,
                                new GcnKernels.FloatProduct[] { new(at[1], at[4], at[6], (uint)widths[0]), new(at[2], 0UL, at[7], (uint)widths[1]), new(at[3], at[5], at[8], (uint)widths[2]) }
                                    .AsSpan(0, count)));
                    ProductsCase($"gemv_multi_f32, {count} product(s), {m} rows, k {K} ({isa})", inputs, [.. Enumerable.Range(6, count)], expected, Run, worst, "gemv_multi_f32");
                }
            }
        }

        RowsReport(worst);
    }

    // The sampler's statistics against the CPU's: every token id (the drawn one and the five alternatives) and the ids
    // equal, the probabilities and entropy within the tolerance; the largest error.
    private static float SamplerCompare(float[] expectedIds, float[] ids, float[] expected, float[] actual, string what)
    {
        Check(expectedIds.AsSpan().SequenceEqual(ids), $"{what}: tokens {string.Join(' ', ids)}, the CPU's {string.Join(' ', expectedIds)}");
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            int field = i % 13;
            if (field is 0 or 3 or 5 or 7 or 9 or 11)
            {
                Check(expected[i] == actual[i], $"{what}: step {i / 13 / ids.Length}, row {i / 13 % ids.Length}: statistic {field} (an id) is {actual[i]}, the CPU's {expected[i]}"
                    + $" (statistics [{string.Join(", ", actual.Skip(i - field).Take(13))}], the CPU's [{string.Join(", ", expected.Skip(i - field).Take(13))}])");
                continue;
            }

            float error = MathF.Abs(expected[i] - actual[i]) / MathF.Max(1f, MathF.Abs(expected[i]));
            Check(error <= RowsTolerance, $"{what}: step {i / 13 / ids.Length}, row {i / 13 % ids.Length}: statistic {field} is {actual[i]:R}, the CPU's {expected[i]:R}");
            worst = MathF.Max(worst, error);
        }

        return worst;
    }

    internal static void GcnDecodingSampler(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(60);
        const int Rows = 5, Steps = 2;
        const uint Seed = 1234;

        // Logits of `vocabulary` tokens a row: random, or in quarter steps (tied scores); then the special rows: +∞ (two of
        // them), all NaN, all -∞, NaN beside a best score.
        float[] Logits(int vocabulary, bool ties)
        {
            var logits = new float[Rows * vocabulary];
            for (int i = 0; i < logits.Length; i++)
            {
                float value = (random.NextSingle() * 2 - 1) * 6;
                logits[i] = ties ? MathF.Round(value * 4) / 4 : value;
            }

            return logits;
        }

        void Special(float[] logits, int vocabulary)
        {
            logits[vocabulary + 17] = float.PositiveInfinity;
            logits[vocabulary + vocabulary / 2] = float.PositiveInfinity;
            Array.Fill(logits, float.NaN, 2 * vocabulary, vocabulary);
            Array.Fill(logits, float.NegativeInfinity, 3 * vocabulary, vocabulary);
            for (int j = 0; j < vocabulary; j += 3)
            {
                logits[4 * vocabulary + j] = float.NaN;
            }

            logits[4 * vocabulary + 4] = 30f;
        }

        // The CPU's draws over two steps (ids, statistics), then the kernels' on the interpreter twice (another grid and wave
        // order), the same bits, compared.
        void Case(string what, float[] logits, int vocabulary, float temperature, int topK, float topP, float minP, bool candidates, bool greedy)
        {
            int blocks = (vocabulary + GcnKernels.CandidateSlice - 1) / GcnKernels.CandidateSlice, slots = blocks * GcnKernels.CandidateSlots;
            var settings = GcnKernels.SamplerSettings.Of(vocabulary, vocabulary, 0, temperature, topK, topP, minP, Seed, Rows);

            // Inputs: logits, ids, stats, step 0, step 1, candidate scores, candidate ids, flags, parts, counters.
            float[][] inputs =
            [
                logits, new float[Rows], new float[Steps * Rows * 13], [0f], [1f], new float[Rows * slots], new float[Rows * slots], new float[Rows * blocks],
                new float[Rows * blocks * 3], new float[Rows],
            ];
            float[][] expected = RowsOnCpu(inputs, (cpu, s) =>
            {
                for (int step = 0; step < Steps; step++)
                {
                    cpu.SampleRowsKernel(s[0], s[1], s[2], s[3 + step], Rows, vocabulary, vocabulary, 0, temperature, topK, topP, minP, Seed);
                }
            });
            var form = !candidates ? GcnKernels.SamplerForm.Rows : greedy ? GcnKernels.SamplerForm.Greedy : GcnKernels.SamplerForm.Candidates;
            string kernel = candidates ? $"{(greedy ? "topk_greedy_f32" : "topk_candidates_f32")} + {GcnKernels.SamplerKernel(form)}" : GcnKernels.SamplerKernel(form);
            foreach (var isa in ProductIsas)
            {
                // lastBlock: topk_greedy_f32's last block of each row samples it (one launch a step).
                void Run(GcnMemory memory, ulong[] at, bool second, ulong seed, bool lastBlock)
                {
                    for (int step = 0; step < Steps; step++)
                    {
                        if (candidates)
                        {
                            DecodingPacked(greedy ? "topk_greedy_f32" : "topk_candidates_f32", isa, memory, (second ? Rows * blocks : 3, 1, 1), seed + (ulong)step,
                                bytes => GcnKernels.PackCandidates(bytes, greedy, at[0], at[5], at[6], at[7], at[8], settings, (uint)blocks, at[1], at[2], at[3 + step],
                                    lastBlock ? at[9] : 0UL));
                        }

                        if (!lastBlock)
                        {
                            DecodingPacked(GcnKernels.SamplerKernel(form), isa, memory, (second ? Rows : 2, 1, 1), seed,
                                bytes => GcnKernels.PackSampler(bytes, form, at[0], at[1], at[2], at[3 + step], settings, at[5], at[6], at[7], at[8], (uint)slots));
                        }
                    }
                }

                foreach (bool lastBlock in greedy ? (bool[])[false, true] : [false])
                {
                    string label = lastBlock ? " (the last block samples)" : "";
                    float[][] first = RowsOnGcn(inputs, (m, a) => Run(m, a, false, 1, lastBlock));
                    float[][] again = RowsOnGcn(inputs, (m, a) => Run(m, a, true, 7, lastBlock));
                    RowsSameBits([first[1], first[2]], [again[1], again[2]], $"{what}{label} ({isa})");
                    float error = SamplerCompare(expected[1], first[1], expected[2], first[2], $"{what}{label} ({isa})");
                    worst[kernel + label] = MathF.Max(worst.GetValueOrDefault(kernel + label), error);
                    if (lastBlock)
                    {
                        float[][] launched = RowsOnGcn(inputs, (m, a) => Run(m, a, true, 7, false));
                        foreach (int o in (int[])[1, 2])
                        {
                            Check(MemoryMarshal.AsBytes(launched[o].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(again[o].AsSpan())),
                                $"{what} ({isa}): storage {o} by the last blocks differs from sample_candidates_greedy_f32's bit for bit");
                        }

                        Check(again[9].All(v => BitConverter.SingleToUInt32Bits(v) == 0), $"{what} ({isa}): the last blocks left a counter that is not zero");
                    }
                }
            }
        }

        // The whole row: 700 tokens (random) and 3,000 (tied scores), the settings Vulkan's sampler test takes.
        foreach (var (vocabulary, ties) in ((int, bool)[])[(700, false), (3000, true)])
        {
            float[] logits = Logits(vocabulary, ties);
            Special(logits, vocabulary);
            foreach (var (temperature, topK, topP, minP) in SamplerSettings)
            {
                Case($"sample_rows_f32, {vocabulary} tokens{(ties ? " (tied scores)" : "")}, temperature {temperature}, top-k {topK}, top-p {topP}, min-p {minP}", logits, vocabulary,
                    temperature, topK, topP, minP, candidates: false, greedy: false);
            }
        }

        // The candidates: 9,000 tokens (5 blocks of 2,048), random and tied; a row whose block holds 100 tied best scores
        // (more than its slots: the whole row) and one whose cut-off ties overflow on the first count only.
        foreach (bool ties in (bool[])[false, true])
        {
            const int Vocabulary = 9000;
            float[] logits = Logits(Vocabulary, ties);
            Special(logits, Vocabulary);
            if (!ties)
            {
                Array.Fill(logits, 7f, 3 * Vocabulary + 2048, 100);                    // replaces the all -∞ row: 100 ties in a block
                Array.Fill(logits, 6.5f, 2 * Vocabulary + 4100, 70);                   // replaces the all-NaN row: 70 ties at the cut-off
                logits[2 * Vocabulary + 4000] = 9f;
            }

            foreach (var (temperature, topK, topP, minP) in ((float, int, float, float)[])[(1f, 1, 1f, 0f), (0.7f, 40, 1f, 0f), (0.9f, 20, 0.95f, 0.02f), (1f, 64, 0.8f, 0f), (0.5f, 3, 0.5f, 0.1f)])
            {
                string what = $"{Vocabulary} tokens{(ties ? " (tied scores)" : "")}, temperature {temperature}, top-k {topK}, top-p {topP}, min-p {minP}";
                Case($"the candidates, {what}", logits, Vocabulary, temperature, topK, topP, minP, candidates: true, greedy: false);
                if (topK == 1)
                {
                    Case($"the greedy rows, {what}", logits, Vocabulary, temperature, topK, topP, minP, candidates: true, greedy: true);
                }
            }
        }

        RowsReport(worst);
    }
}
