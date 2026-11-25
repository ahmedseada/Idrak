// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// The product routes (ProductRoutes, IDRAK_ROUTES): products of bfloat16, int8 and q8_0 weights by today's
// kernels (name), by the lanes over weights stored in the file's layout with no fast path (generic), and by the faster
// lane per shape, the matrix units among them (measured); every route gives the product of the weights' values.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ProductRouteGroup =
    [
        ("product routes: bfloat16, float16, int8, q8_0, q4_0, iq4_nl (a code table), int64 (codes with a scale an output row), float64, int2 (2-bit codes, 32 a block with a float16 scale) and mxfp4 (OCP: E2M1 values sharing an E8M0 scale) weights give the product of their values (as ggml decodes them) under the name, generic and measured routes, the measured route with the described lane and the written kernels, the described lane alone and the written kernels alone (1, 3, 8, 9 and 70 rows; widths not a multiple of 4; within float32's bound for k terms, and x's rounding where an integer lane takes the shape)", ProductRoutesNear),
        ("product routes: under the generic and measured routes int8 and bfloat16 weights are kept stored in the file's layout (a row an output; int8 with today's codes and a scale an output)", ProductRoutesStoredLayout),
        ("product routes: the CPU's two lanes over q4_k's, q4_0's, mxfp4's, q6_k's, int64's and int2's stored codes, decoded in registers and by int8 dot products over x rounded to int8 a block of 32 values (int64: over its eight byte planes), its two over bfloat16's stored values (made float32 in registers in even and odd pairs or by a shift) and its two over float64's (float64 FMAs, x widened once or in registers), each give the product of the weights' values (1, 3, 4, 5, 9 and 70 rows; q4_0 rows of 3, 8 and 10 blocks: a last group of 8 short; q6_k rows of 1 and 3 super-blocks; int64 rows of 100, 256 and 544 values, with a scale an output row; float64 rows of 100, 256 and 2600 values: passes over k in spans; int2 and mxfp4 rows of 3, 8 and 10 blocks: a last group of 8 short; bfloat16 rows of 100, 256, 520 and 4200 values: passes over k in spans; the packed bfloat16 product under the measured route, 1, 3, 9 and 70 rows; within float32's bound for k terms, the integer lane also within its rounding of x); the described lane alone (any type read from its description, none of the written kernels) over each of them too and over the grid codebooks iq2_xxs and iq3_xxs (an index giving 8 or 4 magnitudes, the sign masks a vector of factors applied to the int8 codes; rows of 1, 2 and 3 super-blocks), each type's description decoding its stored bytes to its own conversion's values, and over types made of a description alone (3-bit codes in two planes with an E8M0 scale, a table of fractions with a bfloat16 scale, fp6 E2M3 in two planes, 16-bit integers with and without a code offset, 32-bit integers, fp4 E2M1 in blocks of 16 with an E4M3 scale each and a float16 scale shared by 8 blocks stored after them, 3-, 5- and 7-bit codes in a bit stream across byte boundaries)", ProductRoutesNibbleLanes),
        ("product routes: the CPU's lanes over stored weights give the same bits on any CPU: every lane that runs here (q4_k, q4_0, mxfp4, q6_k, bfloat16, int64, float64 and int2 by their written lanes and the described lane, the types made of a description alone by the described lane; 1, 3, 4, 5, 9 and 70 rows of fixed operands) the SHA-256 of its outputs recorded for its format and lane, the packed weights' bytes too", ProductRoutesLaneBits),
        ("product routes: CUDA's described lane alone (any type read from its description; no written kernel, nothing of a weight widened) over q4_k, q4_0, bfloat16, q6_k, int64 (a scale an output row), float64, int2 and mxfp4 as stored, over float16, int8, q8_0 and iq4_nl, over the grid codebooks iq2_xxs and iq3_xxs (an index giving 8 or 4 magnitudes, a vector of signs a group), and over types made of a description alone (3-bit codes in two planes with an E8M0 scale, a table of fractions with a bfloat16 scale, fp6 E2M3 in two planes, 16-bit integers with and without a code offset, 32-bit integers, fp4 E2M1 in blocks of 16 with an E4M3 scale each and a float16 scale shared by 8 blocks stored after them): the rows product (1, 3, 8, 9 and 70 rows; float64 arithmetic for 64- and 32-bit codes) within float32's bound for k terms, and in a bfloat16 run the tensor-core product (9, 70 and 130 rows; k split in 1 and 2) within it of x and the values rounded to bfloat16, and the integer products within it and x's rounding to int8 a block of 32 values: a whole-byte integer code's (int64, int8, 32-bit integers: the byte planes by int8 tensor cores) and narrow codes' (q4_k, q4_0, q6_k, int2, mxfp4, q8_0, iq4_nl and the description-only types whose codes' numbers are whole within int8 at a power of two: made int8 in registers, the factors a chunk of 32 or 16 at a time); every type the narrow lane reads must offer it", ProductRoutesDescribedCuda),
        ("product routes: the matrix-unit lanes (CUDA, a bfloat16 run) give the product of x rounded to bfloat16 and the weights' values (the native kind's, and float16's: rounded to bfloat16 too), for bfloat16, float16, int8, q8_0, q4_0 and iq4_nl (9, 70 and 130 rows; within float32's bound for k terms); the tensor rows (today's tensor-core kernel over bfloat16, float16 and int8 rows as stored, 32-value blocks decoded as the tiles load, the other formats' widened to bfloat16 rows; 128- and 64-row tiles) alone and as one launch over a group sharing x (bfloat16, float16, int8, q8_0, q4_0, Q4_K, IQ2_XXS; 128, 256 and 70 columns; 9, 130 and 300 rows) too", ProductRoutesMatrixUnits),
        ("product routes: q8_0 packs at load as ggml stores it (each value within half its block's step), kept as stored where the device multiplies it and expanded to float32 elsewhere", ProductRoutesQ8Packing),
        ("product routes: the k-quants' super-block descriptions (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) and the grid codebooks' (IQ2_XXS, IQ3_XXS) decode random blocks to the same bits as ggml's decoders", ProductRoutesSuperDecode),
        ("product routes: k-quant and grid-codebook weights (Q4_K, Q6_K, IQ2_XXS, IQ3_XXS) by their CUDA lanes from the description, the few-row kernel (1, 5 rows), the float32 product (70, 130), the native kind in a bfloat16 run (70, 130) and the tensor rows decoding as the tiles load or over widened rows (70, 300; 128- and 64-row tiles), give the product of their values (the bfloat16 lanes': x and the weights rounded to bfloat16; within float32's bound for k terms)", ProductRoutesSuperLanes),
        ("product routes: the CPU's float32 product over B packed once into the tiled kernel's panels (MultiplyPanels, today's path over stored rows), or filled as each tile needs them (MultiplyPanelsFrom, the tiles lane), gives the bits of the product over B [k, n] (70, 130 and 512 rows)", ProductRoutesPanels),
        ("product routes: stored weights sharing an input in one launch (CUDA's grouped few-row kernels: 2 and 3 weights of bfloat16, int8, q8_0, q4_0, iq4_nl, Q4_K, and under the measured route the group's one launches, the written kernels' and the described lane's by each warp split, measured against each product by its own lane, the described lane and the written kernels measured and the described lane alone, Q6_K, int64, float64, int2 and mxfp4 too, each output the bits of its product by the lane the group settled on; the CPU's grouped pass over Q4_K's, q4_0's, Q6_K's, bfloat16's, int64's, float64's, int2's and mxfp4's stored values, x prepared once, with the described lane and the written kernels measured and with the described lane alone) give each the bits of its own product (1, 3 and 8 rows; 70 and 130 in a bfloat16 run, x packed once for the group)", ProductRoutesMany),
        ("product routes: the fused epilogues of stored weights (CUDA, few rows, the measured route with every lane): the gate/up pair's act(gate) · up (SiLU, tanh GELU, ReLU) and a product's residual + y with its RMS normalization, by each fused lane (the written kernels' where they read the type, the described lane's by each warp split), match the same lane's own products composed with them, and through the dispatchers (the lanes competing, no storage converted) the composition or fused lane the shape settled on matches its own products (bfloat16, int8, q4_0, Q4_K, Q6_K, int2, mxfp4, int64 with a scale an output row, float64; 1, 3 and 8 rows; within 1e-5 of the values' magnitude)", ProductRoutesEpilogues),
        ("product routes: a bfloat16 checkpoint loaded under the measured route packs each weight from the file's rows (a bfloat16 weight's bits kept as they are; no float32 matrix, no transposition): every weight and the logits the bits of packing the float32 values, for bfloat16, int8, q8_0, q4_0, q4_k and iq2_xxs, with tied and own heads", ProductRoutesCheckpointRows),
        ("product routes: attention written token-major for the output projection (IMergedHeadsLayout: float32, bfloat16 and int8 caches, plain and windowed) gives the bits of the heads merged after attention, where the device writes so (the CPU; CUDA's tensor-core passes in a bfloat16 run, the bfloat16 cache's staging its halves as they are: the float32 cache's bits over the same values)", ProductRoutesMergedAttention),
        ("product routes: every described super-block and grid type (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS; random blocks as ggml stores them) by the CUDA tensor rows, decoding the values as each tile loads and over widened rows, 128- and 64-row tiles, gives the product of x and the values as ggml decodes them, both rounded to bfloat16 (70 rows; within float32's bound for k terms)", ProductRoutesEveryBlockType),
        ("product routes: scales on x's columns (a stored weight's scales shared by every output: a scale a column, a run of 32 columns, one for the whole tensor) are applied to x once a call, then the product with no scales: under the generic and measured routes the product of x and the weight's values times their columns' scales (q4_0, int8, bfloat16, int2; 1, 3, 9 and 70 rows; within float32's bound for k terms, and x's rounding where an integer lane takes the shape)", ProductRoutesColumnScales),
        ("product routes: IDRAK_ROUTES reads name, generic and measured (any case, blank is name) and IDRAK_ROUTE_LANES all, described and written (any case, blank is all), each refusing anything else naming the variable; the measured route records its choice", ProductRoutesSetting),
        ("product routes: ggml's q4_1, q5_0, q5_1, iq4_xs, tq1_0, tq2_0, iq2_xs, iq2_s, iq3_s, iq1_s and iq1_m described as data (a minimum added, fifth bits a stream, 6-bit scales in two pieces, base-3 digits of a byte read as a fraction, codebook indices of 9 to 12 bits over two or three planes and listed fields, the delta folded into IQ1's codebook, a float16 scale in four pieces) decode random blocks to ggml's values bit for bit, and the CPU's described lane, by the integer arithmetic and every instruction set that runs here, gives the product of those values (1, 3, 4, 5, 9 and 70 rows; within float32's bound for k terms and x's rounding to int8 a block of 32); CUDA's described lane gives it too where it reads the type, and where it does not says why", ProductRoutesGgufDescribed),
    ];

    private static readonly string[] RouteFormats = ["bfloat16", "float16", "int8", "q8_0", "q4_0", "iq4_nl"];

    // The k-quants packed at load here (super-blocks of 256 values: only widths in whole blocks).
    private static readonly string[] SuperFormats = ["q4_k", "q6_k"];

    // The grid-codebook formats packed at load here (IQ: blocks of 256, only widths in whole blocks).
    private static readonly string[] GridFormats = ["iq2_xxs", "iq3_xxs"];

    // Types defined from their bits, layout and scales, packed from float values: int64 (codes with a scale an output row),
    // float64 (binary64 values, no scales), int2 (2-bit codes, 32 a block with a float16 scale), mxfp4 (OCP: 32 E2M1 values
    // sharing an E8M0 scale).
    private static readonly string[] TypeFormats = ["int64", "float64", "int2", "mxfp4"];

    // (k, n): k whole blocks of 32 (q8_0), n not a multiple of 4 or 2 so the packed rows' padding is crossed.
    private static readonly (int K, int N)[] RouteShapes = [(96, 70), (256, 37), (288, 129)];

    private static readonly int[] RouteRows = [1, 3, 8, 9, 70];

    // Runs `body` under `route`, then puts the process's route back.
    // A checkpoint's weights packed from its rows (IWeightSource.ReadOutputRows, ReadOutputBFloat16) against the
    // same checkpoint read only as float32 values (the load before), under the measured route.
    private static void ProductRoutesCheckpointRows(Device device)
    {
        var architecture = PretrainedArchitectures.Get("Qwen3ForCausalLM");
        int runs = 0;
        UnderRoute(ProductRoute.Measured, () =>
        {
            foreach (bool tied in new[] { true, false })
            {
                // Rows of 256 and 512 inputs (4 heads of 64): whole blocks of the k-quants and the grids.
                var spec = SmallSpec with { QkNorm = true, Dim = 256, HeadDim = 64, FfDim = 512, Vocabulary = 300, TieEmbeddings = tied };
                var weights = new RandomWeights(73);
                using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
                {
                    // every weight drawn, for the checkpoint
                }

                string folder = TempFolder();
                try
                {
                    WriteCheckpoint(folder, spec, weights, "Qwen3ForCausalLM", SafeTensorType.BF16, sharded: !tied);
                    int[] ids = [2, 9, 14, 3, 7];
                    using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], device);
                    foreach (string format in new[] { "bfloat16", "int8", "q8_0", "q4_0", "q4_k", "iq2_xxs" })
                    {
                        string what = $"{(tied ? "tied" : "untied")}, {format}";
                        using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, PackedFormatName = format });
                        using var reader = SafeTensorsReader.Open(folder);
                        var build = new DecoderBuildOptions { Device = device, PackedFormatName = format, MaxPositions = model.MaxPositions };
                        using var before = model.Spec.Build(new FloatOnlyWeights(new CheckpointWeights(reader, architecture)), build);
                        var a = before.Parameters().Concat(before.Buffers()).ToList();
                        var b = model.Network.Parameters().Concat(model.Network.Buffers()).ToList();
                        Check(a.Count == b.Count && a.Count > 0, $"{what}: {b.Count} tensors, the float32 values' load made {a.Count}");
                        for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
                        {
                            Check(a[i].Shape.SequenceEqual(b[i].Shape) && SameRouteBits(a[i].ToArray(), b[i].ToArray()),
                                $"{what}: tensor {i} {Tensor.FormatShape(b[i].Shape)} differs from the float32 values' load's {Tensor.FormatShape(a[i].Shape)}");
                        }

                        var stored = model.Network.Descendants().OfType<Linear>().Select(l => l.PackedWeight).OfType<StoredWeight>().ToList();
                        Check(stored.Count > 0, $"{what}: the projections are stored weights ({string.Join(", ", model.Network.Descendants().OfType<Linear>().Select(l => l.PackedWeight?.Name ?? "float32").Distinct())})");
                        Check(SameRouteBits(before.Predict(input).ToArray(), model.Network.Predict(input).ToArray()), $"{what}: the same logits");
                        runs++;
                    }
                }
                finally
                {
                    Directory.Delete(folder, recursive: true);
                }
            }
        });
        Check(runs == 12, $"{runs} loads compared");
    }

    private static bool SameRouteBits(float[] a, float[] b) =>
        System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(a.AsSpan()).SequenceEqual(System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(b.AsSpan()));

    private static void UnderRoute(ProductRoute route, Action body)
    {
        var before = ProductRoutes.Current;
        ProductRoutes.Current = route;
        try
        {
            body();
        }
        finally
        {
            ProductRoutes.Current = before;
        }
    }

    private static float[] RouteValues(Random random, long count) =>
        [.. Enumerable.Range(0, (int)count).Select(_ => random.NextSingle() * 2 - 1)];

    // y = x · W through the weights' own product (the tensor's route), on the device.
    private static float[] RouteProduct(IPackedWeight weight, float[] x, int m, Device device)
    {
        using var scope = new TensorScope();
        using var noGrad = Autograd.NoGrad();
        var input = Tensor.From(x, [m, weight.Rows], device);
        return weight.MatMul(input).ToArray();
    }

    // Runs `body` with `lanes` the measured route's candidates, then puts the process's back.
    private static void UnderLanes(RouteLanes lanes, Action body)
    {
        var before = ProductRoutes.Lanes;
        ProductRoutes.Lanes = lanes;
        try
        {
            body();
        }
        finally
        {
            ProductRoutes.Lanes = before;
        }
    }

    private static void ProductRoutesNear(Device device)
    {
        var random = new Random(37);
        foreach (var (route, lanes) in new[] { (ProductRoute.Name, RouteLanes.All), (ProductRoute.Generic, RouteLanes.All), (ProductRoute.Measured, RouteLanes.All),
            (ProductRoute.Measured, RouteLanes.Described), (ProductRoute.Measured, RouteLanes.Written) })
        {
            UnderLanes(lanes, () => UnderRoute(route, () =>
            {
                foreach (var (k, n) in RouteShapes)
                {
                    var values = RouteValues(random, (long)k * n);
                    foreach (string format in RouteFormats.Concat(SuperFormats).Concat(GridFormats).Concat(TypeFormats))
                    {
                        if (k % (ElementTypes.Find(format)?.BlockSize ?? 1) != 0)
                        {
                            continue;                                       // a super-block format needs whole blocks of k
                        }

                        using var weight = PackedWeightFormats.Pack(format, values, k, n, device);
                        float[] w;
                        using (var expanded = weight.Dequantize())
                        {
                            w = expanded.ToArray();
                        }

                        // The CPU's integer lanes under the measured route (the written ones of q4_k, q4_0, q6_k, int64, int2 and mxfp4; the
                        // described lane's where the type's values are whole multiples of a unit) multiply x rounded to int8 a block of 32 values.
                        bool cpu = device.Backend is Idrak.Abstraction.Devices.Cpu.CpuBackend;
                        bool roundsX = route == ProductRoute.Measured && cpu && (format is "q4_k" or "q4_0" or "q6_k" or "int64" or "int2" or "mxfp4"
                            || lanes != RouteLanes.Written && ElementTypes.Find(format) is { } described && Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedArithmetic(described, k) == "integer");
                        foreach (int m in RouteRows)
                        {
                            var x = RouteValues(random, (long)m * k);
                            var y = RouteProduct(weight, x, m, device);

                            for (int i = 0; i < m; i++)
                            {
                                var steps = roundsX ? HalfSteps(x, i, k) : null;
                                for (int j = 0; j < n; j++)
                                {
                                    double sum = 0, magnitude = 0, rounding = 0;
                                    for (int r = 0; r < k; r++)
                                    {
                                        double term = (double)x[i * k + r] * w[r * n + j];
                                        sum += term;
                                        magnitude += Math.Abs(term);
                                        rounding += steps is null ? 0 : Math.Abs(w[r * n + j]) * steps[r];
                                    }

                                    // A float32 sum of k terms is within k · 2^-24 of their magnitude; twice that, and a tiny absolute floor;
                                    // x rounded, each value within its half step, adds Σ |w| · half step (and the terms grow by as much).
                                    double bound = k * Math.Pow(2, -23) * (magnitude + rounding) + rounding + 1e-30;
                                    double error = Math.Abs(y[i * n + j] - sum);
                                    if (!(error <= bound))
                                    {
                                        Check(false, $"{device}, route {route} (lanes {lanes}), {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} "
                                            + $"(error {error:E2} past the bound {bound:E2}; the weights are {weight.Description})");
                                    }
                                }
                            }
                        }
                    }
                }
            }));
        }
    }

    // Half the step of x's rounding to int8 a block of 32 values (the block's largest magnitude / 127, halved, with a margin
    // for the float32 arithmetic of the rounding) at each value of row i: how far the integer lane moves each value of x.
    private static double[] HalfSteps(float[] x, int i, int k)
    {
        var steps = new double[k];
        for (int b = 0; b < k / 32; b++)
        {
            double most = 0;
            for (int v = 0; v < 32; v++)
            {
                most = Math.Max(most, Math.Abs(x[i * k + b * 32 + v]));
            }

            for (int v = 0; v < 32; v++)
            {
                steps[b * 32 + v] = most / 127 * 0.5001;
            }
        }

        return steps;
    }

    // Each of the CPU's lanes over a 4-bit weight's stored codes (q4_k's super-blocks, q4_0's 32-value blocks: rows of 3, 8
    // and 10 blocks, so whole groups of 8 and short last ones), run directly: the decoded lane (codes decoded in
    // registers) within float32's bound of the product of the weights' values, the integer lane (x rounded to int8 a block
    // of 32 values) within that and its rounding of x.
    private static void ProductRoutesNibbleLanes(Device device)
    {
        if (device.Backend is not Idrak.Abstraction.Devices.Cpu.CpuBackend)
        {
            Skip($"the nibble lanes are the CPU's (not {device})");
            return;
        }

        var random = new Random(83);
        int runs = 0;
        UnderRoute(ProductRoute.Measured, () =>
        {
            foreach (var (format, k, n) in new[] { ("q4_k", 256, 37), ("q4_k", 512, 130), ("q4_0", 96, 37), ("q4_0", 256, 37), ("q4_0", 320, 130),
                ("q6_k", 256, 37), ("q6_k", 768, 130), ("bfloat16", 100, 37), ("bfloat16", 256, 37), ("bfloat16", 520, 130), ("bfloat16", 4200, 9),
                ("int64", 100, 37), ("int64", 256, 37), ("int64", 544, 130), ("float64", 100, 37), ("float64", 256, 37), ("float64", 2600, 130), ("int2", 96, 37), ("int2", 256, 37), ("int2", 320, 130),
                ("mxfp4", 96, 37), ("mxfp4", 256, 37), ("mxfp4", 320, 130), ("iq2_xxs", 256, 37), ("iq2_xxs", 512, 130), ("iq3_xxs", 256, 37), ("iq3_xxs", 768, 130) })
            {
                bool bfloat16 = format == "bfloat16", float64 = format == "float64", codebook = format.StartsWith("iq", StringComparison.Ordinal);
                var type = ElementTypes.Find(format) ?? throw new InvalidOperationException($"{format} is not registered");
                var values = RouteValues(random, (long)k * n);
                using var weight = (StoredWeight)PackedWeightFormats.Pack(format, values, k, n, device);
                float[] w;
                using (var expanded = weight.Dequantize())
                {
                    w = expanded.ToArray();
                }

                var storage = weight.StoredValues.Storage;
                var words = Idrak.Abstraction.Devices.Cpu.CpuBackend.D(storage);
                var rowScales = weight.Scales is { } held ? Idrak.Abstraction.Devices.Cpu.CpuBackend.D(held.Storage) : null;   // int64: one an output row
                DescribedDecodes(type, words, k, n);
                string? arithmetic = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedArithmetic(type, k);
                Check(arithmetic is not null, $"the described lane does not read {type.Name} ({weight.Description}; its description {type.Description})");
                // Every lane that runs on this CPU: the format's two kinds (q4_k, q4_0, q6_k, int64, int2 and mxfp4: decoded and
                // integer; bfloat16: shifted and paired; float64: decoded and mixed, x widened once or in registers), each by
                // the portable instructions and by each instruction set the CPU has; then the described lane's.
                var written = Idrak.Abstraction.Devices.Cpu.CpuBackend.StoredLanesOf(type, k);
                var describedLanes = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedLanesOf(type, k);
                // The grid codebooks (iq2_xxs, iq3_xxs) have no written lane over stored values: the described lane alone.
                Check((written.Count > 0 || codebook) && describedLanes.Count > 0, $"{type.Name} [{k}]: {written.Count} written lanes and {describedLanes.Count} described lanes run here");
                foreach (int m in new[] { 1, 3, 4, 5, 9, 70 })
                {
                    var x = RouteValues(random, (long)m * k);
                    for (int which = 0; which < written.Count + describedLanes.Count; which++)
                    {
                        bool described = which >= written.Count;
                        var y = new float[m * n];
                        bool integer = described ? arithmetic == "integer" : written[which].Second && !bfloat16 && !float64;
                        bool ran = described
                            ? Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedRowLane(which - written.Count, type, words, storage.Length, x, y, m, n, k, rowScales)
                            : Idrak.Abstraction.Devices.Cpu.CpuBackend.StoredRowLane(which, type, words, storage.Length, x, y, m, n, k, rowScales);
                        string lane = described ? $"{describedLanes[which - written.Count]} ({arithmetic})" : written[which].Lane;
                        Check(ran, $"{lane}: does not read {type.Name} ({weight.Description}), though listed as running here");
                        for (int i = 0; i < m; i++)
                        {
                            var steps = integer ? HalfSteps(x, i, k) : null;
                            for (int j = 0; j < n; j++)
                            {
                                double sum = 0, magnitude = 0, rounding = 0;
                                for (int r = 0; r < k; r++)
                                {
                                    double term = (double)x[i * k + r] * w[r * n + j];
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                    rounding += steps is null ? 0 : Math.Abs(w[r * n + j]) * steps[r];
                                }

                                double bound = k * Math.Pow(2, -23) * (magnitude + rounding) + rounding + 1e-30;
                                double error = Math.Abs(y[i * n + j] - sum);
                                if (!(error <= bound))
                                {
                                    Check(false, $"the {lane}, {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} (error {error:E2} past the bound {bound:E2})");
                                }
                            }
                        }

                        runs++;
                    }
                }
            }
        });
        Check(runs > 0, "no lane over stored values ran");
        DescribedOnly(random);

        // The packed bfloat16 product (a tied head's [k, n] layout) under the measured route: only the kernels reading the
        // weights in registers, at few rows and many, within float32's bound of the product of the weights' values.
        foreach (var (k, n) in new[] { (100, 37), (256, 130) })
        {
            var values = RouteValues(random, (long)k * n);
            IPackedWeight? head = null;
            UnderRoute(ProductRoute.Name, () => head = PackedWeightFormats.Pack(PackedFormat.BFloat16, values, k, n, device));
            using var packed = head!;
            float[] w;
            using (var expanded = packed.Dequantize())
            {
                w = expanded.ToArray();
            }

            foreach (int m in new[] { 1, 3, 9, 70 })
            {
                var x = RouteValues(random, (long)m * k);
                float[] y = [];
                UnderRoute(ProductRoute.Measured, () => y = RouteProduct(packed, x, m, device));
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0, magnitude = 0;
                        for (int r = 0; r < k; r++)
                        {
                            double term = (double)x[i * k + r] * w[r * n + j];
                            sum += term;
                            magnitude += Math.Abs(term);
                        }

                        double error = Math.Abs(y[i * n + j] - sum), bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                        if (!(error <= bound))
                        {
                            Check(false, $"the packed bfloat16 product [{k}, {n}], {m} rows, measured route: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} (error {error:E2} past the bound {bound:E2})");
                        }
                    }
                }
            }
        }
    }

    // A type's description read alone (TypeDescription.Decode) gives the values its own conversion gives (ElementType.Decode)
    // for the stored bytes of n rows of k values: the same, or within one float32 rounding (a 64-bit code rounds to float32
    // at once one way, through float64 the other; a super-block's factors multiply in another order).
    private static void DescribedDecodes(ElementType type, float[] words, int k, int n)
    {
        var d = type.Description;
        Check(d is not null, $"{type.Name} has no description");
        if (d is null)
        {
            return;
        }

        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan())[..(int)type.ByteCount((long)k * n)];
        float[] own = new float[k * n], described = new float[k * n];
        type.Decode(bytes, own);
        d.Decode(bytes, described);
        for (int i = 0; i < own.Length; i++)
        {
            float a = own[i], b = described[i];
            if (!(a == b || float.IsNaN(a) && float.IsNaN(b) || Math.Abs(a - b) <= Math.Abs(a) * Math.Pow(2, -22)))
            {
                Check(false, $"{type.Name}: value {i} is {a:R} by its conversion, {b:R} by its description ({d})");
                return;
            }
        }
    }

    // Types made of a description alone (registered nowhere, no conversion and no kernel of their own): a 3-bit code in a
    // 2-bit and a 1-bit plane with an E8M0 scale; a table of 16 fractions with a bfloat16 scale a 64 values; a 6-bit fp E2M3
    // in a 4-bit and a 2-bit plane with an E8M0 scale; 16-bit integers with a code offset of 0.5 and without; 32-bit
    // integers. Each with the row length it is multiplied with.
    private static (string Name, ElementType Type, TypeDescription D, int K)[] DescribedOnlyTypes()
    {
        var e8m0 = ElementType.Float8E8M0.Description!.Code;
        double[] fractions = [-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1];
        var made = new (string Name, TypeDescription D, int K)[]
        {
            ("3-bit codes in two planes, offset -4, an E8M0 scale a 32 values",
                new TypeDescription(32, 13, NumberType.Integer(3, false), [new CodePlane(1, 2, 4), new CodePlane(9, 1, 8)], -4, [new BlockScale(e8m0, 32, 0)]), 96),
            ("a table of 16 fractions, two a byte, a bfloat16 scale a 64 values",
                new TypeDescription(64, 34, NumberType.Entries(4, fractions), [new CodePlane(2, 4, 2)], Scales: [new BlockScale(NumberType.Floating(8, 7), 64, 0)]), 192),
            ("fp6 E2M3 in a 4-bit and a 2-bit plane, an E8M0 scale a 32 values",
                new TypeDescription(32, 25, NumberType.Floating(2, 3, bias: 1, specials: FloatSpecials.None), [new CodePlane(1, 4, 32), new CodePlane(17, 2, 32)],
                    Scales: [new BlockScale(e8m0, 32, 0)]), 320),
            ("16-bit integers plus 0.5", new TypeDescription(1, 2, NumberType.Integer(16, true), [new CodePlane(0, 16, 1)], 0.5), 100),
            ("16-bit integers", new TypeDescription(1, 2, NumberType.Integer(16, true), [new CodePlane(0, 16, 1)]), 100),
            ("32-bit integers", new TypeDescription(1, 4, NumberType.Integer(32, true), [new CodePlane(0, 32, 1)]), 100),
        };

        return [.. made.Select(t => (t.Name, new ElementType { Name = $"described: {t.Name}", Summary = t.Name, BlockSize = t.D.Values, BlockBits = t.D.Bytes * 8, Description = t.D }, t.D, t.K))];
    }

    // Types made of a description alone over the layouts the descriptions read beyond a block's own bytes (registered nowhere,
    // not among the lane-bits test's recorded types): fp4 E2M1 codes in blocks of 16 with an E4M3 scale each and a float16
    // scale shared by each group of 8 blocks, stored after them (a scale level of its own position and count); codes packed
    // across byte boundaries, a continuous stream of bits (3-bit codes, offset -4, an E8M0 scale a 32 values; 5-bit two's
    // complement codes with a bfloat16 scale a 64 values; 7-bit codes, offset -64, an E8M0 scale a 32 values); each with the
    // row length it is multiplied with.
    private static (string Name, ElementType Type, TypeDescription D, int K)[] DescribedLayoutTypes()
    {
        var e2m1 = ElementType.Float4E2M1.Description!.Code;
        var e4m3 = ElementType.Float8E4M3.Description!.Code;
        var e8m0 = ElementType.Float8E8M0.Description!.Code;
        var made = new (string Name, TypeDescription D, int K)[]
        {
            ("3-bit codes in a bit stream, offset -4, an E8M0 scale a 32 values",
                new TypeDescription(32, 13, NumberType.Integer(3, false), [new CodePlane(1, 3, 32, Stream: true)], -4, [new BlockScale(e8m0, 32, 0)]), 96),
            ("5-bit two's complement codes in a bit stream, a bfloat16 scale a 64 values",
                new TypeDescription(64, 42, NumberType.Integer(5, true), [new CodePlane(2, 5, 64, Stream: true)], Scales: [new BlockScale(NumberType.Floating(8, 7), 64, 0)]), 192),
            ("7-bit codes in a bit stream, offset -64, an E8M0 scale a 32 values",
                new TypeDescription(32, 29, NumberType.Integer(7, false), [new CodePlane(1, 7, 32, Stream: true)], -64, [new BlockScale(e8m0, 32, 0)]), 320),
            ("fp4 E2M1 in blocks of 16 with an E4M3 scale each, a float16 scale shared by 8 blocks after them",
                new TypeDescription(16, 9, e2m1, [new CodePlane(1, 4, 2)], Scales: [new BlockScale(e4m3, 16, 0), new BlockScale(NumberType.Floating(5, 10), 128, 72)],
                    Group: 8, GroupBytes: 74), 256),
        };

        return [.. made.Select(t => (t.Name, new ElementType { Name = $"described: {t.Name}", Summary = t.Name, BlockSize = t.D.UnitValues, BlockBits = t.D.UnitBytes * 8, Description = t.D }, t.D, t.K))];
    }

    // n rows of k random values of description d as stored: random codes, each block's (and each group's) scales finite
    // (E8M0 from 2^-7 to 2^3, E4M3 in [0.5, 2), a 16-bit float in [0.5, 2)).
    private static byte[] DescribedOnlyBytes(TypeDescription d, int k, int n, Random random)
    {
        if (d.Group > 1)
        {
            int unitsRow = k / d.UnitValues;
            var grouped = new byte[n * unitsRow * d.UnitBytes];
            random.NextBytes(grouped);
            for (int unit = 0; unit < grouped.Length / d.UnitBytes; unit++)
            {
                foreach (var scale in d.Scales ?? [])
                {
                    for (int v = 0; v < d.UnitValues; v += scale.Values)
                    {
                        int at = unit * d.UnitBytes + d.ScaleBase(scale, v) + scale.Offset + d.ScaleIndex(scale, v) * scale.Type.Bits / 8;
                        if (scale.Type.Bits == 8)
                        {
                            grouped[at] = (byte)(0x30 + random.Next(16));          // E4M3: [0.5, 2)
                        }
                        else
                        {
                            ushort half = (ushort)(0x3800 + random.Next(0x400));   // float16: [0.5, 1)
                            grouped[at] = (byte)half;
                            grouped[at + 1] = (byte)(half >> 8);
                        }
                    }
                }
            }

            return grouped;
        }

        int rowBytes = k / d.Values * d.Bytes;
        var bytes = new byte[n * rowBytes];
        random.NextBytes(bytes);
        for (int block = 0; block < bytes.Length / d.Bytes; block++)
        {
            foreach (var scale in d.Scales ?? [])
            {
                int at = block * d.Bytes + scale.Offset;
                if (scale.Type.Bits == 8)
                {
                    bytes[at] = (byte)random.Next(120, 131);                     // E8M0: 2^-7 to 2^3
                }
                else
                {
                    ushort bits = (ushort)(BitConverter.SingleToInt32Bits(0.5f + 1.5f * random.NextSingle()) >> 16);   // bfloat16 in [0.5, 2)
                    bytes[at] = (byte)bits;
                    bytes[at + 1] = (byte)(bits >> 8);
                }
            }
        }

        return bytes;
    }

    // The CPU's described lane over the types made of a description alone (DescribedOnlyTypes) from their descriptions
    // (random codes, finite scales), within float32's bound of the product of the values the description gives (and of x's
    // rounding where its arithmetic is integer).
    private static void DescribedOnly(Random random)
    {
        var made = DescribedOnlyTypes().Concat(DescribedLayoutTypes()).ToArray();
        int n = 37, ran = 0;
        foreach (var (name, type, d, k) in made)
        {
            Check(d.Valid, $"{name}: the description does not hold together ({d})");
            var bytes = DescribedOnlyBytes(d, k, n, random);
            var words = new float[(bytes.Length + 3) / 4];
            Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
            var w = new float[n * k];
            d.Decode(bytes, w);                                                            // row j's value r at j · k + r
            Check(w.All(float.IsFinite), $"{name}: its random values are not all finite");
            string? arithmetic = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedArithmetic(type, k);
            Check(arithmetic is not null, $"the described lane does not read {name} ({d})");
            foreach (int m in new[] { 1, 4, 5, 9 })
            {
                var x = RouteValues(random, (long)m * k);
                var y = new float[m * n];
                if (!Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedRowLane(type, words, words.Length, x, y, m, n, k))
                {
                    continue;
                }

                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        double sum = 0, magnitude = 0, rounding = 0;
                        for (int r = 0; r < k; r++)
                        {
                            double term = (double)x[i * k + r] * w[j * k + r];
                            sum += term;
                            magnitude += Math.Abs(term);
                            if (arithmetic == "integer")
                            {
                                double most = 0;
                                for (int v = r / 32 * 32; v < Math.Min(k, r / 32 * 32 + 32); v++)
                                {
                                    most = Math.Max(most, Math.Abs(x[i * k + v]));
                                }

                                rounding += Math.Abs(w[j * k + r]) * most / 127 * 0.5001;
                            }
                        }

                        double bound = k * Math.Pow(2, -23) * (magnitude + rounding) + rounding + 1e-30, error = Math.Abs(y[i * n + j] - sum);
                        if (!(error <= bound))
                        {
                            Check(false, $"the described lane ({arithmetic}), {name} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} (error {error:E2} past the bound {bound:E2})");
                        }
                    }
                }

                ran++;
            }
        }

        Check(ran == made.Length * 4, $"the described lane ran {ran} of {made.Length * 4} products of types made of a description alone");
    }

    // ggml's types described as data alone (GgufElementTypes.Descriptions), each with the row length it is multiplied with.
    private static readonly (string Format, int K)[] GgufDescribedFormats =
    [
        ("q4_1", 96), ("q5_0", 96), ("q5_1", 320), ("iq4_xs", 512), ("tq1_0", 256), ("tq2_0", 512), ("iq2_xs", 512), ("iq2_s", 256), ("iq3_s", 512), ("iq1_s", 256),
        ("iq1_m", 512),
    ];

    // Each described ggml type's random blocks (37 rows): the description's values the exact values of ggml's dequantizer
    // (the type's own conversion), bit for bit; then on the CPU the described lane by each lane that runs here (all of them
    // the integer arithmetic), on CUDA its described lane where its plan reads the type (else its reason), each within
    // float32's bound of the product of those values (and x's rounding where x is rounded to int8).
    private static void ProductRoutesGgufDescribed(Device device)
    {
        bool cpu = device.Backend is Idrak.Abstraction.Devices.Cpu.CpuBackend;
        var cuda = device.Backend as Idrak.Gpu.Cuda.CudaBackend;
        if (!cpu && cuda is null)
        {
            Skip($"the described lanes are the CPU's and CUDA's (not {device})");
            return;
        }

        var random = new Random(61);
        int n = 37, ran = 0, declined = 0;
        UnderRoute(ProductRoute.Measured, () => GgufDescribedUnder(device, cuda, random, n, ref ran, ref declined));
        Check(ran > 0 || cuda is not null && declined == GgufDescribedFormats.Length, $"{device}: the described lane ran no product of ggml's described types");
        Console.WriteLine($"{device}: {ran} products of ggml's described types, {declined} types the described lane does not read here");
    }

    private static void GgufDescribedUnder(Device device, Idrak.Gpu.Cuda.CudaBackend? cuda, Random random, int n, ref int ran, ref int declined)
    {
        foreach (var (format, k) in GgufDescribedFormats)
        {
            var type = ElementTypes.Find(format) ?? throw new InvalidOperationException($"{format} is not registered");
            var d = type.Description;
            Check(d is { Valid: true }, $"{format}: no description that holds together ({d})");
            if (d is null)
            {
                continue;
            }

            var bytes = GgufDescribedBytes(d, k, n, random);
            float[] exact = new float[n * k], described = new float[n * k];
            type.Decode(bytes, exact);                                                         // ggml's values: row j's value r at j · k + r
            d.Decode(bytes, described);
            int differ = -1;
            for (int i = 0; i < exact.Length && differ < 0; i++)
            {
                differ = BitConverter.SingleToInt32Bits(exact[i]) == BitConverter.SingleToInt32Bits(described[i]) ? -1 : i;
            }

            Check(differ < 0, $"{format}: value {differ} is {(differ < 0 ? 0 : exact[differ]):R} by ggml's dequantizer, {(differ < 0 ? 0 : described[differ]):R} by its description");
            Check(exact.All(float.IsFinite), $"{format}: its random values are not all finite");
            if (cuda is not null)
            {
                if (Idrak.Gpu.Cuda.CudaBackend.DescribedOf(type) is null)
                {
                    Check(Idrak.Gpu.Cuda.CudaBackend.DescribedReason(type) is { Length: > 0 }, $"{device}: {format}: not read, and no reason given");
                    declined++;
                    continue;
                }

                using var stored = Tensor.FromBytes(bytes, type, [n, k], device);
                ran += DescribedCudaProducts(cuda, device, format, type, stored.Storage, null, exact, k, n, random);
                continue;
            }

            var words = new float[(bytes.Length + 3) / 4];
            Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
            string? arithmetic = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedArithmetic(type, k);
            Check(arithmetic == "integer", $"the described lane reads {format} by {arithmetic ?? "nothing"}, not the integer arithmetic ({d})");
            var lanes = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedLanesOf(type, k);
            foreach (int m in new[] { 1, 3, 4, 5, 9, 70 })
            {
                var x = RouteValues(random, (long)m * k);
                for (int lane = 0; lane < lanes.Count; lane++)
                {
                    var y = new float[m * n];
                    if (!Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedRowLane(lane, type, words, words.Length, x, y, m, n, k))
                    {
                        Check(false, $"{format} [{k}, {n}], {m} rows: its {lanes[lane]} does not run here, though listed");
                        continue;
                    }

                    for (int i = 0; i < m; i++)
                    {
                        var steps = HalfSteps(x, i, k);
                        for (int j = 0; j < n; j++)
                        {
                            double sum = 0, magnitude = 0, rounding = 0;
                            for (int r = 0; r < k; r++)
                            {
                                double term = (double)x[i * k + r] * exact[j * k + r];
                                sum += term;
                                magnitude += Math.Abs(term);
                                rounding += Math.Abs(exact[j * k + r]) * steps[r];
                            }

                            double bound = k * Math.Pow(2, -23) * (magnitude + rounding) + rounding + 1e-30, error = Math.Abs(y[i * n + j] - sum);
                            if (!(error <= bound))
                            {
                                Check(false, $"the {lanes[lane]} ({arithmetic}), {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of ggml's values {sum} (error {error:E2} past the bound {bound:E2})");
                                i = m;
                                break;
                            }
                        }
                    }

                    ran++;
                }
            }
        }
    }

    // n rows of k values of description d as stored: random bytes, each 16-bit float scale and minimum (at its offset or
    // from its pieces) set to a finite value of either sign, magnitude in [2^-4, 2^0).
    private static byte[] GgufDescribedBytes(TypeDescription d, int k, int n, Random random)
    {
        var bytes = new byte[n * (k / d.Values) * d.Bytes];
        random.NextBytes(bytes);
        for (int at = 0; at < bytes.Length; at += d.Bytes)
        {
            var block = bytes.AsSpan(at, d.Bytes);
            foreach (var scale in (d.Scales ?? []).Concat(d.Mins ?? []).Where(s => s.Type is { Kind: NumberKind.Float, Bits: 16 }))
            {
                for (int index = 0; index < d.Values / scale.Values; index++)
                {
                    int bits = random.Next(2) << 15 | random.Next(11, 15) << 10 | random.Next(1 << 10);
                    if (scale.Pieces is { } pieces)
                    {
                        foreach (var piece in pieces[index])
                        {
                            int mask = (1 << piece.Bits) - 1;
                            block[piece.Offset] = (byte)(block[piece.Offset] & ~(mask << piece.Shift) | (bits & mask) << piece.Shift);
                            bits >>= piece.Bits;
                        }

                        continue;
                    }

                    for (int b = 0; b < 16; b++)
                    {
                        long bit = 8L * scale.Offset + 16L * index + b;
                        block[(int)(bit >> 3)] = (byte)(block[(int)(bit >> 3)] & ~(1 << (int)(bit & 7)) | (bits >> b & 1) << (int)(bit & 7));
                    }
                }
            }
        }

        return bytes;
    }

    // The bits of the CPU's lanes over stored weights on fixed operands: each format's weights packed from seeded values, x
    // seeded, 1, 3, 4, 5, 9 and 70 rows; per format and kind of lane the SHA-256 (its first 16 hex digits) of every output's
    // bits, and of the packed weights' bytes; the types made of a description alone (DescribedOnlyTypes) by the described
    // lane. Every lane that runs here is hashed on its own (each kind by the portable instructions and by each instruction
    // set the CPU has), and each must give its kind's bits: a lane computes alike on every CPU, whatever its instructions,
    // so one list holds (LaneBitsRecorded). A lane that does not run here, or a hash not in the list, fails naming it, and
    // the hashes made here are listed to compare.
    private static void ProductRoutesLaneBits(Device device)
    {
        if (device.Backend is not Idrak.Abstraction.Devices.Cpu.CpuBackend)
        {
            Skip($"the lanes over stored weights are the CPU's (not {device})");
            return;
        }

        var random = new Random(97);
        var hashes = new Dictionary<(string Key, string Lane), System.Security.Cryptography.IncrementalHash>();
        var order = new List<(string Key, string Lane)>();
        void Add(string key, string lane, ReadOnlySpan<byte> bytes)
        {
            if (!hashes.TryGetValue((key, lane), out var hash))
            {
                hashes[(key, lane)] = hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
                order.Add((key, lane));
            }

            hash.AppendData(bytes);
        }

        int[] rows = [1, 3, 4, 5, 9, 70];
        UnderRoute(ProductRoute.Measured, () =>
        {
            foreach (var (format, k, n) in new[] { ("q4_k", 512, 37), ("q4_0", 320, 37), ("mxfp4", 320, 37), ("q6_k", 768, 37), ("bfloat16", 520, 37), ("bfloat16", 4200, 9),
                ("int64", 544, 37), ("float64", 2600, 9), ("int2", 320, 37) })
            {
                var type = ElementTypes.Find(format) ?? throw new InvalidOperationException($"{format} is not registered");
                var values = RouteValues(random, (long)k * n);
                using var weight = (StoredWeight)PackedWeightFormats.Pack(format, values, k, n, device);
                var storage = weight.StoredValues.Storage;
                var words = Idrak.Abstraction.Devices.Cpu.CpuBackend.D(storage);
                var rowScales = weight.Scales is { } held ? Idrak.Abstraction.Devices.Cpu.CpuBackend.D(held.Storage) : null;
                Add($"{format} packed", "", System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length)));
                if (rowScales is not null)
                {
                    Add($"{format} packed", "", System.Runtime.InteropServices.MemoryMarshal.AsBytes(rowScales.AsSpan()));
                }

                var written = Idrak.Abstraction.Devices.Cpu.CpuBackend.StoredLanesOf(type, k);
                var described = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedLanesOf(type, k);
                Check(written.Count >= 2 && described.Count >= 1, $"{format} [{k}]: {written.Count} written lanes and {described.Count} described lanes run here");
                foreach (int m in rows)
                {
                    var x = RouteValues(random, (long)m * k);
                    for (int lane = 0; lane < written.Count; lane++)
                    {
                        var y = new float[m * n];
                        Check(Idrak.Abstraction.Devices.Cpu.CpuBackend.StoredRowLane(lane, type, words, storage.Length, x, y, m, n, k, rowScales),
                            $"{format} [{k}, {n}], {m} rows: its {written[lane].Lane} does not run here");
                        Add($"{format} {(written[lane].Second ? "second" : "first")}", written[lane].Lane, System.Runtime.InteropServices.MemoryMarshal.AsBytes(y.AsSpan()));
                    }

                    for (int lane = 0; lane < described.Count; lane++)
                    {
                        var y = new float[m * n];
                        Check(Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedRowLane(lane, type, words, storage.Length, x, y, m, n, k, rowScales),
                            $"{format} [{k}, {n}], {m} rows: its {described[lane]} does not run here");
                        Add($"{format} described", described[lane], System.Runtime.InteropServices.MemoryMarshal.AsBytes(y.AsSpan()));
                    }
                }
            }

            foreach (var (name, type, d, k) in DescribedOnlyTypes())
            {
                int n = 37;
                var bytes = DescribedOnlyBytes(d, k, n, random);
                var words = new float[(bytes.Length + 3) / 4];
                Buffer.BlockCopy(bytes, 0, words, 0, bytes.Length);
                var described = Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedLanesOf(type, k);
                Check(described.Count >= 1, $"{name}: no described lane runs here");
                foreach (int m in rows)
                {
                    var x = RouteValues(random, (long)m * k);
                    for (int lane = 0; lane < described.Count; lane++)
                    {
                        var y = new float[m * n];
                        Check(Idrak.Abstraction.Devices.Cpu.CpuBackend.DescribedRowLane(lane, type, words, words.Length, x, y, m, n, k), $"{name}, {m} rows: its {described[lane]} does not run here");
                        Add($"{name} described", described[lane], System.Runtime.InteropServices.MemoryMarshal.AsBytes(y.AsSpan()));
                    }
                }
            }
        });

        var made = new SortedDictionary<string, string>(StringComparer.Ordinal);
        var differ = new List<string>();
        foreach (var (key, lane) in order)
        {
            using var hash = hashes[(key, lane)];
            string hex = Convert.ToHexString(hash.GetHashAndReset())[..16];
            made.TryAdd(key, hex);                                         // the kind's first lane: the portable one
            if (!LaneBitsRecorded.TryGetValue(key, out var recorded) || recorded != hex)
            {
                differ.Add($"{key}{(lane.Length > 0 ? $" by the {lane}" : "")}: the bits hash to {hex}, recorded {recorded ?? "nothing"}");
            }
        }

        differ.AddRange(LaneBitsRecorded.Keys.Where(key => !made.ContainsKey(key)).Select(key => $"{key}: recorded, not hashed here"));
        Check(differ.Count == 0, $"{differ.Count} of {order.Count} lanes differ from the recorded bits: {string.Join("; ", differ)}"
            + $"{Environment.NewLine}the hashes made here:{Environment.NewLine}{string.Join(Environment.NewLine, made.Select(e => $"        [\"{e.Key}\"] = \"{e.Value}\","))}");
        Console.WriteLine($"{order.Count} lanes hashed (each kind of lane by every instruction set that runs here), every one the recorded bits of its kind");
    }

    // The hashes ProductRoutesLaneBits makes, recorded (the CPU's lanes give these bits on any CPU).
    private static readonly Dictionary<string, string> LaneBitsRecorded = new(StringComparer.Ordinal)
    {
        ["16-bit integers described"] = "AA650437BFD4B7EC",
        ["16-bit integers plus 0.5 described"] = "C758FF37AA9881B3",
        ["3-bit codes in two planes, offset -4, an E8M0 scale a 32 values described"] = "266CB3932659C6D5",
        ["32-bit integers described"] = "BEF4122A2A5C359A",
        ["a table of 16 fractions, two a byte, a bfloat16 scale a 64 values described"] = "B2EFD9742133B719",
        ["bfloat16 described"] = "1F14D9DAE9B8DFDF",
        ["bfloat16 first"] = "1F14D9DAE9B8DFDF",
        ["bfloat16 packed"] = "B2B8ADDE868D62F8",
        ["bfloat16 second"] = "A2C47570E1FE4008",
        ["float64 described"] = "C151C31D72C6C283",
        ["float64 first"] = "C151C31D72C6C283",
        ["float64 packed"] = "B09057F69C11D882",
        ["float64 second"] = "C151C31D72C6C283",
        ["fp6 E2M3 in a 4-bit and a 2-bit plane, an E8M0 scale a 32 values described"] = "275089C20C16CCD2",
        ["int2 described"] = "DD87E37FA24AE3F6",
        ["int2 first"] = "DAB1A16E2310BE7C",
        ["int2 packed"] = "4D39DB18C6D4D6F4",
        ["int2 second"] = "DD87E37FA24AE3F6",
        ["int64 described"] = "21F1BA49477B8C02",
        ["int64 first"] = "2CD5D31DFF2540CE",
        ["int64 packed"] = "ECBA01AB8ED13831",
        ["int64 second"] = "21F1BA49477B8C02",
        ["mxfp4 described"] = "E789C7DEB1AB96BA",
        ["mxfp4 first"] = "7102698DDEA25EDF",
        ["mxfp4 packed"] = "CAECF9C30E7FAD8D",
        ["mxfp4 second"] = "E789C7DEB1AB96BA",
        ["q4_0 described"] = "3D3B7C646E54EC9D",
        ["q4_0 first"] = "5141931BE5AE4177",
        ["q4_0 packed"] = "EED96641580BA4BE",
        ["q4_0 second"] = "3D3B7C646E54EC9D",
        ["q4_k described"] = "2C7667C71A41568D",
        ["q4_k first"] = "EA45628929DC06D0",
        ["q4_k packed"] = "9A51AEFF0C835D8D",
        ["q4_k second"] = "1DB7C46450939FF7",
        ["q6_k described"] = "FE20B8A3603EB411",
        ["q6_k first"] = "C1F28A6BECFD0343",
        ["q6_k packed"] = "CD4B4CE1C48EFDD1",
        ["q6_k second"] = "EAE080B4EDE7E2F1",
    };

    // CUDA's described lane run alone (no written kernel) over q4_k, q4_0, bfloat16, q6_k, int64 (a scale an output row),
    // float64, int2 and mxfp4 as stored, over the routes' other formats, and over the types made of a description alone
    // (DescribedOnlyTypes, uploaded as stored): each through DescribedCudaProducts.
    private static void ProductRoutesDescribedCuda(Device device)
    {
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend cuda)
        {
            Skip($"the CUDA described lane is CUDA's (not {device})");
            return;
        }

        var random = new Random(97);
        int ran = 0;
        UnderRoute(ProductRoute.Measured, () =>
        {
            foreach (var (format, k, n) in new[] { ("q4_k", 256, 37), ("q4_k", 512, 130), ("q4_0", 96, 37), ("q4_0", 320, 130), ("bfloat16", 100, 37), ("bfloat16", 256, 130),
                ("q6_k", 256, 37), ("q6_k", 768, 130), ("int64", 100, 37), ("int64", 544, 130), ("float64", 100, 37), ("float64", 256, 130), ("int2", 96, 37), ("int2", 320, 130),
                ("mxfp4", 96, 37), ("mxfp4", 320, 130), ("float16", 100, 37), ("int8", 96, 37), ("q8_0", 96, 37), ("iq4_nl", 96, 37), ("iq2_xxs", 256, 37), ("iq3_xxs", 512, 130) })
            {
                var type = ElementTypes.Find(format) ?? throw new InvalidOperationException($"{format} is not registered");
                Check(Idrak.Gpu.Cuda.CudaBackend.DescribedOf(type) is not null, $"{device}: the described lane does not read {format}: {Idrak.Gpu.Cuda.CudaBackend.DescribedReason(type)}");
                if (format is "q4_k" or "q4_0" or "q6_k" or "int2" or "mxfp4" or "q8_0" or "iq4_nl" && Idrak.Gpu.Cuda.CudaBackend.DescribedOf(type) is { } narrowPlan)
                {
                    Check(narrowPlan.NarrowChunk > 0, $"{device}: the integer lane over narrow codes does not read {format}: {narrowPlan.NarrowReason}");
                }

                using var weight = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device);
                float[] expanded;
                using (var values = weight.Dequantize())
                {
                    expanded = values.ToArray();                                   // [k, n]
                }

                var w = new float[n * k];
                for (int r = 0; r < k; r++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        w[j * k + r] = expanded[r * n + j];
                    }
                }

                ran += DescribedCudaProducts(cuda, device, $"{format} ({weight.Description})", type, weight.StoredValues.Storage, weight.Scales?.Storage, w, k, n, random,
                    weight.Scales?.ToArray());
            }

            foreach (var (name, type, d, k) in DescribedOnlyTypes().Concat(DescribedLayoutTypes()))
            {
                if (d.Planes.Any(p => p.Stream) && Idrak.Gpu.Cuda.CudaBackend.DescribedOf(type) is null)
                {
                    // Codes across byte boundaries are read by the CPU's described lane; CUDA's plan takes those whose fields
                    // keep one offset a thread, and says why not otherwise.
                    Check(Idrak.Gpu.Cuda.CudaBackend.DescribedReason(type) is { Length: > 0 }, $"{device}: {name}: not read, and no reason given");
                    continue;
                }

                int n = 37;
                var bytes = DescribedOnlyBytes(d, k, n, random);
                var w = new float[n * k];
                d.Decode(bytes, w);
                using var stored = Tensor.FromBytes(bytes, type, [n, k], device);
                ran += DescribedCudaProducts(cuda, device, name, type, stored.Storage, null, w, k, n, random);
            }
        });
        Check(ran > 0, $"{device}: the described lane ran no product");
    }

    // The described lane over `type`'s n stored rows of k values (`stored`; a scale an output row where `scales` are
    // given, `rowScales` their values), against the values `w` (row j's value r at j · k + r): the rows product at 1, 3, 8,
    // 9 and 70 rows within float32's bound for k terms; in a bfloat16 run the tensor-core product (9, 70 and 130 rows; k
    // split in 1 and 2; a float32 plan's) within it of the product of x and the values before the row's scale rounded to
    // bfloat16, then times the scale (either neighbour where a value lies within two units of float32's last place of a
    // rounding tie: the lane rounds its own float32 value of the description). Returns the products run.
    private static int DescribedCudaProducts(Idrak.Gpu.Cuda.CudaBackend cuda, Device device, string name, ElementType type, Storage stored, Storage? scales, float[] w, int k, int n,
        Random random, float[]? rowScales = null)
    {
        int ran = 0;
        var plan = Idrak.Gpu.Cuda.CudaBackend.DescribedOf(type);
        bool wide = plan?.Wide ?? false, whole = plan?.IntegerBytes > 0 && type.ByteCount(k) % 16 == 0, narrow = plan?.IntegerBytes == 0 && plan.NarrowChunk > 0 && k % 2 == 0;
        // Lanes: 0 the rows product (`splits`: log2 of the warps a group of 8 weight rows takes), 1 the tensor-core product, 2
        // the integer product (a whole-byte integer code's byte planes, or narrow codes made int8 in registers); `splits` the
        // k splits of the last two.
        foreach (var (lane, m, splits) in new[] { (0, 1, 3), (0, 3, 2), (0, 8, 1), (0, 9, 0), (0, 70, 3), (0, 1, 0), (1, 9, 1), (1, 70, 1), (1, 130, 1), (1, 130, 2),
            (2, 9, 1), (2, 70, 1), (2, 130, 1), (2, 130, 2) })
        {
            bool mma = lane == 1, integer = lane == 2;
            if (mma && wide || integer && !(whole || narrow))
            {
                continue;                                                       // float64 arithmetic: no tensor-core product
            }

            var x = RouteValues(random, (long)m * k);
            float[] y;
            using (var scope = new TensorScope())
            using (var mixed = lane > 0 ? (IDisposable)MixedPrecision.BFloat16() : null)
            {
                var input = Tensor.From(x, [m, k], device);
                var output = Tensor.Empty([m, n], device);
                if (!cuda.StoredDescribed(type, input.Storage, stored, scales, output.Storage, m, n, k, mma, lane == 0 ? 1 : splits, integer, lane == 0 ? splits : 3))
                {
                    Check(false, $"{device}: the described lane's {(integer ? "integer" : mma ? "tensor-core" : "rows")} product does not take {name} [{k}, {n}], {m} rows ({cuda.TypedUnavailableReason})");
                    continue;
                }

                y = output.ToArray();
            }

            for (int i = 0; i < m; i++)
            {
                var steps = integer ? BlockHalfSteps(x, i, k) : null;
                for (int j = 0; j < n; j++)
                {
                    double sum = 0, magnitude = 0, slack = 0;
                    for (int r = 0; r < k; r++)
                    {
                        float wv = w[j * k + r], xv = x[i * k + r];
                        if (steps is not null)
                        {
                            slack += Math.Abs(wv) * (r < steps.Length ? steps[r] : 0);
                        }

                        double rowScale = rowScales?[j] ?? 1;
                        if (mma)
                        {
                            float code = rowScales is null ? wv : (float)(wv / rowScale);
                            float wb = Bf16(code), xb = Bf16(xv);
                            double up = Bf16(MathF.BitIncrement(MathF.BitIncrement(code))), down = Bf16(MathF.BitDecrement(MathF.BitDecrement(code)));
                            slack += Math.Abs((double)xb * rowScale) * Math.Max(Math.Abs(up - wb), Math.Abs(down - wb));
                            (wv, xv) = ((float)(wb * rowScale), xb);
                        }

                        double term = (double)xv * wv;
                        sum += term;
                        magnitude += Math.Abs(term);
                    }

                    double bound = k * Math.Pow(2, -23) * (magnitude + slack) + slack + 1e-30, error = Math.Abs(y[i * n + j] - sum);
                    if (!(error <= bound))
                    {
                        string product = integer ? $"integer product (k split in {splits})" : mma ? $"tensor-core product (k split in {splits})" : "rows product";
                        Check(false, $"{device}: the described lane's {product}, {name} [{k}, {n}], {m} rows: "
                            + $"y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} (error {error:E2} past the bound {bound:E2})");
                        return ran;
                    }
                }
            }

            ran++;
        }

        return ran;

        static float Bf16(float v) => BitConverter.Int32BitsToSingle(BFloat16Bits.Round(v) << 16);

        // Half the step of x's rounding to int8 a block of 32 values (a last block shorter), with a margin for its float32
        // arithmetic, at each value of row i.
        static double[] BlockHalfSteps(float[] x, int i, int k)
        {
            var steps = new double[k];
            for (int b = 0; b * 32 < k; b++)
            {
                int end = Math.Min(k, b * 32 + 32);
                double most = 0;
                for (int v = b * 32; v < end; v++)
                {
                    most = Math.Max(most, Math.Abs(x[i * k + v]));
                }

                for (int v = b * 32; v < end; v++)
                {
                    steps[v] = most / 127 * 0.5001;
                }
            }

            return steps;
        }
    }

    private static void ProductRoutesStoredLayout(Device device)
    {
        var random = new Random(41);
        int k = 96, n = 37;
        var values = RouteValues(random, (long)k * n);
        foreach (var route in new[] { ProductRoute.Generic, ProductRoute.Measured })
        {
            UnderRoute(route, () =>
            {
                using var bf16 = PackedWeightFormats.Pack("bfloat16", values, k, n, device);
                Check(bf16 is StoredWeight { Type.Name: "bfloat16", Scales: null }, $"{device}, route {route}: bfloat16 packs as {bf16.GetType().Name} ({bf16.Description})");
                using var int8 = PackedWeightFormats.Pack("int8", values, k, n, device);
                Check(int8 is StoredWeight { Type.Name: "int8", Scales: not null }, $"{device}, route {route}: int8 packs as {int8.GetType().Name} ({int8.Description})");
                using var today = UnderName(() => PackedWeightFormats.Pack("int8", values, k, n, device));
                float[] a, b;
                using (var x = int8.Dequantize())
                using (var y = today.Dequantize())
                {
                    (a, b) = (x.ToArray(), y.ToArray());
                }

                int differs = Enumerable.Range(0, a.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]), -1);
                Check(differs < 0, differs < 0 ? "" : $"{device}, route {route}: the stored int8 weight's value {differs} is {a[differs]:R}, today's {b[differs]:R}");
            });
        }
    }

    // `make` under the named route (today's packing), the process's route put back after.
    private static IPackedWeight UnderName(Func<IPackedWeight> make)
    {
        IPackedWeight? made = null;
        UnderRoute(ProductRoute.Name, () => made = make());
        return made!;
    }

    private static void ProductRoutesMatrixUnits(Device device)
    {
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend cuda)
        {
            Skip($"the matrix-unit lane is CUDA's (not {device})");
            return;
        }

        var random = new Random(53);
        int ran = 0;
        UnderRoute(ProductRoute.Generic, () =>
        {
            foreach (var (k, n) in new[] { (96, 70), (256, 130) })
            {
                var values = RouteValues(random, (long)k * n);
                foreach (string format in RouteFormats)
                {
                    using var weight = (StoredWeight)PackedWeightFormats.Pack(format, values, k, n, device);
                    float[] w;
                    using (var expanded = weight.Dequantize())
                    {
                        w = expanded.ToArray();
                    }

                    foreach (var (lane, m) in new[] { 0, 1, 2, 3, 4 }.SelectMany(lane => new[] { 9, 70, 130 }.Select(m => (lane, m))))
                    {
                        if (lane == 3 && format != "int8")
                        {
                            continue;                                       // the raw lane reads signed 8-bit codes only
                        }

                        bool native = lane > 0;
                        var x = RouteValues(random, (long)m * k);
                        float[] y;
                        using (var scope = new TensorScope())
                        using (MixedPrecision.BFloat16())
                        {
                            var input = Tensor.From(x, [m, k], device);
                            var output = Tensor.Empty([m, n], device);
                            bool took = lane switch
                            {
                                4 => cuda.StoredTensorRows(input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, output.Storage, m, n, k),
                                3 => cuda.StoredNativeRaw(weight.Type, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, output.Storage, m, n, k),
                                2 => cuda.StoredNativeDirect(weight.Type, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, output.Storage, m, n, k),
                                1 => cuda.StoredNative(weight.Type, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, output.Storage, m, n, k),
                                _ => cuda.StoredMatrixUnits(weight.Type, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, output.Storage, m, n, k),
                            };
                            if (!took && lane >= 2)
                            {
                                Check(false, $"{device}: the {(lane == 4 ? "tensor-row" : lane == 3 ? "raw native" : "direct native")} lane does not take {format} [{k}] ({cuda.TypedUnavailableReason})");
                                continue;
                            }

                            if (!took)
                            {
                                Skip($"{device}: the matrix-unit lane does not take {format} here ({cuda.TypedUnavailableReason})");
                                return;
                            }

                            y = output.ToArray();
                            ran++;
                        }

                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                // The native lane multiplies float16 weights in the float16 kind (x rounded to float16, the
                                // weights as they are); every other lane and type in bfloat16.
                                // The raw lane multiplies int8 codes exactly (each output's scale applied to its sum).
                                // The tensor rows too over int8 (codes widened exactly, the scale applied to each output's sum); over
                                // the other formats they read the weights widened to bfloat16, as the native kind.
                                bool f16Kind = lane is 1 or 4 && format == "float16", rawCodes = lane == 3 || lane == 4 && format == "int8";
                                double sum = 0, magnitude = 0;
                                for (int r = 0; r < k; r++)
                                {
                                    float wv = w[r * n + j], wb = f16Kind || rawCodes ? wv : native || format == "float16" ? BitConverter.Int32BitsToSingle(BFloat16Bits.Round(wv) << 16) : wv;
                                    float xv = x[i * k + r], xr = f16Kind ? (float)(Half)xv : BitConverter.Int32BitsToSingle(BFloat16Bits.Round(xv) << 16);
                                    double term = (double)xr * wb;
                                    sum += term;
                                    magnitude += Math.Abs(term);
                                }

                                double bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                                double error = Math.Abs(y[i * n + j] - sum);
                                if (!(error <= bound))
                                {
                                    Check(false, $"{device}: {(lane == 4 ? "the tensor rows" : lane == 3 ? "the raw native kind" : lane == 2 ? "the direct native kind" : native ? "the native kind" : "matrix units")}, {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of x rounded to bfloat16 "
                                        + $"and the weights {sum} (error {error:E2} past {bound:E2})");
                                }
                            }
                        }
                    }
                }
            }
        });
        Check(ran > 0, $"{device}: the matrix-unit lane ran no product");

        // The tensor rows' one launch over a group sharing x (widths 128, 256 and 70: whole column tiles but the last's),
        // each output the product of x rounded to bfloat16 and its weights' values.
        UnderRoute(ProductRoute.Measured, () =>
        {
            int k = 256;
            foreach (string format in new[] { "bfloat16", "float16", "int8", "q8_0", "q4_0", "q4_k", "iq2_xxs" })
            {
                var weights = new[] { 128, 256, 70 }.Select(n => (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device)).ToArray();
                try
                {
                    var values = weights.Select(w =>
                    {
                        using var expanded = w.Dequantize();
                        return expanded.ToArray();
                    }).ToArray();
                    foreach (int count in new[] { 2, 3 })
                    {
                        foreach (var (m, tile) in new[] { (9, 128), (130, 64), (300, 128), (300, 64) })
                        {
                            var x = RouteValues(random, (long)m * k);
                            using var mixed = MixedPrecision.BFloat16();
                            Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = tile;
                            using var scope = new TensorScope();
                            var input = Tensor.From(x, [m, k], device);
                            var outputs = weights.Take(count).Select(w => Tensor.Empty([m, w.Columns], device)).ToArray();
                            var products = weights.Take(count).Select((w, j) => (w.StoredValues.Storage, w.Scales?.Storage, outputs[j].Storage, w.Columns)).ToArray();
                            bool grouped = cuda.StoredTensorRowsMany(input.Storage, m, k, products);
                            Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = null;
                            if (!grouped)
                            {
                                Check(false, $"{device}: {format}, {count} weights, {m} rows: the tensor rows take no group ({cuda.TypedUnavailableReason})");
                                continue;
                            }

                            for (int j = 0; j < count; j++)
                            {
                                int n = weights[j].Columns;
                                float[] y = outputs[j].ToArray(), w = values[j];
                                bool wrong = false;
                                for (int i = 0; i < m && !wrong; i++)
                                {
                                    for (int c = 0; c < n && !wrong; c++)
                                    {
                                        double sum = 0, magnitude = 0;
                                        for (int r = 0; r < k; r++)
                                        {
                                            // Weights read as stored (bfloat16, float16, int8 codes and scales) or widened to bfloat16 rows;
                                            // x rounded to bfloat16, or to float16 for float16's products.
                                            float wv = format is "bfloat16" or "float16" or "int8" ? w[r * n + c] : BitConverter.Int32BitsToSingle(BFloat16Bits.Round(w[r * n + c]) << 16);
                                            float xv = format == "float16" ? (float)(Half)x[i * k + r] : BitConverter.Int32BitsToSingle(BFloat16Bits.Round(x[i * k + r]) << 16);
                                            double term = (double)xv * wv;
                                            sum += term;
                                            magnitude += Math.Abs(term);
                                        }

                                        double error = Math.Abs(y[i * n + c] - sum), bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                                        if (!(error <= bound))
                                        {
                                            Check(false, $"{device}: the tensor rows' group, {format}, {count} weights, {m} rows: weight {j}'s y[{i}, {c}] = {y[i * n + c]}, the product {sum} (error {error:E2} past {bound:E2})");
                                            wrong = true;
                                        }
                                    }
                                }
                            }

                            ran++;
                        }
                    }
                }
                finally
                {
                    foreach (var w in weights)
                    {
                        w.Dispose();
                    }
                }
            }
        });
    }

    private static void ProductRoutesQ8Packing(Device device)
    {
        var random = new Random(43);
        int k = 96, n = 37;
        var values = RouteValues(random, (long)k * n);
        var type = ElementTypes.Get("q8_0");
        Check(type.Scaled is { Values: 32, Bytes: 34, CodeBits: 8, SignedCodes: true, CodeBias: 0, ScaleOffset: 0, CodeOffset: 2 },
            $"q8_0 describes its blocks: {type.Scaled}");
        foreach (var route in new[] { ProductRoute.Name, ProductRoute.Generic })
        {
            UnderRoute(route, () =>
            {
                using var weight = PackedWeightFormats.Pack("q8_0", values, k, n, device);
                bool stored = weight is StoredWeight;
                bool multiplies = WeightStorage.Multiplies(device.Backend, type);
                Check(stored == multiplies, $"{device}, route {route}: q8_0 {(stored ? "kept as stored" : "expanded")}, but the device {(multiplies ? "multiplies" : "does not multiply")} it ({weight.Description})");
                Check(!device.IsGpu ? stored : stored == (route != ProductRoute.Name), $"{device}, route {route}: q8_0 is {weight.Description}");
                float[] w;
                using (var expanded = weight.Dequantize())
                {
                    w = expanded.ToArray();
                }

                // Each value within half its block's step, and the float16 scale's rounding over 127 codes (column j's block b holds rows
                // 32b ... 32b + 31).
                for (int j = 0; j < n; j++)
                {
                    for (int b = 0; b < k / 32; b++)
                    {
                        float amax = 0f;
                        for (int r = 32 * b; r < 32 * b + 32; r++)
                        {
                            amax = MathF.Max(amax, MathF.Abs(values[r * n + j]));
                        }

                        double step = (double)(float)(Half)(amax / 127f);
                        for (int r = 32 * b; r < 32 * b + 32; r++)
                        {
                            double error = Math.Abs(w[r * n + j] - values[r * n + j]);
                            if (!(error <= 0.6 * step + 1e-7))
                            {
                                Check(false, $"{device}: q8_0 value ({r}, {j}) is {w[r * n + j]}, stored from {values[r * n + j]} with a step of {step}");
                            }
                        }
                    }
                }
            });
        }
    }

    private static void ProductRoutesSuperDecode(Device device)
    {
        var random = new Random(61);
        foreach (string name in new[] { "iq2_xxs", "iq3_xxs" })
        {
            var type = ElementTypes.Find(name);
            if (type?.Grid is not { } grid)
            {
                Check(false, $"{name}: no grid description ({type?.Describe() ?? "not registered"})");
                continue;
            }

            const int count = 6;
            var raw = new byte[count * grid.Bytes];
            random.NextBytes(raw);
            for (int b = 0; b < count; b++)
            {
                System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(b * grid.Bytes + grid.ScaleOffset), (Half)(random.NextSingle() * 0.2f - 0.1f));
            }

            var byGrid = new float[count * grid.Values];
            var byGgml = new float[count * grid.Values];
            grid.Decode(raw, byGrid);
            type.Decode(raw, byGgml);
            int wrong = Enumerable.Range(0, byGgml.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(byGrid[i]) != BitConverter.SingleToInt32Bits(byGgml[i]), -1);
            Check(wrong < 0, wrong < 0 ? "" : $"{name}: value {wrong} decodes to {byGrid[wrong]:R} by the description, {byGgml[wrong]:R} by ggml's decoder");
        }

        foreach (string name in new[] { "q2_k", "q3_k", "q4_k", "q5_k", "q6_k" })
        {
            var type = ElementTypes.Find(name);
            if (type?.Super is not { } super)
            {
                Check(false, $"{name}: no super-block description ({type?.Describe() ?? "not registered"})");
                continue;
            }

            const int blocks = 6;
            var bytes = new byte[blocks * super.Bytes];
            random.NextBytes(bytes);
            for (int b = 0; b < blocks; b++)
            {
                foreach (int offset in new[] { super.ScaleOffset, super.MinOffset })
                {
                    if (offset >= 0)
                    {
                        System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(b * super.Bytes + offset), (Half)(random.NextSingle() * 0.2f - 0.1f));
                    }
                }
            }

            var described = new float[blocks * super.Values];
            var ggml = new float[blocks * super.Values];
            super.Decode(bytes, described);
            type.Decode(bytes, ggml);
            int differs = Enumerable.Range(0, ggml.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(described[i]) != BitConverter.SingleToInt32Bits(ggml[i]), -1);
            Check(differs < 0, differs < 0 ? "" : $"{name}: value {differs} (block {differs / super.Values}, {differs % super.Values} in it) decodes to {described[differs]:R} by the description, {ggml[differs]:R} by ggml's decoder");
        }
    }

    private static void ProductRoutesSuperLanes(Device device)
    {
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend cuda)
        {
            Skip($"the k-quants' generated lanes are CUDA's (not {device}); the CPU's q4_k lanes over the stored codes have their own test");
            return;
        }

        var random = new Random(67);
        UnderRoute(ProductRoute.Measured, () =>
        {
            foreach (string format in SuperFormats.Concat(GridFormats))
            {
                int k = 512, n = 70;
                var values = RouteValues(random, (long)k * n);
                using var weight = (StoredWeight)PackedWeightFormats.Pack(format, values, k, n, device);
                float[] w;
                using (var expanded = weight.Dequantize())
                {
                    w = expanded.ToArray();
                }

                // rows: 0 none, 3 the tensor rows decoding as the tiles load, 4 over widened rows (each with both row tiles).
                foreach (var (native, m, direct, rows) in new[] { (false, 1, false, 0), (false, 5, false, 0), (false, 70, false, 0), (false, 130, false, 0), (true, 70, false, 0),
                             (true, 130, false, 0), (true, 70, true, 0), (true, 130, true, 0), (true, 70, false, 3), (true, 300, false, 3), (true, 70, false, 4), (true, 300, false, 4) })
                foreach (int tile in rows == 0 ? [0] : new[] { 128, 64 })
                {
                    if (direct && GridFormats.Contains(format))
                    {
                        continue;                                           // the direct native lane decodes no grid codebook
                    }

                    var x = RouteValues(random, (long)m * k);
                    float[] y;
                    using (var scope = new TensorScope())
                    using (native ? (IDisposable)MixedPrecision.BFloat16() : null)
                    {
                        var input = Tensor.From(x, [m, k], device);
                        var output = Tensor.Empty([m, n], device);
                        Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = tile == 0 ? null : tile;
                        bool ran;
                        try
                        {
                            ran = rows > 0 ? cuda.StoredTensorRows(input.Storage, weight.StoredValues.Storage, null, output.Storage, m, n, k, widen: rows == 4)
                                : direct ? cuda.StoredNativeDirect(weight.Type, input.Storage, weight.StoredValues.Storage, null, output.Storage, m, n, k)
                                : cuda.StoredSuperLane(weight.Type, input.Storage, weight.StoredValues.Storage, output.Storage, m, n, k, native);
                        }
                        finally
                        {
                            Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = null;
                        }

                        if (!ran)
                        {
                            Check(false, $"{device}: {format}'s lanes do not run ({cuda.TypedUnavailableReason})");
                            return;
                        }

                        y = output.ToArray();
                    }

                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < n; j++)
                        {
                            double sum = 0, magnitude = 0;
                            for (int r = 0; r < k; r++)
                            {
                                float xv = x[i * k + r], wv = w[r * n + j];
                                if (native)
                                {
                                    xv = BitConverter.Int32BitsToSingle(BFloat16Bits.Round(xv) << 16);
                                    wv = BitConverter.Int32BitsToSingle(BFloat16Bits.Round(wv) << 16);
                                }

                                double term = (double)xv * wv;
                                sum += term;
                                magnitude += Math.Abs(term);
                            }

                            double bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                            double error = Math.Abs(y[i * n + j] - sum);
                            if (!(error <= bound))
                            {
                                Check(false, $"{device}: {format} [{k}, {n}], {m} rows{(rows == 3 ? $", the tensor rows decoding ({tile}-row tiles)" : rows == 4 ? $", the tensor rows widened ({tile}-row tiles)" : native ? ", the native kind" : "")}: y[{i}, {j}] = {y[i * n + j]}, the product {sum} (error {error:E2} past {bound:E2})");
                            }
                        }
                    }
                }
            }
        });
    }

    private static void ProductRoutesPanels(Device device)
    {
        int nr = Idrak.Abstraction.Devices.Cpu.CpuMatMul.PanelColumns;
        if (nr == 0)
        {
            Skip("this CPU has no tiled kernel (no panels)");
            return;
        }

        var random = new Random(71);
        int ran = 0;
        foreach (var (m, n, k) in new[] { (70, 4 * nr, 96), (130, 6 * nr, 256), (512, 8 * nr, 128) })
        {
            if (!Idrak.Abstraction.Devices.Cpu.CpuMatMul.TakesPanels(m, n, k))
            {
                continue;
            }

            var a = RouteValues(random, (long)m * k);
            var b = RouteValues(random, (long)k * n);
            var panels = new float[k * n];
            for (int q = 0; q < n / nr; q++)
            {
                for (int r = 0; r < k; r++)
                {
                    for (int c = 0; c < nr; c++)
                    {
                        panels[q * k * nr + r * nr + c] = b[r * n + q * nr + c];
                    }
                }
            }

            var expected = new float[m * n];
            var got = new float[m * n];
            var tiled = new float[m * n];
            Idrak.Abstraction.Devices.Cpu.CpuMatMul.Multiply(a, b, expected, m, n, k, false, false, 0f);
            Idrak.Abstraction.Devices.Cpu.CpuMatMul.MultiplyPanels(a, panels, got, m, n, k);
            Idrak.Abstraction.Devices.Cpu.CpuMatMul.MultiplyPanelsFrom(a, new FilledPanels(b, n, k, nr), tiled, m, n, k);
            int differs = Enumerable.Range(0, got.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i]), -1);
            Check(differs < 0, differs < 0 ? "" : $"[{m}, {k}] · [{k}, {n}]: element {differs} is {got[differs]:R} over panels, {expected[differs]:R} over B");
            int differsTiled = Enumerable.Range(0, tiled.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(tiled[i]) != BitConverter.SingleToInt32Bits(expected[i]), -1);
            Check(differsTiled < 0, differsTiled < 0 ? "" : $"[{m}, {k}] · [{k}, {n}]: element {differsTiled} is {tiled[differsTiled]:R} over panels filled in the tiles, {expected[differsTiled]:R} over B");
            ran++;
        }

        Check(ran > 0, "no shape took the panels");
    }

    private static void ProductRoutesMany(Device device)
    {
        // The CPU groups q4_k's, q4_0's, q6_k's, bfloat16's, int64's, float64's, int2's and mxfp4's products (its lanes over the stored values); CUDA every type here.
        bool cpu = device.Backend is Idrak.Abstraction.Devices.Cpu.CpuBackend;
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend && !cpu)
        {
            Skip($"the grouped kernels are CUDA's and the CPU's (not {device})");
            return;
        }

        var random = new Random(73);
        foreach (var lanes in new[] { RouteLanes.All, RouteLanes.Described })
        {
            UnderLanes(lanes, () => ProductRoutesManyUnder(device, cpu, random));
        }
    }

    // ProductRoutesMany under the process's lanes (the CPU's: the described lane and the written kernels measured, or the
    // described lane alone).
    private static void ProductRoutesManyUnder(Device device, bool cpu, Random random)
    {
        UnderRoute(ProductRoute.Measured, () =>
        {
            int k = 256;
            foreach (string format in cpu ? new[] { "q4_k", "q4_0", "q6_k", "bfloat16", "int64", "float64", "int2", "mxfp4" }
                : new[] { "bfloat16", "int8", "q8_0", "q4_0", "iq4_nl", "q4_k", "q6_k", "int64", "float64", "int2", "mxfp4" })
            {
                var weights = new[] { 70, 37, 129 }.Select(n => (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device)).ToArray();
                try
                {
                    foreach (int count in new[] { 2, 3 })
                    {
                        foreach (int m in new[] { 1, 3, 8, 70, 130 })
                        {
                            var x = RouteValues(random, (long)m * k);
                            using var mixed = m > 8 ? (IDisposable)MixedPrecision.BFloat16() : null;   // prompt sizes: x packed once for the group
                            using var scope = new TensorScope();
                            var input = Tensor.From(x, [m, k], device);
                            var outputs = weights.Take(count).Select(w => Tensor.Empty([m, w.Columns], device)).ToArray();
                            var products = weights.Take(count).Select((w, j) => (w.StoredValues.Storage, w.Scales?.Storage, outputs[j].Storage, w.Columns)).ToArray();
                            if (!device.Backend.StoredMatMulManyKernel(input.Storage, m, k, products))
                            {
                                Check(false, $"{device}: {format}, {count} weights, {m} rows: no grouped launch");
                                continue;
                            }

                            // CUDA's few rows: the group's measured lane (each product by its own, or one launch of a lane: then each
                            // output that lane's product's bits).
                            int? groupLane = device.Backend is Idrak.Gpu.Cuda.CudaBackend cuda && m <= 8 ? cuda.StoredGroupChoice(input.Storage, m, k, products) : null;
                            for (int j = 0; j < count; j++)
                            {
                                float[] together = outputs[j].ToArray();
                                float[] alone = groupLane is { } lane && lane > 0 && device.Backend is Idrak.Gpu.Cuda.CudaBackend own
                                    ? OwnLaneProduct(own, weights[j], x, m, Idrak.Gpu.Cuda.CudaBackend.StoredEpilogueProductLane(weights[j].Type, lane), device)
                                    : RouteProduct(weights[j], x, m, device);
                                int differs = Enumerable.Range(0, alone.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(together[i]) != BitConverter.SingleToInt32Bits(alone[i]), -1);
                                Check(differs < 0, differs < 0 ? "" : $"{device}: {format}, {count} weights, {m} rows (group lane {groupLane?.ToString() ?? "own"}): weight {j}'s output {differs} is {together[differs]:R} grouped, {alone[differs]:R} alone");
                            }
                        }
                    }
                }
                finally
                {
                    foreach (var w in weights)
                    {
                        w.Dispose();
                    }
                }
            }
        });
    }

    // A stored weight's product by one of CUDA's measured-route lanes (CudaBackend.StoredRoutedLane), on the device.
    private static float[] OwnLaneProduct(Idrak.Gpu.Cuda.CudaBackend cuda, StoredWeight weight, float[] x, int m, int lane, Device device)
    {
        using var scope = new TensorScope();
        var input = Tensor.From(x, [m, weight.Rows], device);
        var y = Tensor.Empty([m, weight.Columns], device);
        Check(cuda.StoredRoutedLane(lane, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, y.Storage, m, weight.Columns, weight.Rows),
            $"{device}: {weight.Type.Name}: product lane {lane} did not run");
        return y.ToArray();
    }

    private static void ProductRoutesColumnScales(Device device)
    {
        if (device.Backend is not (Idrak.Abstraction.Devices.Cpu.CpuBackend or Idrak.Gpu.Cuda.CudaBackend))
        {
            Skip($"scales on x's columns are the CPU's and CUDA's stored products (not {device})");
            return;
        }

        var random = new Random(89);
        int runs = 0;
        foreach (var route in new[] { ProductRoute.Generic, ProductRoute.Measured })
        {
            UnderRoute(route, () =>
            {
                int k = 256, n = 37;
                foreach (string format in new[] { "q4_0", "int8", "bfloat16", "int2" })
                {
                    // The weight's stored bytes (packed on the CPU, no scales of its own), then the same bytes with scales on
                    // its columns on the device.
                    var type = ElementTypes.Find(format) ?? throw new InvalidOperationException($"{format} is not registered");
                    using var plain = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, Device.Cpu);
                    if (plain.Scales is not null)
                    {
                        continue;                                              // a type packed with scales of its own
                    }

                    float[] w;
                    using (var expanded = plain.Dequantize())
                    {
                        w = expanded.ToArray();                                // [k, n]
                    }

                    var words = Idrak.Abstraction.Devices.Cpu.CpuBackend.D(plain.StoredValues.Storage);
                    var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(words.AsSpan())[..(int)type.ByteCount((long)k * n)].ToArray();
                    foreach (int block in new[] { 1, 32, k })
                    {
                        int count = (k + block - 1) / block;
                        var scales = Enumerable.Range(0, count).Select(_ => 0.25f + random.NextSingle()).ToArray();
                        var layout = new StoredScales(n, block, count);
                        using var weight = StoredWeight.FromStored(type, bytes, k, n, new StoredScaling(scales, null, layout), device);
                        foreach (int m in new[] { 1, 3, 9, 70 })
                        {
                            var x = RouteValues(random, (long)m * k);
                            var y = RouteProduct(weight, x, m, device);
                            for (int i = 0; i < m; i++)
                            {
                                var steps = HalfSteps(x, i, k);
                                for (int j = 0; j < n; j++)
                                {
                                    double sum = 0, magnitude = 0, rounding = 0;
                                    for (int r = 0; r < k; r++)
                                    {
                                        double value = (double)w[r * n + j] * scales[r / block], term = x[i * k + r] * value;
                                        sum += term;
                                        magnitude += Math.Abs(term);
                                        rounding += Math.Abs(value) * steps[r];
                                    }

                                    // x's rounding allowed (an integer lane may take the shape), twice: x times the scales is rounded too.
                                    double bound = k * Math.Pow(2, -22) * (magnitude + rounding) + 2 * rounding + 1e-30, error = Math.Abs(y[i * n + j] - sum);
                                    if (!(error <= bound))
                                    {
                                        Check(false, $"{device}: {route} route, {format} with a scale every {block} columns, {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the scaled values {sum} (error {error:E2} past {bound:E2})");
                                    }
                                }
                            }

                            runs++;
                        }
                    }
                }
            });
        }

        Check(runs > 0, $"{device}: no product with scales on x's columns ran");
    }

    private static void ProductRoutesEpilogues(Device device)
    {
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend cuda)
        {
            Skip($"the fused epilogues of stored weights are CUDA's (not {device})");
            return;
        }

        static float Act(float g, int kind) => kind switch
        {
            0 => g / (1f + MathF.Exp(-g)),
            1 => 0.5f * g * (1f + MathF.Tanh(0.7978845608f * (g + 0.044715f * g * g * g))),
            _ => MathF.Max(g, 0f),
        };

        void Near(float got, double expected, double magnitude, string what)
        {
            double error = Math.Abs(got - expected);
            if (!(error <= 1e-5 * magnitude + 1e-6))
            {
                Check(false, $"{device}: {what}: {got:R}, composed {expected:R} (error {error:E2})");
            }
        }

        // A product by one of the measured route's lanes (CudaBackend.StoredRoutedLane), on the device.
        float[] LaneProduct(StoredWeight weight, float[] x, int m, int lane)
        {
            using var scope = new TensorScope();
            var input = Tensor.From(x, [m, weight.Rows], device);
            var y = Tensor.Empty([m, weight.Columns], device);
            Check(cuda.StoredRoutedLane(lane, input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, y.Storage, m, weight.Columns, weight.Rows),
                $"{device}: {weight.Type.Name}: product lane {lane} did not run");
            return y.ToArray();
        }

        // The gate/up pair's hidden by `lane` (null: through the dispatcher, the lanes competing), no storage converted.
        float[]? Gated(StoredWeight gate, StoredWeight up, float[] x, int m, int kind, int? lane, string what)
        {
            int k = gate.Rows, n = gate.Columns;
            using var scope = new TensorScope();
            var input = Tensor.From(x, [m, k], device);
            var gateOut = Tensor.Empty([m, n], device);
            var upOut = Tensor.Empty([m, n], device);
            var hidden = Tensor.Empty([m, n], device);
            (Storage, Storage?, Storage, int)[] pair = [(gate.StoredValues.Storage, gate.Scales?.Storage, gateOut.Storage, n), (up.StoredValues.Storage, up.Scales?.Storage, upOut.Storage, n)];
            // As Linear calls it: the stored weights reach the kernel as they are, no conversion counted.
            var seen = new ConversionsSeen();
            bool ran;
            using (Telemetry.Subscribe(seen))
            {
                ran = lane is { } own ? cuda.StoredGatedFused(kind, input.Storage, m, k, pair, hidden.Storage, own)
                    : device.Backend.StoredMatMulGatedPair(kind, input.Storage, m, k, pair, hidden.Storage);
            }

            Check(seen.Events.Count == 0, $"{device}: {what}: the gate/up pair converted {seen.Events.Count} storages ({string.Join("; ", seen.Events.Take(3))})");
            Check(ran, $"{device}: {what}: no gate/up pair");
            return ran ? hidden.ToArray() : null;
        }

        // The residual and norm pass by `lane` (null: through the dispatcher): (sum, normalized), no storage converted.
        (float[] Sum, float[] Normalized)? AddNorm(StoredWeight weight, float[] x, int m, float[] residualValues, float[] gain, int? lane, string what)
        {
            int k = weight.Rows, n = weight.Columns;
            using var scope = new TensorScope();
            var input = Tensor.From(x, [m, k], device);
            var residual = Tensor.From(residualValues, [m, n], device);
            var gains = Tensor.From(gain, [n], device);
            var y = Tensor.Empty([m, n], device);
            var sum = Tensor.Empty([m, n], device);
            var normalized = Tensor.Empty([m, n], device);
            var seen = new ConversionsSeen();
            bool ran;
            using (Telemetry.Subscribe(seen))
            {
                ran = lane is { } own
                    ? cuda.StoredAddNormFused(input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, y.Storage, m, n, k, residual.Storage, sum.Storage, gains.Storage,
                        normalized.Storage, 1e-6f, 0.5f, own)
                    : device.Backend.StoredMatMulAddRmsNorm(input.Storage, weight.StoredValues.Storage, weight.Scales?.Storage, y.Storage, m, n, k, residual.Storage, sum.Storage,
                        gains.Storage, normalized.Storage, 1e-6f, 0.5f);
            }

            // The stored weight reaches the kernel as it is (a weight converted every call was a decoding step's 6 ms).
            Check(seen.Events.Count == 0, $"{device}: {what}: the residual and norm product converted {seen.Events.Count} storages ({string.Join("; ", seen.Events.Take(3))})");
            Check(ran, $"{device}: {what}: no residual and norm pass");
            return ran ? (sum.ToArray(), normalized.ToArray()) : null;
        }

        void CheckGated(float[]? h, float[] g, float[] u, int kind, string what)
        {
            for (int i = 0; h is not null && i < h.Length; i++)
            {
                double expected = (double)Act(g[i], kind) * u[i];
                Near(h[i], expected, Math.Abs(expected) + Math.Abs(g[i] * u[i]), $"{what}: hidden {i}");
            }
        }

        void CheckAddNorm((float[] Sum, float[] Normalized)? got, float[] y, float[] residualValues, float[] gain, int m, int n, string what)
        {
            if (got is not var (s, o))
            {
                return;
            }

            for (int i = 0; i < m; i++)
            {
                double squares = 0;
                for (int j = 0; j < n; j++)
                {
                    double v = (double)residualValues[i * n + j] + y[i * n + j];
                    squares += v * v;
                    Near(s[i * n + j], v, Math.Abs(v) + 1, $"{what}: sum [{i}, {j}]");
                }

                double scale = 1 / Math.Sqrt(squares / n + 1e-6);
                for (int j = 0; j < n; j++)
                {
                    double v = (double)residualValues[i * n + j] + y[i * n + j], expected = v * scale * (gain[j] + 0.5);
                    Near(o[i * n + j], expected, Math.Abs(expected) + 1, $"{what}: normalized [{i}, {j}]");
                }
            }
        }

        // Each fused lane (the written kernels' where they read the type, the described lane's by each warp split of its rows
        // product) against the same lane's own products composed with the pass, so both sides add in one order; then every
        // lane competing (the measured route, all lanes), the dispatcher's result against the lane it settled on: the
        // composition (each product by its own measured lane, RouteProduct) or that fused lane's own products.
        var random = new Random(79);
        int fusedRuns = 0, describedRuns = 0;
        UnderLanes(RouteLanes.All, () => UnderRoute(ProductRoute.Measured, () =>
        {
            int k = 256, n = 96;
            foreach (string format in new[] { "bfloat16", "int8", "q4_0", "q4_k", "q6_k", "int2", "mxfp4", "int64", "float64" })
            {
                using var gate = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device);
                using var up = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device);
                var gain = RouteValues(random, n);
                foreach (int m in new[] { 1, 3, 8 })
                {
                    var x = RouteValues(random, (long)m * k);
                    var residualValues = RouteValues(random, (long)m * n);
                    var gatedLanes = cuda.StoredEpilogueCandidates(0, gate.Type, gate.Scales is not null, m, k);
                    var normLanes = cuda.StoredEpilogueCandidates(1, gate.Type, gate.Scales is not null, m, k);
                    Check(Idrak.Gpu.Cuda.CudaBackend.DescribedOf(gate.Type) is null || gatedLanes.Any(l => l >= 10) && normLanes.Any(l => l >= 10),
                        $"{device}: {format}, {m} rows: the described lane reads the type but offers no fused pass (gate/up lanes {string.Join(", ", gatedLanes)}; norm lanes {string.Join(", ", normLanes)})");
                    foreach (int lane in gatedLanes.Where(l => l != 0))
                    {
                        int productLane = Idrak.Gpu.Cuda.CudaBackend.StoredEpilogueProductLane(gate.Type, lane);
                        float[] g = LaneProduct(gate, x, m, productLane), u = LaneProduct(up, x, m, productLane);
                        foreach (int kind in new[] { 0, 1, 2 })
                        {
                            string what = $"{format}, {m} rows, fused lane {lane}, activation {kind}";
                            CheckGated(Gated(gate, up, x, m, kind, lane, what), g, u, kind, what);
                        }

                        fusedRuns++;
                        describedRuns += lane >= 10 ? 1 : 0;
                    }

                    foreach (int lane in normLanes.Where(l => l != 0))
                    {
                        int productLane = Idrak.Gpu.Cuda.CudaBackend.StoredEpilogueProductLane(gate.Type, lane);
                        string what = $"{format}, {m} rows, fused lane {lane}";
                        CheckAddNorm(AddNorm(gate, x, m, residualValues, gain, lane, what), LaneProduct(gate, x, m, productLane), residualValues, gain, m, n, what);
                        fusedRuns++;
                    }

                    // Every lane competing: the result of the lane the shape settled on.
                    foreach (int kind in new[] { 0, 1, 2 })
                    {
                        string what = $"{format}, {m} rows, dispatched, activation {kind}";
                        var h = Gated(gate, up, x, m, kind, null, what);
                        int chosen = cuda.StoredEpilogueChoice(0, kind, gate.Type, gate.Scales is not null, m, n, k) ?? 0;
                        if (chosen == 0)
                        {
                            // The composition: the pair's products as their group settled on (each by its own lane, or one
                            // launch of a lane), then the pass.
                            using var scope = new TensorScope();
                            var input = Tensor.From(x, [m, k], device);
                            (Storage, Storage?, Storage, int)[] pair = [(gate.StoredValues.Storage, gate.Scales?.Storage, Tensor.Empty([m, n], device).Storage, n),
                                (up.StoredValues.Storage, up.Scales?.Storage, Tensor.Empty([m, n], device).Storage, n)];
                            chosen = -(cuda.StoredGroupChoice(input.Storage, m, k, pair) ?? 0);
                        }

                        int productLane = Idrak.Gpu.Cuda.CudaBackend.StoredEpilogueProductLane(gate.Type, Math.Abs(chosen));
                        float[] g = chosen == 0 ? RouteProduct(gate, x, m, device) : LaneProduct(gate, x, m, productLane);
                        float[] u = chosen == 0 ? RouteProduct(up, x, m, device) : LaneProduct(up, x, m, productLane);
                        CheckGated(h, g, u, kind, $"{what} ({(chosen > 0 ? $"fused lane {chosen}" : $"composed, group lane {-chosen}")})");
                    }

                    {
                        string what = $"{format}, {m} rows, dispatched";
                        var got = AddNorm(gate, x, m, residualValues, gain, null, what);
                        int chosen = cuda.StoredEpilogueChoice(1, gate.Scales is null ? 0 : 1, gate.Type, gate.Scales is not null, m, n, k) ?? 0;
                        float[] y = chosen == 0 ? RouteProduct(gate, x, m, device) : LaneProduct(gate, x, m, Idrak.Gpu.Cuda.CudaBackend.StoredEpilogueProductLane(gate.Type, chosen));
                        CheckAddNorm(got, y, residualValues, gain, m, n, $"{what} (lane {chosen})");
                    }
                }
            }
        }));
        Check(fusedRuns > 0 && describedRuns > 0, $"{device}: {fusedRuns} fused passes compared, {describedRuns} of them the described lane's");
    }

    // B's panels filled as a tile asks for them (the tiles lane's way), from B [k, n].
    private readonly struct FilledPanels(float[] b, int n, int k, int nr) : Idrak.Abstraction.Devices.Cpu.CpuMatMul.IPanelSource
    {
        public bool Fills => true;

        public ref float Panel(int panel, float[] buffer, float[] rows)
        {
            for (int r = 0; r < k; r++)
            {
                for (int c = 0; c < nr; c++)
                {
                    buffer[r * nr + c] = b[r * n + panel * nr + c];
                }
            }

            return ref buffer[0];
        }
    }

    // Attention written token-major (IMergedHeadsLayout.AttendMerged) against the layout's own pass with the heads merged
    // after it, for each built-in cache format, plain and with a window: the same bits, each query head's row where an
    // output projection reads it. Where the device writes so is its own (Backend.MergesAttention): the CPU for every
    // format and variant, CUDA in a bfloat16 run for the float32 and bfloat16 caches' tensor-core passes, plain only;
    // there the bfloat16 cache's pass (its halves staged as they are) also gives the float32 cache's bits over the same
    // values, and both stay near the float32 reference.
    private static void ProductRoutesMergedAttention(Device device)
    {
        const int Sequences = 2, KvHeads = 2, Group = 2, Steps = 16, Dim = 64, Capacity = 40;
        float scale = 1f / MathF.Sqrt(Dim);
        var random = new Random(91);

        // bfloat16 values (the low halves cleared), so the float32 and bfloat16 caches hold the same ones.
        float[] Values(int count, bool halves) => [.. Enumerable.Range(0, count).Select(_ =>
        {
            float v = (float)(random.NextDouble() * 2 - 1);
            return halves ? BitConverter.Int32BitsToSingle(BitConverter.SingleToInt32Bits(v) & unchecked((int)0xFFFF0000)) : v;
        })];

        int rows = Sequences * KvHeads, queries = Group * Steps;
        float[] q = Values(rows * queries * Dim, halves: false), keys = Values(rows * Steps * Dim, halves: true), values = Values(rows * Steps * Dim, halves: true);
        bool cuda = device.Backend is Idrak.Gpu.Cuda.CudaBackend;
        int merged = 0;
        var tensorCore = new Dictionary<KeyValueFormat, float[]>();
        foreach (var precision in cuda ? new[] { MatMulPrecision.Float32, MatMulPrecision.BFloat16 } : [MatMulPrecision.Float32])
        {
            using var mixed = MixedPrecision.Use(precision);
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.BFloat16, KeyValueFormat.Int8 })
            {
                foreach (var variant in new[] { default(AttentionVariant), new AttentionVariant(5, 0f) })
                {
                    var layout = KeyValueLayouts.For(format);
                    string what = $"{format} cache, {(variant.IsPlain ? "plain" : "a window of 5")}, {precision} products on {device}";
                    using var context = new DecodingContext(device, Sequences, Capacity, layout);
                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    context.BeginStep(Steps);
                    var cache = context.CacheFor(layout, rows, Dim);
                    var qt = Tensor.From(q, [rows, queries, Dim], device);
                    layout.Write(Tensor.From(keys, [rows, Steps, Dim], device), Tensor.From(values, [rows, Steps, Dim], device), cache, context.PositionOf(cache));
                    var heads = variant.IsPlain
                        ? layout.Attend(qt, cache, context, Steps, scale, decoderKernels: true)
                        : ((ISlidingWindowLayout)layout).AttendVariant(qt, cache, context, Steps, scale, variant)!;
                    var written = cache.MergedHeadsLayout!.AttendMerged(qt, cache, context, Steps, scale, KvHeads, variant);
                    bool expected = !cuda || precision == MatMulPrecision.BFloat16 && variant.IsPlain && format != KeyValueFormat.Int8;
                    Check(written is not null == expected, $"{what}: written token-major {(written is null ? "nowhere" : "here")}, expected {(expected ? "here" : "nowhere")}");
                    if (written is null)
                    {
                        context.EndStep(Steps);
                        continue;
                    }

                    // Head h = b · kv + j's row i = g · steps + s goes to step s of sequence b, query head j · group + g.
                    float[] h = heads.ToArray(), want = new float[h.Length];
                    for (int b = 0; b < Sequences; b++)
                    {
                        for (int j = 0; j < KvHeads; j++)
                        {
                            for (int g = 0; g < Group; g++)
                            {
                                for (int s = 0; s < Steps; s++)
                                {
                                    int from = ((b * KvHeads + j) * queries + g * Steps + s) * Dim, to = ((b * Steps + s) * KvHeads * Group + j * Group + g) * Dim;
                                    Array.Copy(h, from, want, to, Dim);
                                }
                            }
                        }
                    }

                    float[] got = written.ToArray();
                    int differ = Enumerable.Range(0, got.Length).Count(i => BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(want[i]));
                    Check(written.Shape.SequenceEqual([Sequences, Steps, KvHeads * Group * Dim]) && differ == 0,
                        $"{what}: shape {Tensor.FormatShape(written.Shape)}, {differ} of {got.Length} values differ from the heads merged after attention");
                    if (cuda && variant.IsPlain && precision == MatMulPrecision.BFloat16)
                    {
                        tensorCore[format] = got;
                    }

                    merged++;
                    context.EndStep(Steps);
                }
            }
        }

        if (cuda && tensorCore.TryGetValue(KeyValueFormat.Float32, out var f32) && tensorCore.TryGetValue(KeyValueFormat.BFloat16, out var bf16))
        {
            int differ = Enumerable.Range(0, f32.Length).Count(i => BitConverter.SingleToInt32Bits(f32[i]) != BitConverter.SingleToInt32Bits(bf16[i]));
            Check(differ == 0, $"the bfloat16 cache's tensor-core pass gives the float32 cache's bits over the same values ({differ} of {f32.Length} differ)");

            // Both near the float32 reference (the CPU's pass over the same values): bfloat16 queries and weights.
            float[] reference;
            using (new TensorScope())
            using (Autograd.NoGrad())
            using (var context = new DecodingContext(Device.Cpu, Sequences, Capacity, KeyValueLayouts.For(KeyValueFormat.Float32)))
            {
                var layout = KeyValueLayouts.For(KeyValueFormat.Float32);
                context.BeginStep(Steps);
                var cache = context.CacheFor(layout, rows, Dim);
                layout.Write(Tensor.From(keys, [rows, Steps, Dim], Device.Cpu), Tensor.From(values, [rows, Steps, Dim], Device.Cpu), cache, context.PositionOf(cache));
                reference = cache.MergedHeadsLayout!.AttendMerged(Tensor.From(q, [rows, queries, Dim], Device.Cpu), cache, context, Steps, scale, KvHeads, default)!.ToArray();
            }

            float worst = Enumerable.Range(0, f32.Length).Max(i => MathF.Abs(bf16[i] - reference[i]));
            Check(worst < 0.03f, $"the bfloat16 cache's tensor-core pass within 0.03 of the float32 reference (worst {worst})");
        }

        Check(merged > 0, $"some pass on {device} wrote token-major");
    }

    // Every described super-block and grid type, also those with no encoder here (Q2_K, Q3_K, Q5_K), from random blocks as
    // ggml stores them (their scales set to small floats): the CUDA tensor rows decoding the values as each tile loads, and
    // over the rows widened for the call, with 128- and 64-row tiles, against x and the values as ggml decodes them, both
    // rounded to bfloat16.
    private static void ProductRoutesEveryBlockType(Device device)
    {
        if (device.Backend is not Idrak.Gpu.Cuda.CudaBackend cuda)
        {
            Skip($"the tensor rows are CUDA's (not {device})");
            return;
        }

        var random = new Random(97);
        const int k = 512, n = 130, m = 70;
        int ran = 0;
        foreach (string name in new[] { "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "iq2_xxs", "iq3_xxs" })
        {
            var type = ElementTypes.Find(name);
            if (type is null || type.Super is null && type.Grid is null)
            {
                Check(false, $"{name}: no super-block or grid description ({type?.Describe() ?? "not registered"})");
                continue;
            }

            int values = type.Super?.Values ?? type.Grid!.Values, bytes = type.Super?.Bytes ?? type.Grid!.Bytes;
            int[] scales = type.Super is { } super ? [super.ScaleOffset, super.MinOffset] : [type.Grid!.ScaleOffset];
            int blocks = n * (k / values);
            var raw = new byte[blocks * bytes];
            random.NextBytes(raw);
            for (int b = 0; b < blocks; b++)
            {
                foreach (int offset in scales.Where(o => o >= 0))
                {
                    System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(b * bytes + offset), (Half)(random.NextSingle() * 0.02f - 0.01f));
                }
            }

            var decoded = new float[n * k];
            type.Decode(raw, decoded);                                      // n rows of k values, as stored
            using var weight = StoredWeight.FromStored(type, raw, k, n, null, device);
            foreach (bool widen in new[] { false, true })
            {
                foreach (int tile in new[] { 128, 64 })
                {
                    var x = RouteValues(random, (long)m * k);
                    float[] y;
                    using (var scope = new TensorScope())
                    using (MixedPrecision.BFloat16())
                    {
                        var input = Tensor.From(x, [m, k], device);
                        var output = Tensor.Empty([m, n], device);
                        Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = tile;
                        bool took;
                        try
                        {
                            took = cuda.StoredTensorRows(input.Storage, weight.StoredValues.Storage, null, output.Storage, m, n, k, widen);
                        }
                        finally
                        {
                            Idrak.Gpu.Cuda.CudaBackend.RowsTileOverride = null;
                        }

                        if (!took)
                        {
                            Check(false, $"{device}: the tensor rows ({(widen ? "widened" : "decoding")}) do not take {name} ({cuda.TypedUnavailableReason})");
                            continue;
                        }

                        y = output.ToArray();
                    }

                    bool wrong = false;
                    for (int i = 0; i < m && !wrong; i++)
                    {
                        for (int j = 0; j < n && !wrong; j++)
                        {
                            double sum = 0, magnitude = 0;
                            for (int r = 0; r < k; r++)
                            {
                                double term = (double)BitConverter.Int32BitsToSingle(BFloat16Bits.Round(x[i * k + r]) << 16)
                                              * BitConverter.Int32BitsToSingle(BFloat16Bits.Round(decoded[j * k + r]) << 16);
                                sum += term;
                                magnitude += Math.Abs(term);
                            }

                            double error = Math.Abs(y[i * n + j] - sum), bound = k * Math.Pow(2, -23) * magnitude + 1e-30;
                            if (!(error <= bound))
                            {
                                Check(false, $"{device}: {name}, the tensor rows {(widen ? "widened" : "decoding")} ({tile}-row tiles): y[{i}, {j}] = {y[i * n + j]}, the product {sum} (error {error:E2} past {bound:E2})");
                                wrong = true;
                            }
                        }
                    }

                    ran++;
                }
            }
        }

        Check(ran == 7 * 4, $"{device}: {ran} of {7 * 4} tensor-row runs");
    }

    private static void ProductRoutesSetting(Device device)
    {
        Check(ProductRoutes.Parse(null) == ProductRoute.Name && ProductRoutes.Parse(" ") == ProductRoute.Name && ProductRoutes.Parse("NAME") == ProductRoute.Name,
            "blank and 'name' are the named route");
        Check(ProductRoutes.Parse("Generic") == ProductRoute.Generic && ProductRoutes.Parse("measured") == ProductRoute.Measured,
            "generic and measured are read in any case");
        string? refused = null;
        try
        {
            ProductRoutes.Parse("fast");
        }
        catch (InvalidOperationException e)
        {
            refused = e.Message;
        }

        Check(refused is not null && refused.Contains(ProductRoutes.EnvironmentVariable, StringComparison.Ordinal) && refused.Contains("fast", StringComparison.Ordinal),
            $"an unknown route is refused naming the variable and the value: {refused}");
        Check(ProductRoutes.ParseLanes(null) == RouteLanes.All && ProductRoutes.ParseLanes(" ") == RouteLanes.All && ProductRoutes.ParseLanes("ALL") == RouteLanes.All
            && ProductRoutes.ParseLanes("Described") == RouteLanes.Described && ProductRoutes.ParseLanes("written") == RouteLanes.Written,
            "IDRAK_ROUTE_LANES reads all (or blank), described and written in any case");
        string? refusedLanes = null;
        try
        {
            ProductRoutes.ParseLanes("fast");
        }
        catch (InvalidOperationException e)
        {
            refusedLanes = e.Message;
        }

        Check(refusedLanes is not null && refusedLanes.Contains(ProductRoutes.LanesVariable, StringComparison.Ordinal) && refusedLanes.Contains("fast", StringComparison.Ordinal),
            $"unknown lanes are refused naming the variable and the value: {refusedLanes}");

        if (Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE") is { } autotune && autotune.Trim() is "0" or "false" or "off" or "no")
        {
            return;                                                       // nothing is measured: the named lane is taken without a record
        }

        var random = new Random(47);
        int k = 128, n = 96, m = 5;
        var values = RouteValues(random, (long)k * n);
        UnderRoute(ProductRoute.Measured, () =>
        {
            ProductRoutes.ForgetChoices();
            using var weight = PackedWeightFormats.Pack("int8", values, k, n, device);
            RouteProduct(weight, RouteValues(random, (long)m * k), m, device);
            Check(ProductRoutes.Choices.Count > 0, $"{device}: the measured route recorded no choice for an int8 product of {m} rows ({string.Join("; ", ProductRoutes.Choices)})");
        });
    }
}
