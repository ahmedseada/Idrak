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
        ("product routes: bfloat16, float16, int8, q8_0, q4_0, iq4_nl (a code table), int64 (codes with a scale an output row), float64, int2 (2-bit codes, 32 a block with a float16 scale) and mxfp4 (OCP: E2M1 values sharing an E8M0 scale) weights give the product of their values (as ggml decodes them) under the name, generic and measured routes (1, 3, 8, 9 and 70 rows; widths not a multiple of 4; within float32's bound for k terms)", ProductRoutesNear),
        ("product routes: under the generic and measured routes int8 and bfloat16 weights are kept stored in the file's layout (a row an output; int8 with today's codes and a scale an output)", ProductRoutesStoredLayout),
        ("product routes: the CPU's two lanes over q4_k's, q4_0's, mxfp4's, q6_k's, int64's and int2's stored codes, decoded in registers and by int8 dot products over x rounded to int8 a block of 32 values (int64: over its eight byte planes), its two over bfloat16's stored values (made float32 in registers in even and odd pairs or by a shift) and its two over float64's (float64 FMAs, x widened once or in registers), each give the product of the weights' values (1, 3, 4, 5, 9 and 70 rows; q4_0 rows of 3, 8 and 10 blocks: a last group of 8 short; q6_k rows of 1 and 3 super-blocks; int64 rows of 100, 256 and 544 values, with a scale an output row; float64 rows of 100, 256 and 2600 values: passes over k in spans; int2 and mxfp4 rows of 3, 8 and 10 blocks: a last group of 8 short; bfloat16 rows of 100, 256, 520 and 4200 values: passes over k in spans; the packed bfloat16 product under the measured route, 1, 3, 9 and 70 rows; within float32's bound for k terms, the integer lane also within its rounding of x)", ProductRoutesNibbleLanes),
        ("product routes: the matrix-unit lanes (CUDA, a bfloat16 run) give the product of x rounded to bfloat16 and the weights' values (the native kind's, and float16's: rounded to bfloat16 too), for bfloat16, float16, int8, q8_0, q4_0 and iq4_nl (9, 70 and 130 rows; within float32's bound for k terms); the tensor rows (today's tensor-core kernel over bfloat16, float16 and int8 rows as stored, 32-value blocks decoded as the tiles load, the other formats' widened to bfloat16 rows; 128- and 64-row tiles) alone and as one launch over a group sharing x (bfloat16, float16, int8, q8_0, q4_0, Q4_K, IQ2_XXS; 128, 256 and 70 columns; 9, 130 and 300 rows) too", ProductRoutesMatrixUnits),
        ("product routes: q8_0 packs at load as ggml stores it (each value within half its block's step), kept as stored where the device multiplies it and expanded to float32 elsewhere", ProductRoutesQ8Packing),
        ("product routes: the k-quants' super-block descriptions (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) and the grid codebooks' (IQ2_XXS, IQ3_XXS) decode random blocks to the same bits as ggml's decoders", ProductRoutesSuperDecode),
        ("product routes: k-quant and grid-codebook weights (Q4_K, Q6_K, IQ2_XXS, IQ3_XXS) by their CUDA lanes from the description, the few-row kernel (1, 5 rows), the float32 product (70, 130), the native kind in a bfloat16 run (70, 130) and the tensor rows decoding as the tiles load or over widened rows (70, 300; 128- and 64-row tiles), give the product of their values (the bfloat16 lanes': x and the weights rounded to bfloat16; within float32's bound for k terms)", ProductRoutesSuperLanes),
        ("product routes: the CPU's float32 product over B packed once into the tiled kernel's panels (MultiplyPanels, today's path over stored rows), or filled as each tile needs them (MultiplyPanelsFrom, the tiles lane), gives the bits of the product over B [k, n] (70, 130 and 512 rows)", ProductRoutesPanels),
        ("product routes: stored weights sharing an input in one launch (CUDA's grouped few-row kernels: 2 and 3 weights of bfloat16, int8, q8_0, q4_0, iq4_nl, Q4_K; the CPU's grouped pass over Q4_K's, q4_0's, Q6_K's, bfloat16's, int64's, float64's, int2's and mxfp4's stored values, x prepared once) give each the bits of its own product (1, 3 and 8 rows; 70 and 130 in a bfloat16 run, x packed once for the group)", ProductRoutesMany),
        ("product routes: the fused epilogues of stored weights (CUDA, few rows): the gate/up pair's act(gate) · up (SiLU, tanh GELU, ReLU) and a product's residual + y with its RMS normalization match the products composed with them (bfloat16, int8, q4_0, Q4_K; 1, 3 and 8 rows; within 1e-5 of the values' magnitude), through the dispatchers with no storage converted", ProductRoutesEpilogues),
        ("product routes: a bfloat16 checkpoint loaded under the measured route packs each weight from the file's rows (a bfloat16 weight's bits kept as they are; no float32 matrix, no transposition): every weight and the logits the bits of packing the float32 values, for bfloat16, int8, q8_0, q4_0, q4_k and iq2_xxs, with tied and own heads", ProductRoutesCheckpointRows),
        ("product routes: attention written token-major for the output projection (IMergedHeadsLayout: float32, bfloat16 and int8 caches, plain and windowed) gives the bits of the heads merged after attention, where the device writes so (the CPU; CUDA's tensor-core passes in a bfloat16 run, the bfloat16 cache's staging its halves as they are: the float32 cache's bits over the same values)", ProductRoutesMergedAttention),
        ("product routes: every described super-block and grid type (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS; random blocks as ggml stores them) by the CUDA tensor rows, decoding the values as each tile loads and over widened rows, 128- and 64-row tiles, gives the product of x and the values as ggml decodes them, both rounded to bfloat16 (70 rows; within float32's bound for k terms)", ProductRoutesEveryBlockType),
        ("product routes: IDRAK_ROUTES reads name, generic and measured (any case, blank is name) and refuses anything else naming the variable; the measured route records its choice", ProductRoutesSetting),
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

    private static void ProductRoutesNear(Device device)
    {
        var random = new Random(37);
        foreach (var route in new[] { ProductRoute.Name, ProductRoute.Generic, ProductRoute.Measured })
        {
            UnderRoute(route, () =>
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

                        // The CPU's integer lane (q4_k, q4_0, q6_k and int64 under the measured route) multiplies x rounded to int8 a block of 32 values.
                        bool roundsX = route == ProductRoute.Measured && device.Backend is Idrak.Abstraction.Devices.Cpu.CpuBackend && format is "q4_k" or "q4_0" or "q6_k" or "int64" or "int2" or "mxfp4";
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
                                        Check(false, $"{device}, route {route}, {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} "
                                            + $"(error {error:E2} past the bound {bound:E2}; the weights are {weight.Description})");
                                    }
                                }
                            }
                        }
                    }
                }
            });
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
                ("mxfp4", 96, 37), ("mxfp4", 256, 37), ("mxfp4", 320, 130) })
            {
                bool bfloat16 = format == "bfloat16", float64 = format == "float64";
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
                foreach (int m in new[] { 1, 3, 4, 5, 9, 70 })
                {
                    var x = RouteValues(random, (long)m * k);
                    // q4_k, q4_0, q6_k and int64: the decoded and the integer lane; bfloat16: the shifted and the paired lane;
                    // float64: the decoded and the mixed lane (x widened once, or in registers).
                    foreach (bool second in new[] { false, true })
                    {
                        var y = new float[m * n];
                        bool integer = second && !bfloat16 && !float64;
                        bool ran = Idrak.Abstraction.Devices.Cpu.CpuBackend.StoredRowLane(second, type, words, storage.Length, x, y, m, n, k, rowScales);
                        if (!ran)
                        {
                            Check(integer, $"the {(bfloat16 ? "bfloat16" : float64 && second ? "mixed" : "decoded")} lane does not read {type.Name} ({weight.Description})");
                            continue;
                        }

                        string lane = bfloat16 ? second ? "paired" : "shifted" : float64 && second ? "mixed" : integer ? "integer" : "decoded";
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
                                    Check(false, $"the {lane} lane, {format} [{k}, {n}], {m} rows: y[{i}, {j}] = {y[i * n + j]}, the product of the values {sum} (error {error:E2} past the bound {bound:E2})");
                                }
                            }
                        }

                        runs++;
                    }
                }
            }
        });
        Check(runs > 0, "no lane over stored values ran");

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
        UnderRoute(ProductRoute.Measured, () =>
        {
            int k = 256;
            foreach (string format in cpu ? new[] { "q4_k", "q4_0", "q6_k", "bfloat16", "int64", "float64", "int2", "mxfp4" } : new[] { "bfloat16", "int8", "q8_0", "q4_0", "iq4_nl", "q4_k" })
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

                            for (int j = 0; j < count; j++)
                            {
                                float[] together = outputs[j].ToArray(), alone = RouteProduct(weights[j], x, m, device);
                                int differs = Enumerable.Range(0, alone.Length).FirstOrDefault(i => BitConverter.SingleToInt32Bits(together[i]) != BitConverter.SingleToInt32Bits(alone[i]), -1);
                                Check(differs < 0, differs < 0 ? "" : $"{device}: {format}, {count} weights, {m} rows: weight {j}'s output {differs} is {together[differs]:R} grouped, {alone[differs]:R} alone");
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

        var random = new Random(79);
        UnderRoute(ProductRoute.Measured, () =>
        {
            int k = 256, n = 96;
            foreach (string format in new[] { "bfloat16", "int8", "q4_0", "q4_k" })
            {
                using var gate = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device);
                using var up = (StoredWeight)PackedWeightFormats.Pack(format, RouteValues(random, (long)k * n), k, n, device);
                var gain = RouteValues(random, n);
                foreach (int m in new[] { 1, 3, 8 })
                {
                    var x = RouteValues(random, (long)m * k);
                    float[] g = RouteProduct(gate, x, m, device), u = RouteProduct(up, x, m, device);
                    foreach (var (kind, fused) in new[] { 0, 1, 2 }.SelectMany(kind => new[] { (kind, false), (kind, true) }))
                    {
                        using var scope = new TensorScope();
                        var input = Tensor.From(x, [m, k], device);
                        var gateOut = Tensor.Empty([m, n], device);
                        var upOut = Tensor.Empty([m, n], device);
                        var hidden = Tensor.Empty([m, n], device);
                        (Storage, Storage?, Storage, int)[] pair = [(gate.StoredValues.Storage, gate.Scales?.Storage, gateOut.Storage, n), (up.StoredValues.Storage, up.Scales?.Storage, upOut.Storage, n)];
                        // Through the dispatcher (as Linear calls it): the stored weights reach the kernel as they are, no conversion counted.
                        var seen = new ConversionsSeen();
                        bool ran;
                        using (Telemetry.Subscribe(seen))
                        {
                            ran = fused ? cuda.StoredGatedFused(kind, input.Storage, m, k, pair, hidden.Storage)
                                : device.Backend.StoredMatMulGatedPair(kind, input.Storage, m, k, pair, hidden.Storage);
                        }

                        Check(seen.Events.Count == 0, $"{device}: {format}, {m} rows: the gate/up pair converted {seen.Events.Count} storages ({string.Join("; ", seen.Events.Take(3))})");
                        if (!ran)
                        {
                            Check(false, $"{device}: {format}, {m} rows: no fused gate/up pair");
                            continue;
                        }

                        var h = hidden.ToArray();
                        for (int i = 0; i < m * n; i++)
                        {
                            double expected = (double)Act(g[i], kind) * u[i];
                            Near(h[i], expected, Math.Abs(expected) + Math.Abs(g[i] * u[i]), $"{format}, {m} rows, activation {kind}: hidden {i}");
                        }
                    }

                    foreach (bool fused in new[] { false, true })
                    using (var scope = new TensorScope())
                    {
                        var input = Tensor.From(x, [m, k], device);
                        var residualValues = RouteValues(random, (long)m * n);
                        var residual = Tensor.From(residualValues, [m, n], device);
                        var gains = Tensor.From(gain, [n], device);
                        var y = Tensor.Empty([m, n], device);
                        var sum = Tensor.Empty([m, n], device);
                        var normalized = Tensor.Empty([m, n], device);
                        var seen = new ConversionsSeen();
                        bool ran;
                        using (Telemetry.Subscribe(seen))
                        {
                            ran = fused ? cuda.StoredAddNormFused(input.Storage, gate.StoredValues.Storage, gate.Scales?.Storage, y.Storage, m, n, k,
                                    residual.Storage, sum.Storage, gains.Storage, normalized.Storage, 1e-6f, 0.5f)
                                : device.Backend.StoredMatMulAddRmsNorm(input.Storage, gate.StoredValues.Storage, gate.Scales?.Storage, y.Storage, m, n, k,
                                    residual.Storage, sum.Storage, gains.Storage, normalized.Storage, 1e-6f, 0.5f);
                        }

                        // The stored weight reaches the kernel as it is (a weight converted every call was a decoding step's 6 ms).
                        Check(seen.Events.Count == 0, $"{device}: {format}, {m} rows: the residual and norm product converted {seen.Events.Count} storages ({string.Join("; ", seen.Events.Take(3))})");
                        if (!ran)
                        {
                            Check(false, $"{device}: {format}, {m} rows: no fused residual and norm");
                            continue;
                        }

                        float[] s = sum.ToArray(), o = normalized.ToArray();
                        for (int i = 0; i < m; i++)
                        {
                            double squares = 0;
                            for (int j = 0; j < n; j++)
                            {
                                double v = (double)residualValues[i * n + j] + g[i * n + j];
                                squares += v * v;
                                Near(s[i * n + j], v, Math.Abs(v) + 1, $"{format}, {m} rows: sum [{i}, {j}]");
                            }

                            double scale = 1 / Math.Sqrt(squares / n + 1e-6);
                            for (int j = 0; j < n; j++)
                            {
                                double v = (double)residualValues[i * n + j] + g[i * n + j], expected = v * scale * (gain[j] + 0.5);
                                Near(o[i * n + j], expected, Math.Abs(expected) + 1, $"{format}, {m} rows: normalized [{i}, {j}]");
                            }
                        }
                    }
                }
            }
        });
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
