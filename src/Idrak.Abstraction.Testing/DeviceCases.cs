// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The device check's built-in cases: together they call every operation of the device contract on fixed shapes (the
/// edge sizes 1 and odd primes among them) and on random ones, and check the CPU's results against plain loops where the
/// arithmetic is simple. The cases are public so an app can run a part of them (<see cref="DeviceCheckOptions.Filter"/>)
/// or add its own beside them.
/// </summary>
public static partial class DeviceCases
{
    /// <summary>Every built-in case.</summary>
    public static IReadOnlyList<DeviceCase> All { get; } =
    [
        new("element-wise: every unary function and its gradient, against plain loops", UnaryFunctions, random: true),
        new("element-wise: binary operations, extremes, powers, clamping, selection and their gradients", BinaryOperations, random: true),
        new("element-wise: affine maps, accumulation, row vectors, sums, dropout and gradient norms", Accumulation, random: true),
        new("matrix products: every transpose, beta 0 and 1, against plain loops", MatrixProducts),
        new("matrix products: batched, many sharing an input, column sums, with bias and low-rank terms", MoreProducts, random: true),
        new("strided products on matrix units, with a bias", StridedProducts, random: true),
        new("large tensors: a million elements and a 1000 x 1000 product (parallel and multi-block paths)", LargeTensors),
        new("softmax, log-softmax, scaled and masked softmax, arg-max, class match and token cross-entropy", SoftmaxFamily, random: true),
        new("normalization: batch, group, layer and RMS norms, their gradients and fused forms", Normalization, random: true),
        new("rotary positions: rope forward and back, normalized rope, pairs and attention heads", RotaryPositions, random: true),
        new("embeddings: gather, bfloat16 gather, one-hot and scatter-add", Embeddings, random: true),
        new("convolution and pooling: im2col, col2im, max pooling and its gradients", ConvolutionAndPooling, random: true),
        new("layout: permutations, axis sums and broadcasts", Layout, random: true),
        new("optimizer steps: SGD, Adam, 8-bit Adam and fused AdamW", OptimizerSteps, random: true),
        new("packed weights: int8, 4-bit and bfloat16 products, dequantization and packing", PackedWeights, random: true),
        new("packed weights: fused products (many, gated, with the residual norm, low-rank, transposed, FP8)", FusedPackedProducts, random: true),
        new("gated activations: SiLU, GELU and ReLU gates, their gradients and bfloat16 forms", GatedActivations, random: true),
        new("decoding: masks and key/value caches in float32, int8 and bfloat16", KeyValueCaches, random: true),
        new("attention: decoding, tiled, int8 and bfloat16 caches, rows, segments, strided, windows and soft-caps", Attention, random: true),
        new("sampling: penalties, top-k, top-p and min-p draws, and the token history", Sampling, random: true),
        new("autograd: a small network's forward and backward passes through tensors", Autograd, random: true),
    ];

    // ------------------------------------------------------------------ helpers

    private static float[] Read(Storage storage) => DeviceCaseContext.Read(storage);

    private static Storage Random(DeviceCaseContext c, int n, float scale = 1f) => c.Storage(c.Values(n, scale));

    private static float[] PositiveValues(DeviceCaseContext c, int n, float scale = 3f) => [.. c.Values(n, scale).Select(v => MathF.Abs(v) + 0.1f)];

    // Integers in [0, count) as floats (indices, token ids).
    private static Storage Indices(DeviceCaseContext c, int n, int count) => c.Storage([.. Enumerable.Range(0, n).Select(_ => (float)c.Random.Next(count))]);

    // Fixed sizes, then one random size, then (Large) a million and three.
    private static int[] Sizes(DeviceCaseContext c, params int[] sizes) =>
        [.. sizes, c.Size(2, 300), .. c.Large ? [1_000_003] : Array.Empty<int>()];

    // The bfloat16 bits of x, rounded to nearest, ties to even (as PackBFloat16).
    private static ushort BFloat16(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    // values [rows, cols] as bfloat16 pairs per row: ⌈cols / 2⌉ words a row, column 2w in the low half of word w.
    private static float[] PackRows(float[] values, int rows, int cols)
    {
        int words = (cols + 1) / 2;
        var packed = new float[rows * words];
        for (int r = 0; r < rows; r++)
        {
            for (int w = 0; w < words; w++)
            {
                uint low = BFloat16(values[r * cols + 2 * w]), high = 2 * w + 1 < cols ? BFloat16(values[r * cols + 2 * w + 1]) : 0u;
                packed[r * words + w] = BitConverter.UInt32BitsToSingle(low | (high << 16));
            }
        }

        return packed;
    }

    // ------------------------------------------------------------------ element-wise

    private static Func<float, float>? Reference(UnaryOp op) => op switch
    {
        UnaryOp.Sigmoid => v => 1f / (1f + MathF.Exp(-v)),
        UnaryOp.Tanh => MathF.Tanh,
        UnaryOp.Relu => v => MathF.Max(v, 0f),
        UnaryOp.Square => v => v * v,
        UnaryOp.Abs => MathF.Abs,
        UnaryOp.Exp => MathF.Exp,
        UnaryOp.Log => MathF.Log,
        UnaryOp.Sqrt => MathF.Sqrt,
        UnaryOp.Sin => MathF.Sin,
        UnaryOp.Cos => MathF.Cos,
        UnaryOp.Silu => v => v / (1f + MathF.Exp(-v)),
        UnaryOp.Sign => v => MathF.Sign(v),
        UnaryOp.Gelu => v => 0.5f * v * (1f + MathF.Tanh(0.7978845608f * (v + 0.044715f * v * v * v))),
        _ => null,
    };

    private static void UnaryFunctions(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int n in Sizes(c, 1, 37))
        {
            foreach (var op in Enum.GetValues<UnaryOp>())
            {
                var xs = op is UnaryOp.Log or UnaryOp.Sqrt ? PositiveValues(c, n) : c.Values(n, 4f);
                var x = c.Storage(xs);
                var y = c.Zeros(n);
                b.Unary(op, x, y, n);
                if (Reference(op) is { } f)
                {
                    c.ExpectClose([.. xs.Select(f)], Read(y), 1e-4f, $"{op} of {n} values");
                }

                b.UnaryBackward(op, x, y, Random(c, n), Random(c, n), n);
            }
        }

        // Saturation: large inputs give exactly the limits.
        var s = c.Storage([-100f, 100f, -1000f, 1000f]);
        var sy = c.Zeros(4);
        b.Unary(UnaryOp.Sigmoid, s, sy, 4);
        c.ExpectClose([0f, 1f, 0f, 1f], Read(sy), 1e-6f, "sigmoid saturation");
        b.Unary(UnaryOp.Tanh, s, sy, 4);
        c.ExpectClose([-1f, 1f, -1f, 1f], Read(sy), 1e-6f, "tanh saturation");
    }

    private static void BinaryOperations(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int n in Sizes(c, 1, 37))
        {
            float[] av = c.Values(n, 4f), bv = c.Values(n, 4f);
            Storage a = c.Storage(av), bs = c.Storage(bv), y = c.Zeros(n);
            foreach (var op in Enum.GetValues<BinaryOp>())
            {
                b.Binary(op, a, bs, y, n);
                Func<float, float, float> f = op switch
                {
                    BinaryOp.Add => (p, q) => p + q,
                    BinaryOp.Sub => (p, q) => p - q,
                    BinaryOp.Mul => (p, q) => p * q,
                    BinaryOp.Maximum => MathF.Max,
                    _ => MathF.Min,
                };
                c.ExpectClose([.. av.Zip(bv, f)], Read(y), 1e-6f, $"{op} of {n} values");
            }

            foreach (var op in new[] { BinaryOp.Maximum, BinaryOp.Minimum })
            {
                b.ExtremumBackward(op, a, bs, Random(c, n), Random(c, n), Random(c, n), n);
                b.ExtremumBackward(op, a, bs, Random(c, n), null, Random(c, n), n);
                b.ExtremumBackward(op, a, bs, Random(c, n), Random(c, n), null, n);
            }

            var positive = c.Storage(PositiveValues(c, n));
            foreach (float exponent in new[] { 2f, 0.5f, -1.5f, 3f })
            {
                b.Pow(positive, y, n, exponent);
                b.PowBackward(positive, Random(c, n), Random(c, n), n, exponent);
            }

            b.Clamp(a, y, n, -1f, 2f);
            b.ClampBackward(a, Random(c, n), Random(c, n), n, -1f, 2f);
            b.Clamp(a, y, n, float.NegativeInfinity, 0.5f);
            b.ClampBackward(a, Random(c, n), Random(c, n), n, 0f, float.PositiveInfinity);

            var condition = c.Storage([.. Enumerable.Range(0, n).Select(_ => (float)c.Random.Next(2))]);
            b.Where(condition, a, bs, y, n);
            b.WhereBackward(condition, Random(c, n), Random(c, n), Random(c, n), n);
            b.WhereBackward(condition, Random(c, n), null, Random(c, n), n);
            b.WhereBackward(condition, Random(c, n), Random(c, n), null, n);
        }
    }

    private static void Accumulation(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int n in Sizes(c, 1, 37))
        {
            var xv = c.Values(n);
            var x = c.Storage(xv);
            var y = c.Zeros(n);
            b.Fill(y, n, 0.25f);
            b.Affine(x, y, n, 3f, -2f);
            c.ExpectClose([.. xv.Select(v => 3f * v - 2f)], Read(y), 1e-5f, "affine");
            b.Axpy(x, Random(c, n), n, -0.5f);
            b.MulAdd(x, Random(c, n), Random(c, n), n);
            var sum = c.Zeros(1);
            b.Sum(x, sum, n, 1f);
            c.ExpectClose([(float)xv.Sum(v => (double)v)], Read(sum), 1e-4f, "sum");
            b.Sum(x, sum, n, 1f / n);
            c.ExpectClose([(float)xv.Average(v => (double)v)], Read(sum), 1e-5f, "mean");
            b.AxpyAt(sum, Random(c, 5), c.Random.Next(5), 0.5f);
            b.AddBroadcastScalar(sum, Random(c, n), n, 2f);
            b.InvSqrt(c.Storage(PositiveValues(c, n)), y, n, 1e-5f);
            var total = c.Storage([0.5f]);
            b.SumSquares(x, total, n);
            var factor = c.Zeros(1);
            b.ClipFactor(total, factor, 0.1f);
            b.ClipFactor(c.Zeros(1), factor, 1f);
            foreach (float p in new[] { 0f, 0.25f, 0.5f })
            {
                uint seed = (uint)c.Random.Next();
                b.Dropout(x, y, n, p, seed);
                b.DropoutBackward(Random(c, n), Random(c, n), n, p, seed);
                b.AddDropout(Random(c, n), x, c.Zeros(n), n, p, seed);
            }
        }

        foreach (var (rows, cols) in new[] { (1, 1), (3, 17), (c.Size(1, 64), c.Size(1, 300)) })
        {
            var a = Random(c, rows * cols);
            b.AddRowVector(a, Random(c, cols), c.Zeros(rows * cols), rows, cols);
            b.SumRows(a, Random(c, cols), rows, cols);
        }
    }

    // ------------------------------------------------------------------ products

    private static void MatrixProducts(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (var (m, n, k) in new[] { (1, 1, 1), (3, 5, 2), (17, 33, 9), (64, 48, 70), (130, 67, 91), (1100, 1030, 21), (1, 700, 300), (8, 300, 77), (16, 129, 65) })
        {
            foreach (bool ta in new[] { false, true })
            {
                foreach (bool tb in new[] { false, true })
                {
                    foreach (float beta in new[] { 0f, 1f })
                    {
                        float[] a = c.Values(m * k), bm = c.Values(k * n), cv = c.Values(m * n);
                        var expected = new float[m * n];
                        for (int i = 0; i < m; i++)
                        {
                            for (int j = 0; j < n; j++)
                            {
                                double acc = 0;
                                for (int p = 0; p < k; p++)
                                {
                                    acc += (double)a[ta ? p * m + i : i * k + p] * bm[tb ? j * k + p : p * n + j];
                                }

                                expected[i * n + j] = (float)(acc + beta * cv[i * n + j]);
                            }
                        }

                        var output = c.Storage(cv);
                        b.MatMul(c.Storage(a), c.Storage(bm), output, m, n, k, ta, tb, beta);
                        c.ExpectClose(expected, Read(output), 1e-4f, $"matmul m={m} n={n} k={k} transA={ta} transB={tb} beta={beta}");
                    }
                }
            }
        }
    }

    private static void MoreProducts(DeviceCaseContext c)
    {
        var b = c.Backend;
        int batch = c.Size(1, 5), m = c.Size(1, 40), n = c.Size(1, 40), k = c.Size(1, 40);
        foreach (bool ta in new[] { false, true })
        {
            foreach (bool tb in new[] { false, true })
            {
                b.BatchedMatMul(Random(c, batch * m * k), Random(c, batch * k * n), Random(c, batch * m * n), batch, m, n, k, ta, tb, c.Random.Next(2));
            }
        }

        // Several products of one input (the query, key and value projections), with and without biases.
        var input = Random(c, m * k);
        int[] widths = [c.Size(1, 48), c.Size(1, 48), c.Size(1, 48)];
        var products = widths.Select((w, i) => (Weight: Random(c, k * w), Bias: i == 1 ? null : Random(c, w), Output: c.Zeros(m * w), Columns: w)).ToArray();
        b.MatMulMany(input, m, k, products);

        // Column sums of a strided block (a slice's bias gradient).
        int ld = n + c.Random.Next(4);
        b.SumColumns(Random(c, (m + 1) * ld), ld, ld, Random(c, n), m, n);

        // c = a·b + bias in one pass.
        Storage a = Random(c, m * k), bw = Random(c, k * n), bias = Random(c, n), y = Random(c, m * n);
        c.Composed(Ops.MatMulBias, cpu => cpu.MatMulBias(a, bw, bias, y, m, n, k), cpu =>
        {
            cpu.MatMul(a, bw, y, m, n, k, false, false, 0f);
            cpu.AddRowVector(y, bias, y, m, n);
        });

        // c = beta·c + a·op(b) + u·op(v) (a LoRA adapter's term).
        int rank = c.Size(1, 16);
        foreach (bool tb in new[] { false, true })
        {
            float beta = c.Random.Next(2);
            Storage u = Random(c, m * rank), v = Random(c, rank * n), lowRank = Random(c, m * n), lb = Random(c, k * n);
            c.Composed(Ops.MatMulLowRank, cpu => cpu.MatMulLowRank(a, lb, lowRank, m, n, k, tb, beta, u, v, rank), cpu =>
            {
                cpu.MatMul(a, lb, lowRank, m, n, k, false, tb, beta);
                cpu.MatMul(u, v, lowRank, m, n, rank, false, tb, 1f);
            });
        }
    }

    private static void StridedProducts(DeviceCaseContext c)
    {
        var b = c.Backend;
        int m = c.Size(1, 48), n = c.Size(1, 48), k = c.Size(1, 48);
        foreach (bool ta in new[] { false, true })
        {
            foreach (bool tb in new[] { false, true })
            {
                int lda = (ta ? m : k) + c.Random.Next(3), ldb = (tb ? k : n) + c.Random.Next(3), ldc = n + c.Random.Next(3);
                int aRows = ta ? k : m, bRows = tb ? n : k;
                long aOffset = c.Random.Next(4), bOffset = c.Random.Next(4), cOffset = c.Random.Next(4);
                Storage a = Random(c, (int)aOffset + aRows * lda), bs = Random(c, (int)bOffset + bRows * ldb), y = Random(c, (int)cOffset + m * ldc);
                foreach (var bias in new[] { null, Random(c, n) })
                {
                    float beta = c.Random.Next(2);
                    c.Composed(Ops.GemmStrided, cpu => cpu.GemmStrided(a, aOffset, lda, ta, bs, bOffset, ldb, tb, y, cOffset, ldc, m, n, k, beta, bias),
                        cpu => GemmReference(cpu, a, aOffset, lda, ta, bs, bOffset, ldb, tb, y, cOffset, ldc, m, n, k, beta, bias));
                }
            }
        }
    }

    // GemmStrided (no epilogue) by plain loops on the host, written into the CPU storage.
    private static void GemmReference(Backend cpu, Storage a, long aOffset, int lda, bool ta, Storage bs, long bOffset, int ldb, bool tb, Storage y, long cOffset,
        int ldc, int m, int n, int k, float beta, Storage? bias)
    {
        float[] av = Read(a), bv = Read(bs), yv = Read(y);
        float[]? biasValues = bias is null ? null : Read(bias);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double acc = 0;
                for (int p = 0; p < k; p++)
                {
                    acc += (double)av[aOffset + (ta ? p * lda + i : i * lda + p)] * bv[bOffset + (tb ? j * ldb + p : p * ldb + j)];
                }

                long at = cOffset + i * ldc + j;
                yv[at] = beta * yv[at] + (float)acc + (biasValues?[j] ?? 0f);
            }
        }

        cpu.Upload(yv, y);
    }

    // Moved here from tests/Idrak.Tests ("large tensors"): the parallel and multi-block paths.
    private static void LargeTensors(DeviceCaseContext c)
    {
        const int N = 1_000_003;
        var x = c.Values(N);
        using var tx = Tensor.From(x, c.Device);
        double expectedSum = x.Sum(v => (double)v);
        c.ExpectClose([(float)expectedSum], [tx.Sum().Item()], 1e-2f, "large sum");
        var sigmoid = tx.Sigmoid().ToArray();
        for (int i = 0; i < N; i += 9973)
        {
            c.ExpectClose([1f / (1f + MathF.Exp(-x[i]))], [sigmoid[i]], 1e-5f, $"large sigmoid[{i}]");
        }

        using var matrix = Tensor.From(x.AsSpan(0, 1000 * 1000), [1000, 1000], c.Device);
        using var ones = Tensor.Ones([1000, 1], c.Device);
        var rowSums = matrix.MatMul(ones).ToArray();
        for (int r = 0; r < 1000; r += 97)
        {
            double expected = 0;
            for (int col = 0; col < 1000; col++)
            {
                expected += x[r * 1000 + col];
            }

            c.ExpectClose([(float)expected], [rowSums[r]], 1e-3f, $"large matmul row {r}");
        }
    }

    // ------------------------------------------------------------------ rows

    private static void SoftmaxFamily(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (var (rows, cols) in new[] { (1, 1), (3, 17), (2, 1000), (c.Size(1, 64), c.Size(1, 300)) })
        {
            int n = rows * cols;
            var x = Random(c, n, 6f);
            var y = c.Zeros(n);
            foreach (bool log in new[] { false, true })
            {
                b.Softmax(x, y, rows, cols, log);
                b.SoftmaxBackward(y, Random(c, n), Random(c, n), rows, cols, log);
            }

            int maskRows = Math.Max(1, rows / 2);
            var mask = c.Storage([.. Enumerable.Range(0, maskRows * cols).Select(i => c.Random.Next(4) == 0 ? -1e9f : c.Random.NextSingle())]);
            b.ScaleMaskSoftmax(x, null, y, rows, cols, 1, 0.125f);
            b.ScaleMaskSoftmax(x, mask, y, rows, cols, maskRows, 0.5f);
            b.ArgMax(x, c.Zeros(rows), rows, cols);

            var targets = c.Zeros(n);
            var targetValues = new float[n];
            for (int r = 0; r < rows; r++)
            {
                targetValues[r * cols + c.Random.Next(cols)] = 1f;
            }

            b.Upload(targetValues, targets);
            b.ClassMatch(x, targets, c.Zeros(rows), rows, cols, 0.5f);

            var weights = c.Storage([.. Enumerable.Range(0, rows).Select(r => r % 3 == 2 ? 0f : 1f)]);
            b.SoftmaxCrossEntropyRows(Random(c, n, 4f), Indices(c, rows, cols), weights, c.Zeros(rows), rows, cols, 1f / rows);
        }

        b.ClassMatch(Random(c, 9), c.Storage([.. Enumerable.Range(0, 9).Select(i => (float)(i % 2))]), c.Zeros(9), 9, 1, 0.25f);
    }

    private static void Normalization(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (var (outer, groups, inner) in new[] { (2, 1, 1), (2, 3, 5), (c.Size(2, 8), c.Size(1, 16), c.Size(1, 64)) })
        {
            int n = outer * groups * inner;
            var x = Random(c, n, 3f);
            Storage mean = c.Zeros(groups), variance = c.Zeros(groups), invStd = c.Zeros(groups), y = c.Zeros(n);
            b.NormStats(x, mean, variance, invStd, outer, groups, inner, 1e-5f);
            b.NormApply(x, mean, invStd, y, outer, groups, inner);
            b.NormBackward(Random(c, n), y, Random(c, groups), Random(c, groups), invStd, Random(c, n), outer, groups, inner);
            foreach (bool accumulate in new[] { false, true })
            {
                b.GroupScaleShift(x, Random(c, groups), Random(c, groups), Random(c, n), n, groups, inner, accumulate);
                b.GroupScaleShift(x, null, Random(c, groups), Random(c, n), n, groups, inner, accumulate);
                b.GroupScaleShift(x, Random(c, groups), null, Random(c, n), n, groups, inner, accumulate);
            }

            b.GroupReduce(x, Random(c, n), Random(c, groups), Random(c, groups), outer, groups, inner);
            b.GroupReduce(x, null, Random(c, groups), null, outer, groups, inner);
        }

        foreach (var (rows, cols) in new[] { (1, 1), (3, 17), (c.Size(1, 32), c.Size(2, 300)) })
        {
            int n = rows * cols;
            Storage x = Random(c, n, 3f), gamma = Random(c, cols), beta = Random(c, cols), y = c.Zeros(n), stats = c.Zeros(2 * rows), inv = c.Zeros(rows);
            b.LayerNormFused(x, gamma, beta, y, rows, cols, 1e-5f);
            b.LayerNormTrain(x, gamma, beta, y, stats, rows, cols, 1e-5f);
            b.LayerNormBackward(x, gamma, Random(c, n), stats, Random(c, n), Random(c, cols), Random(c, cols), rows, cols);
            b.LayerNormBackward(x, gamma, Random(c, n), stats, null, Random(c, cols), Random(c, cols), rows, cols);
            b.LayerNormBackward(x, gamma, Random(c, n), stats, Random(c, n), null, null, rows, cols);
            b.RmsNorm(x, y, inv, rows, cols, 1e-6f);
            b.RmsNormBackward(Random(c, n), y, inv, Random(c, n), rows, cols);
            foreach (float offset in new[] { 0f, 1f })
            {
                b.RmsNormAffine(x, gamma, y, rows, cols, 1e-6f, offset);
                b.AddRmsNormAffine(x, Random(c, n), c.Zeros(n), gamma, y, rows, cols, 1e-6f, offset);
            }

            b.BiasGelu(x, Random(c, cols), y, n, cols);
        }
    }

    // cos and sin of position · θ^(-p / half) for positions [0, positions) and pairs [0, half).
    private static (Storage Cos, Storage Sin) RotaryTables(DeviceCaseContext c, int positions, int half)
    {
        var cos = new float[positions * half];
        var sin = new float[positions * half];
        for (int t = 0; t < positions; t++)
        {
            for (int p = 0; p < half; p++)
            {
                double angle = t * Math.Pow(10_000, -(double)p / half);
                cos[t * half + p] = (float)Math.Cos(angle);
                sin[t * half + p] = (float)Math.Sin(angle);
            }
        }

        return (c.Storage(cos), c.Storage(sin));
    }

    private static void RotaryPositions(DeviceCaseContext c)
    {
        var b = c.Backend;
        int batch = c.Size(1, 3), steps = c.Size(1, 9), heads = c.Size(1, 4), dim = 2 * c.Size(1, 32), maxPositions = 64;
        var (cos, sin) = RotaryTables(c, maxPositions, dim / 2);
        int rows = batch * steps * heads, n = rows * dim;
        var positions = c.Storage([.. Enumerable.Range(0, steps).Select(t => (float)((t * 7 + 3) % maxPositions))]);
        foreach (bool interleaved in new[] { false, true })
        {
            foreach (int half in new[] { dim / 2, Math.Max(1, dim / 4) })
            {
                var (hc, hs) = half == dim / 2 ? (cos, sin) : RotaryTables(c, maxPositions, half);
                foreach (float sign in new[] { 1f, -1f })
                {
                    var xv = c.Values(n);
                    b.Rope(c.Storage(xv), c.Storage(xv), hc, hs, positions, rows, heads, steps, dim, half, interleaved, sign);
                }

                b.RmsNormRope(Random(c, n), Random(c, dim), hc, hs, positions, c.Zeros(n), rows, dim, 1e-6f, 1f, heads, steps, half, interleaved);
                int heads2 = c.Size(1, 3), rows2 = batch * steps * heads2;
                b.RmsNormRopePair(Random(c, n), Random(c, dim), c.Zeros(n), rows, 1e-6f, 0f, heads,
                    Random(c, rows2 * dim), Random(c, dim), c.Zeros(rows2 * dim), rows2, 1e-5f, 1f, heads2, hc, hs, positions, dim, steps, half, interleaved);
            }
        }
    }

    private static void Embeddings(DeviceCaseContext c)
    {
        var b = c.Backend;
        int vocabulary = c.Size(1, 100), dim = c.Size(1, 64), count = c.Size(1, 40);
        var table = c.Values(vocabulary * dim);
        var indices = Indices(c, count, vocabulary);
        b.Gather(c.Storage(table), indices, c.Zeros(count * dim), count, dim, vocabulary);
        b.GatherBFloat16(c.Storage(PackRows(table, vocabulary, dim)), indices, c.Zeros(count * dim), count, dim, vocabulary);
        b.OneHot(indices, c.Zeros(count * vocabulary), count, vocabulary);
        b.ScatterAdd(Random(c, count * dim), indices, Random(c, vocabulary * dim), count, dim, vocabulary);
    }

    private static void ConvolutionAndPooling(DeviceCaseContext c)
    {
        var b = c.Backend;
        ConvGeometry[] convolutions =
        [
            new(1, 1, 1, 1, 1, 1, 1, 1, 0, 0), new(2, 3, 7, 5, 3, 3, 1, 1, 1, 1), new(1, 2, 9, 9, 3, 3, 2, 2, 0, 0),
            new(c.Size(1, 3), c.Size(1, 4), c.Size(3, 12), c.Size(3, 12), 3, 2, c.Size(1, 2), c.Size(1, 2), c.Random.Next(2), c.Random.Next(2)),
        ];
        foreach (var g in convolutions)
        {
            int input = g.N * g.C * g.H * g.W, cols = g.Positions * g.PatchSize;
            b.Im2Col(Random(c, input), c.Zeros(cols), g);
            b.Col2Im(Random(c, cols), Random(c, input), g);
        }

        ConvGeometry[] pools = [new(1, 1, 2, 2, 2, 2, 2, 2, 0, 0), new(2, 3, 8, 6, 2, 2, 2, 2, 0, 0), new(c.Size(1, 3), c.Size(1, 4), c.Size(3, 12), c.Size(3, 12), 3, 3, 2, 1, 0, 0)];
        foreach (var g in pools)
        {
            int input = g.N * g.C * g.H * g.W, count = g.N * g.C * g.OH * g.OW;
            Storage y = c.Zeros(count), argmax = c.Zeros(count);
            b.MaxPool(Random(c, input), y, argmax, g);
            b.MaxPoolBackward(Random(c, count), argmax, Random(c, input), count);
            b.MaxPoolBackward(Random(c, count), argmax, Random(c, input), g);
        }
    }

    private static void Layout(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int rank in new[] { 2, 3, 4, c.Size(2, 6) })
        {
            int[] shape = [.. Enumerable.Range(0, rank).Select(_ => c.Size(1, rank > 4 ? 4 : 7))];
            int[] permutation = [.. Enumerable.Range(0, rank).OrderBy(_ => c.Random.Next())];
            var strides = new int[rank];
            for (int d = rank - 1, s = 1; d >= 0; s *= shape[d], d--)
            {
                strides[d] = s;
            }

            int n = shape.Aggregate(1, (p, q) => p * q);
            int[] outShape = [.. permutation.Select(p => shape[p])], inStrides = [.. permutation.Select(p => strides[p])];
            var x = Random(c, n);
            b.Permute(x, c.Zeros(n), outShape, inStrides, false);
            b.Permute(x, Random(c, n), outShape, inStrides, true);
        }

        foreach (var (outer, dim, inner) in new[] { (1, 1, 1), (2, 3, 5), (c.Size(1, 8), c.Size(1, 32), c.Size(1, 16)) })
        {
            var x = Random(c, outer * dim * inner);
            b.SumAxis(x, c.Zeros(outer * inner), outer, dim, inner, 1f, false);
            b.SumAxis(x, Random(c, outer * inner), outer, dim, inner, 1f / dim, true);
            b.BroadcastAxis(Random(c, outer * inner), Random(c, outer * dim * inner), outer, dim, inner, 0.5f);
        }
    }

    private static void OptimizerSteps(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (int n in Sizes(c, 1, 37))
        {
            b.SgdStep(Random(c, n), Random(c, n), null, n, 0.1f, 0f);
            b.SgdStep(Random(c, n), Random(c, n), Random(c, n), n, 0.05f, 0.9f);
            b.AdamStep(Random(c, n), Random(c, n), Random(c, n), c.Storage(PositiveValues(c, n, 0.1f)), n, 1e-3f, 0.9f, 0.999f, 1e-8f);

            // 8-bit moments: a byte per element (four per float), codes into a sorted map (256 signed, then 256 unsigned),
            // one scale per block of 256 for m, then for v.
            int blocks = (n + EightBitMoments.BlockSize - 1) / EightBitMoments.BlockSize;
            var map = new float[512];
            for (int i = 0; i < 256; i++)
            {
                map[i] = -1f + 2f * i / 255f;
                map[256 + i] = i / 255f;
            }

            Storage Bytes() => c.Storage([.. Enumerable.Range(0, (n + 3) / 4).Select(_ => BitConverter.Int32BitsToSingle(c.Random.Next() & 0x7F7F7F7F))]);
            b.AdamStep8Bit(Random(c, n), Random(c, n), Bytes(), Bytes(), c.Storage(PositiveValues(c, 2 * blocks, 0.01f)), c.Storage(map), n,
                1e-3f, 0.9f, 0.999f, 1e-8f, 0.5f, 0.999f);
        }

        // AdamW over several tensors, clipped to a norm.
        int[] sizes = [c.Size(1, 64), c.Size(1, 300), 1];
        var tensors = sizes.Select(n => (P: Random(c, n), G: Random(c, n), M: Random(c, n), V: c.Storage(PositiveValues(c, n, 0.1f)), N: n)).ToArray();
        IDisposable? cache = null;
        try
        {
            b.FusedAdamW(tensors, ref cache, 0.5f, 1e-3f, 0.999f, 0.9f, 0.999f, 1e-8f, true);
        }
        finally
        {
            cache?.Dispose();
        }
    }
}
