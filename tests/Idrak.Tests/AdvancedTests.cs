// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// Tests for classification, normalization, embeddings, convolution, recurrent and attention layers,
// N-D tensor operations, schedulers and number-type interop.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Advanced =
    [
        ("gradient: exp, log, gelu", d =>
        {
            GradCheck(d, [3, 4], x => x.Exp().Sum(), scale: 0.5f);
            GradCheck(d, [3, 4], x => (x.Abs() + 0.5f).Log().Sum());
            GradCheck(d, [3, 4], x => x.Gelu().Sum(), scale: 2f);
        }),
        ("softmax rows are distributions and match reference", SoftmaxReference),
        ("gradient: softmax, log-softmax", d =>
        {
            using var w = Tensor.From(RandomArray(new Random(11), 12), [3, 4], d);
            GradCheck(d, [3, 4], x => (x.Softmax() * w).Sum(), scale: 2f);
            GradCheck(d, [3, 4], x => (x.LogSoftmax() * w).Sum(), scale: 2f);
        }),
        ("gradient: cross-entropy (with label smoothing)", d =>
        {
            using var targets = Tensor.From(new float[,] { { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, { 1, 0, 0 } }, d);
            GradCheck(d, [4, 3], x => Losses.CrossEntropy(x, targets), scale: 2f);
            GradCheck(d, [4, 3], x => Losses.CrossEntropy(x, targets, labelSmoothing: 0.1f), scale: 2f);
        }),
        ("gradient: binary cross-entropy (probabilities and logits)", d =>
        {
            using var targets = Tensor.From([1f, 0f, 1f, 0f, 1f, 1f], [6, 1], d);
            GradCheck(d, [6, 1], x => Losses.BinaryCrossEntropy(x.Sigmoid(), targets), scale: 2f);
            GradCheck(d, [6, 1], x => Losses.BinaryCrossEntropyWithLogits(x, targets), scale: 2f);
            using var logits = Tensor.From([2f, -1f, 0.5f, -3f, 0.1f, 4f], [6, 1], d);
            AssertClose([Losses.BinaryCrossEntropy(logits.Sigmoid(), targets).Item()], [Losses.BinaryCrossEntropyWithLogits(logits, targets).Item()], 1e-4f, "BCE forms agree");
        }),
        ("argmax and accuracy", ArgMaxAndAccuracy),
        ("one-hot and sparse cross-entropy match dense", d =>
        {
            using var labels = Tensor.From(new float[,] { { 2, 0 }, { 1, 2 } }, d);
            var oneHot = Tensor.OneHot(labels, 3);
            Check(oneHot.Shape.SequenceEqual([2, 2, 3]), "one-hot shape");
            AssertClose([0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], oneHot.ToArray(), 0, "one-hot values");
            using (var wide = Tensor.OneHot(Tensor.From([49_999f, 0f, 12_345f], [3], d), 50_000))
            {
                var values = wide.ToArray();
                Check(values.Sum() == 3 && values[49_999] == 1 && values[50_000] == 1 && values[100_000 + 12_345] == 1, "one-hot over 50,000 classes");
            }

            using var logits = Tensor.From(RandomArray(new Random(40), 12, 2f), [2, 2, 3], d);
            AssertClose([Losses.CrossEntropy(logits, oneHot).Item()], [Losses.SparseCrossEntropy(logits, labels).Item()], 1e-6f, "sparse == dense");
            GradCheck(d, [2, 2, 3], x => Losses.SparseCrossEntropy(x, labels, labelSmoothing: 0.1f), scale: 2f);
            AssertClose([Metric.Accuracy.BatchMean(logits, oneHot).Item()], [Metric.SparseAccuracy.BatchMean(logits, labels).Item()], 0, "sparse accuracy");
        }),
        ("batched matmul matches reference and gradients", BatchedMatMul),
        ("gradient: N-D × 2-D matmul (shared weight)", d =>
        {
            using var w = Tensor.From(RandomArray(new Random(12), 4 * 3), [4, 3], d);
            GradCheck(d, [2, 5, 4], x => x.MatMul(w).Tanh().Sum());
        }),
        ("permute, transpose, narrow, concat, stack", ShapeOps),
        ("gradient: permute, narrow, concat, sum/mean over a dimension", d =>
        {
            using var w = Tensor.From(RandomArray(new Random(13), 24), [4, 3, 2], d);
            GradCheck(d, [2, 3, 4], x => (x.Permute(2, 1, 0) * w).Sum());
            GradCheck(d, [3, 5], x => x.Narrow(1, 1, 3).Square().Sum());
            GradCheck(d, [3, 2], x => (Tensor.Concat([x, x.Square(), x], 1) * Tensor.Ones([3, 6], d)).Tanh().Sum());
            GradCheck(d, [2, 3, 4], x => x.Sum(1).Square().Sum() + x.Mean(-1, keepDim: true).Tanh().Sum());
        }),
        ("batchnorm normalizes per channel; running stats; eval mode", BatchNormBehaviour),
        ("gradient: batchnorm (input and parameters)", d =>
        {
            var bn = new BatchNorm(3, device: d);
            GradCheck(d, [4, 3, 2, 2], x => (bn.Forward(x) * Tensor.From(RandomArray(new Random(14), 48), [4, 3, 2, 2], d)).Sum(), scale: 2f);
            ParameterGradCheck(bn, Tensor.From(RandomArray(new Random(15), 24, 2f), [8, 3], d), y => y.Tanh().Sum());
        }),
        ("layernorm normalizes rows; gradients", d =>
        {
            var ln = new LayerNorm(5, device: d);
            using var x = Tensor.From(RandomArray(new Random(16), 30, 3f), [2, 3, 5], d);
            var row = ln.Forward(x).ToArray().AsSpan(0, 5).ToArray();
            AssertClose([0f], [row.Average()], 1e-4f, "row mean");
            AssertClose([1f], [row.Select(v => v * v).Average()], 1e-3f, "row variance");
            using var w = Tensor.From(RandomArray(new Random(17), 30), [2, 3, 5], d);
            GradCheck(d, [2, 3, 5], t => (ln.Forward(t) * w).Sum(), scale: 2f);
            ParameterGradCheck(ln, x, y => (y * w).Sum());
        }),
        ("embedding lookup and gradient", EmbeddingBehaviour),
        ("conv2d matches direct convolution", ConvReference),
        ("gradient: conv2d (stride 2, padding 1) input and weights", d =>
        {
            var conv = new Conv2d(2, 3, 3, stride: 2, padding: 1, device: d, random: new Random(18));
            GradCheck(d, [2, 2, 5, 5], x => conv.Forward(x).Tanh().Sum());
            ParameterGradCheck(conv, Tensor.From(RandomArray(new Random(19), 2 * 2 * 5 * 5), [2, 2, 5, 5], d), y => y.Tanh().Sum());
        }),
        ("maxpool matches reference; gradient", MaxPoolBehaviour),
        ("gradient: LSTM and GRU (inputs and weights)", d =>
        {
            var lstm = new LSTM(3, 4, returnSequences: true, device: d, random: new Random(20));
            GradCheck(d, [2, 3, 3], x => lstm.Forward(x).Sum());
            ParameterGradCheck(lstm, Tensor.From(RandomArray(new Random(21), 18), [2, 3, 3], d), y => y.Square().Sum());
            var gru = new GRU(3, 4, device: d, random: new Random(22));
            GradCheck(d, [2, 3, 3], x => gru.Forward(x).Sum());
            ParameterGradCheck(gru, Tensor.From(RandomArray(new Random(23), 18), [2, 3, 3], d), y => y.Square().Sum());
        }),
        ("gradient: multi-head attention (causal) and transformer layer", d =>
        {
            var attention = new MultiHeadAttention(8, 2, causal: true, device: d, random: new Random(24));
            GradCheck(d, [2, 3, 8], x => attention.Forward(x).Tanh().Sum());
            var block = new TransformerEncoderLayer(8, 2, ffDim: 16, dropout: 0f, device: d, random: new Random(25));
            GradCheck(d, [2, 3, 8], x => block.Forward(x).Tanh().Sum());
            ParameterGradCheck(block, Tensor.From(RandomArray(new Random(26), 48), [2, 3, 8], d), y => y.Tanh().Sum());
        }),
        ("causal attention ignores future positions", CausalMasking),
        ("softmax classifier learns spirals (cross-entropy, BatchNorm, AdamW, cosine)", SpiralClassification),
        ("CNN learns to tell stripes apart", ConvolutionLearns),
        ("LSTM and transformer learn a sequence task", SequenceModelsLearn),
        ("schedulers, weight decay and gradient clipping", OptimizerFeatures),
        ("other number types in and out", NumberTypes),
        ("save and load include buffers (BatchNorm running stats)", SaveLoadBuffers),
    ];

    /// <summary>Checks d(loss)/d(parameter) for a sample of entries of every parameter, by central differences.</summary>
    private static void ParameterGradCheck(Module module, Tensor input, Func<Tensor, Tensor> loss)
    {
        foreach (var p in module.Parameters())
        {
            p.ZeroGrad();
        }

        loss(module.Forward(input)).Backward();
        const float h = 1e-2f;
        foreach (var (p, index) in module.Parameters().Select((p, i) => (p, i)))
        {
            var analytic = p.Grad!.ToArray();
            var values = p.ToArray();
            int step = Math.Max(1, values.Length / 12);
            for (int i = 0; i < values.Length; i += step)
            {
                float original = values[i];
                values[i] = original + h;
                p.Load(values);
                float plus;
                using (Autograd.NoGrad())
                {
                    plus = loss(module.Forward(input)).Item();
                }

                values[i] = original - h;
                p.Load(values);
                float minus;
                using (Autograd.NoGrad())
                {
                    minus = loss(module.Forward(input)).Item();
                }

                values[i] = original;
                p.Load(values);
                AssertClose([(plus - minus) / (2 * h)], [analytic[i]], 3e-2f, $"{module} parameter {index} element {i}");
            }
        }
    }

    private static void SoftmaxReference(Device device)
    {
        var x = RandomArray(new Random(30), 4 * 7, 5f);
        using var t = Tensor.From(x, [4, 7], device);
        var y = t.Softmax().ToArray();
        var logY = t.LogSoftmax().ToArray();
        for (int r = 0; r < 4; r++)
        {
            var row = x.AsSpan(r * 7, 7).ToArray();
            double max = row.Max(), sum = row.Sum(v => Math.Exp(v - max));
            for (int j = 0; j < 7; j++)
            {
                double expected = Math.Exp(row[j] - max) / sum;
                AssertClose([(float)expected], [y[r * 7 + j]], 1e-5f, "softmax");
                AssertClose([(float)Math.Log(expected)], [logY[r * 7 + j]], 1e-4f, "log-softmax");
            }
        }

        using var huge = Tensor.From([1000f, 0f, -1000f], [1, 3], device);
        AssertClose([1f, 0f, 0f], huge.Softmax().ToArray(), 1e-6f, "softmax is stable for large inputs");
    }

    private static void ArgMaxAndAccuracy(Device device)
    {
        using var p = Tensor.From(new float[,] { { 0.1f, 0.7f, 0.2f }, { 0.5f, 0.2f, 0.3f }, { 0.2f, 0.2f, 0.6f }, { 0.3f, 0.4f, 0.3f } }, device);
        using var t = Tensor.From(new float[,] { { 0, 1, 0 }, { 1, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } }, device);
        AssertClose([1f, 0f, 2f, 1f], p.ArgMax().ToArray(), 0, "argmax");
        AssertClose([0.75f], [Metric.Accuracy.BatchMean(p, t).Item()], 1e-6f, "accuracy");
        using var probabilities = Tensor.From([0.9f, 0.2f, 0.6f, 0.4f], [4, 1], device);
        using var labels = Tensor.From([1f, 0f, 0f, 0f], [4, 1], device);
        AssertClose([0.75f], [Metric.BinaryAccuracy().BatchMean(probabilities, labels).Item()], 1e-6f, "binary accuracy");
    }

    private static void BatchedMatMul(Device device)
    {
        var random = new Random(31);
        const int B = 3, M = 5, K = 4, N = 6;
        foreach (bool ta in new[] { false, true })
        {
            foreach (bool tb in new[] { false, true })
            {
                var a = RandomArray(random, B * M * K);
                var b = RandomArray(random, B * K * N);
                using var ta_ = Tensor.From(a, ta ? [B, K, M] : [B, M, K], device);
                using var tb_ = Tensor.From(b, tb ? [B, N, K] : [B, K, N], device);
                var c = ta_.MatMul(tb_, ta, tb).ToArray();
                for (int batch = 0; batch < B; batch++)
                {
                    for (int i = 0; i < M; i++)
                    {
                        for (int j = 0; j < N; j++)
                        {
                            double acc = 0;
                            for (int p = 0; p < K; p++)
                            {
                                float av = a[batch * M * K + (ta ? p * M + i : i * K + p)];
                                float bv = b[batch * K * N + (tb ? j * K + p : p * N + j)];
                                acc += av * bv;
                            }

                            AssertClose([(float)acc], [c[(batch * M + i) * N + j]], 1e-4f, $"bmm transA={ta} transB={tb}");
                        }
                    }
                }

                using var other = Tensor.From(b, tb ? [B, N, K] : [B, K, N], device);
                GradCheck(device, ta ? [B, K, M] : [B, M, K], x => x.MatMul(other, ta, tb).Tanh().Sum());
                using var left = Tensor.From(a, ta ? [B, K, M] : [B, M, K], device);
                GradCheck(device, tb ? [B, N, K] : [B, K, N], x => left.MatMul(x, ta, tb).Tanh().Sum());
            }
        }
    }

    private static void ShapeOps(Device device)
    {
        var values = Enumerable.Range(0, 24).Select(i => (float)i).ToArray();
        using var x = Tensor.From(values, [2, 3, 4], device);
        var p = x.Permute(2, 0, 1);
        Check(p.Shape.SequenceEqual([4, 2, 3]), "permute shape");
        var pv = p.ToArray();
        for (int a = 0; a < 4; a++)
        {
            for (int b = 0; b < 2; b++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Check(pv[(a * 2 + b) * 3 + c] == values[(b * 3 + c) * 4 + a], "permute values");
                }
            }
        }

        var t = x.Transpose(0, 2).ToArray();
        Check(t[(1 * 3 + 2) * 2 + 1] == values[(1 * 3 + 2) * 4 + 1], "transpose values");
        AssertClose([5, 6, 9, 10, 17, 18, 21, 22], x.Narrow(1, 1, 2).Narrow(2, 1, 2).ToArray(), 0, "narrow");
        var cat = Tensor.Concat([x.Narrow(2, 0, 1), x.Narrow(2, 3, 1)], 2);
        AssertClose([0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23], cat.ToArray(), 0, "concat");
        var stacked = Tensor.Stack([x.Narrow(1, 0, 1).Reshape(2, 4), x.Narrow(1, 2, 1).Reshape(2, 4)], 1);
        Check(stacked.Shape.SequenceEqual([2, 2, 4]), "stack shape");
        AssertClose([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23], stacked.ToArray(), 0, "stack");
        AssertClose([12, 15, 18, 21, 48, 51, 54, 57], x.Sum(1).ToArray(), 0, "sum over dim 1");
        AssertClose([1.5f, 5.5f, 9.5f, 13.5f, 17.5f, 21.5f], x.Mean(-1).ToArray(), 1e-6f, "mean over last dim");
        Check(x.Flatten().Shape.SequenceEqual([2, 12]), "flatten");
        using var bias = Tensor.From([100f, 200f, 300f, 400f], device);
        Check((x + bias).ToArray()[5] == 205f, "trailing-dimension broadcast add");
    }

    private static void BatchNormBehaviour(Device device)
    {
        var bn = new BatchNorm(2, momentum: 0.5f, device: device);
        var x = RandomArray(new Random(32), 4 * 2 * 3, 4f);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += (i / 3 % 2) * 10;   // channel 1 is shifted by +10
        }

        using var input = Tensor.From(x, [4, 2, 3], device);
        var y = bn.Forward(input).ToArray();
        for (int c = 0; c < 2; c++)
        {
            var channel = Enumerable.Range(0, 4).SelectMany(n => Enumerable.Range(0, 3).Select(i => y[(n * 2 + c) * 3 + i])).ToArray();
            AssertClose([0f], [channel.Average()], 1e-4f, $"channel {c} mean");
            AssertClose([1f], [channel.Select(v => v * v).Average()], 1e-3f, $"channel {c} variance");
        }

        var running = bn.RunningMean.ToArray();
        Check(running[1] > 4f && running[0] < 2f, $"running means {running[0]}, {running[1]}");
        bn.Eval();
        using var eval = bn.Forward(input);
        Check(!eval.ToArray().SequenceEqual(y), "evaluation mode uses running statistics");
    }

    private static void EmbeddingBehaviour(Device device)
    {
        var embedding = new Embedding(5, 3, device: device, random: new Random(33));
        using var ids = Tensor.From(new float[,] { { 0, 4 }, { 4, 2 } }, device);
        var table = embedding.Weight.ToArray();
        var y = embedding.Forward(ids);
        Check(y.Shape.SequenceEqual([2, 2, 3]), "embedding shape");
        AssertClose([.. table[12..15], .. table[6..9]], y.ToArray()[6..12], 0, "embedding rows");
        embedding.Weight.ZeroGrad();
        (y * 2f).Sum().Backward();
        var grad = embedding.Weight.Grad!.ToArray();
        AssertClose([2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 4, 4, 4], grad, 0, "embedding gradient accumulates repeated ids");
    }

    private static void ConvReference(Device device)
    {
        var random = new Random(34);
        const int N = 2, C = 3, H = 6, W = 5, OC = 4, K = 3, S = 2, P = 1;
        var conv = new Conv2d(C, OC, K, stride: S, padding: P, device: device, random: random);
        using var biasValues = Tensor.From(RandomArray(random, OC), device);
        conv.Bias!.Load(biasValues.ToArray());
        var x = RandomArray(random, N * C * H * W);
        using var input = Tensor.From(x, [N, C, H, W], device);
        var y = conv.Forward(input);
        int oh = (H + 2 * P - K) / S + 1, ow = (W + 2 * P - K) / S + 1;
        Check(y.Shape.SequenceEqual([N, OC, oh, ow]), $"conv shape {Tensor.FormatShape(y.Shape)}");
        var w = conv.Weight.ToArray();
        var b = conv.Bias.ToArray();
        var actual = y.ToArray();
        for (int n = 0; n < N; n++)
        {
            for (int o = 0; o < OC; o++)
            {
                for (int i = 0; i < oh; i++)
                {
                    for (int j = 0; j < ow; j++)
                    {
                        double acc = b[o];
                        for (int c = 0; c < C; c++)
                        {
                            for (int ki = 0; ki < K; ki++)
                            {
                                for (int kj = 0; kj < K; kj++)
                                {
                                    int r = i * S - P + ki, q = j * S - P + kj;
                                    if (r >= 0 && r < H && q >= 0 && q < W)
                                    {
                                        acc += w[o * C * K * K + (c * K + ki) * K + kj] * x[((n * C + c) * H + r) * W + q];
                                    }
                                }
                            }
                        }

                        AssertClose([(float)acc], [actual[((n * OC + o) * oh + i) * ow + j]], 1e-4f, "conv value");
                    }
                }
            }
        }
    }

    private static void MaxPoolBehaviour(Device device)
    {
        var pool = new MaxPool2d(2);
        using var x = Tensor.From(new float[] { 1, 5, 2, 0, 3, 4, 8, 1, 9, 2, 6, 7, 0, 1, 3, 2 }, [1, 1, 4, 4], device, requiresGrad: true);
        var y = pool.Forward(x);
        AssertClose([5, 8, 9, 7], y.ToArray(), 0, "maxpool values");
        (y * Tensor.From([1f, 2f, 3f, 4f], [1, 1, 2, 2], device)).Sum().Backward();
        AssertClose([0, 1, 0, 0, 0, 0, 2, 0, 3, 0, 0, 4, 0, 0, 0, 0], x.Grad!.ToArray(), 0, "maxpool gradient");
        // Max is not differentiable at ties, so spread the inputs apart with a fixed shuffled offset per
        // element (spacing 0.5 > random range 0.2): every window then has a unique, stable maximum.
        Tensor Spread(Tensor t)
        {
            var offsets = Enumerable.Range(0, t.Size).Select(i => 0.5f * i).ToArray();
            new Random(38).Shuffle(offsets);
            return t + Tensor.From(offsets, t.Shape, device);
        }

        using var weights = Tensor.From(RandomArray(new Random(39), 24), [2, 2, 2, 3], device);
        GradCheck(device, [2, 2, 4, 6], t => (new MaxPool2d(2, 2).Forward(Spread(t)) * weights).Sum(), scale: 0.1f);
        GradCheck(device, [1, 2, 5, 5], t => new MaxPool2d(3, 2, padding: 1).Forward(Spread(t)).Sum(), scale: 0.1f);
    }

    private static void CausalMasking(Device device)
    {
        var attention = new MultiHeadAttention(4, 2, causal: true, device: device, random: new Random(35));
        var a = RandomArray(new Random(36), 12);
        var b = (float[])a.Clone();
        for (int i = 8; i < 12; i++)
        {
            b[i] += 5f;   // change only the last position
        }

        using (Autograd.NoGrad())
        {
            var ya = attention.Forward(Tensor.From(a, [1, 3, 4], device)).ToArray();
            var yb = attention.Forward(Tensor.From(b, [1, 3, 4], device)).ToArray();
            AssertClose(ya[..8], yb[..8], 1e-5f, "earlier positions must not see the future");
            Check(!ya[8..].SequenceEqual(yb[8..]), "the last position sees its own input");
        }
    }

    private static Dataset Spirals(int perClass, int classes, int seed)
    {
        var random = new Random(seed);
        var features = new float[perClass * classes, 2];
        var labels = new int[perClass * classes];
        for (int c = 0; c < classes; c++)
        {
            for (int i = 0; i < perClass; i++)
            {
                int row = c * perClass + i;
                double r = i / (double)perClass;
                double angle = c * 2 * Math.PI / classes + r * 4 + random.NextDouble() * 0.2;
                features[row, 0] = (float)(r * Math.Cos(angle));
                features[row, 1] = (float)(r * Math.Sin(angle));
                labels[row] = c;
            }
        }

        return Dataset.FromClassLabels(features, labels, classes);
    }

    private static void SpiralClassification(Device device)
    {
        var data = Spirals(100, 3, 1);
        var random = new Random(2);
        using var model = new Sequential
        {
            new Linear(2, 64, device: device, random: random), new BatchNorm(64, device: device), new ReLU(),
            new Linear(64, 64, device: device, random: random), new ReLU(),
            new Linear(64, 3, device: device, random: random),
        };
        using var optimizer = new AdamW(model.Parameters(), 0.02f, weightDecay: 1e-4f);
        var trainer = new Trainer(model, optimizer, (p, t) => Losses.CrossEntropy(p, t))
        {
            Metrics = { Metric.Accuracy },
            Scheduler = new CosineAnnealing(optimizer, totalEpochs: 150, warmupEpochs: 5),
        };
        trainer.Fit(new DataLoader(data, 32, shuffle: true, device: device, seed: 3), epochs: 150);
        var result = trainer.Evaluate(new DataLoader(data, 300, device: device));
        Check(result.Metrics["accuracy"] > 0.95, $"spiral accuracy {result.Metrics["accuracy"]:P1}");
    }

    private static void ConvolutionLearns(Device device)
    {
        // 8x8 images with a horizontal (class 0) or vertical (class 1) bright line at a random position, plus noise.
        var random = new Random(4);
        const int Count = 200;
        var features = new float[Count, 64];
        var labels = new int[Count];
        for (int s = 0; s < Count; s++)
        {
            labels[s] = s % 2;
            int line = random.Next(8);
            for (int i = 0; i < 64; i++)
            {
                int row = i / 8, col = i % 8;
                features[s, i] = (labels[s] == 0 ? row == line : col == line) ? 1f : 0f;
                features[s, i] += (random.NextSingle() - 0.5f) * 0.3f;
            }
        }

        var data = Dataset.FromClassLabels(features, labels, 2).WithFeatureShape(1, 8, 8);
        var r = new Random(5);
        using var model = new Sequential
        {
            new Conv2d(1, 8, 3, padding: 1, device: device, random: r), new BatchNorm(8, device: device), new ReLU(),
            new MaxPool2d(2),
            new Conv2d(8, 8, 3, padding: 1, device: device, random: r), new ReLU(),
            new GlobalAveragePool2d(),
            new Linear(8, 2, device: device, random: r),
        };
        using var optimizer = new Adam(model.Parameters(), 0.01f);
        var trainer = new Trainer(model, optimizer, (p, t) => Losses.CrossEntropy(p, t)) { Metrics = { Metric.Accuracy } };
        trainer.Fit(new DataLoader(data, 32, shuffle: true, device: device, seed: 6), epochs: 25);
        var accuracy = trainer.Evaluate(new DataLoader(data, 200, device: device)).Metrics["accuracy"];
        Check(accuracy > 0.95, $"CNN accuracy {accuracy:P1}");
    }

    private static void SequenceModelsLearn(Device device)
    {
        // Sequences of 8 tokens from a vocabulary of 6; the class is whether token 5 appears more often than token 4.
        var random = new Random(7);
        const int Samples = 400, Length = 8, Vocabulary = 6;
        var features = new float[Samples, Length];
        var labels = new int[Samples];
        for (int s = 0; s < Samples; s++)
        {
            for (int t = 0; t < Length; t++)
            {
                features[s, t] = random.Next(Vocabulary);
            }

            int Count(int token) => Enumerable.Range(0, Length).Count(t => features[s, t] == token);
            if (Count(4) == Count(5))
            {
                features[s, random.Next(Length)] = random.Next(2) == 0 ? 4 : 5;   // break the tie
            }

            labels[s] = Count(5) > Count(4) ? 1 : 0;
        }

        var (train, test) = Dataset.FromClassLabels(features, labels, 2).Split(0.8, seed: 8);
        var models = new (string Name, Func<Random, Module> Create)[]
        {
            ("LSTM", r => new Sequential
            {
                new Embedding(Vocabulary, 16, device, r),
                new LSTM(16, 32, device: device, random: r),
                new Linear(32, 2, device: device, random: r),
            }),
            ("Transformer", r => new Sequential
            {
                new Embedding(Vocabulary, 16, device, r),
                new PositionalEncoding(Length, 16, device),
                new TransformerEncoderLayer(16, 2, ffDim: 32, dropout: 0f, device: device, random: r),
                new Lambda(x => x.Mean(1), "MeanOverTime"),
                new Linear(16, 2, device: device, random: r),
            }),
        };

        foreach (var (name, create) in models)
        {
            using var model = create(new Random(9));
            using var optimizer = new Adam(model.Parameters(), 0.01f);
            var trainer = new Trainer(model, optimizer, (p, t) => Losses.CrossEntropy(p, t)) { Metrics = { Metric.Accuracy }, MaxGradientNorm = 5f };
            trainer.Fit(new DataLoader(train, 32, shuffle: true, device: device, seed: 10), epochs: 40);
            var accuracy = trainer.Evaluate(new DataLoader(test, 100, device: device)).Metrics["accuracy"];
            Check(accuracy > 0.85, $"{name} test accuracy {accuracy:P1}");
        }
    }

    private static void OptimizerFeatures(Device device)
    {
        using var p = Tensor.From([1f, 2f], device, requiresGrad: true);
        using var sgd = new Sgd([p], 0.1f);
        var step = new StepDecay(sgd, stepSize: 2, gamma: 0.5f);
        var rates = new List<float> { sgd.LearningRate };
        for (int i = 0; i < 4; i++)
        {
            step.Step();
            rates.Add(sgd.LearningRate);
        }

        AssertClose([0.1f, 0.1f, 0.05f, 0.05f, 0.025f], [.. rates], 1e-6f, "step decay");
        using var adam = new Adam([p], 1f);
        var cosine = new CosineAnnealing(adam, totalEpochs: 10, minLearningRate: 0f, warmupEpochs: 2);
        Check(adam.LearningRate < 0.5f, "warm-up starts low");
        for (int i = 0; i < 10; i++)
        {
            cosine.Step();
        }

        AssertClose([0f], [adam.LearningRate], 1e-6f, "cosine ends at the minimum");

        // AdamW with zero gradients only decays the weights.
        using var w = Tensor.From([10f, -10f], device, requiresGrad: true);
        using var adamW = new AdamW([w], learningRate: 0.1f, weightDecay: 0.5f);
        (w * 0f).Sum().Backward();
        adamW.Step();
        AssertClose([9.5f, -9.5f], w.ToArray(), 1e-5f, "decoupled weight decay");

        using var q = Tensor.From([3f, 4f], device, requiresGrad: true);
        using var clipper = new Sgd([q], 0f);
        (q * q).Sum().Backward();   // gradient (6, 8), norm 10
        double norm = clipper.ClipGradientNorm(1f);
        AssertClose([10f], [(float)norm], 1e-5f, "norm before clipping");
        AssertClose([0.6f, 0.8f], q.Grad!.ToArray(), 1e-5f, "clipped gradient");

        // The same on the device (no host read of the norm), across several tensors; unchanged within the limit.
        using var r1 = Tensor.From([3f, 0f], device, requiresGrad: true);
        using var r2 = Tensor.From([4f], device, requiresGrad: true);
        using var onDevice = new Sgd([r1, r2], 0f);
        ((r1 * r1).Sum() + (r2 * r2).Sum()).Backward();   // gradients (6, 0) and (8): norm 10
        onDevice.ClipGradientNormOnDevice(1f);
        AssertClose([0.6f, 0f], r1.Grad!.ToArray(), 1e-5f, "clipped on the device (first tensor)");
        AssertClose([0.8f], r2.Grad!.ToArray(), 1e-5f, "clipped on the device (second tensor)");
        onDevice.ClipGradientNormOnDevice(5f);
        AssertClose([0.6f, 0f], r1.Grad!.ToArray(), 1e-5f, "within the limit: unchanged");
        r1.ZeroGrad();
        r2.ZeroGrad();
        onDevice.ClipGradientNormOnDevice(1f);
        AssertClose([0f], r2.Grad!.ToArray(), 0f, "zero gradients stay zero");

        // ClipAndStep (every tensor in a few fused passes where the device has them) matches clipping then stepping, and
        // leaves the gradients zeroed; sizes across chunk boundaries, with and without clipping and weight decay.
        var rng = new Random(33);
        int[] sizes = [5, 4096, 4097, 9000, 1];
        var initial = sizes.Select(n => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray()).ToArray();
        var targets = sizes.Select(n => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray()).ToArray();
        foreach (var (maxNorm, decay) in new[] { (0.5f, 0.1f), (0f, 0f), (1e6f, 0.01f) })
        {
            float[][] Train(bool fused)
            {
                var parameters = initial.Select((v, i) => Tensor.From(v, [sizes[i]], device, requiresGrad: true)).ToList();
                using var optimizer = new AdamW(parameters, learningRate: 0.05f, weightDecay: decay);
                for (int step = 0; step < 3; step++)
                {
                    optimizer.ZeroGrad();
                    using (var scope = new TensorScope())
                    {
                        var loss = parameters.Select((p, i) => ((p - Tensor.From(targets[i], [sizes[i]], device)).Square().Sum())).Aggregate((a, b) => a + b);
                        loss.Backward();
                    }

                    if (fused)
                    {
                        optimizer.ClipAndStep(maxNorm);
                    }
                    else
                    {
                        if (maxNorm > 0f)
                        {
                            optimizer.ClipGradientNormOnDevice(maxNorm);
                        }

                        optimizer.Step();
                    }
                }

                if (fused)
                {
                    Check(parameters.All(p => p.Grad is null || p.Grad.ToArray().All(g => g == 0f)), "ClipAndStep leaves the gradients zeroed");
                }

                var values = parameters.Select(p => p.ToArray()).ToArray();
                parameters.ForEach(p => p.Dispose());
                return values;
            }

            var separate = Train(fused: false);
            var together = Train(fused: true);
            for (int i = 0; i < sizes.Length; i++)
            {
                AssertClose(separate[i], together[i], 1e-5f, $"ClipAndStep (max norm {maxNorm}, decay {decay}), tensor {i} of {sizes[i]}");
            }
        }
    }

    private static void NumberTypes(Device device)
    {
        using var fromDouble = Tensor.From<double>([1.5, 2.5, -3.25], [3], device);
        AssertClose([1.5f, 2.5f, -3.25f], fromDouble.ToArray(), 0, "double in");
        using var fromInt = Tensor.From(new int[,] { { 1, 2 }, { 3, 4 } }, device);
        Check(fromInt.Shape.SequenceEqual([2, 2]), "int[,] in");
        Check(fromInt.ToArray<int>().SequenceEqual([1, 2, 3, 4]), "int out");
        using var fromHalf = Tensor.From<Half>([(Half)0.5f, (Half)(-2f)], [2], device);
        Check(fromHalf.ToArray<double>().SequenceEqual([0.5, -2.0]), "Half in, double out");
        using var big = Tensor.From([300f, -5f, 1.9f], device);
        Check(big.ToArray<byte>().SequenceEqual(new byte[] { 255, 0, 1 }), "byte out saturates and truncates");
    }

    private static void SaveLoadBuffers(Device device)
    {
        string path = Path.GetTempFileName();
        try
        {
            using var a = new Sequential { new Linear(3, 4, device: device), new BatchNorm(4, device: device) };
            using (var x = Tensor.From(RandomArray(new Random(37), 30, 3f), [10, 3], device))
            {
                a.Forward(x);   // updates running statistics
            }

            a.Save(path);
            using var b = new Sequential { new Linear(3, 4, device: device), new BatchNorm(4, device: device) };
            b.Load(path);
            foreach (var (ta, tb) in a.Parameters().Concat(a.Buffers()).Zip(b.Parameters().Concat(b.Buffers())))
            {
                AssertClose(ta.ToArray(), tb.ToArray(), 0, "saved tensor");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
