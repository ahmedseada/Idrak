// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Benches;

/// <summary>
/// Element-wise operations, reductions, softmax and layer norm at three sizes: microseconds a call (the warm median of
/// rounds of Settings.Calls calls queued back to back with one device sync, so the sync and the driver's submission
/// latency are shared, not charged to each 10 µs kernel), the bandwidth that implies (bytes read and written over the
/// time), the launches each takes, forward and with its backward step. A fused operation shows as fewer launches and a
/// higher bandwidth than the chain it replaced.
/// </summary>
internal static class OpsBench
{
    private const string Section = "ops";

    public static void Run(Device device, Settings settings, Report report)
    {
        int[] sizes = settings.Size switch
        {
            BenchSize.Small => [1 << 14, 1 << 18],
            BenchSize.Medium => [1 << 16, 1 << 20, 1 << 23],
            _ => [1 << 18, 1 << 22, 1 << 25],
        };
        var random = new Random(7);
        Console.WriteLine(Bench.Line("operation [elements]", ("µs warm", ""), ("GB/s", ""), ("launches", ""), ("µs fwd+bwd", ""), ("launches", "")));
        foreach (int n in sizes)
        {
            using var x = Bench.RandomTensor(random, [n], device, 2f);
            using var y = Bench.RandomTensor(random, [n], device, 2f);
            int rows = Math.Max(1, n / 1024);
            using var matrix = Bench.RandomTensor(random, [rows, 1024], device, 2f);
            using var norm = new LayerNorm(1024, device: device);
            (string Name, Func<Tensor, Tensor> Op, Tensor Input, long Bytes)[] cases =
            [
                ("exp", t => t.Exp(), x, 8L * n),
                ("tanh", t => t.Tanh(), x, 8L * n),
                ("gelu (tanh form)", t => t.Gelu(), x, 8L * n),
                ("gelu (erf)", t => t.GeluErf(), x, 8L * n),
                ("leaky relu 0.1", t => t.LeakyRelu(0.1f), x, 8L * n),
                ("add", t => t + y, x, 12L * n),
                ("mul", t => t * y, x, 12L * n),
                ("x * 2 + 1 (affine)", t => t * 2f + 1f, x, 8L * n),
                ("square then sum", t => t.Square().Sum(), x, 4L * n),
                ("sum over rows [rows, 1024]", t => t.Sum(1), matrix, 4L * n),
                ("softmax [rows, 1024]", t => t.Softmax(), matrix, 8L * n),
                ("layer norm [rows, 1024]", t => norm.Forward(t), matrix, 8L * n),
                ("weight norm [rows, 1024]", t => Tensor.WeightNorm(WeightNormGain(t), t), matrix, 8L * n),
            ];
            int calls = settings.Calls;
            foreach (var (name, op, input, bytes) in cases)
            {
                string label = $"{name} [{n:N0}]";
                void Forward()
                {
                    for (int c = 0; c < calls; c++)
                    {
                        using var result = op(input);
                    }
                }

                var (_, warmRound) = Bench.Time(device, Forward, settings.Repeats, settings.Warmup);
                double warm = warmRound / calls;
                var census = Bench.Count(() => op(input));
                double warmUs = warm * 1000, gbs = bytes / (warm / 1000) / 1e9;

                // The same with gradients recorded and the backward step run.
                using var tracked = Bench.RandomTensor(random, input.Shape, device, 2f, requiresGrad: true);
                void Both()
                {
                    for (int c = 0; c < calls; c++)
                    {
                        // The result freed with its loss each call, so the calls reuse one block instead of each taking a new one.
                        using var result = op(tracked);
                        using var loss = result.Sum();
                        loss.Backward();
                        tracked.ReleaseGrad();
                    }
                }

                var (_, warmBothRound) = Bench.Time(device, Both, settings.Repeats, settings.Warmup);
                double warmBoth = warmBothRound / calls;
                var bothCensus = Bench.Count(Both);
                long bothLaunches = bothCensus.Count / calls;
                Console.WriteLine(Bench.Line(label, ("", Bench.F(warmUs, "0.0")), ("", Bench.F(gbs, "0.0")), ("", census.Count.ToString()),
                    ("", Bench.F(warmBoth * 1000, "0.0")), ("", bothLaunches.ToString())));
                report.Add(Section, label, device, "warm", warmUs, "µs");
                report.Add(Section, label, device, "bandwidth", gbs, "GB/s");
                report.Add(Section, label, device, "launches", census.Count, "launches");
                report.Add(Section, label, device, "fwd+bwd warm", warmBoth * 1000, "µs");
                report.Add(Section, label, device, "fwd+bwd launches", bothLaunches, "launches");
            }
        }
    }

    private static Tensor WeightNormGain(Tensor direction) => Tensor.Ones([direction.Shape[0]], direction.Device);

    /// <summary>Matrix products and convolutions at the shapes training and decoding run: TFLOPS and milliseconds a call.</summary>
    public static void RunProducts(Device device, Settings settings, Report report)
    {
        const string ProductsSection = "matmul";
        (int M, int K, int N)[] shapes = settings.Size switch
        {
            BenchSize.Small => [(256, 256, 256), (1, 1024, 4096), (512, 1024, 1024)],
            BenchSize.Medium => [(1024, 1024, 1024), (1, 2048, 8192), (180, 2048, 8192), (4096, 1024, 4096)],
            _ => [(4096, 4096, 4096), (1, 4096, 32000), (512, 4096, 16384), (8192, 2048, 8192)],
        };
        var random = new Random(11);
        Console.WriteLine(Bench.Line("product m x k x n", ("ms warm", ""), ("TFLOPS", ""), ("ms cold", "")));
        foreach (var (m, k, n) in shapes)
        {
            using var a = Bench.RandomTensor(random, [m, k], device, 0.5f);
            using var b = Bench.RandomTensor(random, [k, n], device, 0.5f);
            var (cold, warm) = Bench.Time(device, () => a.MatMul(b), settings.Repeats, settings.Warmup);
            double tflops = 2.0 * m * k * n / (warm / 1000) / 1e12;
            string label = $"{m} x {k} x {n}";
            Console.WriteLine(Bench.Line(label, ("", Bench.F(warm, "0.000")), ("", Bench.F(tflops, "0.00")), ("", Bench.F(cold, "0.0"))));
            report.Add(ProductsSection, label, device, "warm", warm, "ms");
            report.Add(ProductsSection, label, device, "rate", tflops, "TFLOPS");
        }

        // Convolutions as networks run them: a backbone's 3 x 3, a stem's strided 7 x 7, and a speech decoder's long 1-D filters.
        (string Name, int Batch, int In, int Out, int H, int W, int Kernel, int Stride, int Padding)[] convolutions = settings.Size switch
        {
            BenchSize.Small => [("3x3 32->32 on 32x32", 4, 32, 32, 32, 32, 3, 1, 1), ("1-D k11 32->32 on 1x4096", 1, 32, 32, 1, 4096, 11, 1, 5)],
            BenchSize.Medium => [("3x3 64->64 on 56x56", 8, 64, 64, 56, 56, 3, 1, 1), ("7x7 s2 3->64 on 224x224", 8, 3, 64, 224, 224, 7, 2, 3), ("1-D k11 128->128 on 1x8192", 2, 128, 128, 1, 8192, 11, 1, 5)],
            _ => [("3x3 128->128 on 56x56", 32, 128, 128, 56, 56, 3, 1, 1), ("7x7 s2 3->64 on 224x224", 32, 3, 64, 224, 224, 7, 2, 3), ("1-D k11 512->512 on 1x8192", 4, 512, 512, 1, 8192, 11, 1, 5)],
        };
        Console.WriteLine(Bench.Line("convolution (forward, then with both gradients)", ("ms warm", ""), ("TFLOPS", ""), ("ms fwd+bwd", ""), ("launches", "")));
        foreach (var (name, batch, inChannels, outChannels, h, w, kernel, stride, padding) in convolutions)
        {
            using var conv = h == 1
                ? new Conv2d(inChannels, outChannels, (1, kernel), (1, stride), (0, padding), device: device, random: random)
                : new Conv2d(inChannels, outChannels, kernel, stride, padding, device: device, random: random);
            using var input = Bench.RandomTensor(random, [batch, inChannels, h, w], device, 1f, requiresGrad: true);
            var (_, warm) = Bench.Time(device, () =>
            {
                using var _ = Autograd.NoGrad();
                conv.Forward(input);
            }, settings.Repeats, settings.Warmup);
            int oh = (h + 2 * (h == 1 ? 0 : padding) - (h == 1 ? 1 : kernel)) / (h == 1 ? 1 : stride) + 1, ow = (w + 2 * padding - kernel) / stride + 1;
            double flops = 2.0 * batch * outChannels * oh * ow * inChannels * (h == 1 ? kernel : kernel * kernel);
            void Both()
            {
                using var loss = conv.Forward(input).Sum();
                loss.Backward();
                foreach (var p in conv.Parameters())
                {
                    p.ReleaseGrad();
                }

                input.ReleaseGrad();
            }

            var (_, warmBoth) = Bench.Time(device, Both, settings.Repeats, settings.Warmup);
            var census = Bench.Count(Both);
            Console.WriteLine(Bench.Line(name, ("", Bench.F(warm, "0.000")), ("", Bench.F(flops / (warm / 1000) / 1e12, "0.00")), ("", Bench.F(warmBoth, "0.000")), ("", census.Count.ToString())));
            report.Add(ProductsSection, "conv " + name, device, "warm", warm, "ms");
            report.Add(ProductsSection, "conv " + name, device, "fwd+bwd warm", warmBoth, "ms");
            report.Add(ProductsSection, "conv " + name, device, "fwd+bwd launches", census.Count, "launches");
        }
    }
}
