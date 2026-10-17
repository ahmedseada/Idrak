// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Benches;

/// <summary>
/// A decoder built from a spec with random weights: its training step (the token cross-entropy over random tokens, the
/// backward pass, an AdamW step) and its decoding token by token through a key/value cache. Milliseconds a step, launches
/// a step, peak memory, and the step with activation checkpointing (what memory it saves at what cost).
/// </summary>
internal static class DecoderBench
{
    private static DecoderSpec Spec(BenchSize size) => size switch
    {
        BenchSize.Small => new() { Vocabulary = 4096, Dim = 256, Layers = 4, Heads = 4, KvHeads = 2, HeadDim = 64, FfDim = 1024, MaxPositions = 2048 },
        BenchSize.Medium => new() { Vocabulary = 32000, Dim = 1024, Layers = 12, Heads = 16, KvHeads = 8, HeadDim = 64, FfDim = 4096, MaxPositions = 4096 },
        _ => new() { Vocabulary = 32000, Dim = 2048, Layers = 24, Heads = 16, KvHeads = 8, HeadDim = 128, FfDim = 8192, MaxPositions = 4096 },
    };

    private static (int Batch, int Tokens) TrainingShape(BenchSize size) => size switch
    {
        BenchSize.Small => (4, 128),
        BenchSize.Medium => (8, 512),
        _ => (8, 1024),
    };

    public static void RunTraining(Device device, Settings settings, Report report)
    {
        const string Section = "train-decoder";
        var spec = Spec(settings.Size);
        var (batch, tokens) = TrainingShape(settings.Size);
        var random = new Random(3);
        Sequential model;
        try
        {
            model = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device });
        }
        catch (ResourceLimitExceededException e)
        {
            Console.WriteLine($"   skipped: the {settings.Size.ToString().ToLowerInvariant()} model does not fit {device} ({e.Message}); IDRAK_BENCH_SIZE names a smaller one");
            return;
        }

        using var disposeModel = model;
        model.Train();
        long parameters = model.Parameters().Sum(p => (long)p.Size);
        using var optimizer = new AdamW(model.Parameters(), 1e-4f);
        int layers = model.Count - 1;
        var head = model[^1];

        // The batch the size names, halved until the device holds a step (the row says which), as a trainer does.
        while (true)
        {
            Console.WriteLine($"   {parameters / 1e6:F1} M parameters, batch {batch} x {tokens} tokens");
            float[] ids = [.. Enumerable.Range(0, batch * tokens).Select(_ => (float)random.Next(spec.Vocabulary))];
            float[] next = [.. Enumerable.Range(0, batch * tokens).Select(_ => (float)random.Next(spec.Vocabulary))];
            using var input = Tensor.From(ids, [batch, tokens], device);
            using var targets = Tensor.From(next, [batch * tokens], device);
            using var weights = Tensor.Ones([batch * tokens], device);
            int rows = batch * tokens;

            void Step(bool checkpointing)
            {
                optimizer.ZeroGrad();
                var hidden = checkpointing ? Checkpointed(model, input, layers) : model.ForwardFirst(input, layers);
                var flat = hidden.Reshape(rows, spec.Dim);
                using var loss = Losses.TokenCrossEntropy(flat, h => head.Forward(h), targets, weights, 1f / rows);
                loss.Backward();
                optimizer.ClipAndStep(1f);
            }

            try
            {
                foreach (bool checkpointing in (bool[])[false, true])
                {
                    string label = checkpointing ? $"step, checkpointed blocks, batch {batch}" : $"step, batch {batch}";
                    var (cold, warm) = Bench.Time(device, () => Step(checkpointing), settings.Repeats, settings.Warmup);
                    var census = Bench.Count(() => Step(checkpointing));
                    long peak = Bench.PeakBytes(device, () => Step(checkpointing));
                    double tokensPerSecond = rows / (warm / 1000);
                    Console.WriteLine(Bench.Line(label, ("ms warm", Bench.F(warm, "0.0")), ("ms cold", Bench.F(cold, "0")), ("launches", census.Count.ToString()),
                        ("peak", Bench.Mib(peak)), ("tok/s", Bench.F(tokensPerSecond, "0"))));
                    report.Add(Section, label, device, "warm", warm, "ms");
                    report.Add(Section, label, device, "cold", cold, "ms");
                    report.Add(Section, label, device, "launches", census.Count, "launches");
                    report.Add(Section, label, device, "peak", peak / 1048576.0, "MiB");
                    report.Add(Section, label, device, "throughput", tokensPerSecond, "tok/s");
                    if (!checkpointing)
                    {
                        Console.WriteLine("   launches a step by operation (forward and backward apart, the top twelve):");
                        foreach (var (operation, count, elements, _) in census.ByCount.Take(12))
                        {
                            Console.WriteLine(Bench.Line("   " + operation, ("/step", count.ToString()), ("MiB out", Bench.F(elements * 4.0 / 1048576, "0"))));
                        }
                    }
                }

                return;
            }
            catch (ResourceLimitExceededException e) when (batch > 1)
            {
                optimizer.ZeroGrad();
                ComputeResources.ReleaseCachedMemory(device);
                Console.WriteLine($"   batch {batch} x {tokens} does not fit {device}: {FirstClause(e.Message)}; the batch halves");
                batch /= 2;
            }
        }
    }

    // The message up to its first colon or full stop: the device and "is out of memory", not the whole remedy.
    private static string FirstClause(string message)
    {
        int cut = message.IndexOfAny([':', '.']);
        return cut > 0 ? message[..cut] : message;
    }

    // Each block checkpointed: only its output kept, the block run again in the backward pass.
    private static Tensor Checkpointed(Sequential model, Tensor input, int layers)
    {
        var x = model[0].Forward(input);
        for (int i = 1; i < layers; i++)
        {
            x = model[i].ForwardCheckpointed(x);
        }

        return x;
    }

    public static void RunDecoding(Device device, Settings settings, Report report)
    {
        const string Section = "infer-decoder";
        var spec = Spec(settings.Size);
        int prompt = settings.Size == BenchSize.Small ? 64 : 256, generated = settings.Size == BenchSize.Small ? 64 : 128;
        var random = new Random(5);
        using var model = spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device });
        model.Eval();
        float[] ids = [.. Enumerable.Range(0, prompt).Select(_ => (float)random.Next(spec.Vocabulary))];
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            using var context = new DecodingContext(device, 1, prompt + generated + 1, format);
            double promptMs;
            using (Autograd.NoGrad())
            using (new TensorScope())
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                using var logits = model.ForwardCached(Tensor.From(ids, [1, prompt], device), context);
                device.Backend.Synchronize();
                promptMs = watch.Elapsed.TotalMilliseconds;
            }

            long launches = 0;
            var watchAll = System.Diagnostics.Stopwatch.StartNew();
            using (Autograd.NoGrad())
            {
                for (int t = 0; t < generated; t++)
                {
                    float id = random.Next(spec.Vocabulary);
                    if (t == generated - 1)
                    {
                        launches = Bench.Count(() => model.ForwardCached(Tensor.From([id], [1, 1], device), context)).Count;
                    }
                    else
                    {
                        using var scope = new TensorScope();
                        model.ForwardCached(Tensor.From([id], [1, 1], device), context);
                    }
                }

                device.Backend.Synchronize();
            }

            double tokensPerSecond = generated / watchAll.Elapsed.TotalSeconds;
            long cacheBytes = ComputeResources.GetMemoryUsage(device).InUse;
            string label = $"cache {format}, prompt {prompt}, {generated} tokens";
            Console.WriteLine(Bench.Line(label, ("tok/s", Bench.F(tokensPerSecond, "0.0")), ("launches/token", launches.ToString()), ("ms prompt", Bench.F(promptMs, "0.0")), ("in use", Bench.Mib(cacheBytes))));
            report.Add(Section, label, device, "decoding", tokensPerSecond, "tok/s");
            report.Add(Section, label, device, "launches a token", launches, "launches");
            report.Add(Section, label, device, "prompt", promptMs, "ms");
        }
    }

    /// <summary>Deterministic random weights for a spec: norms near one, the rest small.</summary>
    private sealed class RandomWeights(int seed) : IWeightSource
    {
        private readonly Random _random = new(seed);

        public float[] Read(string name, IReadOnlyList<int> shape)
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            bool gain = name.EndsWith("norm.weight", StringComparison.Ordinal);
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = gain ? 0.8f + 0.4f * _random.NextSingle() : (_random.NextSingle() * 2 - 1) * 0.05f;
            }

            return values;
        }
    }
}
