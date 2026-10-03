// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak demo xor|spirals|shapes|gpt</c>: trains one of the samples (samples/Idrak.Samples.Xor, Spirals,
/// ShapeRecognition, GptTraining) at a small size in seconds on the chosen device, and prints the device, the result
/// and the speed, to show the library works there.
/// </summary>
internal sealed class DemoCommand : Command
{
    /// <summary>The demos, with what each trains.</summary>
    public static readonly IReadOnlyDictionary<string, string> Demos = new Dictionary<string, string>
    {
        ["xor"] = "a 2-8-1 network learns XOR (full batch, 600 steps)",
        ["spirals"] = "an MLP separates three interleaved spirals (600 points, 40 epochs)",
        ["shapes"] = "a small CNN tells circles, squares, triangles and crosses apart (16 x 16 images, 4 epochs)",
        ["gpt"] = "a one-layer character GPT learns a sentence and writes it back (200 steps)",
    };

    public override string Name => "demo";

    public override string Summary => "Trains a small built-in sample in seconds on the device, with the speed";

    public override string Usage =>
        "xor|spirals|shapes|gpt [options]\n\n" +
        string.Concat(Demos.Select(d => $"  {d.Key,-8} {d.Value}\n")) + "\n" +
        "Exits with 1 when the demo does not reach its usual result (a sign something is wrong on this device).\n\n" +
        "Examples:\n" +
        "  idrak demo xor\n" +
        "  idrak demo shapes -d vulkan:0\n" +
        "  idrak demo gpt -d cuda:0 -j\n\n" +
        "Environment: IDRAK_DISABLE_CUDA, IDRAK_DISABLE_VULKAN, IDRAK_DISABLE_HIP (which backends are tried), IDRAK_MATMUL\n" +
        "(product precision), IDRAK_AUTOTUNE and the tuning caches (kernel choices).";

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, $"the demo ({string.Join(", ", Demos.Keys)})").ToLowerInvariant();
        if (!Demos.ContainsKey(name))
        {
            throw new UsageException($"Unknown demo '{name}'; choose one of {string.Join(", ", Demos.Keys)}.");
        }

        var device = context.Device;
        context.Write($"Device: {device} - {device.Name}");
        context.Write($"Demo:   {Demos[name]}");
        var watch = Stopwatch.StartNew();
        var result = name switch
        {
            "xor" => Xor(device),
            "spirals" => Spirals(device),
            "shapes" => Shapes(device),
            _ => Gpt(device),
        };
        double seconds = watch.Elapsed.TotalSeconds;
        double rate = result.Samples / Math.Max(seconds, 1e-9);
        context.Write($"Result: {result.Summary}");
        if (result.Sample is { } sample)
        {
            context.Write($"Wrote:  \"{sample}\"");
        }

        context.Write($"Speed:  {seconds:F2} s, {rate:N0} {result.Unit}/s{(result.Passed ? "" : "  (below the usual result)")}");
        context.WriteJson(new JsonObject
        {
            ["demo"] = name,
            ["device"] = device.ToString(),
            ["deviceName"] = device.Name,
            ["metric"] = result.Metric,
            ["value"] = result.Value,
            ["summary"] = result.Summary,
            ["sample"] = result.Sample,
            ["seconds"] = Math.Round(seconds, 3),
            [result.Unit + "PerSecond"] = Math.Round(rate, 1),
            ["passed"] = result.Passed,
        });
        return result.Passed ? ExitCodes.Ok : ExitCodes.Failed;
    }

    /// <summary>What a demo reached: its metric, the samples (or tokens) it processed, and whether that is the usual result.</summary>
    internal sealed record DemoResult(string Metric, double Value, string Summary, long Samples, string Unit, bool Passed, string? Sample = null);

    private static DemoResult Xor(Device device)
    {
        const int Steps = 600;
        using var model = Network.Input(2).Seed(42).Linear(8).Tanh().Linear(1).Sigmoid().OnDevice(device).Build();
        using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);
        using var x = Tensor.From([0f, 0, 0, 1, 1, 0, 1, 1], [4, 2], device);
        using var y = Tensor.From([0f, 1, 1, 0], [4, 1], device);
        float loss = 0f;
        for (int step = 0; step < Steps; step++)
        {
            using var scope = new TensorScope();
            var l = Losses.MeanSquaredError(model.Forward(x), y);
            optimizer.ZeroGrad();
            l.Backward();
            optimizer.Step();
            if (step == Steps - 1)
            {
                loss = l.Item();
            }
        }

        model.Eval();
        using var outputs = model.Predict(x);
        var values = outputs.ToArray();
        int correct = Enumerable.Range(0, 4).Count(i => (values[i] >= 0.5f ? 1 : 0) == (i is 1 or 2 ? 1 : 0));
        return new DemoResult("correct", correct, $"{correct}/4 correct, loss {loss:F4}", 4L * Steps, "samples", correct == 4);
    }

    private static DemoResult Spirals(Device device)
    {
        const int Classes = 3, PerClass = 200, Epochs = 40;
        var random = new Random(1);
        var features = new float[Classes * PerClass, 2];
        var labels = new int[Classes * PerClass];
        for (int c = 0; c < Classes; c++)
        {
            for (int i = 0; i < PerClass; i++)
            {
                int row = c * PerClass + i;
                double radius = i / (double)PerClass;
                double angle = c * 2 * Math.PI / Classes + radius * 5 + random.NextDouble() * 0.25;
                (features[row, 0], features[row, 1], labels[row]) = ((float)(radius * Math.Cos(angle)), (float)(radius * Math.Sin(angle)), c);
            }
        }

        var (train, test) = Dataset.FromClassLabels(features, labels, Classes).Split(0.8, seed: 2);
        using var model = Network.Input(2).Seed(3).Linear(64).ReLU().Linear(64).ReLU().Linear(Classes).OnDevice(device).Build();
        using var optimizer = new AdamW(model.Parameters(), learningRate: 0.01f, weightDecay: 1e-4f);
        var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Metrics = { Metric.Accuracy } };
        trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 4), Epochs);
        double accuracy = trainer.Evaluate(new DataLoader(test, 512, device: device)).Metrics["accuracy"];
        return new DemoResult("accuracy", accuracy, $"test accuracy {accuracy * 100:F1}% on {test.Count} points",
            (long)train.Count * Epochs, "samples", accuracy >= 0.8);
    }

    private static DemoResult Shapes(Device device)
    {
        const int Size = 16, Epochs = 4;
        var train = DrawShapes(1600, Size, seed: 1);
        var test = DrawShapes(400, Size, seed: 2);
        using var model = Network.Image(1, Size, Size).Seed(3)
            .Conv2d(8, 3, padding: 1).ReLU().MaxPool2d(2)
            .Conv2d(16, 3, padding: 1).ReLU().MaxPool2d(2)
            .Flatten().Linear(32).ReLU().Linear(4).OnDevice(device).Build();
        using var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f);
        var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Metrics = { Metric.Accuracy } };
        trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 4), Epochs);
        double accuracy = trainer.Evaluate(new DataLoader(test, 200, device: device)).Metrics["accuracy"];
        return new DemoResult("accuracy", accuracy, $"test accuracy {accuracy * 100:F1}% on {test.Count} images (4 classes)",
            (long)train.Count * Epochs, "images", accuracy >= 0.5);
    }

    // The ShapeRecognition sample's images: outlines of circles, squares, triangles and crosses with noise.
    private static Dataset DrawShapes(int count, int size, int seed)
    {
        var random = new Random(seed);
        var pixels = new float[count, size * size];
        var labels = new int[count];
        for (int s = 0; s < count; s++)
        {
            int shape = labels[s] = s % 4;
            float r = 3 + random.NextSingle() * 3.5f;
            float cx = r + random.NextSingle() * (size - 1 - 2 * r), cy = r + random.NextSingle() * (size - 1 - 2 * r);
            float brightness = 0.6f + random.NextSingle() * 0.4f;
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dx = x - cx, dy = y - cy;
                    bool on = shape switch
                    {
                        0 => MathF.Abs(MathF.Sqrt(dx * dx + dy * dy) - r) < 0.8f,
                        1 => MathF.Abs(MathF.Max(MathF.Abs(dx), MathF.Abs(dy)) - r) < 0.7f,
                        2 => dy <= r && dy >= -r && MathF.Abs(dx) <= (dy + r) / 2 && MathF.Abs(dx) >= (dy + r) / 2 - 1.2f
                             || MathF.Abs(dy - r) < 0.7f && MathF.Abs(dx) <= r,
                        _ => (MathF.Abs(dx) < 0.8f || MathF.Abs(dy) < 0.8f) && MathF.Max(MathF.Abs(dx), MathF.Abs(dy)) <= r,
                    };
                    pixels[s, y * size + x] = (on ? brightness : 0f) + (random.NextSingle() - 0.5f) * 0.2f;
                }
            }
        }

        return Dataset.FromClassLabels(pixels, labels, 4).WithFeatureShape(1, size, size);
    }

    private static DemoResult Gpt(Device device)
    {
        const string Text = "idrak runs on every device. ";
        const int Context = 32, Batch = 16, Steps = 200;
        string corpus = string.Concat(Enumerable.Repeat(Text, 40));
        var tokenizer = new CharTokenizer(new string([.. Text.Distinct().Order()]));
        var ids = tokenizer.Encode(corpus);
        using var model = Architectures.Gpt(tokenizer.VocabularySize, Context, dim: 32, heads: 2, layers: 1, ffDim: 64, dropout: 0f).Seed(5).OnDevice(device).Build();
        using var optimizer = new Adam(model.Parameters(), learningRate: 0.003f);
        var random = new Random(6);
        float first = 0f, last = 0f;
        for (int step = 0; step < Steps; step++)
        {
            var inputs = new float[Batch * Context];
            var targets = new float[Batch * Context];
            for (int b = 0; b < Batch; b++)
            {
                int start = random.Next(ids.Count - Context - 1);
                for (int t = 0; t < Context; t++)
                {
                    inputs[b * Context + t] = ids[start + t];
                    targets[b * Context + t] = ids[start + t + 1];
                }
            }

            using var scope = new TensorScope();
            var loss = Losses.SparseCrossEntropy(model.Forward(Tensor.From(inputs, [Batch, Context], device)), Tensor.From(targets, [Batch, Context], device));
            optimizer.ZeroGrad();
            loss.Backward();
            optimizer.Step();
            last = loss.Item();
            first = step == 0 ? last : first;
        }

        model.Eval();
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 24 };
        string written = "idrak " + new TextGenerator(model, tokenizer, Context) { KeepCache = false }.Generate("idrak ", options).Text;
        return new DemoResult("loss", last, $"loss {first:F3} -> {last:F3}", (long)Batch * Context * Steps, "tokens", last < 0.5f * first, written.TrimEnd());
    }
}
