// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Image classification with a convolutional network: 16x16 grayscale drawings of circles, squares,
// triangles and crosses at random positions and sizes, with noise. Conv2d, BatchNorm, MaxPool2d, Dropout.
//
//   dotnet run -c Release --project samples/Idrak.Samples.ShapeRecognition            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.ShapeRecognition -- --predict --input "circle,cross,square"
//        draw new random images of those shapes and classify them with the saved model

using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Samples;
using Idrak.Training;

if (SampleOptions.Parse(args) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels));

const int Size = 16;
string[] shapes = ["circle", "square", "triangle", "cross"];
var init = new Random(3);
using var model = new Sequential
{
    new Conv2d(1, 16, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(16, device: device), new ReLU(),
    new MaxPool2d(2),                                                                        // 16x16 -> 8x8
    new Conv2d(16, 32, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(32, device: device), new ReLU(),
    new MaxPool2d(2),                                                                        // 8x8 -> 4x4
    new Flatten(),
    new Linear(32 * 4 * 4, 64, device: device, random: init), new ReLU(), new Dropout(0.2f, init),
    new Linear(64, shapes.Length, device: device, random: init),
};
model.Name = "shape-cnn";

string modelPath = options.ModelPath("shapes.weights");
if (options.PredictOnly)
{
    // Inference mode: draw fresh images of the requested shapes and classify them with the saved model.
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    model.Load(modelPath);
    Console.WriteLine($"Loaded {modelPath}\n");
    var requested = (options.Input ?? "circle,square,triangle,cross").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
    int[] labels = [.. requested.Select(name => Array.IndexOf(shapes, name.ToLowerInvariant()))];
    if (labels.Any(l => l < 0))
    {
        Console.Error.WriteLine($"error: shapes must be among: {string.Join(", ", shapes)}.");
        return 2;
    }

    var images = Draw(labels.Length, seed: Environment.TickCount, labels);
    using var batch = Tensor.From(images.Features, [images.Count, 1, Size, Size], device);
    using var logits = model.Predict(batch);
    Show(images, logits.ToArray2D(), images.Count);
    return 0;
}

var train = Draw(4000, seed: 1);
var test = Draw(800, seed: 2);
Console.WriteLine($"{train.Count} training and {test.Count} test images of {Size}x{Size} pixels, {shapes.Length} classes\n");

int epochs = options.Epochs ?? 12;
using var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f);
var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
{
    Metrics = { Metric.Accuracy },
    Scheduler = new CosineAnnealing(optimizer, epochs),
};
trainer.Fit(new DataLoader(train, options.BatchSize ?? 64, shuffle: true, device: device, seed: 4), epochs,
    validation: new DataLoader(test, 400, device: device));

var result = trainer.Evaluate(new DataLoader(test, 400, device: device));
Console.WriteLine($"\nTest accuracy: {result.Metrics["accuracy"]:P1}\n");

model.Save(modelPath);
Console.WriteLine($"Saved the model to {modelPath} (test it with --predict)\n");

// Show a few test images with the model's guess and confidence.
Show(test, trainer.Predict(test), 4);

return result.Metrics["accuracy"] > 0.9 ? 0 : 1;

// Prints each image as ASCII art with the model's guess, confidence and the true shape.
void Show(Dataset data, float[,] scores, int count)
{
    for (int i = 0; i < count; i++)
    {
        var logits = Enumerable.Range(0, shapes.Length).Select(c => scores[i, c]).ToArray();
        float max = logits.Max();
        var exp = logits.Select(v => MathF.Exp(v - max)).ToArray();
        int guess = Array.IndexOf(logits, max);
        int actual = data.GetTargets(i).IndexOf(1f);
        Console.WriteLine($"Image {i + 1}: predicted {shapes[guess]} ({exp[guess] / exp.Sum():P0}), actually {shapes[actual]}");
        var pixels = data.GetFeatures(i);
        for (int y = 0; y < Size; y += 2)
        {
            var line = new char[Size];
            for (int x = 0; x < Size; x++)
            {
                float v = (pixels[y * Size + x] + pixels[(y + 1) * Size + x]) / 2;
                line[x] = v switch { > 0.6f => '#', > 0.3f => '+', > 0.15f => '.', _ => ' ' };
            }

            Console.WriteLine("    " + new string(line));
        }
    }
}

// Renders `count` images, cycling through the shape classes (or using the given labels).
Dataset Draw(int count, int seed, int[]? only = null)
{
    var random = new Random(seed);
    var pixels = new float[count, Size * Size];
    var labels = new int[count];
    for (int s = 0; s < count; s++)
    {
        int shape = only?[s] ?? s % shapes.Length;
        labels[s] = shape;
        float r = 3 + random.NextSingle() * 3.5f;
        float cx = r + random.NextSingle() * (Size - 1 - 2 * r), cy = r + random.NextSingle() * (Size - 1 - 2 * r);
        float brightness = 0.6f + random.NextSingle() * 0.4f;
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                float dx = x - cx, dy = y - cy;
                bool on = shape switch
                {
                    0 => MathF.Abs(MathF.Sqrt(dx * dx + dy * dy) - r) < 0.8f,                                  // circle outline
                    1 => MathF.Max(MathF.Abs(dx), MathF.Abs(dy)) is var m && MathF.Abs(m - r) < 0.7f,          // square outline
                    2 => dy <= r && dy >= -r && MathF.Abs(dx) <= (dy + r) / 2 && MathF.Abs(dx) >= (dy + r) / 2 - 1.2f
                         || MathF.Abs(dy - r) < 0.7f && MathF.Abs(dx) <= r,                                    // triangle outline
                    _ => (MathF.Abs(dx) < 0.8f || MathF.Abs(dy) < 0.8f) && MathF.Max(MathF.Abs(dx), MathF.Abs(dy)) <= r, // cross
                };
                pixels[s, y * Size + x] = (on ? brightness : 0f) + (random.NextSingle() - 0.5f) * 0.2f;
            }
        }
    }

    return Dataset.FromClassLabels(pixels, labels, shapes.Length, shapes).WithFeatureShape(1, Size, Size);
}
