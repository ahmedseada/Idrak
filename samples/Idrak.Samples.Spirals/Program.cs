// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Multi-class classification: three interleaved spirals that no straight line can separate.
// Softmax + cross-entropy, BatchNorm, AdamW with cosine learning-rate schedule, accuracy metric.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Spirals            (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Spirals -- --predict --input "0.5,0.2;-0.3,-0.6"
//        classify points (x,y in [-1, 1]) with the saved model

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
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 20));

const int Classes = 3, PerClass = 300;
string[] classNames = ["red", "green", "blue"];

// ---------------------------------------------------------------- model: outputs raw scores (logits)
var init = new Random(3);
using var model = new Sequential
{
    new Linear(2, 128, device: device, random: init), new BatchNorm(128, device: device), new ReLU(),
    new Linear(128, 64, device: device, random: init), new BatchNorm(64, device: device), new ReLU(),
    new Linear(64, Classes, device: device, random: init),
};
model.Name = "spiral-classifier";

string modelPath = options.ModelPath("spirals.weights");
float accuracy = 1f;
if (options.PredictOnly)
{
    // Inference mode: load the saved classifier and classify the given points.
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    model.Load(modelPath);
    Console.WriteLine($"Loaded {modelPath}\n");
    var points = (options.Input ?? "0,0;0.5,0.2;-0.3,-0.6;0.8,-0.5;-0.9,0.4")
        .Split(';', StringSplitOptions.RemoveEmptyEntries)
        .Select(p => p.Split(',').Select(v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray()).ToArray();
    var input = new float[points.Length, 2];
    for (int i = 0; i < points.Length; i++)
    {
        (input[i, 0], input[i, 1]) = (points[i][0], points[i][1]);
    }

    using var pointTensor = Tensor.From(input, device);
    using var pointLogits = model.Predict(pointTensor);
    using var pointProbabilities = pointLogits.Softmax();
    var pp = pointProbabilities.ToArray();
    Console.WriteLine("     x      y  | class  | " + string.Join("  ", classNames.Select(n => n.PadLeft(6))));
    for (int i = 0; i < points.Length; i++)
    {
        int best = Enumerable.Range(0, Classes).MaxBy(c => pp[i * Classes + c]);
        Console.WriteLine($"  {input[i, 0],5:F2}  {input[i, 1],5:F2} | {classNames[best],-6} | " +
                          string.Join("  ", Enumerable.Range(0, Classes).Select(c => $"{pp[i * Classes + c],6:P0}")));
    }
}
else
{
    // ---------------------------------------------------------------- data: 3 spirals, 300 points each
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
            features[row, 0] = (float)(radius * Math.Cos(angle));
            features[row, 1] = (float)(radius * Math.Sin(angle));
            labels[row] = c;
        }
    }

    var data = Dataset.FromClassLabels(features, labels, Classes, classNames);
    var (train, test) = data.Split(0.8, seed: 2);

    int epochs = options.Epochs ?? 200;
    using var optimizer = new AdamW(model.Parameters(), learningRate: 0.01f, weightDecay: 1e-4f);
    var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
    {
        Metrics = { Metric.Accuracy },
        Scheduler = new CosineAnnealing(optimizer, totalEpochs: epochs, warmupEpochs: 5),
    };

    trainer.Fit(new DataLoader(train, options.BatchSize ?? 64, shuffle: true, device: device, seed: 4), epochs,
        validation: new DataLoader(test, 512, device: device));

    // ---------------------------------------------------------------- evaluate
    var result = trainer.Evaluate(new DataLoader(test, 512, device: device));
    Console.WriteLine($"\nTest accuracy: {result.Metrics["accuracy"]:P1}  (cross-entropy {result.Loss:F4})\n");

    var scores = trainer.Predict(test);
    var confusion = new int[Classes, Classes];
    for (int i = 0; i < test.Count; i++)
    {
        int actual = test.GetTargets(i).IndexOf(1f);
        int predicted = Enumerable.Range(0, Classes).MaxBy(c => scores[i, c]);
        confusion[actual, predicted]++;
    }

    Console.WriteLine("Confusion matrix (rows = actual, columns = predicted):");
    Console.WriteLine("        " + string.Concat(classNames.Select(n => n.PadLeft(7))));
    for (int a = 0; a < Classes; a++)
    {
        Console.WriteLine($"  {classNames[a],-6}" + string.Concat(Enumerable.Range(0, Classes).Select(p => confusion[a, p].ToString().PadLeft(7))));
    }

    model.Save(modelPath);
    Console.WriteLine($"\nSaved the model to {modelPath} (test it with --predict)");
    accuracy = (float)result.Metrics["accuracy"];
}

// ---------------------------------------------------------------- decision map with probabilities
const int Width = 60, Height = 26;
var grid = new float[Width * Height, 2];
for (int y = 0; y < Height; y++)
{
    for (int x = 0; x < Width; x++)
    {
        grid[y * Width + x, 0] = -1.1f + 2.2f * x / (Width - 1);
        grid[y * Width + x, 1] = 1.1f - 2.2f * y / (Height - 1);
    }
}

using var gridTensor = Tensor.From(grid, device);
using var logits = model.Predict(gridTensor);
using var probabilities = logits.Softmax();
var p = probabilities.ToArray();
Console.WriteLine("\nDecision map (upper case = at least 90% confident):");
for (int y = 0; y < Height; y++)
{
    var line = new char[Width];
    for (int x = 0; x < Width; x++)
    {
        int cell = y * Width + x;
        int best = Enumerable.Range(0, Classes).MaxBy(c => p[cell * Classes + c]);
        char symbol = "rgb"[best];
        line[x] = p[cell * Classes + best] >= 0.9f ? char.ToUpperInvariant(symbol) : symbol;
    }

    Console.WriteLine("  " + new string(line));
}

return accuracy > 0.9 ? 0 : 1;
