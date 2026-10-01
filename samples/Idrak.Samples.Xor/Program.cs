// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Learns XOR, the classic problem a single linear layer cannot solve.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Xor               GPU if available, else CPU
//   dotnet run -c Release --project samples/Idrak.Samples.Xor -- --cpu      force the CPU
//   dotnet run -c Release --project samples/Idrak.Samples.Xor -- --cuda     force the GPU
//   dotnet run -c Release --project samples/Idrak.Samples.Xor -- --predict --input "1,0;1,1"
//                                                                     load the saved model and evaluate inputs
//   dotnet run -c Release --project samples/Idrak.Samples.Xor -- --help     all options

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

// Telemetry: print one line every 250 epochs (plus the first and last), and optionally log everything to a file.
using var console = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels, epochInterval: 250));
await using var fileLog = options.LogFile is { } path ? new JsonLinesLogger(path, TelemetryLevel.All) : null;
using var fileSubscription = fileLog is null ? null : Telemetry.Subscribe(fileLog);

// The four XOR cases: the output is 1 when exactly one input is 1.
var data = Dataset.FromArrays(
    features: new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } },
    targets: new float[,] { { 0 }, { 1 }, { 1 }, { 0 } },
    featureNames: ["a", "b"],
    targetNames: ["a xor b"]);

// 2 inputs -> 8 hidden units (tanh) -> 1 output (sigmoid, so it reads as a probability).
var random = new Random(42);
using var model = new Sequential
{
    new Linear(2, 8, device: device, random: random),
    new Tanh(),
    new Linear(8, 1, device: device, random: random),
    new Sigmoid(),
};
model.Name = "xor";

string modelPath = options.ModelPath("xor.weights");
if (options.PredictOnly)
{
    // Inference mode: load the trained weights and evaluate the given inputs ("a,b;a,b;...").
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    model.Load(modelPath);
    var pairs = (options.Input ?? "0,0;0,1;1,0;1,1").Split(';', StringSplitOptions.RemoveEmptyEntries)
        .Select(p => p.Split(',').Select(v => float.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray()).ToArray();
    var inputs = new float[pairs.Length, 2];
    for (int i = 0; i < pairs.Length; i++)
    {
        inputs[i, 0] = pairs[i][0];
        inputs[i, 1] = pairs[i][1];
    }

    var outputs = model.Predict(inputs);
    Console.WriteLine($"Loaded {modelPath}\n\n  A     B   | output  | rounded");
    for (int i = 0; i < pairs.Length; i++)
    {
        Console.WriteLine($"  {inputs[i, 0],-4}  {inputs[i, 1],-4}| {outputs[i, 0]:F4}  |   {(outputs[i, 0] >= 0.5f ? 1 : 0)}");
    }

    return 0;
}

using var optimizer = new Adam(model.Parameters(), learningRate: 0.05f);
var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
{
    Metrics = { Metric.MeanAbsoluteError },
};

var loader = new DataLoader(data, batchSize: options.BatchSize ?? 4, device: device);
trainer.Fit(loader, epochs: options.Epochs ?? 2000);
model.Save(modelPath);
Console.WriteLine($"Saved the model to {modelPath} (test it with --predict)");

// Inference.
var predictions = model.Predict(data.FeaturesToArray());
var x = data.FeaturesToArray();
var y = data.TargetsToArray();
int correct = 0;

Console.WriteLine("\n A  B | expected | predicted | rounded");
Console.WriteLine("------+----------+-----------+--------");
for (int i = 0; i < data.Count; i++)
{
    int rounded = predictions[i, 0] >= 0.5f ? 1 : 0;
    bool ok = rounded == (int)y[i, 0];
    correct += ok ? 1 : 0;
    Console.WriteLine($" {x[i, 0]}  {x[i, 1]} |    {y[i, 0]}     |  {predictions[i, 0]:F4}   |   {rounded} {(ok ? "✓" : "✗")}");
}

Console.WriteLine($"\n{correct}/{data.Count} correct");
return correct == data.Count ? 0 : 1;
