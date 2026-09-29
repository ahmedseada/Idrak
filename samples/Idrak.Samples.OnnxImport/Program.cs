// Imports an ONNX model made by another framework into Idrak layers and runs it on Idrak's own
// backend (CPU or CUDA). With a "<file>.expected.json" next to the model (inputs and the other framework's outputs,
// as written by tools/pytorch/xor_to_onnx.py), it checks that Idrak computes the same values, then saves the
// model as a Idrak package (.ikm), reloads it and checks it again.
//
//   python tools/pytorch/xor_to_onnx.py --out xor.onnx
//   dotnet run -c Release --project samples/Idrak.Samples.OnnxImport -- --onnx xor.onnx --cuda

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Onnx;
using Idrak.Samples;

if (SampleOptions.Parse(args, ("onnx", "the .onnx file to import (required)"), ("tolerance", "largest allowed difference (default: the expected file's, else 1e-5)")) is not { } options)
{
    return 0;
}

string path = options.Get("onnx") ?? throw new ArgumentException("Pass --onnx <file>.");
var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}");

var clock = Stopwatch.StartNew();
using var imported = OnnxImport.Load(path, device);
Console.WriteLine($"Imported {Path.GetFileName(path)} in {clock.Elapsed.TotalMilliseconds:F1} ms:");
if (imported.Model is GraphModule graph)
{
    Console.WriteLine($"  a graph of {graph.Nodes.Count} nodes (skip connections, branches or shape arithmetic):");
    foreach (var node in graph.Nodes)
    {
        Console.WriteLine($"    {node.Output,-28} = {(node.Layer is { } layer ? layer.ToString() : node.Op)}({string.Join(", ", node.Inputs)})");
    }
}
else
{
    foreach (var layer in (Sequential)imported.Model)
    {
        Console.WriteLine($"  {layer}");
    }
}

foreach (var note in imported.Notes)
{
    Console.WriteLine($"  note: {note}");
}

foreach (var (key, value) in imported.Metadata)
{
    Console.WriteLine($"  metadata {key} = {value}");
}

string expectedPath = path + ".expected.json";
if (!File.Exists(expectedPath))
{
    Console.WriteLine($"\nNo {Path.GetFileName(expectedPath)}: nothing to compare with.");
    return 0;
}

var expected = JsonNode.Parse(File.ReadAllText(expectedPath))!;
float tolerance = options.Get("tolerance") is { } given ? float.Parse(given, System.Globalization.CultureInfo.InvariantCulture)
    : expected["tolerance"] is { } stored ? (float)stored : 1e-5f;
var inputs = expected["inputs"]!.AsArray().Select(row => row!.AsArray().Select(v => (float)v!).ToArray()).ToList();
var reference = expected["outputs"]!.AsArray().Select(row => row!.AsArray().Select(v => (float)v!).ToArray()).ToList();
string source = expected["torch"] is { } version ? $"PyTorch {version}" : "other framework";
Console.WriteLine($"\nReference: {source} " +
    $"(trained on {expected["trainedOn"] ?? "?"}, {expected["exporter"] ?? "?"} exporter)");

// Every sample is stored flat; the model gets them as one batch shaped [N, ..inputShape] (for example [4, 3, 32, 32]).
int[] sampleShape = expected["inputShape"] is JsonArray storedShape ? [.. storedShape.Select(d => (int)d!)] : [.. imported.InputShape ?? [inputs[0].Length]];

float[][] Run(Module model)
{
    using var scope = new TensorScope();
    using var batch = Tensor.From([.. inputs.SelectMany(i => i)], [inputs.Count, .. sampleShape], device);
    var result = model.Predict(batch).ToArray();
    int width = result.Length / inputs.Count;
    return [.. Enumerable.Range(0, inputs.Count).Select(i => result[(i * width)..((i + 1) * width)])];
}

float Compare(string name, float[][] actual)
{
    float worst = 0;
    bool small = inputs[0].Length <= 4 && reference[0].Length <= 2;
    Console.WriteLine(small ? $"\n{name}:\n  input            reference     Idrak   difference" : $"\n{name}:");
    for (int i = 0; i < inputs.Count; i++)
    {
        float sampleWorst = 0;
        for (int j = 0; j < reference[i].Length; j++)
        {
            float difference = MathF.Abs(actual[i][j] - reference[i][j]);
            sampleWorst = MathF.Max(sampleWorst, difference);
            if (small)
            {
                Console.WriteLine($"  {$"[{string.Join(", ", inputs[i])}]",-16} {reference[i][j],10:F6}   {actual[i][j],10:F6}   {difference:E1}");
            }
        }

        if (!small)
        {
            int Top(float[] v) => Array.IndexOf(v, v.Max());
            Console.WriteLine($"  sample {i}: {reference[i].Length} outputs, largest difference {sampleWorst:E1}, " +
                $"top output {Top(reference[i])} (reference) / {Top(actual[i])} (Idrak)");
        }

        worst = MathF.Max(worst, sampleWorst);
    }

    Console.WriteLine($"  largest difference {worst:E2} ({(worst <= tolerance ? "match" : "MISMATCH")}, tolerance {tolerance:E0})");
    return worst;
}

float importedWorst = Compare($"Imported model on {device}", Run(imported.Model));

string packagePath = Path.ChangeExtension(path, ".ikm");
imported.SavePackage(packagePath);
using var predictor = Predictor.Load(packagePath, device).Build();
float packageWorst = Compare($"Saved as {Path.GetFileName(packagePath)} and reloaded on {device}",
    [.. inputs.Select(input => predictor.Predict(input))]);

clock.Restart();
const int Repeats = 1000;
for (int i = 0; i < Repeats; i++)
{
    Run(imported.Model);
}

Console.WriteLine($"\n{Repeats} batches of {inputs.Count}: {clock.Elapsed.TotalMilliseconds / Repeats:F3} ms per batch on {device}");
return importedWorst <= tolerance && packageWorst <= tolerance ? 0 : 1;
