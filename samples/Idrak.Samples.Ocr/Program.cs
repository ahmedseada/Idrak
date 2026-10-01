// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Optical character recognition: a CNN learns to read the 36 characters 0-9 and A-Z from randomly
// distorted renderings, then reads whole text lines by segmenting them into characters.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Ocr                                   train, save, read (add --cpu / --cuda)
//   dotnet run -c Release --project samples/Idrak.Samples.Ocr -- --predict --input "HELLO WORLD 2026"
//                                                                    render that text and read it with the saved model
//   dotnet run -c Release --project samples/Idrak.Samples.Ocr -- --predict --image page.pgm    read your own image
//   dotnet run -c Release --project samples/Idrak.Samples.Ocr -- --predict --save-image line.pgm  export the demo line

using System.Diagnostics;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Samples;
using Idrak.Samples.Ocr;
using Idrak.Training;

if (SampleOptions.Parse(args,
        ("image", "PGM image to read (one or more lines of printed text)"),
        ("save-image", "write the rendered demo line to this PGM file")) is not { } options)
{
    return 0;
}

var device = options.Device;
Console.WriteLine($"Device: {device} - {device.Name}\n");
using var logger = Telemetry.Subscribe(new ConsoleLogger(options.LogLevels));

const int Cell = Renderer.Cell;
string alphabet = Font.Characters;

// ---------------------------------------------------------------- model: CNN over 20x20 character images
var init = new Random(1);
using var model = new Sequential
{
    new Conv2d(1, 32, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(32, device: device), new ReLU(),
    new MaxPool2d(2),                                                                           // 20x20 -> 10x10
    new Conv2d(32, 64, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(64, device: device), new ReLU(),
    new MaxPool2d(2),                                                                           // 10x10 -> 5x5
    new Flatten(),
    new Linear(64 * 5 * 5, 128, device: device, random: init), new ReLU(), new Dropout(0.3f, init),
    new Linear(128, alphabet.Length, device: device, random: init),
};
model.Name = "ocr-cnn";

string modelPath = options.ModelPath("ocr.weights");
if (options.PredictOnly)
{
    // Inference mode: load the saved recognizer and go straight to reading.
    if (!options.RequireModel(modelPath))
    {
        return 1;
    }

    model.Load(modelPath);
    Console.WriteLine($"Loaded {modelPath}");
}
else
{
    // ---------------------------------------------------------------- data: distorted renderings of every character
    var stopwatch = Stopwatch.StartNew();
    var train = Characters(perClass: 300, seed: 2);
    var test = Characters(perClass: 60, seed: 3);
    Console.WriteLine($"Rendered {train.Count:N0} training and {test.Count:N0} test characters in {stopwatch.ElapsedMilliseconds} ms\n");

    int epochs = options.Epochs ?? 10;
    using var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f, weightDecay: 1e-4f);
    var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
    {
        Metrics = { Metric.Accuracy },
        Scheduler = new CosineAnnealing(optimizer, epochs, warmupEpochs: 1),
    };
    trainer.Fit(new DataLoader(train, options.BatchSize ?? 64, shuffle: true, device: device, seed: 4), epochs,
        validation: new DataLoader(test, 500, device: device));

    var result = trainer.Evaluate(new DataLoader(test, 500, device: device));
    Console.WriteLine($"\nCharacter accuracy on unseen renderings: {result.Metrics["accuracy"]:P2}");
    ReportConfusions(trainer.Predict(test), test);

    model.Save(modelPath);
    Console.WriteLine($"Saved the model to {modelPath} (test it with --predict --input \"YOUR TEXT\")");
    MeasureLines();
}

// ---------------------------------------------------------------- demo
if (options.Get("image") is { } imagePath)
{
    var (pixels, width, height) = Pgm.Read(imagePath);
    Console.WriteLine($"\n{imagePath} ({width}x{height}):");
    Preview(pixels, width, height);
    Console.WriteLine($"Text: {Read(pixels, width, height)}");
}
else
{
    string text = (options.Input ?? "IDRAK READS 2026").ToUpperInvariant();
    var (pixels, width) = Renderer.RenderLine(text, new Random(6));
    Console.WriteLine($"\nRendered line \"{text}\":");
    Preview(pixels, width, Cell);
    Console.WriteLine($"Recognized: {Read(pixels, width, Cell)}");
    if (options.Get("save-image") is { } savePath)
    {
        Pgm.Write(savePath, pixels, width, Cell);
        Console.WriteLine($"Saved the line to {savePath}; read it back with --image {savePath}");
    }
}

return 0;

// Reads 40 random rendered text lines and reports character accuracy (edit distance) and exact-line rate.
void MeasureLines()
{
    string[] words = ["HELLO", "WORLD", "NEURAL", "SHARP", "DOTNET", "CODE", "IMAGE", "TEXT", "READ", "MODEL", "LAYER", "TENSOR", "GPU", "CUDA", "PIXEL", "QUICK", "BROWN", "FOX", "JUMPS", "OVER", "LAZY", "DOG"];
    var random = new Random(5);
    int totalChars = 0, errors = 0, exactLines = 0;
    const int Lines = 40;
    for (int i = 0; i < Lines; i++)
    {
        string truth = string.Join(' ', Enumerable.Range(0, random.Next(2, 5)).Select(_ => random.Next(4) == 0
            ? random.Next(10, 99999).ToString()
            : words[random.Next(words.Length)]));
        var (pixels, width) = Renderer.RenderLine(truth, random);
        string read = Read(pixels, width, Cell);
        totalChars += truth.Length;
        errors += EditDistance(truth, read);
        exactLines += truth == read ? 1 : 0;
    }

    Console.WriteLine($"\nRead {Lines} random text lines: {1 - errors / (double)totalChars:P2} character accuracy, {exactLines}/{Lines} lines exactly right");
}

// Renders perClass distorted images of every character, as a [N, 1, 20, 20] classification dataset.
Dataset Characters(int perClass, int seed)
{
    var rng = new Random(seed);
    int count = perClass * alphabet.Length;
    var pixels = new float[count, Cell * Cell];
    var labels = new int[count];
    var image = new float[Cell * Cell];
    for (int s = 0; s < count; s++)
    {
        labels[s] = s % alphabet.Length;
        Array.Clear(image);
        Renderer.RenderCharacter(labels[s], image, rng);
        for (int i = 0; i < image.Length; i++)
        {
            pixels[s, i] = image[i];
        }
    }

    return Dataset.FromClassLabels(pixels, labels, alphabet.Length, [.. alphabet.Select(c => c.ToString())]).WithFeatureShape(1, Cell, Cell);
}

// Segments an image into characters, classifies them all in one batch, and reassembles the text.
string Read(float[] pixels, int width, int height)
{
    var glyphs = Segmenter.Segment(pixels, width, height);
    if (glyphs.Count == 0)
    {
        return "";
    }

    var batch = new float[glyphs.Count * Cell * Cell];
    for (int i = 0; i < glyphs.Count; i++)
    {
        glyphs[i].Pixels.CopyTo(batch, i * Cell * Cell);
    }

    using var input = Tensor.From(batch, [glyphs.Count, 1, Cell, Cell], device);
    using var logits = model.Predict(input);
    using var best = logits.ArgMax();
    var classes = best.ToArray();
    var text = new System.Text.StringBuilder();
    for (int i = 0; i < glyphs.Count; i++)
    {
        text.Append(glyphs[i].NewLineBefore ? "\n" : glyphs[i].SpaceBefore ? " " : "");
        text.Append(alphabet[(int)classes[i]]);
    }

    return text.ToString();
}

void ReportConfusions(float[,] scores, Dataset data)
{
    var mistakes = new Dictionary<(char Actual, char Predicted), int>();
    for (int i = 0; i < data.Count; i++)
    {
        int actual = data.GetTargets(i).IndexOf(1f);
        int predicted = Enumerable.Range(0, alphabet.Length).MaxBy(c => scores[i, c]);
        if (actual != predicted)
        {
            var key = (alphabet[actual], alphabet[predicted]);
            mistakes[key] = mistakes.GetValueOrDefault(key) + 1;
        }
    }

    if (mistakes.Count > 0)
    {
        Console.WriteLine("Most common confusions: " + string.Join(", ",
            mistakes.OrderByDescending(m => m.Value).Take(6).Select(m => $"{m.Key.Actual}→{m.Key.Predicted} ×{m.Value}")));
    }
}

static void Preview(float[] pixels, int width, int height)
{
    for (int y = 0; y < height; y += 2)
    {
        var line = new char[Math.Min(width, 110)];
        for (int x = 0; x < line.Length; x++)
        {
            float v = Math.Max(pixels[y * width + x], y + 1 < height ? pixels[(y + 1) * width + x] : 0f);
            line[x] = v switch { > 0.6f => '#', > 0.35f => '+', _ => ' ' };
        }

        Console.WriteLine("  " + new string(line).TrimEnd());
    }
}

static int EditDistance(string a, string b)
{
    var previous = Enumerable.Range(0, b.Length + 1).ToArray();
    for (int i = 1; i <= a.Length; i++)
    {
        var current = new int[b.Length + 1];
        current[0] = i;
        for (int j = 1; j <= b.Length; j++)
        {
            current[j] = Math.Min(Math.Min(current[j - 1] + 1, previous[j] + 1), previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
        }

        previous = current;
    }

    return previous[b.Length];
}
