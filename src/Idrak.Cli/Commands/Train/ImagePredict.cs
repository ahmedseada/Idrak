// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data.Abstractions;
using Idrak.Models.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak predict</c> on an image model: classes with their probabilities, boxes, per-class pixel counts (and mask PNGs)
/// or feature vectors (and .npy files), through the model's own preprocessing. Images whose preprocessing gives the same
/// size run as one batch, as many at once as the device has room for (<see cref="MeasuredBatches"/>).
/// </summary>
internal static class ImagePredict
{
    private sealed record Result(string File, int Width, int Height)
    {
        public double[]? Probabilities { get; set; }

        public IReadOnlyList<Detection>? Boxes { get; set; }

        public SegmentationMask? Mask { get; set; }

        public float[]? Features { get; set; }

        public int[]? FeatureShape { get; set; }

        public string? Written { get; set; }
    }

    /// <summary>Runs <paramref name="model"/> on the images <paramref name="inputs"/> name and prints or writes the predictions.</summary>
    public static int Run(CommandContext context, string modelPath, ImageModel model, IReadOnlyList<string> inputs)
    {
        var files = ImageModelFiles.Expand(inputs);
        if (files.Count == 0)
        {
            throw new UsageException($"No images in {string.Join(", ", inputs)} (the codecs read {string.Join(", ", ImageFiles.DecodedExtensions)}).");
        }

        string? decoder = context.Option("--decoder");
        if (decoder is not null && model.Task != ImageTask.Detection)
        {
            throw new UsageException($"--decoder is for detectors; {modelPath} is a {Task(model.Task)} model.");
        }

        var device = context.Device;
        var labels = model.Labels;
        int top = context.IntOption("--top", model.Task == ImageTask.Classification ? Math.Min(5, labels?.Count ?? 5) : 1);
        float threshold = Number(context, "--threshold", 0.25f), iou = Number(context, "--iou", 0.5f);
        int most = context.IntOption("--max", 100);
        int? ceiling = context.Option("--batch") is null ? null : context.IntOption("--batch", 1);
        if (top <= 0 || most <= 0 || ceiling is <= 0)
        {
            throw new UsageException("--top, --max and --batch need positive numbers.");
        }

        string? output = context.Option("--out");
        bool folderOut = model.Task is ImageTask.Segmentation or ImageTask.Features;
        if (output is not null && folderOut)
        {
            Directory.CreateDirectory(output);
        }

        ModelDetector? detector = model.Task != ImageTask.Detection ? null
            : new ModelDetector(model.Network, model.Preprocessor, decoder ?? model.Decoder!, model.DecoderSettings,
                new DetectorOptions { MinScore = threshold, IouThreshold = iou, MaxDetections = most, Classes = labels }, device);
        ModelSegmenter? segmenter = model.Task == ImageTask.Segmentation ? new ModelSegmenter(model.Network, model.Preprocessor, device) : null;

        // Each image's network size from its header, then the runs of one size in measured batches.
        var sizes = new (int Height, int Width)[files.Count];
        var results = new Result[files.Count];
        for (int i = 0; i < files.Count; i++)
        {
            var info = ImageCodecs.ReadInfo(files[i]) ?? throw new InvalidDataException($"{files[i]}: not an image format of the registered codecs ({string.Join(", ", ImageCodecs.Names)}).");
            sizes[i] = model.Preprocessor.OutputSize(info.Height, info.Width);
            results[i] = new Result(files[i], info.Width, info.Height);
        }

        var batches = new MeasuredBatches(device, ceiling);
        int runs = 0;
        using (var progress = new ProgressLine(context, "predicting", files.Count))
        {
            foreach (var group in Enumerable.Range(0, files.Count).GroupBy(i => sizes[i]))
            {
                int[] members = [.. group];
                batches.Run(members.Length, (start, count) =>
                {
                    var chunk = members.AsSpan(start, count).ToArray();
                    var images = chunk.Select(i => ImageCodecs.Decode(files[i])).ToList();
                    switch (model.Task)
                    {
                        case ImageTask.Detection:
                            var found = detector!.Detect(images);
                            for (int k = 0; k < chunk.Length; k++)
                            {
                                results[chunk[k]].Boxes = found[k];
                            }

                            break;
                        case ImageTask.Segmentation:
                            var masks = segmenter!.Segment(images);
                            for (int k = 0; k < chunk.Length; k++)
                            {
                                results[chunk[k]].Mask = masks[k];
                            }

                            break;
                        default:
                            Outputs(model, images, chunk, results, device);
                            break;
                    }

                    runs++;
                    progress.Advance(count);
                });
            }

            progress.Finish();
        }

        if (output is not null && folderOut)
        {
            Write(model, results, output);
        }

        Print(context, model, results, top, inputs);
        if (output is not null && !folderOut)
        {
            RowFiles.Write(Rows(model, results, top), output);
            context.Write($"{results.Length:N0} images' predictions written to {output}");
        }
        else if (output is not null)
        {
            context.Write($"{results.Count(r => r.Written is not null):N0} {(model.Task == ImageTask.Segmentation ? "masks" : ".npy files")} written to {output}");
        }

        if (batches.Halvings > 0)
        {
            context.Detail($"out of device memory {batches.Halvings} time(s): batches halved to {batches.Size} images");
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = modelPath,
            ["architecture"] = model.Architecture,
            ["task"] = Task(model.Task),
            ["device"] = device.ToString(),
            ["images"] = results.Length,
            ["batches"] = runs,
            ["batch"] = batches.Size,
            ["output"] = output,
            ["predictions"] = new JsonArray([.. Rows(model, results, top, nested: true).Select(r => (JsonNode)r)]),
        });
        return ExitCodes.Ok;
    }

    /// <summary>The task's name in output: classification, detection, segmentation or features.</summary>
    public static string Task(ImageTask task) => task.ToString().ToLowerInvariant();

    private static float Number(CommandContext context, string option, float fallback) => context.Option(option) is not { } text ? fallback
        : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) && v is >= 0 and <= 1 ? v
        : throw new UsageException($"{option} needs a number from 0 to 1, not '{text}'.");

    // Classification and features: the preprocessed pixels as one batch, the network's outputs per image.
    private static void Outputs(ImageModel model, List<ImageData> images, int[] chunk, Result[] results, Device device)
    {
        var pixels = images.Select(model.Preprocessor.Pixels).ToList();
        var (height, width) = model.Preprocessor.OutputSize(images[0].Height, images[0].Width);
        int per = pixels[0].Length, channels = per / (height * width);
        var batch = new float[pixels.Count * per];
        for (int k = 0; k < pixels.Count; k++)
        {
            pixels[k].CopyTo(batch, k * per);
        }

        float[] values;
        int[] shape;
        using (var x = Tensor.From(batch, [pixels.Count, channels, height, width], device))
        using (var y = model.Network.Predict(x))
        {
            values = y.ToArray();
            shape = [.. y.Shape[1..]];
        }

        int size = values.Length / pixels.Count;
        for (int k = 0; k < chunk.Length; k++)
        {
            var row = values.AsSpan(k * size, size);
            if (model.Task == ImageTask.Classification)
            {
                results[chunk[k]].Probabilities = Softmax(row);
            }
            else
            {
                results[chunk[k]].Features = row.ToArray();
                results[chunk[k]].FeatureShape = shape;
            }
        }
    }

    private static double[] Softmax(ReadOnlySpan<float> values)
    {
        float max = float.NegativeInfinity;
        foreach (float v in values)
        {
            max = Math.Max(max, v);
        }

        var result = new double[values.Length];
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            sum += result[i] = Math.Exp(values[i] - max);
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    // Mask PNGs (the grey level is the class) or .npy feature files, one per image, named after it.
    private static void Write(ImageModel model, Result[] results, string folder)
    {
        var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var result in results)
        {
            string stem = Path.GetFileNameWithoutExtension(result.File), extension = model.Task == ImageTask.Segmentation ? ".png" : ".npy";
            string name = stem + extension;
            for (int n = 2; !used.Add(name); n++)
            {
                name = $"{stem}-{n}{extension}";
            }

            string path = Path.Combine(folder, name);
            if (result.Mask is { } mask)
            {
                if (mask.ClassCount > 256)
                {
                    throw new NotSupportedException($"{mask.ClassCount} classes do not fit an 8-bit mask image.");
                }

                var levels = new float[mask.Width * mask.Height];
                var labels = mask.Labels;
                for (int i = 0; i < levels.Length; i++)
                {
                    levels[i] = labels[i] / 255f;
                }

                ImageEncoders.Save(path, new ImageData(levels, 1, mask.Height, mask.Width));
                result.Written = path;
            }
            else if (result.Features is { } features)
            {
                ImageModelFiles.WriteNpy(path, features, result.FeatureShape!);
                result.Written = path;
            }
        }
    }

    private static string Label(ImageModel model, int index) => model.Labels is { } labels && index < labels.Count ? labels[index] : index.ToString(CultureInfo.InvariantCulture);

    // A file as the table shows it: relative to the folder it was found in, shortened from the front.
    private static string Shown(string file, IReadOnlyList<string> inputs)
    {
        string? folder = inputs.FirstOrDefault(i => Directory.Exists(i) && Path.GetFullPath(file).StartsWith(Path.GetFullPath(i), StringComparison.Ordinal));
        return RowFiles.CellEnd(folder is null ? Path.GetFileName(file) : Path.GetRelativePath(folder, file));
    }

    private static string Percent(double p) => (p * 100).ToString("0.0", CultureInfo.InvariantCulture) + "%";

    private static string Coordinate(float v) => v.ToString("0.0", CultureInfo.InvariantCulture);

    private static void Print(CommandContext context, ImageModel model, Result[] results, int top, IReadOnlyList<string> inputs)
    {
        string input = model.InputShape is { } shape ? $"[{string.Join(", ", shape)}]" : "sized by each image";
        context.Write($"Model     {model.Architecture} · {Task(model.Task)}{(model.Labels is { } l ? $", {l.Count} classes" : "")} · input {input} · {context.Device}");
        switch (model.Task)
        {
            case ImageTask.Classification:
                context.Table(top > 1 ? ["File", "Class", "Probability", "Top"] : ["File", "Class", "Probability"], results.Select(r =>
                {
                    var ranked = Ranked(r.Probabilities!).ToList();
                    IReadOnlyList<string> row = [Shown(r.File, inputs), Label(model, ranked[0].Index), Percent(ranked[0].Probability)];
                    return top > 1 ? [.. row, string.Join(", ", ranked.Take(top).Select(p => $"{Label(model, p.Index)} {Percent(p.Probability)}"))] : row;
                }));
                break;
            case ImageTask.Detection:
                context.Table(["File", "Label", "Score", "X", "Y", "Width", "Height"], results.SelectMany(r => r.Boxes!.Select(d => (IReadOnlyList<string>)
                [
                    Shown(r.File, inputs), d.Label ?? Label(model, d.Class), d.Score.ToString("0.000", CultureInfo.InvariantCulture),
                    Coordinate(d.Box.X), Coordinate(d.Box.Y), Coordinate(d.Box.Width), Coordinate(d.Box.Height),
                ])));
                int boxes = results.Sum(r => r.Boxes!.Count);
                context.Write($"{boxes:N0} box{(boxes == 1 ? "" : "es")} in {results.Length:N0} image{(results.Length == 1 ? "" : "s")}");
                break;
            case ImageTask.Segmentation:
                context.Table(["File", "Class", "Pixels", "Share"], results.SelectMany(r =>
                {
                    var counts = r.Mask!.Counts();
                    double all = Math.Max(1, r.Mask.Width * r.Mask.Height);
                    return counts.Select((n, c) => (n, c)).Where(p => p.n > 0).Select(p => (IReadOnlyList<string>)
                        [Shown(r.File, inputs), Label(model, p.c), p.n.ToString(CultureInfo.InvariantCulture), Percent(p.n / all)]);
                }));
                break;
            default:
                context.Table(["File", "Shape", "Size", "Norm"], results.Select(r => (IReadOnlyList<string>)
                [
                    Shown(r.File, inputs), $"[{string.Join(", ", r.FeatureShape!)}]", r.Features!.Length.ToString(CultureInfo.InvariantCulture),
                    Math.Sqrt(r.Features.Sum(v => (double)v * v)).ToString("0.####", CultureInfo.InvariantCulture),
                ]));
                break;
        }
    }

    private static IEnumerable<(int Index, double Probability)> Ranked(double[] probabilities) =>
        probabilities.Select((p, i) => (i, p)).OrderByDescending(p => p.p).ThenBy(p => p.i);

    // The predictions as rows (-o FILE, one per image or per box) or, nested, one object per image (the JSON document).
    private static IEnumerable<JsonObject> Rows(ImageModel model, Result[] results, int top, bool nested = false)
    {
        foreach (var r in results)
        {
            var row = new JsonObject { ["file"] = r.File };
            switch (model.Task)
            {
                case ImageTask.Classification:
                    var ranked = Ranked(r.Probabilities!).ToList();
                    row["class"] = Label(model, ranked[0].Index);
                    row["probability"] = Math.Round(ranked[0].Probability, 6);
                    if (top > 1 || nested)
                    {
                        row["top"] = new JsonArray([.. ranked.Take(top).Select(p => (JsonNode)new JsonObject { ["class"] = Label(model, p.Index), ["probability"] = Math.Round(p.Probability, 6) })]);
                    }

                    yield return row;
                    break;
                case ImageTask.Detection:
                    var boxes = r.Boxes!.Select(d => new JsonObject
                    {
                        ["label"] = d.Label ?? Label(model, d.Class), ["class"] = d.Class, ["score"] = Math.Round(d.Score, 6),
                        ["x"] = Math.Round(d.Box.X, 3), ["y"] = Math.Round(d.Box.Y, 3), ["width"] = Math.Round(d.Box.Width, 3), ["height"] = Math.Round(d.Box.Height, 3),
                    }).ToList();
                    if (nested)
                    {
                        row["boxes"] = new JsonArray([.. boxes]);
                        yield return row;
                    }
                    else
                    {
                        foreach (var box in boxes)
                        {
                            var flat = new JsonObject { ["file"] = r.File };
                            foreach (var (key, value) in box)
                            {
                                flat[key] = value?.DeepClone();
                            }

                            yield return flat;
                        }
                    }

                    break;
                case ImageTask.Segmentation:
                    var counts = r.Mask!.Counts();
                    row["width"] = r.Mask.Width;
                    row["height"] = r.Mask.Height;
                    row["classes"] = new JsonArray([.. counts.Select((n, c) => (n, c)).Where(p => p.n > 0)
                        .Select(p => (JsonNode)new JsonObject { ["class"] = Label(model, p.c), ["index"] = p.c, ["pixels"] = p.n })]);
                    row["mask"] = r.Written;
                    yield return row;
                    break;
                default:
                    row["shape"] = new JsonArray([.. r.FeatureShape!.Select(d => (JsonNode)d)]);
                    row["size"] = r.Features!.Length;
                    row["npy"] = r.Written;
                    if (nested && r.Written is null)
                    {
                        row["values"] = new JsonArray([.. r.Features.Select(v => (JsonNode)Math.Round(v, 6))]);
                    }

                    yield return row;
                    break;
            }
        }
    }
}
