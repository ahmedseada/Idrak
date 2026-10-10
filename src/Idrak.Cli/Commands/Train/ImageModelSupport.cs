// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// What <c>idrak predict</c> and <c>idrak train</c> share for image models: telling a checkpoint from a package or a
/// builder JSON, loading it through its family (<see cref="ImageModels.Load"/>, the family from a plug-in), a package of
/// <c>idrak train</c> as the same description, the image files of the arguments, and the refusal of a name no registry
/// holds.
/// </summary>
internal static class ImageModelFiles
{
    /// <summary>
    /// Whether <paramref name="path"/> names an image model checkpoint read through its family: a folder, an .onnx or
    /// .safetensors file, or a config.json (not a model package or a builder JSON).
    /// </summary>
    public static bool IsCheckpoint(string path) =>
        Directory.Exists(path)
        || File.Exists(path) && (path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) || path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                                 || Path.GetFileName(path).Equals("config.json", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// The image model at <paramref name="path"/> on the command's device, through the family its checkpoint names (a
    /// plug-in's); with <paramref name="weights"/>, the weights <c>idrak train</c> wrote (.ikw) read over the checkpoint's.
    /// <paramref name="trainable"/> (and weights to read) keeps every layer in float32.
    /// </summary>
    public static ImageModel Load(CommandContext context, string path, string? weights, bool trainable)
    {
        if (!File.Exists(path) && !Directory.Exists(path))
        {
            throw new UsageException($"{path} not found: give a model folder (config.json and safetensors, or an .onnx file), a .safetensors or .onnx file, or a model package (.ikm).");
        }

        if (weights is not null && !File.Exists(weights))
        {
            throw new UsageException($"--weights {weights}: no such file (the .ikw weights idrak train writes).");
        }

        var model = ImageModels.Load(path, new ImageModelOptions
        {
            Device = context.Device,
            Weights = trainable || weights is not null ? EncoderWeights.Float32 : EncoderWeights.AsStored,
        });
        if (weights is not null)
        {
            try
            {
                model.Network.Load(weights);
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        return model;
    }

    /// <summary>
    /// A package <c>idrak train</c> wrote for a detector, a segmenter or a backbone, as an image model: its network, the
    /// resize to its input size with values in [0, 1] (as it was trained), its classes and decoder.
    /// </summary>
    public static ImageModel FromPackage(ModelPackageReader package, JsonObject meta, Device device)
    {
        var builder = package.Network();
        if (builder.InputKind != InputKind.Image)
        {
            throw new InvalidDataException("The package's network does not take images.");
        }

        int channels = builder.InputShape[0], height = builder.InputShape[1], width = builder.InputShape[2];
        var task = (string?)meta["task"] switch
        {
            "detection" => ImageTask.Detection,
            "segmentation" => ImageTask.Segmentation,
            "features" => ImageTask.Features,
            _ => ImageTask.Classification,
        };
        var network = package.BuildNetwork(device: device);
        network.Eval();
        return new ImageModel
        {
            Architecture = (string?)meta["architecture"] ?? network.Name ?? "network",
            Task = task,
            Network = network,
            Preprocessor = Resize(channels, height, width),
            Channels = channels,
            Labels = meta["classes"] is JsonArray classes && classes.Count > 0 ? [.. classes.Select(c => (string)c!)] : null,
            Decoder = (string?)meta["decoder"],
            DecoderSettings = meta["decoderSettings"] as JsonObject,
        };
    }

    /// <summary>A resize to <paramref name="height"/> x <paramref name="width"/>, values in [0, 1], grey kept grey for one channel.</summary>
    public static ImagePreprocessor Resize(int channels, int height, int width) => new()
    {
        Height = height,
        Width = width,
        Normalize = false,
        Grayscale = channels == 1,
        ConvertRgb = channels != 1,
    };

    /// <summary>Whether <paramref name="e"/> is a registry's refusal of a name it does not hold (a family, a decoder, a head).</summary>
    public static bool NotRegistered(Exception e) => e is NotSupportedException && e.Message.Contains(" is not registered", StringComparison.Ordinal);

    /// <summary>Prints a registry's refusal with the remedy on the command line, and gives the failure's exit code.</summary>
    public static int Refuse(CommandContext context, Exception e)
    {
        context.Error(Messages.T("{0} Load its plug-in with -P (--plugin).", e.Message));
        return ExitCodes.Failed;
    }

    /// <summary>
    /// The image files <paramref name="inputs"/> name: files as they are, folders searched (with their subfolders) for the
    /// extensions of the registered codecs, and file names with * or ? matched in their folder; in that order, each once.
    /// </summary>
    public static IReadOnlyList<string> Expand(IEnumerable<string> inputs)
    {
        var files = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (string input in inputs)
        {
            IEnumerable<string> found;
            if (Directory.Exists(input))
            {
                found = Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories).Where(ImageFiles.IsDecoded).Order(StringComparer.Ordinal);
            }
            else if (input.IndexOfAny(['*', '?']) >= 0)
            {
                string folder = Path.GetDirectoryName(input) is { Length: > 0 } d ? d : ".";
                found = Directory.Exists(folder) ? Directory.EnumerateFiles(folder, Path.GetFileName(input)).Order(StringComparer.Ordinal) : [];
            }
            else if (File.Exists(input))
            {
                found = [input];
            }
            else
            {
                throw new UsageException($"Image not found: {input}");
            }

            files.AddRange(found.Where(seen.Add));
        }

        return files;
    }

    /// <summary>Writes a NumPy .npy file of float32 <paramref name="values"/> shaped <paramref name="shape"/> (format 1.0, little-endian, C order).</summary>
    public static void WriteNpy(string path, ReadOnlySpan<float> values, IReadOnlyList<int> shape)
    {
        string dims = shape.Count == 1 ? $"{shape[0]}," : string.Join(", ", shape.Select(d => d.ToString(CultureInfo.InvariantCulture)));
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}";
        int unpadded = 10 + header.Length + 1;                                  // magic, version, length, header, newline
        header = header.PadRight(header.Length + (64 - unpadded % 64) % 64) + "\n";
        using var file = File.Create(path);
        file.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0]);
        file.Write(BitConverter.GetBytes((ushort)header.Length));
        file.Write(Encoding.ASCII.GetBytes(header));
        foreach (float v in values)
        {
            file.Write(BitConverter.GetBytes(v));
        }
    }
}

/// <summary>
/// How many images go through the network at once, measured on the running device: the first batch is one image, whose
/// peak memory sets the rest (as many as fit in half the memory the device reports free); a batch the device has no room
/// for is halved and run again. <c>--batch N</c> sets a ceiling instead of the measurement.
/// </summary>
internal sealed class MeasuredBatches(Device device, int? ceiling)
{
    private int? _fits = ceiling;
    private bool _measured = ceiling is not null;

    /// <summary>Times a batch ran out of memory and was halved.</summary>
    public int Halvings { get; private set; }

    /// <summary>The batch size settled on (after the measurement), or null before it.</summary>
    public int? Size => _measured ? _fits : null;

    /// <summary>Runs <paramref name="count"/> items through <paramref name="run"/> (start, length), in measured batches.</summary>
    public void Run(int count, Action<int, int> run)
    {
        for (int start = 0; start < count;)
        {
            int n = Math.Min(count - start, _measured ? _fits ?? count : 1);
            try
            {
                var backend = device.Backend;
                long before = backend.GetMemoryUsage().InUse;
                backend.ResetPeakMemoryUsage();
                run(start, n);
                if (!_measured)
                {
                    long perItem = Math.Max(1, (backend.GetMemoryUsage().Peak - before) / n);
                    _fits = backend.AvailableMemory() is { } free ? (int)Math.Clamp(free / 2 / perItem, 1, int.MaxValue) : null;
                    _measured = true;
                }

                start += n;
            }
            catch (ResourceLimitExceededException) when (n > 1)
            {
                _fits = n / 2;
                _measured = true;
                Halvings++;
            }
        }
    }
}
