// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;
using Idrak.Onnx;

namespace Idrak.Models;

/// <summary>
/// Loads image models (classifiers, detectors, segmenters, backbones) through their registered family
/// (<see cref="ImageModelFamilies"/>). Idrak.Vision turns the result into a predictor ready for images
/// (<c>model.Classifier()</c>, <c>model.Detector()</c>, <c>model.Segmenter()</c>).
/// </summary>
/// <remarks>
/// <code>
/// ImageModelFamilies.Register(new MyResNetFamily());                // a plug-in or the app brings the family
/// using var model = ImageModels.Load("models/resnet-tiny");         // config.json + model.safetensors (+ preprocessor_config.json)
/// var classifier = model.Classifier();                              // Idrak.Vision: images in, classes out
/// </code>
/// </remarks>
public static class ImageModels
{
    /// <summary>
    /// Reads the image model at <paramref name="path"/>: a model folder (config.json and safetensors weights, or any folder a
    /// registered <see cref="CheckpointFormats">checkpoint format</see> reads), a .safetensors file in such a folder, an .onnx
    /// file, or a folder holding one .onnx file and no safetensors. The architecture is config.json's ("architectures") or,
    /// for an ONNX file without one, its "architecture" metadata; the family registered under that name reads the rest.
    /// Tensors are read one at a time as the family makes its layers, so the checkpoint is never held whole in memory.
    /// </summary>
    /// <exception cref="NotSupportedException">No family is registered under the architecture; the message names <see cref="ImageModelFamilies"/>.</exception>
    /// <exception cref="InvalidDataException">The checkpoint names no architecture, or its family cannot read it.</exception>
    public static ImageModel Load(string path, ImageModelOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        options ??= new ImageModelOptions();
        if (OnnxFile(path) is { } onnx)
        {
            return LoadOnnx(path, onnx, options);
        }

        string target = File.Exists(path) && (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || Path.GetFileName(path) == "config.json")
            ? Path.GetDirectoryName(Path.GetFullPath(path))! : path;
        var format = CheckpointFormats.For(target);
        string folder = format.Prepare(target);
        var config = ReadConfig(folder) ?? throw new InvalidDataException($"{folder} has no config.json (it names the image model's architecture).");
        string architecture = options.Architecture ?? ArchitectureOf(config)
            ?? throw new InvalidDataException($"{Path.Combine(folder, "config.json")} names no architecture (\"architectures\"); pass ImageModelOptions.Architecture.");
        var family = ImageModelFamilies.Get(architecture);                    // refused before any weight is opened
        var notes = new List<string>(format.Notes(folder));
        using var tensors = format.Open(folder);
        var checkpoint = new ImageCheckpoint(architecture, config, path, folder, tensors, null, options, notes);
        return Finish(family.Read(checkpoint), checkpoint);
    }

    private static ImageModel LoadOnnx(string path, string file, ImageModelOptions options)
    {
        string folder = Path.GetDirectoryName(Path.GetFullPath(file))!;
        var config = ReadConfig(folder);
        ImportedNetwork? imported = null;
        bool taken = false;
        try
        {
            string? architecture = options.Architecture ?? (config is null ? null : ArchitectureOf(config));
            if (architecture is null || config is null)
            {
                // The file's own metadata: "architecture", and "config" (a config.json's text) when no config.json came with it.
                imported = OnnxImport.Load(file, options.Device);
                if (config is null && imported.Metadata.TryGetValue("config", out var text))
                {
                    config = JsonNode.Parse(text) as JsonObject ?? throw new InvalidDataException($"{file}'s \"config\" metadata is not a JSON object.");
                }

                architecture ??= (imported.Metadata.TryGetValue("architecture", out var name) && name.Length > 0 ? name
                    : config is not null ? ArchitectureOf(config) : null)
                    ?? throw new InvalidDataException(
                        $"{file} names no architecture: give a config.json beside it, an \"architecture\" metadata entry, or ImageModelOptions.Architecture.");
            }

            var family = ImageModelFamilies.Get(architecture);
            var notes = new List<string>();
            ImportedNetwork Import()
            {
                if (imported is null)
                {
                    imported = OnnxImport.Load(file, options.Device);
                }

                if (!taken)
                {
                    notes.AddRange(imported.Notes);
                    taken = true;                                               // the family's network now: the model disposes it
                }

                return imported;
            }

            var checkpoint = new ImageCheckpoint(architecture, config ?? [], path, folder, null, Import, options, notes);
            return Finish(family.Read(checkpoint), checkpoint);
        }
        finally
        {
            if (!taken)
            {
                imported?.Dispose();
            }
        }
    }

    // Checks what the family described and adds the checkpoint's notes.
    private static ImageModel Finish(ImageModel model, ImageCheckpoint checkpoint)
    {
        if (model is null)
        {
            throw new InvalidOperationException($"Image model family '{checkpoint.Architecture}' read {checkpoint.Path} as nothing.");
        }

        string? problem = model.Network is null ? "no network"
            : model.Preprocessor is null ? "no preprocessing"
            : model.Channels <= 0 ? $"{model.Channels} input channels"
            : model.Task == ImageTask.Detection && string.IsNullOrEmpty(model.Decoder) ? "a detector without a decoder name"
            : model.Labels is { Count: 0 } ? "an empty list of labels"
            : !Enum.IsDefined(model.Task) ? $"task {model.Task}"
            : null;
        if (problem is not null)
        {
            model.Network?.Dispose();
            throw new InvalidOperationException($"Image model family '{checkpoint.Architecture}' described {checkpoint.Path} with {problem}.");
        }

        model.Network!.Eval();
        return model with { Notes = [.. checkpoint.Notes, .. model.Notes.Where(n => !checkpoint.Notes.Contains(n))] };
    }

    // The .onnx file a path names: the file itself, or a folder's only .onnx file (model.onnx first) when it has no safetensors.
    private static string? OnnxFile(string path)
    {
        if (File.Exists(path))
        {
            return path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) ? path : null;
        }

        if (!Directory.Exists(path) || Directory.EnumerateFiles(path, "*.safetensors").Any() || File.Exists(Path.Combine(path, "model.safetensors.index.json")))
        {
            return null;
        }

        string preferred = Path.Combine(path, "model.onnx");
        return File.Exists(preferred) ? preferred : Directory.GetFiles(path, "*.onnx") is [var only] ? only : null;
    }

    private static JsonObject? ReadConfig(string folder)
    {
        string file = Path.Combine(folder, "config.json");
        return File.Exists(file)
            ? JsonNode.Parse(File.ReadAllText(file)) as JsonObject ?? throw new InvalidDataException($"{file} is not a JSON object.")
            : null;
    }

    private static string? ArchitectureOf(JsonObject config) =>
        config["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : (string?)config["architecture"];
}
