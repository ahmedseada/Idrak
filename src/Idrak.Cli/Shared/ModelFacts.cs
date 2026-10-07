// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

namespace Idrak.Cli.Shared;

/// <summary>
/// What a model is, from its folder without loading it: the config, the architecture's <see cref="DecoderSpec"/> (when
/// the family is registered), the tensors' shapes, the chat template and the license. Used by show and memory.
/// </summary>
internal sealed class ModelFacts
{
    public required JsonObject Config { get; init; }

    public required string Architecture { get; init; }

    public string? ModelType { get; init; }

    public DecoderSpec? Spec { get; init; }

    public bool Loadable => Problem is null;

    public string? Problem { get; init; }

    public required long Parameters { get; init; }

    /// <summary>Each tensor's name and element count.</summary>
    public required IReadOnlyList<(string Name, int[] Shape)> Tensors { get; init; }

    public required string Format { get; init; }

    public required long WeightBytes { get; init; }

    public string Rope { get; init; } = "none";

    public string Windows { get; init; } = "none";

    public string? Template { get; init; }

    public string? TemplateSource { get; init; }

    public string? ToolCallFormat { get; init; }

    public string? License { get; init; }

    public required IReadOnlyList<(string Name, long Bytes)> Files { get; init; }

    public required IReadOnlyList<string> Notes { get; init; }

    /// <summary>Reads what <paramref name="local"/> holds.</summary>
    public static ModelFacts Read(ModelCache.Local local)
    {
        string folder = local.Folder;
        string configPath = Path.Combine(folder, "config.json");
        if (!File.Exists(configPath))
        {
            throw new InvalidOperationException($"{folder} has no config.json; it is not a model folder in the Hugging Face layout.");
        }

        var config = JsonNode.Parse(File.ReadAllText(configPath)) as JsonObject ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        string architecture = (string?)config["architectures"]?[0] ?? "(none named)";
        var notes = new List<string>();
        DecoderSpec? spec = null;
        string? problem = null;
        try
        {
            spec = PretrainedArchitectures.Get(architecture).Spec(config, notes);
        }
        catch (Exception e) when (e is NotSupportedException or KeyNotFoundException or InvalidDataException or InvalidOperationException or NullReferenceException or FormatException)
        {
            problem = e.Message;
        }

        var tensors = new List<(string, int[])>();
        string format;
        long weightBytes;
        if (local.File is { } gguf)
        {
            format = ModelCache.GgufFormat(gguf);
            weightBytes = new FileInfo(gguf).Length;
        }
        else
        {
            format = ModelCache.SafeTensorsFormat(folder);
            weightBytes = Directory.GetFiles(folder, "*.safetensors").Sum(f => new FileInfo(f).Length);
        }

        try
        {
            using var store = ModelCache.OpenTensors(local.File is null ? folder : local.Folder);
            tensors.AddRange(store.Names.Select(n => (n, store.ShapeOf(n))));
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or KeyNotFoundException)
        {
            notes.Add($"weights not readable: {e.Message}");
        }

        string? template = null, templateSource = null, toolCalls = null;
        try
        {
            var tokenizer = File.Exists(Path.Combine(folder, "tokenizer.json")) ? BpeTokenizer.Load(folder) : null;
            if (JinjaChatTemplate.Load(folder, tokenizer) is { } chat)
            {
                template = chat.Source;
                templateSource = File.Exists(Path.Combine(folder, "chat_template.jinja")) ? "chat_template.jinja"
                    : File.Exists(Path.Combine(folder, "chat_template.json")) ? "chat_template.json" : "tokenizer_config.json";
                toolCalls = chat.ToolCallFormatName;
            }
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            notes.Add($"chat template not readable: {e.Message}");
        }

        IEnumerable<string> files = Directory.GetFiles(folder);
        if (local.File is not null)
        {
            files = files.Append(local.File);
        }

        return new ModelFacts
        {
            Config = config,
            Architecture = architecture,
            ModelType = (string?)config["model_type"],
            Spec = spec,
            Problem = problem,
            Parameters = tensors.Sum(t => t.Item2.Aggregate(1L, (a, b) => a * b)),
            Tensors = tensors,
            Format = format,
            WeightBytes = weightBytes,
            Rope = spec is null ? "unknown" : DescribeRope(spec.Rope),
            Windows = spec is null ? "unknown" : DescribeWindows(spec),
            Template = template,
            TemplateSource = templateSource,
            ToolCallFormat = toolCalls,
            License = FindLicense(folder, config, local.File),
            Files = [.. files.Select(f => (Path.GetFileName(f), new FileInfo(f).Length)).OrderBy(f => f.Item1, StringComparer.Ordinal)],
            Notes = notes,
        };
    }

    private static string DescribeRope(RopeSettings? rope)
    {
        if (rope is null)
        {
            return "none";
        }

        string text = $"theta {rope.Theta.ToString("G", CultureInfo.InvariantCulture)}";
        if (rope.RotaryDim is { } dim)
        {
            text += $", {dim} rotary dimensions";
        }

        if (rope.Scaling is { } scaling)
        {
            var parameters = scaling.Parameters;
            text += $", scaling {scaling.Type}" + (parameters["factor"] is { } factor ? $" (factor {factor.ToJsonString()})" : "");
        }

        return text;
    }

    private static string DescribeWindows(DecoderSpec spec)
    {
        if (spec.SlidingWindow is not { } window)
        {
            return "none (full attention in every layer)";
        }

        int windowed = spec.SlidingWindowLayers?.Count(w => w) ?? spec.Layers;
        return $"{window:N0} positions on {windowed} of {spec.Layers} layers";
    }

    // The license from the model card's front matter (README.md "license:"), the GGUF metadata, or the config.
    private static string? FindLicense(string folder, JsonObject config, string? gguf)
    {
        string readme = Path.Combine(folder, "README.md");
        if (File.Exists(readme))
        {
            var lines = File.ReadLines(readme).Take(80).ToList();
            if (lines.Count > 0 && lines[0].Trim() == "---")
            {
                foreach (string line in lines.Skip(1).TakeWhile(l => l.Trim() != "---"))
                {
                    if (line.StartsWith("license:", StringComparison.OrdinalIgnoreCase))
                    {
                        return line["license:".Length..].Trim().Trim('"', '\'');
                    }
                }
            }
        }

        if (gguf is not null)
        {
            try
            {
                using var file = GgufFile.Open(gguf);
                if (file.Get("general.license", "") is { Length: > 0 } license)
                {
                    return license;
                }
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException)
            {
            }
        }

        return (string?)config["license"];
    }
}
