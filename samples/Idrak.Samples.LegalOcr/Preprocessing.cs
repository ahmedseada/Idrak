// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;

namespace Idrak.Samples.LegalOcr;

/// <summary>An input of a preprocessing step as the page shows it: the library's description and the app's suggested value.</summary>
public sealed record TransformInput(string Key, string Label, string Input, double? Min, double? Max, double? Step, IReadOnlyList<string> Choices, bool Optional,
    string? Help, string? Default);

/// <summary>A preprocessing step the page can add: a registered image transform, its inputs and the app's suggested values.</summary>
public sealed record TransformInfo(string Name, string Label, string Summary, IReadOnlyList<TransformInput> Inputs);

/// <summary>A page to preview: the scan (a data URL or base64) or an evaluation page, and the steps.</summary>
public sealed record PreviewRequest(string? Image = null, int? Page = null, string? Preprocessing = null, bool? PanAndScan = null);

/// <summary>
/// The Read tab's preprocessing editor's server side: the registered image transforms with the inputs each describes
/// (the library's <see cref="IImageTransform.Parameters"/>) and the app's suggested values for them
/// (<see cref="LegalOcrSettings.PreprocessingDefaults"/>), and a preview of the steps on a page without a model.
/// </summary>
public sealed class Preprocessing(LegalOcrSettings settings, ILogger<Preprocessing> logger)
{
    /// <summary>The transforms the page can add, in the registry's order, with the app's suggested values.</summary>
    public IReadOnlyList<TransformInfo> Transforms()
    {
        var list = new List<TransformInfo>();
        foreach (string name in ImageTransforms.Names)
        {
            var transform = ImageTransforms.Get(name);
            var defaults = Defaults(name);
            var inputs = transform.Parameters.Select(p => new TransformInput(p.Key, p.Label, p.Input.ToString().ToLowerInvariant(), p.Min, p.Max, p.Step, p.Choices, p.Optional,
                p.Help, defaults.TryGetValue(p.Key, out var value) ? value : null)).ToList();
            list.Add(new TransformInfo(transform.Name, transform.Label, transform.Summary, inputs));
        }

        return list;
    }

    /// <summary>
    /// <paramref name="image"/> after <paramref name="pipeline"/>, as the model's encoder will be given it, with its size,
    /// the time the steps took and, with pan and scan, the crops read beside it.
    /// </summary>
    /// <exception cref="ArgumentException">A step the library refuses (the message names it).</exception>
    public object Preview(ChatImage image, string? pipeline, bool panAndScan)
    {
        var steps = ImageTransformPipeline.Parse(pipeline);
        var original = ChatImageDecoder.Decode(image);
        var clock = Stopwatch.StartNew();
        var seen = steps.Apply(original);
        double seconds = clock.Elapsed.TotalSeconds;
        var crops = panAndScan
            ? (Gemma3PanAndScan.Default with { Enabled = true }).Crops(seen.Height, seen.Width).Select(c => new { top = c.Top, left = c.Left, height = c.Height, width = c.Width }).ToList()
            : null;
        return new
        {
            image = "data:image/png;base64," + Convert.ToBase64String(ImageEncoders.Get("png").Encode(seen)),
            width = seen.Width, height = seen.Height, channels = seen.Channels, originalWidth = original.Width, originalHeight = original.Height,
            preprocessing = steps.ToString(), milliseconds = Math.Round(seconds * 1000, 1), panAndScan, crops,
        };
    }

    // The app's suggested values for a transform, by input key ("" the value), from its setting; none when unset or refused.
    private Dictionary<string, string> Defaults(string name)
    {
        var values = new Dictionary<string, string>(StringComparer.Ordinal);
        if (!settings.PreprocessingDefaults.TryGetValue(name, out var text) || string.IsNullOrWhiteSpace(text))
        {
            return values;
        }

        try
        {
            var step = ImageTransformPipeline.Parse(text).Steps.Single(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (step.Value is { } value)
            {
                values[""] = value;
            }

            foreach (var (key, option) in step.Options)
            {
                values[key] = option;
            }
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            logger.LogWarning("PreprocessingDefaults:{Name} = \"{Text}\" is not one {Name} step: {Message}", name, text, name, ex.Message);
        }

        return values;
    }
}
