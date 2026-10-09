// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Vision.Abstractions;

/// <summary>What a detection decoder is made for (<see cref="DetectionDecoderFactory"/>).</summary>
/// <param name="InputHeight">The network's input height in pixels (the boxes' coordinate space), or 0 when it follows each image.</param>
/// <param name="InputWidth">The network's input width in pixels, or 0 when it follows each image.</param>
/// <param name="Labels">The class names in output order, or null when unknown.</param>
/// <param name="Settings">The decoder's settings (strides, box layout, ...): from the model family's config, or the app's; empty when none.</param>
public sealed record DetectionDecoderContext(int InputHeight, int InputWidth, IReadOnlyList<string>? Labels, JsonObject Settings)
{
    /// <summary>The number of classes, or 0 when the labels are unknown.</summary>
    public int Classes => Labels?.Count ?? 0;

    /// <summary>The setting <paramref name="name"/> as text, or <paramref name="otherwise"/>.</summary>
    public string Text(string name, string otherwise) => Settings[name] is JsonValue v && v.TryGetValue(out string? s) ? s : otherwise;

    /// <summary>The setting <paramref name="name"/> as a number, or <paramref name="otherwise"/>.</summary>
    public float Number(string name, float otherwise) => Settings[name] is JsonValue v && v.TryGetValue(out double d) ? (float)d : otherwise;

    /// <summary>The setting <paramref name="name"/> as a flag, or <paramref name="otherwise"/>.</summary>
    public bool Flag(string name, bool otherwise) => Settings[name] is JsonValue v && v.TryGetValue(out bool b) ? b : otherwise;

    /// <summary>The setting <paramref name="name"/> as a list of whole numbers (strides), or null when it is not given.</summary>
    public int[]? Integers(string name) => Settings[name] is JsonArray a ? [.. a.Select(v => (int)v!)] : null;
}

/// <summary>Makes the <see cref="DetectionDecoder"/> for one detector (its input size, classes and settings).</summary>
public delegate DetectionDecoder DetectionDecoderFactory(DetectionDecoderContext context);

/// <summary>
/// The detection decoders, by name: how a detection network's outputs become boxes (Idrak.Vision's <c>ModelDetector</c>
/// takes one by name, an image model family names its own). A decoder is part of a model family (its anchors or grid, its
/// box encoding, how its scores are stored), so it is registered by the plug-in or app that brings the family; the library
/// registers only <see cref="BoxesScores"/>, for networks whose outputs are already boxes and scores. An unregistered name
/// is refused, naming this registry.
/// </summary>
public static class DetectionDecoders
{
    /// <summary>
    /// "boxes-scores": the network outputs one row per candidate, 4 box coordinates then one score per class ([N, 4 + C];
    /// settings: "layout" "rows" (the default) or "columns" for [4 + C, N]; "box" "xyxy" (the default), "xywh" or
    /// "cxcywh"; "normalized" true when coordinates are fractions of the input's width and height; "scores"
    /// "probabilities" (the default) or "logits", which a sigmoid turns into probabilities). Each row gives one detection:
    /// its best class and that class's score.
    /// </summary>
    public const string BoxesScores = "boxes-scores";

    private static readonly SlotTable<string, DetectionDecoderFactory> Registry = BuiltIn();

    private static SlotTable<string, DetectionDecoderFactory> BuiltIn()
    {
        var table = new SlotTable<string, DetectionDecoderFactory>(nameof(DetectionDecoders), Guard, StringComparer.Ordinal);
        table.RegisterDefault(BoxesScores, BoxesScoresDecoder.Create);
        return table;
    }

    /// <summary>Registers the decoder <paramref name="name"/>; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, DetectionDecoderFactory factory)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(factory);
        Registry.Register(name, factory, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's decoder <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered decoder names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The decoder <paramref name="name"/>'s factory; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered; the message names <see cref="Register"/>.</exception>
    public static DetectionDecoderFactory Get(string name) =>
        Registry.TryGet(name, out var factory) ? factory
            : throw new NotSupportedException($"Detection decoder '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); register it with "
                + "DetectionDecoders.Register (a plug-in or the app that brings the model family; the library registers only generic ones).");

    /// <summary>The library's decoder <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static DetectionDecoderFactory? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the decoder <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's decoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the detections are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The decoder <paramref name="name"/> made for <paramref name="context"/>.</summary>
    public static DetectionDecoder Create(string name, DetectionDecoderContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return Get(name)(context) ?? throw new InvalidOperationException($"Detection decoder '{name}' made no decoder.");
    }

    // An app's decoder under its policy, per call: the outputs are copied once (a span cannot be held), both decoders read the copy.
    private static DetectionDecoderFactory Guard(Slot slot, DetectionDecoderFactory app, DetectionDecoderFactory library) => context =>
    {
        DetectionDecoder? own = null, theirs = null;
        return (outputs, shape) =>
        {
            var copy = outputs.ToArray();
            return slot.Call<IReadOnlyList<Detection>>(() => [.. (own ??= app(context))(copy, shape)], () => [.. (theirs ??= library(context))(copy, shape)], Compare);
        };
    };

    private static string? Compare(IReadOnlyList<Detection> a, IReadOnlyList<Detection> b)
    {
        if (a.Count != b.Count)
        {
            return $"{a.Count} and {b.Count} detections";
        }

        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Class != b[i].Class || Math.Abs(a[i].Score - b[i].Score) > 1e-4f || a[i].Box.IntersectionOverUnion(b[i].Box) < 0.999f)
            {
                return $"detection {i}: {a[i]} and {b[i]}";
            }
        }

        return null;
    }
}

// "boxes-scores": rows of 4 box coordinates and C class scores, read as they are.
internal static class BoxesScoresDecoder
{
    public static DetectionDecoder Create(DetectionDecoderContext context)
    {
        bool columns = context.Text("layout", "rows") switch
        {
            "rows" => false,
            "columns" => true,
            var other => throw new NotSupportedException($"boxes-scores layout '{other}': \"rows\" ([N, 4 + C]) or \"columns\" ([4 + C, N])."),
        };
        string box = context.Text("box", "xyxy");
        if (box is not ("xyxy" or "xywh" or "cxcywh"))
        {
            throw new NotSupportedException($"boxes-scores box '{box}': \"xyxy\", \"xywh\" or \"cxcywh\".");
        }

        bool logits = context.Text("scores", "probabilities") switch
        {
            "probabilities" => false,
            "logits" => true,
            var other => throw new NotSupportedException($"boxes-scores scores '{other}': \"probabilities\" or \"logits\"."),
        };
        bool normalized = context.Flag("normalized", false);
        float sx = normalized ? context.InputWidth : 1, sy = normalized ? context.InputHeight : 1;
        if (normalized && (sx <= 0 || sy <= 0))
        {
            throw new NotSupportedException("boxes-scores with normalized coordinates needs the network's input size.");
        }

        return (outputs, shape) =>
        {
            if (shape.Count != 2)
            {
                throw new ArgumentException($"boxes-scores reads [N, 4 + C] (or [4 + C, N]) outputs per image, not [{string.Join(", ", shape)}].");
            }

            int rows = columns ? shape[1] : shape[0], width = columns ? shape[0] : shape[1], classes = width - 4;
            if (classes < 1 || (context.Classes > 0 && classes != context.Classes))
            {
                throw new ArgumentException($"boxes-scores: {width} values per candidate for {(context.Classes > 0 ? context.Classes : "at least one")} classes (4 + classes).");
            }

            var found = new List<Detection>(rows);
            for (int r = 0; r < rows; r++)
            {
                int step = columns ? rows : 1, first = columns ? r : r * width;     // value j of row r is outputs[first + j * step]
                int best = 0;
                float score = outputs[first + 4 * step];
                for (int c = 1; c < classes; c++)
                {
                    float s = outputs[first + (4 + c) * step];
                    if (s > score)
                    {
                        (best, score) = (c, s);
                    }
                }

                if (logits)
                {
                    score = 1f / (1f + MathF.Exp(-score));
                }

                float a = outputs[first] * sx, b = outputs[first + step] * sy, c2 = outputs[first + 2 * step] * sx, d = outputs[first + 3 * step] * sy;
                var rectangle = box switch
                {
                    "xyxy" => BoundingBox.FromCorners(a, b, c2, d),
                    "xywh" => new BoundingBox(a, b, c2, d),
                    _ => BoundingBox.FromCenter(a, b, c2, d),
                };
                found.Add(new Detection(rectangle, best, score));
            }

            return found;
        };
    }
}
