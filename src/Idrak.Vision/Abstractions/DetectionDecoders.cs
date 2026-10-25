// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Vision.Abstractions;

/// <summary>What a detection decoder (and its training head) is made for (<see cref="IDetectionDecoderKind"/>).</summary>
/// <param name="InputHeight">The network's input height in pixels (the boxes' coordinate space), or 0 when it follows each image.</param>
/// <param name="InputWidth">The network's input width in pixels, or 0 when it follows each image.</param>
/// <param name="Labels">The class names in output order, or null when unknown.</param>
/// <param name="Settings">
/// The decoder's settings (strides, box layout, ...), each a key of the kind's <see cref="IConfigurableKind.Keys"/>: from
/// the model family's config, or the app's; empty when none.
/// </param>
public sealed record DetectionDecoderRequest(int InputHeight, int InputWidth, IReadOnlyList<string>? Labels, JsonObject Settings)
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

/// <summary>
/// A kind of detection decoder, registered in <see cref="DetectionDecoders"/> under its <see cref="IKind.Name"/>: makes
/// the <see cref="DetectionDecoder"/> of one detector (its input size, classes and settings, the settings' keys declared
/// as <see cref="IConfigurableKind.Keys"/>). A decoder whose detector can be trained implements
/// <see cref="IDetectionHeadKind"/> too.
/// </summary>
public interface IDetectionDecoderKind : IKind<DetectionDecoderRequest, DetectionDecoder>;

/// <summary>
/// A detection decoder kind with its training side (<see cref="DetectionHead"/>): the network's outputs read as the
/// candidates its decoder turns into detections, but as tensors that carry gradients. Decoder and head are one kind, so
/// the two read the outputs the same way.
/// </summary>
public interface IDetectionHeadKind : IDetectionDecoderKind
{
    /// <summary>The head for one detector (its input size, classes and settings, as its decoder gets them).</summary>
    DetectionHead CreateHead(DetectionDecoderRequest request);
}

/// <summary>
/// The detection decoders, by name: how a detection network's outputs become boxes (Idrak.Vision's <c>ModelDetector</c>
/// takes one by name, an image model family names its own), and, for a kind that has one (<see cref="IDetectionHeadKind"/>),
/// how its detector trains (<see cref="CreateHead"/>). A decoder is part of a model family (its anchors or grid, its box
/// encoding, how its scores are stored), so it is registered by the plug-in or app that brings the family; the library
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

    private static readonly KindTable<IDetectionDecoderKind> Registry = BuiltIn();

    private static KindTable<IDetectionDecoderKind> BuiltIn()
    {
        var table = new KindTable<IDetectionDecoderKind>(nameof(DetectionDecoders), "Detection decoder",
            "register it with DetectionDecoders.Register (a plug-in or the app that brings the model family; the library registers only generic ones)", Guard,
            StringComparer.Ordinal);
        table.RegisterDefault(new BoxesScoresDecoder());
        return table;
    }

    /// <summary>Registers <paramref name="kind"/> under its name; under a library name it replaces the library's, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDetectionDecoderKind kind) => Registry.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's decoder <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered decoder names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The decoder kind <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered; the message names <see cref="Register"/>.</exception>
    public static IDetectionDecoderKind Get(ReadOnlySpan<char> name) => Registry.Get(name);

    /// <summary>The decoder kind <paramref name="name"/>, or null.</summary>
    public static IDetectionDecoderKind? Find(ReadOnlySpan<char> name) => Registry.Find(name);

    /// <summary>The library's decoder <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDetectionDecoderKind? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the decoder <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's decoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the detections are compared). A head is never guarded: it
    /// is the app's kind's own.
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The decoder <paramref name="name"/> made for <paramref name="request"/>.</summary>
    /// <exception cref="ArgumentException">A setting the kind does not take (the message names it and the kind's keys).</exception>
    public static DetectionDecoder Create(string name, DetectionDecoderRequest request)
    {
        var kind = Checked(name, request);
        return kind.Create(request) ?? throw new InvalidOperationException($"Detection decoder '{name}' made no decoder.");
    }

    /// <summary>The training head of the decoder <paramref name="name"/> made for <paramref name="request"/>.</summary>
    /// <exception cref="NotSupportedException">No decoder of that name is registered, or it has no head (it is not an <see cref="IDetectionHeadKind"/>).</exception>
    public static DetectionHead CreateHead(string name, DetectionDecoderRequest request)
    {
        var kind = Checked(name, request) as IDetectionHeadKind
            ?? throw new NotSupportedException($"Detection decoder '{name}' has no training head (it is not an IDetectionHeadKind): its detector cannot be trained here; "
                + "register a kind with its head under that name with DetectionDecoders.Register (the plug-in or app that brings the model family).");
        return kind.CreateHead(request) ?? throw new InvalidOperationException($"Detection decoder '{name}' made no head.");
    }

    /// <summary>Whether the decoder <paramref name="name"/> is registered and has a training head.</summary>
    public static bool Trainable(ReadOnlySpan<char> name) => Registry.Find(name) is IDetectionHeadKind;

    // The kind, with the request's settings checked against its keys.
    private static IDetectionDecoderKind Checked(string name, DetectionDecoderRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var kind = Get(name);
        foreach (var (key, _) in request.Settings)
        {
            if (!kind.Keys.Contains(key))
            {
                throw new ArgumentException($"Detection decoder '{name}' does not take the setting '{key}'; it takes: {(kind.Keys.Count == 0 ? "none" : string.Join(", ", kind.Keys))}.",
                    nameof(request));
            }
        }

        return kind;
    }

    // An app's decoder under its policy, per call: the outputs are copied once (a span cannot be held), both decoders read the copy.
    private static IDetectionDecoderKind Guard(Slot slot, IDetectionDecoderKind app, IDetectionDecoderKind library) =>
        app is IDetectionHeadKind head ? new GuardedHeadKind(slot, head, library) : new GuardedKind(slot, app, library);

    private class GuardedKind(Slot slot, IDetectionDecoderKind app, IDetectionDecoderKind library) : IDetectionDecoderKind
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public DetectionDecoder Create(DetectionDecoderRequest request)
        {
            DetectionDecoder? own = null, theirs = null;
            return (outputs, shape) =>
            {
                var copy = outputs.ToArray();
                return slot.Call<IReadOnlyList<Detection>>(() => [.. (own ??= app.Create(request))(copy, shape)], () => [.. (theirs ??= library.Create(request))(copy, shape)],
                    Compare);
            };
        }
    }

    // The head is the app's own: a head and a decoder from two registrations would read the outputs differently.
    private sealed class GuardedHeadKind(Slot slot, IDetectionHeadKind own, IDetectionDecoderKind fallback) : GuardedKind(slot, own, fallback), IDetectionHeadKind
    {
        public DetectionHead CreateHead(DetectionDecoderRequest request) => Head.CreateHead(request);

        private IDetectionHeadKind Head { get; } = own;
    }

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
internal sealed class BoxesScoresDecoder : IDetectionDecoderKind
{
    public string Name => DetectionDecoders.BoxesScores;

    public string Summary => "outputs that are boxes and scores already: [N, 4 + C] rows (layout, box, normalized, scores)";

    public IReadOnlySet<string> Keys { get; } = new OptionKeys("layout", "box", "normalized", "scores");

    public DetectionDecoder Create(DetectionDecoderRequest context)
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
        int kind = box switch { "xyxy" => 0, "xywh" => 1, _ => 2 };      // decided once, not per candidate
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

            // [4 + C, N]: each class's scores are a row, so the best class of every candidate is found row by row (in
            // memory order, the same strict comparison in class order), not by striding down a column per candidate.
            float[]? bestScores = null;
            int[]? bestClasses = null;
            if (columns && rows > 0)
            {
                bestScores = outputs.Slice(4 * rows, rows).ToArray();
                bestClasses = new int[rows];
                for (int c = 1; c < classes; c++)
                {
                    var scores = outputs.Slice((4 + c) * rows, rows);
                    for (int r = 0; r < rows; r++)
                    {
                        if (scores[r] > bestScores[r])
                        {
                            (bestClasses[r], bestScores[r]) = (c, scores[r]);
                        }
                    }
                }
            }

            var found = new List<Detection>(rows);
            for (int r = 0; r < rows; r++)
            {
                int step = columns ? rows : 1, first = columns ? r : r * width;     // value j of row r is outputs[first + j * step]
                int best = 0;
                float score;
                if (bestScores is not null)
                {
                    (best, score) = (bestClasses![r], bestScores[r]);
                }
                else
                {
                    score = outputs[first + 4];
                    for (int c = 1; c < classes; c++)
                    {
                        float s = outputs[first + 4 + c];
                        if (s > score)
                        {
                            (best, score) = (c, s);
                        }
                    }
                }

                if (logits)
                {
                    score = 1f / (1f + MathF.Exp(-score));
                }

                float a = outputs[first] * sx, b = outputs[first + step] * sy, c2 = outputs[first + 2 * step] * sx, d = outputs[first + 3 * step] * sy;
                var rectangle = kind switch
                {
                    0 => BoundingBox.FromCorners(a, b, c2, d),
                    1 => new BoundingBox(a, b, c2, d),
                    _ => BoundingBox.FromCenter(a, b, c2, d),
                };
                found.Add(new Detection(rectangle, best, score));
            }

            return found;
        };
    }
}
