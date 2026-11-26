// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Vision.Abstractions;

/// <summary>What a detection decoder (and its training head) is made for (<c>IDetectionDecoderKind</c>).</summary>
/// <param name="InputHeight">The network's input height in pixels (the boxes' coordinate space), or 0 when it follows each image.</param>
/// <param name="InputWidth">The network's input width in pixels, or 0 when it follows each image.</param>
/// <param name="Labels">The class names in output order, or null when unknown.</param>
/// <param name="Settings">
/// The decoder's settings (strides, box layout, ...), each a key the kind reads: from
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
/// The detection decoders: how a detection network's outputs become boxes (an image model family names its own), and,
/// for a kind that has one (<c>IDetectionHeadKind</c>), how its detector trains. A decoder is part of a model family (its
/// anchors or grid, its box encoding, how its scores are stored), so it comes with the plug-in or app that brings the
/// family; the library has only <see cref="BoxesScores"/>, for networks whose outputs are already boxes and scores.
/// Picking one by its name is not on the new design yet.
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
}

// "boxes-scores": rows of 4 box coordinates and C class scores, read as they are.
internal sealed class BoxesScoresDecoder
{
    public string Name => DetectionDecoders.BoxesScores;

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
                    score = 1f / (1f + PortableMathF.Exp(-score));
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
