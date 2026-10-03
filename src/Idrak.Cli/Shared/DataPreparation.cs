// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Data;
using TensorData = Idrak.Data.Dataset;

namespace Idrak.Cli.Shared;

/// <summary>
/// Applies a data preparation (<c>prep.json</c>, format <c>idrak-prep/1</c>, written by <c>idrak suggest</c>) to the
/// rows or images it was made for, giving the feature and target tensors a network trains on. The kinds:
/// <list type="bullet">
/// <item><c>table</c>: "features" is a list of <c>{"column", "type": "number", "fill", "mean", "std"}</c> (missing cells
/// take the fill value, then the value is standard-scaled) and <c>{"column", "type": "onehot", "values"}</c> (one 0/1
/// feature per listed value; other and missing values are all zeros).</item>
/// <item><c>text</c>: "column" holds the text; "tokenizer" is <c>{"type": "words", "lowercase", "length",
/// "vocabulary"}</c>: lower-cased runs of letters and digits, id = position in the vocabulary (0 padding, 1 unknown),
/// cut or padded to "length".</item>
/// <item><c>images</c>: "channels", "height", "width": every image converted and resized to that shape, values in [0, 1]
/// (the library's <see cref="ImageFolderSource"/>); "classes" in folder order; "augment" lists what a trainer may add
/// (not applied here).</item>
/// <item><c>language-model</c>: "tokenizer" is <c>{"type": "characters", "length", "vocabulary"}</c>; each conversation
/// is rendered as "role: content" lines, cut into windows of length + 1 characters: the inputs and the next characters.</item>
/// </list>
/// "target" is <c>{"column", "type": "number", "log", "mean", "std"}</c> (the value, its natural logarithm when "log",
/// standard-scaled) or <c>{"column", "type": "classes", "classes"}</c> (one-hot rows). Rows without a target are skipped.
/// "split" is <c>{"validation", "seed"}</c>; "balance": "oversample" repeats the rows of rare classes in the training
/// part until every class has as many as the largest.
/// </summary>
internal static class DataPreparation
{
    /// <summary>The prepared data and its training / validation parts.</summary>
    public sealed record Prepared(TensorData All, TensorData Train, TensorData Validation);

    /// <summary>Prepares <paramref name="profile"/>'s rows or images as <paramref name="prep"/> says, and splits them.</summary>
    public static Prepared Apply(JsonObject prep, DataProfile profile)
    {
        var data = (string?)prep["kind"] switch
        {
            "table" => Table(prep, profile.Rows),
            "text" => Text(prep, profile.Rows),
            "images" => Images(prep, profile.Images),
            "language-model" => LanguageModel(prep, profile.Conversations),
            var other => throw new InvalidDataException($"prep.json: unknown kind '{other}'."),
        };
        if (data.Count < 2)
        {
            throw new InvalidDataException($"Only {data.Count} usable rows: too few to train and validate.");
        }

        double validation = (double?)prep["split"]?["validation"] ?? 0.2;
        int seed = (int?)prep["split"]?["seed"] ?? 1;
        var (train, held) = data.Split(1 - validation, seed, removeDuplicates: false);
        if (held.Count == 0)
        {
            (train, held) = data.Split(0.5, seed, removeDuplicates: false);
        }

        if ((string?)prep["balance"] == "oversample")
        {
            train = Oversample(train);
        }

        return new Prepared(data, train, held);
    }

    /// <summary>The class index of a cell under a "classes" target, or -1.</summary>
    public static int ClassOf(JsonNode? cell, JsonArray classes)
    {
        string? key = Key(cell);
        for (int i = 0; key is not null && i < classes.Count; i++)
        {
            if ((string?)classes[i] == key)
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>A cell as the text the class list and one-hot values use (numbers in invariant form), or null when missing.</summary>
    public static string? Key(JsonNode? cell) => cell switch
    {
        null => null,
        JsonValue v when v.TryGetValue(out string? s) => string.IsNullOrWhiteSpace(s) ? null : s,
        JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False => v.GetValueKind() == JsonValueKind.True ? "true" : "false",
        JsonValue v when v.GetValueKind() == JsonValueKind.Number => ParseNumber(v).ToString(CultureInfo.InvariantCulture),
        _ => cell.ToJsonString(),
    };

    /// <summary>A cell as a number, or null when missing or not a number.</summary>
    public static double? Number(JsonNode? cell) => cell switch
    {
        JsonValue v when v.GetValueKind() == JsonValueKind.Number => ParseNumber(v),
        JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False => v.GetValueKind() == JsonValueKind.True ? 1 : 0,
        JsonValue v when v.TryGetValue(out string? s) && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double p) => p,
        _ => null,
    };

    /// <summary>A JSON number (from a file or made in code) as a double.</summary>
    public static double ParseNumber(JsonValue value) => double.Parse(value.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture);

    /// <summary>The words of a text as the "words" tokenizer splits them.</summary>
    public static IEnumerable<string> Words(string text, bool lowercase = true)
    {
        var sb = new StringBuilder();
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c))
            {
                sb.Append(lowercase ? char.ToLowerInvariant(c) : c);
            }
            else if (sb.Length > 0)
            {
                yield return sb.ToString();
                sb.Clear();
            }
        }

        if (sb.Length > 0)
        {
            yield return sb.ToString();
        }
    }

    private static TensorData Table(JsonObject prep, List<JsonObject> rows)
    {
        var features = prep["features"]?.AsArray() ?? [];
        var names = new List<string>();
        foreach (var f in features.OfType<JsonObject>())
        {
            if ((string?)f["type"] == "onehot")
            {
                names.AddRange(f["values"]!.AsArray().Select(v => $"{(string?)f["column"]}={(string?)v}"));
            }
            else
            {
                names.Add((string)f["column"]!);
            }
        }

        var x = new List<float>();
        var y = new List<float>();
        var target = prep["target"]!.AsObject();
        int count = 0;
        foreach (var row in rows)
        {
            if (!Target(target, row[(string)target["column"]!], y))
            {
                continue;
            }

            foreach (var f in features.OfType<JsonObject>())
            {
                var cell = row[(string)f["column"]!];
                if ((string?)f["type"] == "onehot")
                {
                    string? key = Key(cell);
                    x.AddRange(f["values"]!.AsArray().Select(v => key is not null && (string?)v == key ? 1f : 0f));
                }
                else
                {
                    double value = Number(cell) ?? (double?)f["fill"] ?? 0;
                    double std = (double?)f["std"] ?? 1;
                    x.Add((float)((value - ((double?)f["mean"] ?? 0)) / (std == 0 ? 1 : std)));
                }
            }

            count++;
        }

        return TensorData.FromFlat([.. x], [.. y], count, names, TargetNames(target));
    }

    private static TensorData Text(JsonObject prep, List<JsonObject> rows)
    {
        var tokenizer = prep["tokenizer"]!.AsObject();
        int length = (int)tokenizer["length"]!;
        bool lowercase = (bool?)tokenizer["lowercase"] ?? true;
        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        int id = 0;
        foreach (var word in tokenizer["vocabulary"]!.AsArray())
        {
            vocabulary.TryAdd((string)word!, id++);
        }

        var target = prep["target"]!.AsObject();
        string column = (string)prep["column"]!;
        var x = new List<float>();
        var y = new List<float>();
        int count = 0;
        foreach (var row in rows)
        {
            if (!Target(target, row[(string)target["column"]!], y))
            {
                continue;
            }

            var ids = Words(Key(row[column]) ?? "", lowercase).Take(length).Select(w => vocabulary.TryGetValue(w, out int i) ? i : 1).ToList();
            x.AddRange(ids.Select(i => (float)i));
            x.AddRange(Enumerable.Repeat(0f, length - ids.Count));
            count++;
        }

        return TensorData.FromFlat([.. x], [.. y], count, [.. Enumerable.Range(0, length).Select(i => $"t{i}")], TargetNames(target)).WithFeatureShape(length);
    }

    private static TensorData Images(JsonObject prep, List<ImageItem> images)
    {
        int c = (int)prep["channels"]!, h = (int)prep["height"]!, w = (int)prep["width"]!;
        string[] classes = [.. prep["classes"]!.AsArray().Select(n => (string)n!)];
        var decoded = images.Where(i => ImageFiles.IsDecoded(i.Path)).Select(i => (i.Path, i.Class)).ToList();
        if (decoded.Count == 0)
        {
            throw new InvalidDataException($"None of the images can be decoded here ({string.Join(", ", ImageCodecs.Names)} are read; JPEG needs a registered codec).");
        }

        var source = new ImageFolderSource(decoded, classes, c, h, w);
        return TensorData.FromSource(source, [.. Enumerable.Range(0, c * h * w).Select(i => $"p{i}")], classes);
    }

    private static TensorData LanguageModel(JsonObject prep, List<JsonObject> conversations)
    {
        var tokenizer = prep["tokenizer"]!.AsObject();
        int length = (int)tokenizer["length"]!;
        var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
        int id = 0;
        foreach (var symbol in tokenizer["vocabulary"]!.AsArray())
        {
            vocabulary.TryAdd((string)symbol!, id++);
        }

        var x = new List<float>();
        var y = new List<float>();
        int count = 0;
        foreach (var conversation in conversations)
        {
            var ids = DataProfile.Render(conversation).Select(ch => vocabulary.TryGetValue(ch.ToString(), out int i) ? i : 1).ToList();
            ids.AddRange(Enumerable.Repeat(0, Math.Max(0, length + 1 - ids.Count)));     // short texts padded to one window
            for (int start = 0; start + length + 1 <= ids.Count; start += length)
            {
                x.AddRange(ids.Skip(start).Take(length).Select(i => (float)i));
                y.AddRange(ids.Skip(start + 1).Take(length).Select(i => (float)i));
                count++;
            }
        }

        string[] positions = [.. Enumerable.Range(0, length).Select(i => $"t{i}")];
        return TensorData.FromFlat([.. x], [.. y], count, positions, positions).WithFeatureShape(length);
    }

    // Appends the target of one row; false when it is missing (or not a known class).
    private static bool Target(JsonObject target, JsonNode? cell, List<float> y)
    {
        if ((string?)target["type"] == "classes")
        {
            var classes = target["classes"]!.AsArray();
            int index = ClassOf(cell, classes);
            if (index < 0)
            {
                return false;
            }

            for (int i = 0; i < classes.Count; i++)
            {
                y.Add(i == index ? 1 : 0);
            }

            return true;
        }

        if (Number(cell) is not { } value || (bool?)target["log"] == true && value <= 0)
        {
            return false;
        }

        if ((bool?)target["log"] == true)
        {
            value = Math.Log(value);
        }

        double std = (double?)target["std"] ?? 1;
        y.Add((float)((value - ((double?)target["mean"] ?? 0)) / (std == 0 ? 1 : std)));
        return true;
    }

    private static List<string> TargetNames(JsonObject target) => (string?)target["type"] == "classes"
        ? [.. target["classes"]!.AsArray().Select(n => (string)n!)]
        : [(string)target["column"]!];

    // Rows of rare classes repeated (in order, round robin) until each class has as many rows as the largest.
    private static TensorData Oversample(TensorData data)
    {
        int classes = data.TargetCount;
        if (classes < 2)
        {
            return data;
        }

        var byClass = Enumerable.Range(0, classes).Select(_ => new List<int>()).ToArray();
        for (int i = 0; i < data.Count; i++)
        {
            byClass[data.GetTargets(i).IndexOf(1f) is var c and >= 0 ? c : 0].Add(i);
        }

        int largest = byClass.Max(l => l.Count);
        var indices = new List<int>();
        foreach (var list in byClass.Where(l => l.Count > 0))
        {
            for (int i = 0; i < largest; i++)
            {
                indices.Add(list[i % list.Count]);
            }
        }

        return data.Subset([.. indices]);
    }
}
