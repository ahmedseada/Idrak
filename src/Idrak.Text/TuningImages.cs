// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Text;

/// <summary>
/// How a vision-language fine-tune prepares its images, the same for training and evaluation records: the image transform
/// pipeline (<c>Transforms</c>, the same text as <c>--image-transform</c>), the family's vision options
/// (<see cref="VisionOptions"/>, as <c>--vision-option</c>; checked against the family's keys with
/// <see cref="ThrowIfUnknown"/>, as generation does) and <see cref="Grayscale"/>. Saved next to the adapters
/// (<see cref="FileName"/>), so <c>run</c> and <c>serve</c> can prepare images as the model was tuned on them. Immutable;
/// equal when the pipeline's text, the options and the grayscale switch are.
/// </summary>
/// <example>
/// <code>
/// var images = TuningImages.Parse("max_width=1024,contrast=1.5", ["do_pan_and_scan=true"], grayscale: true);
/// images.Save(adapterFolder);                       // tuning_images.json
/// var same = TuningImages.Read(adapterFolder);      // what run and serve apply
/// </code>
/// </example>
public sealed record TuningImages
{
    /// <summary>The file's name in the adapter folder.</summary>
    public const string FileName = "tuning_images.json";

    private const string Format = "idrak-tuning-images/1";

    /// <summary>No transforms, no options, colour: images as decoded, the family's defaults.</summary>
    public static TuningImages None { get; } = new();

    /// <summary>The vision family's own options for every image (its keys; the family's defaults for the others).</summary>
    public VisionOptions VisionOptions { get; init; } = VisionOptions.Empty;

    /// <summary>
    /// Turn images to grayscale first (<see cref="Idrak.Imaging.ChatImageDecoder.Grayscale(ImageData)"/>), as the command
    /// line's <c>--grayscale</c> does.
    /// </summary>
    public bool Grayscale { get; init; }

    /// <summary>
    /// The vision family the preparation was saved for (<c>PretrainedVision.Family</c>; the tuner records it), or null when
    /// not recorded. Its vision options are that family's, so an adapter folder's preparation applies to a model of that
    /// family only (<see cref="ThrowIfOtherFamily"/>). Not part of equality: it names whose options they are, not how images
    /// are prepared.
    /// </summary>
    public string? Family { get; init; }

    /// <summary>Whether this changes nothing: no options, colour.</summary>
    public bool IsEmpty => VisionOptions.Count == 0 && !Grayscale;

    // Naming image transforms as a preparation's steps is not on the new design yet. Null, empty or "none" is no
    // transform.
    private static void NoTransforms(string? text)
    {
        if (!string.IsNullOrWhiteSpace(text) && !text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"The image transforms '{text}' by name are not on the new design yet.");
        }
    }

    /// <summary>The preparation the command line's options describe.</summary>
    /// <param name="transforms">The image transforms as text (<c>--image-transform</c>); null, empty or "none" for none (others are not on the new design yet).</param>
    /// <param name="visionOptions">KEY=VALUE items (<c>--vision-option</c>).</param>
    /// <param name="grayscale">Grayscale first (<c>--grayscale</c>).</param>
    /// <exception cref="NotSupportedException">Image transforms: by name, they are not on the new design yet.</exception>
    /// <exception cref="FormatException">A vision option is not KEY=VALUE.</exception>
    public static TuningImages Parse(string? transforms, IEnumerable<string>? visionOptions = null, bool grayscale = false)
    {
        NoTransforms(transforms);
        return new()
        {
            VisionOptions = visionOptions is null ? VisionOptions.Empty : VisionOptions.Parse(visionOptions),
            Grayscale = grayscale,
        };
    }

    /// <summary>
    /// Throws unless every vision option is one the family <paramref name="family"/> takes (<paramref name="accepted"/>),
    /// naming them.
    /// </summary>
    /// <exception cref="ArgumentException">An option the family does not take.</exception>
    public void ThrowIfUnknown(string family, IReadOnlyCollection<string> accepted) => VisionOptions.ThrowIfUnknown(family, accepted);

    /// <summary>
    /// Throws unless the preparation was saved for the vision family <paramref name="family"/> (the model's
    /// <c>PretrainedVision.Family</c>; null for a text-only model) or names none (<see cref="Family"/> null).
    /// </summary>
    /// <param name="family">The model's vision family, or null for a text-only model.</param>
    /// <param name="where">What the preparation is (an adapter folder), for the message.</param>
    /// <exception cref="InvalidOperationException">Another family: the message names both.</exception>
    public void ThrowIfOtherFamily(string? family, string where = "the preparation")
    {
        if (Family is { } saved && !string.Equals(saved, family, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{where} was tuned on a model of the vision family {saved}, and this model is "
                + (family is null ? "a text-only one" : $"of the vision family {family}") + $": load it on a {saved} model (the base model it was tuned from).");
        }
    }

    /// <summary><paramref name="image"/> turned to grayscale when <see cref="Grayscale"/> asks for it (decoded pixels in and out).</summary>
    public ImageData Apply(ImageData image) => Grayscale ? Idrak.Imaging.ChatImageDecoder.Grayscale(image) : image;

    /// <summary>
    /// The preparation as JSON: <c>{"format", "image_transforms", "vision_options", "grayscale"}</c> (no transforms: "",
    /// the options as an object of strings), and <c>"family"</c> when <see cref="Family"/> is set.
    /// </summary>
    public JsonObject ToJson()
    {
        var json = new JsonObject
        {
            ["format"] = Format,
            ["image_transforms"] = "",
            ["vision_options"] = VisionOptions.ToJson(),
            ["grayscale"] = Grayscale,
        };
        if (Family is not null)
        {
            json["family"] = Family;
        }

        return json;
    }

    /// <summary>The preparation in <paramref name="json"/> (as <see cref="ToJson"/> writes it; missing keys: none).</summary>
    /// <exception cref="InvalidDataException">Another format, or a value of the wrong kind.</exception>
    /// <exception cref="NotSupportedException">Image transforms: by name, they are not on the new design yet.</exception>
    public static TuningImages FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if (json["format"] is { } format && (string?)format != Format)
        {
            throw new InvalidDataException($"Unknown tuning images format '{format.ToJsonString()}' (expected {Format}).");
        }

        try
        {
            NoTransforms(json["image_transforms"] switch
            {
                JsonValue value when value.TryGetValue(out string? text) => text,
                JsonArray { Count: > 0 } steps => steps.ToJsonString(),
                _ => null,
            });
            return new TuningImages
            {
                VisionOptions = json["vision_options"] switch
                {
                    null => VisionOptions.Empty,
                    JsonObject o => VisionOptions.FromJson(o),
                    var other => throw new InvalidDataException($"\"vision_options\" is an object, not {other.ToJsonString()}."),
                },
                Grayscale = json["grayscale"] switch
                {
                    null => false,
                    JsonValue v when v.GetValueKind() is JsonValueKind.True or JsonValueKind.False => v.GetValue<bool>(),
                    var other => throw new InvalidDataException($"\"grayscale\" is true or false, not {other.ToJsonString()}."),
                },
                Family = json["family"] switch
                {
                    null => null,
                    JsonValue v when v.GetValueKind() == JsonValueKind.String => v.GetValue<string>(),
                    var other => throw new InvalidDataException($"\"family\" is a text, not {other.ToJsonString()}."),
                },
            };
        }
        catch (FormatException ex)
        {
            throw new InvalidDataException(ex.Message, ex);
        }
    }

    /// <summary>Whether <paramref name="folder"/> has a saved preparation (<see cref="FileName"/>).</summary>
    public static bool Exists(string folder) => File.Exists(Path.Combine(folder, FileName));

    /// <summary>The preparation saved in <paramref name="folder"/>, or null when it has none.</summary>
    /// <exception cref="InvalidDataException">The file is not a saved preparation; the message names it.</exception>
    public static TuningImages? Read(string folder)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        string path = Path.Combine(folder, FileName);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            return FromJson(JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? throw new InvalidDataException("not a JSON object"));
        }
        catch (Exception ex) when (ex is JsonException or InvalidDataException or ArgumentException)
        {
            throw new InvalidDataException($"{path}: {ex.Message}", ex);
        }
    }

    /// <summary>Writes the preparation into <paramref name="folder"/> (created if needed) as <see cref="FileName"/>.</summary>
    public void Save(string folder)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, FileName), ToJson().ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }));
    }

    /// <summary>Equal when the vision options and the grayscale switch are (<see cref="Family"/> aside).</summary>
    public bool Equals(TuningImages? other) =>
        other is not null && Grayscale == other.Grayscale && VisionOptions.ToString() == other.VisionOptions.ToString();

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Grayscale, VisionOptions.ToString());

    /// <summary>"transforms grayscale,contrast=1.5; vision options do_pan_and_scan=true", or "images as decoded".</summary>
    public override string ToString() => IsEmpty ? "images as decoded"
        : $"grayscale {(Grayscale ? "yes" : "no")}; vision options {VisionOptions}";
}
