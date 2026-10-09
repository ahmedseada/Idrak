// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Nlp;

/// <summary>
/// How a vision-language fine-tune prepares its images, the same for training and evaluation records: the image transform
/// pipeline (<see cref="Transforms"/>, the same text as <c>--image-transform</c>), the family's vision options
/// (<see cref="VisionOptions"/>, as <c>--vision-option</c>; checked against the family's keys with
/// <see cref="ThrowIfUnknown"/>, as generation does) and <see cref="Grayscale"/>. Saved next to the adapters
/// (<see cref="FileName"/>), so <c>run</c> and <c>serve</c> can prepare images as the model was tuned on them. Immutable;
/// equal when the pipeline's text, the options and the grayscale switch are.
/// </summary>
/// <example>
/// <code>
/// var images = TuningImages.Parse("max_width=1024,contrast=1.5", ["do_pan_and_scan=true"], grayscale: true);
/// images.ThrowIfUnknown(model.Vision!.Family, model.Vision.VisionOptionKeys);
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

    /// <summary>The image transforms, in order (registered in <see cref="ImageTransforms"/>).</summary>
    public ImageTransformPipeline Transforms { get; init; } = ImageTransformPipeline.Empty;

    /// <summary>The vision family's own options for every image (its keys; the family's defaults for the others).</summary>
    public VisionOptions VisionOptions { get; init; } = VisionOptions.Empty;

    /// <summary>
    /// Turn images to grayscale first (the <c>grayscale</c> transform before <see cref="Transforms"/>, unless they already
    /// have one), as the command line's <c>--grayscale</c> does.
    /// </summary>
    public bool Grayscale { get; init; }

    /// <summary>Whether this changes nothing: no transforms, no options, colour.</summary>
    public bool IsEmpty => Transforms.IsEmpty && VisionOptions.Count == 0 && !Grayscale;

    /// <summary>The transforms every image goes through: <see cref="Transforms"/>, after <c>grayscale</c> when <see cref="Grayscale"/> asks for it.</summary>
    public ImageTransformPipeline Pipeline =>
        Grayscale && !Transforms.Contains("grayscale") ? ImageTransformPipeline.Parse("grayscale").Then(Transforms) : Transforms;

    /// <summary>The preparation the command line's options describe.</summary>
    /// <param name="transforms">The pipeline as text (<c>--image-transform</c>); null, empty or "none" for none.</param>
    /// <param name="visionOptions">KEY=VALUE items (<c>--vision-option</c>).</param>
    /// <param name="grayscale">Grayscale first (<c>--grayscale</c>).</param>
    /// <exception cref="ArgumentException">An unknown transform or option of one, or a bad value.</exception>
    /// <exception cref="FormatException">A vision option is not KEY=VALUE.</exception>
    public static TuningImages Parse(string? transforms, IEnumerable<string>? visionOptions = null, bool grayscale = false) => new()
    {
        Transforms = ImageTransformPipeline.Parse(transforms),
        VisionOptions = visionOptions is null ? VisionOptions.Empty : VisionOptions.Parse(visionOptions),
        Grayscale = grayscale,
    };

    /// <summary>
    /// Throws unless every vision option is one the family <paramref name="family"/> takes (<paramref name="accepted"/>:
    /// <c>PretrainedVision.VisionOptionKeys</c>), naming them.
    /// </summary>
    /// <exception cref="ArgumentException">An option the family does not take.</exception>
    public void ThrowIfUnknown(string family, IReadOnlyCollection<string> accepted) => VisionOptions.ThrowIfUnknown(family, accepted);

    /// <summary><paramref name="image"/> after <see cref="Pipeline"/> (decoded pixels in and out).</summary>
    public ImageData Apply(ImageData image) => Pipeline.Apply(image);

    /// <summary>
    /// The preparation as JSON: <c>{"format", "image_transforms", "vision_options", "grayscale"}</c> (the transforms as
    /// their text, the options as an object of strings).
    /// </summary>
    public JsonObject ToJson() => new()
    {
        ["format"] = Format,
        ["image_transforms"] = Transforms.ToString(),
        ["vision_options"] = VisionOptions.ToJson(),
        ["grayscale"] = Grayscale,
    };

    /// <summary>The preparation in <paramref name="json"/> (as <see cref="ToJson"/> writes it; missing keys: none).</summary>
    /// <exception cref="InvalidDataException">Another format, or a value of the wrong kind.</exception>
    /// <exception cref="ArgumentException">An unknown transform (the registered ones are named).</exception>
    public static TuningImages FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if (json["format"] is { } format && (string?)format != Format)
        {
            throw new InvalidDataException($"Unknown tuning images format '{format.ToJsonString()}' (expected {Format}).");
        }

        try
        {
            return new TuningImages
            {
                Transforms = ImageTransformPipeline.FromJson(json["image_transforms"]),
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

    /// <summary>Equal when the transforms' text, the vision options and the grayscale switch are.</summary>
    public bool Equals(TuningImages? other) =>
        other is not null && Transforms.Equals(other.Transforms) && Grayscale == other.Grayscale && VisionOptions.ToString() == other.VisionOptions.ToString();

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Transforms, Grayscale, VisionOptions.ToString());

    /// <summary>"transforms grayscale,contrast=1.5; vision options do_pan_and_scan=true", or "images as decoded".</summary>
    public override string ToString() => IsEmpty ? "images as decoded"
        : $"transforms {(Pipeline.IsEmpty ? "none" : Pipeline.ToString())}; vision options {VisionOptions}";
}
