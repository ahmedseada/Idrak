// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Data;

/// <summary>
/// One step done to a decoded image before a model's own preprocessing (turn it grey, shrink it, raise its contrast,
/// re-encode it): decoded pixels in, decoded pixels out. Register one with <see cref="ImageTransforms.Register"/>; a
/// pipeline (<see cref="ImageTransformPipeline"/>) names transforms in the order they run. The library's are Pillow's
/// operations, byte for byte (core registers them): <c>grayscale</c>, <c>max_width</c>, <c>max_height</c>,
/// <c>contrast</c>, <c>brightness</c>, <c>sharpness</c>, <c>autocontrast</c>, <c>invert</c>, <c>jpeg</c>. A transform knows nothing of
/// models: a fine-tune that wants its scans prepared a certain way is given a pipeline by the application.
/// </summary>
public interface IImageTransform
{
    /// <summary>The name a pipeline uses (<c>contrast</c>); names compare ignoring case.</summary>
    string Name { get; }

    /// <summary>One line on what it does and what its value is (listed with the registered transforms).</summary>
    string Summary { get; }

    /// <summary>The option keys it takes besides its value (<c>resample</c> for a resize); none for most.</summary>
    IReadOnlyCollection<string> Keys { get; }

    /// <summary>Throws <see cref="ArgumentException"/> when <paramref name="step"/>'s value or options are not ones it takes (checked when a pipeline is parsed).</summary>
    void Check(ImageTransformStep step);

    /// <summary>The image after this step (a new image, or <paramref name="image"/> itself when nothing changes).</summary>
    ImageData Apply(ImageData image, ImageTransformStep step);
}

/// <summary>
/// One step of a pipeline: a transform's name, its value (<c>1024</c> in <c>max_width=1024</c>; null when none is given)
/// and its options (<c>resample=bicubic</c>). Immutable.
/// </summary>
public sealed class ImageTransformStep
{
    private readonly SortedDictionary<string, string> _options;

    /// <summary>The step <paramref name="name"/> with <paramref name="value"/> and <paramref name="options"/> (keys ignore case).</summary>
    public ImageTransformStep(string name, string? value = null, IEnumerable<KeyValuePair<string, string>>? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name.Trim().ToLowerInvariant();
        Value = value?.Trim();
        _options = new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, text) in options ?? [])
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(key);
            _options[key.Trim().ToLowerInvariant()] = text?.Trim() ?? throw new ArgumentException($"The option '{key}' of {Name} has no value.", nameof(options));
        }
    }

    /// <summary>The transform's name (lower case).</summary>
    public string Name { get; }

    /// <summary>The value, as written; null when the step has none.</summary>
    public string? Value { get; }

    /// <summary>The options, by key (lower case).</summary>
    public IReadOnlyDictionary<string, string> Options => _options;

    /// <summary>This step with the option <paramref name="key"/> set to <paramref name="value"/>.</summary>
    public ImageTransformStep With(string key, string value) => new(Name, Value, _options.Append(KeyValuePair.Create(key, value)));

    /// <summary>The value as a finite number; <paramref name="otherwise"/> when there is none (null: a value is required).</summary>
    /// <exception cref="ArgumentException">No value where one is required, or a value that is not a finite number.</exception>
    public double Number(double? otherwise = null) => Value is null or ""
        ? otherwise ?? throw new ArgumentException($"{Name} needs a value ({Name}=NUMBER).")
        : double.TryParse(Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v) ? v
        : throw new ArgumentException($"{Name} takes a number, not '{Value}'.");

    /// <summary>The value as a whole number; <paramref name="otherwise"/> when there is none (null: a value is required).</summary>
    /// <exception cref="ArgumentException">No value where one is required, or a value that is not a whole number.</exception>
    public int Integer(int? otherwise = null) => Value is null or ""
        ? otherwise ?? throw new ArgumentException($"{Name} needs a value ({Name}=N).")
        : int.TryParse(Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v
        : throw new ArgumentException($"{Name} takes a whole number, not '{Value}'.");

    /// <summary>The option <paramref name="key"/>, or <paramref name="otherwise"/> when not given.</summary>
    public string? Option(string key, string? otherwise = null) => _options.TryGetValue(key, out var v) ? v : otherwise;

    /// <summary>Throws unless the step has no value.</summary>
    /// <exception cref="ArgumentException">The step has a value.</exception>
    public void ThrowIfValue()
    {
        if (!string.IsNullOrEmpty(Value))
        {
            throw new ArgumentException($"{Name} takes no value (given '{Value}').");
        }
    }

    /// <summary>Throws unless every option is one of <paramref name="keys"/>, naming them.</summary>
    /// <exception cref="ArgumentException">An option is not in <paramref name="keys"/>.</exception>
    public void ThrowIfUnknownOptions(IReadOnlyCollection<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        foreach (string key in _options.Keys)
        {
            if (!keys.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException(keys.Count == 0
                    ? $"The image transform {Name} takes no options (given '{key}')."
                    : $"The image transform {Name} does not take '{key}'; it takes: {string.Join(", ", keys)}.");
            }
        }
    }

    /// <summary>As a pipeline writes it: <c>name</c>, <c>name=value</c>, then <c>,key=value</c> per option.</summary>
    public override string ToString()
    {
        var text = new StringBuilder(Name);
        if (!string.IsNullOrEmpty(Value))
        {
            text.Append('=').Append(Value);
        }

        foreach (var (key, value) in _options)
        {
            text.Append(',').Append(key).Append('=').Append(value);
        }

        return text.ToString();
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is ImageTransformStep other && other.ToString() == ToString();

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());
}

/// <summary>
/// Image transforms in the order they run, each a registered <see cref="IImageTransform"/>. Written as text,
/// comma-separated: <c>grayscale,max_width=1024,contrast=1.5</c>; an item whose name is an option of the step before it
/// (<c>max_width=1024,resample=bicubic</c>) belongs to that step. As JSON: that text, or an array whose items are such
/// texts or objects <c>{"name": "max_width", "value": 1024, "resample": "lanczos"}</c>. An unknown name or option is an
/// <see cref="ArgumentException"/> naming what is registered. Immutable; equal when their text is.
/// </summary>
public sealed class ImageTransformPipeline : IEquatable<ImageTransformPipeline>
{
    /// <summary>The steps <paramref name="steps"/>, checked against the registry.</summary>
    /// <exception cref="ArgumentException">A step names no registered transform, or one that refuses its value or options.</exception>
    public ImageTransformPipeline(IEnumerable<ImageTransformStep> steps)
    {
        ArgumentNullException.ThrowIfNull(steps);
        Steps = [.. steps];
        foreach (var step in Steps)
        {
            var transform = ImageTransforms.Get(step.Name);
            step.ThrowIfUnknownOptions(transform.Keys);
            transform.Check(step);
        }
    }

    /// <summary>No steps: images as decoded.</summary>
    public static ImageTransformPipeline Empty { get; } = new([]);

    /// <summary>The steps, in order.</summary>
    public IReadOnlyList<ImageTransformStep> Steps { get; }

    /// <summary>Whether there are no steps.</summary>
    public bool IsEmpty => Steps.Count == 0;

    /// <summary>The pipeline written as text (see the class); null, empty or "none" give <see cref="Empty"/>.</summary>
    /// <exception cref="ArgumentException">An unknown transform or option, a bad value, or an item that is not NAME or NAME=VALUE.</exception>
    public static ImageTransformPipeline Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return Empty;
        }

        var steps = new List<(string Name, string? Value, List<KeyValuePair<string, string>> Options)>();
        foreach (string raw in text.Split(','))
        {
            string item = raw.Trim();
            if (item.Length == 0)
            {
                continue;
            }

            int at = item.IndexOf('=', StringComparison.Ordinal);
            string name = (at < 0 ? item : item[..at]).Trim().ToLowerInvariant();
            string? value = at < 0 ? null : item[(at + 1)..].Trim();
            if (name.Length == 0)
            {
                throw new ArgumentException($"An image transform is NAME or NAME=VALUE (such as contrast=1.5), not '{item}'.");
            }

            if (ImageTransforms.Find(name) is not null)
            {
                steps.Add((name, value, []));
                continue;
            }

            if (steps.Count > 0 && value is not null && ImageTransforms.Get(steps[^1].Name).Keys.Contains(name, StringComparer.OrdinalIgnoreCase))
            {
                steps[^1].Options.Add(KeyValuePair.Create(name, value));
                continue;
            }

            string after = steps.Count > 0 && ImageTransforms.Get(steps[^1].Name).Keys is { Count: > 0 } keys
                ? $"; after {steps[^1].Name} it can also be one of its options: {string.Join(", ", keys)}" : "";
            throw new ArgumentException($"Unknown image transform '{name}' (registered: {string.Join(", ", ImageTransforms.Names)}{after}).");
        }

        return new ImageTransformPipeline(steps.Select(s => new ImageTransformStep(s.Name, s.Value, s.Options)));
    }

    /// <summary>
    /// The pipeline in JSON: null, a string (as <see cref="Parse"/>), or an array of strings and objects
    /// (<c>{"name", "value", ...options}</c>).
    /// </summary>
    /// <exception cref="ArgumentException">An unknown transform or option, a bad value, or JSON of another shape.</exception>
    public static ImageTransformPipeline FromJson(JsonNode? json)
    {
        switch (json)
        {
            case null:
                return Empty;
            case JsonValue value when value.GetValueKind() == JsonValueKind.String:
                return Parse(value.GetValue<string>());
            case JsonArray array:
            {
                var steps = new List<ImageTransformStep>();
                foreach (var item in array)
                {
                    if (item is JsonValue text && text.GetValueKind() == JsonValueKind.String)
                    {
                        steps.AddRange(Parse(text.GetValue<string>()).Steps);
                    }
                    else if (item is JsonObject step)
                    {
                        string name = step["name"] is JsonValue n && n.GetValueKind() == JsonValueKind.String ? n.GetValue<string>()
                            : throw new ArgumentException($"An image transform object needs a \"name\": {step.ToJsonString()}.");
                        var options = step.Where(p => p.Key is not "name" and not "value" && p.Value is not null)
                            .Select(p => KeyValuePair.Create(p.Key, Text(p.Value!, p.Key))).ToList();
                        if (ImageTransforms.Find(name) is null)
                        {
                            throw new ArgumentException($"Unknown image transform '{name}' (registered: {string.Join(", ", ImageTransforms.Names)}).");
                        }

                        steps.Add(new ImageTransformStep(name, step["value"] is { } v ? Text(v, "value") : null, options));
                    }
                    else
                    {
                        throw new ArgumentException($"An image transform is a string or an object, not {item?.ToJsonString() ?? "null"}.");
                    }
                }

                return new ImageTransformPipeline(steps);
            }

            default:
                throw new ArgumentException($"Image transforms are a string or an array, not {json.ToJsonString()}.");
        }

        static string Text(JsonNode node, string key) => node.GetValueKind() switch
        {
            JsonValueKind.String => node.GetValue<string>(),
            JsonValueKind.Number => node.ToJsonString(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            var kind => throw new ArgumentException($"The image transform's '{key}' is {kind.ToString().ToLowerInvariant()}: give a string, a number or true/false."),
        };
    }

    /// <summary>This pipeline's steps, then <paramref name="next"/>'s.</summary>
    public ImageTransformPipeline Then(ImageTransformPipeline? next) =>
        next is null || next.IsEmpty ? this : IsEmpty ? next : new ImageTransformPipeline(Steps.Concat(next.Steps));

    /// <summary>Whether a step is the transform <paramref name="name"/>.</summary>
    public bool Contains(string name) => Steps.Any(s => s.Name.Equals(name, StringComparison.OrdinalIgnoreCase));

    /// <summary>The image after every step, in order (the registered transforms at the time of the call).</summary>
    public ImageData Apply(ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        foreach (var step in Steps)
        {
            image = ImageTransforms.Get(step.Name).Apply(image, step);
        }

        return image;
    }

    /// <summary>The pipeline as JSON text (a string; null when empty).</summary>
    public JsonNode? ToJson() => IsEmpty ? null : JsonValue.Create(ToString());

    /// <summary>The steps written as text, comma-separated; "" when empty.</summary>
    public override string ToString() => string.Join(",", Steps);

    /// <inheritdoc />
    public bool Equals(ImageTransformPipeline? other) => other is not null && other.ToString() == ToString();

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as ImageTransformPipeline);

    /// <inheritdoc />
    public override int GetHashCode() => StringComparer.Ordinal.GetHashCode(ToString());
}

/// <summary>
/// The image transforms a pipeline can name, by name (ignoring case). The library's (core registers them as library
/// defaults, Pillow's operations byte for byte): <c>grayscale</c>, <c>max_width</c>, <c>max_height</c>,
/// <c>contrast</c>, <c>brightness</c>, <c>sharpness</c>, <c>autocontrast</c>, <c>invert</c>, <c>jpeg</c>. Register another with
/// <see cref="Register"/>; one of a library name shadows the library's, which <see cref="Unregister"/> brings back.
/// </summary>
public static class ImageTransforms
{
    private static readonly SlotTable<string, IImageTransform> Table =
        new(nameof(ImageTransforms), (slot, app, library) => new GuardedTransform(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static ImageTransforms() => LibraryDefaults.Ensure(typeof(ImageTransforms));

    /// <summary>Registers <paramref name="transform"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImageTransform transform)
    {
        ArgumentNullException.ThrowIfNull(transform);
        ArgumentException.ThrowIfNullOrWhiteSpace(transform.Name);
        if (transform.Name.Contains(',') || transform.Name.Contains('='))
        {
            throw new ArgumentException($"An image transform's name has no ',' or '=' ('{transform.Name}').", nameof(transform));
        }

        Table.Register(transform.Name, transform, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's transform <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The transform registered as <paramref name="name"/>, or null.</summary>
    public static IImageTransform? Find(string name) => Table.Find(name);

    /// <summary>The transform registered as <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">None is: the message names the registered ones.</exception>
    public static IImageTransform Get(string name) => Table.Find(name)
        ?? throw new ArgumentException($"Unknown image transform '{name}' (registered: {string.Join(", ", Names)}); add one with ImageTransforms.Register.");

    /// <summary>The library's transform <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImageTransform? Default(string name) => Table.Default(name);

    /// <summary>Who registered the transform <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's transform <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set; <see cref="SlotPolicy.FallBack"/> retries on the library's).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered transform as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(t => $"{t.Name}: {t.Summary}"));

    private sealed class GuardedTransform(Slot slot, IImageTransform app, IImageTransform library) : IImageTransform
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyCollection<string> Keys => app.Keys;

        public void Check(ImageTransformStep step) => app.Check(step);

        public ImageData Apply(ImageData image, ImageTransformStep step) => slot.Call(() => app.Apply(image, step), () => library.Apply(image, step),
            (a, b) => Comparisons.Exact((a.Channels, a.Height, a.Width), (b.Channels, b.Height, b.Width)) ?? Comparisons.Difference(a.Pixels, b.Pixels, 1e-6f));
    }
}
