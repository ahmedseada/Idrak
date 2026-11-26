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
/// re-encode it): decoded pixels in, decoded pixels out. The library's are Pillow's operations, byte for byte (core
/// registers them): <c>grayscale</c>, <c>max_width</c>, <c>max_height</c>, <c>scale</c>, <c>pad</c>, <c>contrast</c>,
/// <c>brightness</c>, <c>sharpness</c>, <c>autocontrast</c>, <c>equalize</c>, <c>gamma</c>, <c>blur</c>,
/// <c>unsharp</c>, <c>median</c>, <c>min_filter</c>, <c>max_filter</c>, <c>binarize</c>, <c>invert</c>, <c>jpeg</c>;
/// and scan degradations for pages made to look scanned (Idrak.Imaging): <c>rotate</c>, <c>perspective</c>,
/// <c>noise</c>, <c>speckle</c>, <c>dropout</c>, <c>fade</c>, <c>tint</c>, <c>stain</c>, <c>lowres</c>, <c>bleed</c>,
/// and what happens to a paper: <c>wrinkle</c>, <c>fold</c>, <c>smudge</c>, <c>blot</c>, <c>strike</c>,
/// <c>underline</c>, <c>cross</c>, <c>scribble</c>, <c>highlight</c>, <c>tear</c>, <c>holes</c>, <c>scratch</c>,
/// <c>shadow</c>, <c>streak</c>, <c>light</c>, <c>curl</c> (mixed by <see cref="ScanProfiles"/>). Each describes its
/// inputs (<see cref="Parameters"/>), so a program can offer them without knowing the transform. A transform knows
/// nothing of models: a fine-tune that wants its scans prepared a certain way is given a pipeline by the application.
/// </summary>
/// <remarks>
/// Its <c>Name</c> is what a pipeline uses (<c>contrast</c>; names compare ignoring case), its <c>Summary</c> one line on what it does and what its value is, and its <c>Keys</c>
/// the option keys it takes besides its value (<c>resample</c> for a resize; none for most).
/// </remarks>
public interface IImageTransform
{
    /// <summary>Throws <see cref="ArgumentException"/> when <paramref name="step"/>'s value or options are not ones it takes (checked when a pipeline is parsed).</summary>
    void Check(ImageTransformStep step);

    /// <summary>The image after this step (a new image, or <paramref name="image"/> itself when nothing changes).</summary>
    ImageData Apply(ImageData image, ImageTransformStep step);

    /// <summary>
    /// Its inputs, for a program that offers them as controls: the value (key "") and each option, with its kind and
    /// range (the values to start from are the program's); none unless the transform describes them.
    /// </summary>
    IReadOnlyList<ImageTransformParameter> Parameters => [];
}

/// <summary>What kind of input an image transform's value or option is.</summary>
public enum ImageTransformInput
{
    /// <summary>A number (a factor, a radius).</summary>
    Number,

    /// <summary>A whole number (pixels, a size, a percentage).</summary>
    Integer,

    /// <summary>One of <see cref="ImageTransformParameter.Choices"/>.</summary>
    Choice,
}

/// <summary>
/// One input of an image transform (<see cref="IImageTransform.Parameters"/>): its value (<see cref="Key"/> "") or an
/// option, how it is entered and its range. What value to start from is the program's choice, not the transform's.
/// </summary>
/// <param name="Key">"" for the step's value (<c>contrast=1.5</c>), else the option's key (<c>resample</c>).</param>
/// <param name="Label">A short name for people.</param>
/// <param name="Input">How it is entered.</param>
public sealed record ImageTransformParameter(string Key, string Label, ImageTransformInput Input)
{
    /// <summary>The smallest value taken, for numbers.</summary>
    public double? Min { get; init; }

    /// <summary>The largest value taken, for numbers.</summary>
    public double? Max { get; init; }

    /// <summary>A sensible increment for a number control.</summary>
    public double? Step { get; init; }

    /// <summary>The values taken, for <see cref="ImageTransformInput.Choice"/>.</summary>
    public IReadOnlyList<string> Choices { get; init; } = [];

    /// <summary>Whether it may be left out (the transform then does what its summary says, Otsu's threshold for binarize).</summary>
    public bool Optional { get; init; }

    /// <summary>One line on what it changes.</summary>
    public string? Help { get; init; }
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
/// The image transforms a pipeline can name, by name (ignoring case). The library's (core registers them as library
/// defaults, Pillow's operations byte for byte): <c>grayscale</c>, <c>max_width</c>, <c>max_height</c>,
/// <c>contrast</c>, <c>brightness</c>, <c>sharpness</c>, <c>autocontrast</c>, <c>invert</c>, <c>jpeg</c>.
/// </summary>
public static class ImageTransforms
{
}
