// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// A random change to a detection or segmentation sample that keeps its labels true: the image changes and the boxes and
/// masks move with its pixels (a flip, a crop, a rotation), or only the colours change. Deterministic: the same sample
/// and the same random state give the same result. Create the library's by name with <see cref="Augmentations"/>.
/// </summary>
public interface IAugmentation
{
    /// <summary>Its name (<c>flip</c>).</summary>
    string Name { get; }

    /// <summary>
    /// The sample changed (a new sample; <paramref name="sample"/> itself when nothing changes). Draw random numbers from
    /// <paramref name="context"/>'s <see cref="AugmentationContext.Random"/> only, and other samples (a mosaic, a mix) from
    /// <see cref="AugmentationContext.Draw"/>.
    /// </summary>
    AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context);
}

/// <summary>What an augmentation draws from: its random numbers and, for those that combine samples, other samples.</summary>
/// <param name="random">The random numbers (seeded per sample by a loader, so a run repeats exactly).</param>
/// <param name="draw">Gives another sample of the data set, chosen with the random numbers given; null when there is none.</param>
public sealed class AugmentationContext(Random random, Func<Random, AnnotatedImage>? draw = null)
{
    /// <summary>The random numbers.</summary>
    public Random Random { get; } = random ?? throw new ArgumentNullException(nameof(random));

    /// <summary>Whether other samples can be drawn.</summary>
    public bool CanDraw => draw is not null;

    /// <summary>Another sample of the data set, chosen with <see cref="Random"/> (as it is, before augmentation).</summary>
    /// <exception cref="InvalidOperationException">There is no data set to draw from.</exception>
    public AnnotatedImage Draw() => draw is null
        ? throw new InvalidOperationException("This augmentation combines samples (mosaic, mixup): run it where other samples can be drawn, such as Idrak.Vision's AugmentedImageLoader.")
        : draw(Random);

    /// <summary>A context with a new random state seeded by <paramref name="seed"/> and the same samples to draw.</summary>
    public AugmentationContext WithSeed(int seed) => new(new Random(seed), draw);
}

/// <summary>
/// The options an augmentation is created with, by name (ignoring case), as text: numbers, whole numbers, true/false and
/// ranges written <c>low:high</c>. Typed reads name the augmentation and the option when a value is not of the kind asked.
/// </summary>
public sealed class AugmentationOptions
{
    private readonly Dictionary<string, string> _values;

    /// <summary>The options <paramref name="values"/> of the augmentation <paramref name="name"/>.</summary>
    public AugmentationOptions(string name, IEnumerable<KeyValuePair<string, string>>? values = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        _values = new(StringComparer.OrdinalIgnoreCase);
        foreach (var (key, value) in values ?? [])
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(key);
            _values[key.Trim()] = value?.Trim() ?? throw new ArgumentException($"The option '{key}' of {name} has no value.", nameof(values));
        }
    }

    /// <summary>The augmentation's name.</summary>
    public string Name { get; }

    /// <summary>The options as given.</summary>
    public IReadOnlyDictionary<string, string> Values => _values;

    /// <summary>Throws unless every option is one of <paramref name="keys"/>.</summary>
    /// <exception cref="ArgumentException">An option it does not take: the message lists those it does.</exception>
    public AugmentationOptions Allow(params string[] keys)
    {
        foreach (string key in _values.Keys.Where(k => !keys.Contains(k, StringComparer.OrdinalIgnoreCase)))
        {
            throw new ArgumentException(keys.Length == 0 ? $"The augmentation {Name} takes no options (given '{key}')."
                : $"The augmentation {Name} does not take '{key}'; it takes: {string.Join(", ", keys)}.");
        }

        return this;
    }

    /// <summary>The option <paramref name="key"/> as a finite number, or <paramref name="otherwise"/>.</summary>
    public double Number(string key, double otherwise) => _values.TryGetValue(key, out var text) ? Parse(key, text) : otherwise;

    /// <summary>The option <paramref name="key"/> as a whole number, or <paramref name="otherwise"/>.</summary>
    public int Integer(string key, int otherwise) => !_values.TryGetValue(key, out var text) ? otherwise
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) ? value
        : throw new ArgumentException($"The option '{key}' of {Name} is a whole number, not '{text}'.");

    /// <summary>The option <paramref name="key"/> as true or false, or <paramref name="otherwise"/>.</summary>
    public bool Flag(string key, bool otherwise) => !_values.TryGetValue(key, out var text) ? otherwise
        : bool.TryParse(text, out bool value) ? value : text is "1" or "0" ? text == "1"
        : throw new ArgumentException($"The option '{key}' of {Name} is true or false, not '{text}'.");

    /// <summary>The option <paramref name="key"/> as a range <c>low:high</c> (one number <c>v</c> is the range v:v), or <paramref name="otherwise"/>.</summary>
    public (double Low, double High) Range(string key, (double Low, double High) otherwise)
    {
        if (!_values.TryGetValue(key, out var text))
        {
            return otherwise;
        }

        int colon = text.IndexOf(':', StringComparison.Ordinal);
        var range = colon < 0 ? (Parse(key, text), Parse(key, text)) : (Parse(key, text[..colon]), Parse(key, text[(colon + 1)..]));
        return range.Item1 <= range.Item2 ? range : throw new ArgumentException($"The range '{key}' of {Name} runs from low to high, not '{text}'.");
    }

    private double Parse(string key, string text) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) ? value
            : throw new ArgumentException($"The option '{key}' of {Name} is a number, not '{text}'.");

    /// <summary>As a pipeline writes it: <c>name</c> or <c>name(key=value, ...)</c>.</summary>
    public override string ToString() => _values.Count == 0 ? Name
        : $"{Name}({string.Join(", ", _values.OrderBy(p => p.Key, StringComparer.OrdinalIgnoreCase).Select(p => $"{p.Key}={p.Value}"))})";
}

/// <summary>
/// A kind of augmentation: the augmentation for its options.
/// </summary>
public interface IAugmentationKind
{
    /// <summary>The augmentation for <paramref name="options"/>, refusing those it does not take (see <see cref="AugmentationOptions.Allow"/>).</summary>
    IAugmentation Create(AugmentationOptions options);
}

/// <summary>
/// Creates an augmentation from its options (refusing those it does not take, see <see cref="AugmentationOptions.Allow"/>): a kind as a function.
/// </summary>
public delegate IAugmentation AugmentationFactory(AugmentationOptions options);

/// <summary>
/// Runs a pipeline of augmentations of detection and segmentation samples (<see cref="Apply"/>). The library's:
/// <c>flip</c> and <c>shift</c> (core's <c>RandomFlip</c> and <c>RandomShift</c>), and from Idrak.Vision
/// <c>resized-crop</c>, <c>rotation</c>, <c>affine</c>, <c>color-jitter</c>, <c>cutout</c>, <c>mosaic</c>, <c>mixup</c>
/// and <c>resize</c>. Picking them by their names (a pipeline written as text) is not on the new design yet.
/// </summary>
public static class Augmentations
{
    /// <summary>The sample after every augmentation of <paramref name="pipeline"/>, in order.</summary>
    public static AnnotatedImage Apply(IReadOnlyList<IAugmentation> pipeline, AnnotatedImage sample, AugmentationContext context)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(sample);
        ArgumentNullException.ThrowIfNull(context);
        foreach (var augmentation in pipeline)
        {
            sample = augmentation.Apply(sample, context);
        }

        return sample;
    }
}
