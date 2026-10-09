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
    /// <summary>The name it is registered under (<c>flip</c>).</summary>
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

/// <summary>Creates an augmentation from its options (refusing those it does not take, see <see cref="AugmentationOptions.Allow"/>).</summary>
public delegate IAugmentation AugmentationFactory(AugmentationOptions options);

/// <summary>
/// The augmentations of detection and segmentation samples, by name (ignoring case), each a factory taking its options.
/// The library's (registered by the assemblies that hold them, as library defaults): <c>flip</c> and <c>shift</c> (core's
/// <c>RandomFlip</c> and <c>RandomShift</c>), and from Idrak.Vision <c>resized-crop</c>, <c>rotation</c>, <c>affine</c>,
/// <c>color-jitter</c>, <c>cutout</c>, <c>mosaic</c>, <c>mixup</c> and <c>resize</c>. Register another with
/// <see cref="Register"/>; one of a library name shadows the library's, which <see cref="Unregister"/> brings back.
/// A pipeline is written <c>flip, rotation(degrees=10), resized-crop(width=320, height=320, scale=0.5:1)</c>
/// (<see cref="Parse"/>).
/// </summary>
public static class Augmentations
{
    private static readonly SlotTable<string, AugmentationFactory> Table =
        new(nameof(Augmentations), Guard, StringComparer.OrdinalIgnoreCase);

    static Augmentations() => LibraryDefaults.Ensure(typeof(Augmentations));

    /// <summary>Registers the augmentation <paramref name="name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, AugmentationFactory create)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(create);
        if (name.IndexOfAny([',', '(', ')', '=']) >= 0)
        {
            throw new ArgumentException($"An augmentation's name has no ',', '(', ')' or '=' ('{name}').", nameof(name));
        }

        Table.Register(name, create, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's augmentation <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The factory registered as <paramref name="name"/>.</summary>
    /// <exception cref="ArgumentException">None is: the message names the registered ones.</exception>
    public static AugmentationFactory Get(string name) => Table.Find(name)
        ?? throw new ArgumentException($"Unknown augmentation '{name}' (registered: {string.Join(", ", Names)}); add one with Augmentations.Register.");

    /// <summary>The library's factory <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static AugmentationFactory? Default(string name) => Table.Default(name);

    /// <summary>Who registered the augmentation <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's augmentation <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's with the same random state), or whether it only runs beside
    /// the library's (<see cref="SlotPolicy.Shadow"/>: the library's answers, the boxes and labels are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>The augmentation <paramref name="name"/> with <paramref name="options"/> (as text, by name).</summary>
    public static IAugmentation Create(string name, IEnumerable<KeyValuePair<string, string>>? options = null) =>
        Get(name)(new AugmentationOptions(name, options));

    /// <summary>
    /// The augmentations of a pipeline written as text: comma-separated items, each <c>name</c> or
    /// <c>name(key=value, ...)</c>. Null, empty or "none" give none.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown augmentation or option, or an item of another shape.</exception>
    public static IReadOnlyList<IAugmentation> Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        var result = new List<IAugmentation>();
        foreach (string item in SplitTop(text))
        {
            int open = item.IndexOf('(', StringComparison.Ordinal);
            string name = (open < 0 ? item : item[..open]).Trim();
            var options = new List<KeyValuePair<string, string>>();
            if (open >= 0)
            {
                if (!item.EndsWith(')'))
                {
                    throw new ArgumentException($"An augmentation is NAME or NAME(key=value, ...), not '{item}'.");
                }

                foreach (string option in SplitTop(item[(open + 1)..^1]))
                {
                    int equals = option.IndexOf('=', StringComparison.Ordinal);
                    if (equals <= 0)
                    {
                        throw new ArgumentException($"The option '{option}' of {name} is not key=value.");
                    }

                    options.Add(KeyValuePair.Create(option[..equals].Trim(), option[(equals + 1)..].Trim()));
                }
            }

            if (name.Length == 0)
            {
                throw new ArgumentException($"An augmentation is NAME or NAME(key=value, ...), not '{item}'.");
            }

            result.Add(Create(name, options));
        }

        return result;
    }

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

    // Comma-separated items, commas inside parentheses kept.
    private static List<string> SplitTop(string text)
    {
        var items = new List<string>();
        var current = new StringBuilder();
        int depth = 0;
        foreach (char ch in text)
        {
            depth += ch == '(' ? 1 : ch == ')' ? -1 : 0;
            if (depth < 0)
            {
                throw new ArgumentException($"Unbalanced parentheses in '{text}'.");
            }

            if (ch == ',' && depth == 0)
            {
                items.Add(current.ToString().Trim());
                current.Clear();
            }
            else
            {
                current.Append(ch);
            }
        }

        if (depth != 0)
        {
            throw new ArgumentException($"Unbalanced parentheses in '{text}'.");
        }

        items.Add(current.ToString().Trim());
        return [.. items.Where(i => i.Length > 0)];
    }

    // An app's factory under its policy: an app augmentation that cannot be created is a failure like any other (the
    // library's is created instead under FallBack and Shadow).
    private static AugmentationFactory Guard(Slot slot, AugmentationFactory app, AugmentationFactory library) => options =>
    {
        IAugmentation mine;
        try
        {
            mine = app(options);
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            if (slot.Failed(e) || slot.Policy == SlotPolicy.Shadow)
            {
                return library(options);
            }

            throw;
        }

        return new Guarded(slot, mine, options, library);
    };

    // An app's augmentation under its policy: both sides get a random state from the same seed, so a fall-back or a shadow
    // comparison sees the draws the app saw.
    private sealed class Guarded(Slot slot, IAugmentation app, AugmentationOptions options, AugmentationFactory library) : IAugmentation
    {
        private IAugmentation? _library;

        public string Name => app.Name;

        public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
        {
            int seed = context.Random.Next();
            return slot.Call(() => app.Apply(sample, context.WithSeed(seed)), () => (_library ??= library(options)).Apply(sample, context.WithSeed(seed)), Compare);
        }

        private static string? Compare(AnnotatedImage expected, AnnotatedImage actual) =>
            Comparisons.Exact((expected.Width, expected.Height, expected.Count), (actual.Width, actual.Height, actual.Count))
            ?? Comparisons.Difference(expected.Labels, actual.Labels)
            ?? Comparisons.Difference([.. expected.Boxes.SelectMany(b => new[] { b.X, b.Y, b.Width, b.Height })],
                [.. actual.Boxes.SelectMany(b => new[] { b.X, b.Y, b.Width, b.Height })], 1e-3f);
    }
}
